"""Camera-to-camera relative pose estimation for the photo tier (Fix for Issue #1).

Replaces the blind np.vstack of per-photo point clouds with a proper pairwise
registration pipeline:

  1. ORB feature extraction on each image
  2. BFMatcher cross-check to find correspondences
  3. Essential matrix estimation with RANSAC (using per-image intrinsics)
  4. Rotation + translation decomposition
  5. Full chain composition: T_world_from_cam_i = T_world_from_cam_{i-1} @ T_{i-1}_from_i
  6. Optionally: ICP fine-alignment on overlapping point clouds

Design constraints:
  - Pure OpenCV + NumPy – no optional GTSAM dependency required
  - Graceful fallback: if fewer than MIN_MATCHES correspondences are found
    between a pair of images, the transform is taken as identity (the
    points are stacked at the previous camera position rather than at
    the origin), which degrades gracefully.
  - Every recovered pose is tagged with a confidence flag so callers can
    decide how to weight or skip low-confidence estimates.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import cv2

# Minimum inlier matches required to trust an Essential-matrix decomposition.
MIN_MATCHES_FOR_ESSENTIAL = 12
MIN_INLIERS_FOR_ESSENTIAL = 8

# ICP parameters
ICP_MAX_ITERATIONS = 30
ICP_TOLERANCE = 1e-4
ICP_MAX_CORR_DIST = 0.25  # metres


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def _load_gray(image_path: Path | str) -> Optional[np.ndarray]:
    """Load image as grayscale; return None on failure."""
    try:
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None or img.size == 0:
            return None
        # Resize to max 800px wide for speed – ratios preserved
        h, w = img.shape
        if w > 800:
            scale = 800.0 / w
            img = cv2.resize(img, (800, int(h * scale)), interpolation=cv2.INTER_AREA)
        return img
    except Exception:
        return None


def _extract_orb(gray: np.ndarray, n_features: int = 1000) -> Tuple[List, Optional[np.ndarray]]:
    """Extract ORB keypoints and descriptors from a grayscale image."""
    orb = cv2.ORB_create(nfeatures=n_features, scaleFactor=1.2, nlevels=8)
    kps, descs = orb.detectAndCompute(gray, None)
    return kps, descs


# ---------------------------------------------------------------------------
# Correspondence matching
# ---------------------------------------------------------------------------

def _match_features(
    descs_a: np.ndarray,
    descs_b: np.ndarray,
    ratio_threshold: float = 0.75,
) -> List[cv2.DMatch]:
    """Lowe-ratio-test match between two ORB descriptor arrays."""
    if descs_a is None or descs_b is None:
        return []
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    try:
        raw = bf.knnMatch(descs_a, descs_b, k=2)
    except Exception:
        return []
    good: List[cv2.DMatch] = []
    for pair in raw:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio_threshold * n.distance:
            good.append(m)
    return good


# ---------------------------------------------------------------------------
# Essential matrix decomposition
# ---------------------------------------------------------------------------

def _build_K_scaled(
    intrinsics: Dict[str, float],
    original_w: float,
    gray_w: int,
) -> np.ndarray:
    """Build 3×3 camera matrix scaled to the processing resolution."""
    scale = gray_w / original_w
    fx = intrinsics["fx"] * scale
    fy = intrinsics["fy"] * scale
    cx = intrinsics["cx"] * scale
    cy = intrinsics["cy"] * scale
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


def estimate_relative_pose_essential(
    gray_a: np.ndarray,
    gray_b: np.ndarray,
    intrinsics_a: Dict[str, float],
    intrinsics_b: Dict[str, float],
    original_w_a: float,
    original_w_b: float,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Estimate relative rotation R and translation t from image A to image B.

    Returns:
        R     : 3×3 rotation matrix (world-from-a to world-from-b)
        t     : unit 3-vector translation direction (scale ambiguous)
        conf  : confidence in [0, 1] – 0 means fallback identity was used
    """
    kps_a, desc_a = _extract_orb(gray_a)
    kps_b, desc_b = _extract_orb(gray_b)
    matches = _match_features(desc_a, desc_b)

    if len(matches) < MIN_MATCHES_FOR_ESSENTIAL:
        return np.eye(3), np.zeros(3), 0.0

    pts_a = np.float64([kps_a[m.queryIdx].pt for m in matches])
    pts_b = np.float64([kps_b[m.trainIdx].pt for m in matches])

    Ka = _build_K_scaled(intrinsics_a, original_w_a, gray_a.shape[1])
    Kb = _build_K_scaled(intrinsics_b, original_w_b, gray_b.shape[1])

    # Use average K if both are close (same camera)
    K_avg = (Ka + Kb) / 2.0

    try:
        E, inlier_mask = cv2.findEssentialMat(
            pts_a, pts_b, K_avg,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.0,
        )
    except Exception:
        return np.eye(3), np.zeros(3), 0.0

    if E is None or inlier_mask is None:
        return np.eye(3), np.zeros(3), 0.0

    n_inliers = int(inlier_mask.sum())
    if n_inliers < MIN_INLIERS_FOR_ESSENTIAL:
        return np.eye(3), np.zeros(3), 0.0

    _, R, t, _ = cv2.recoverPose(E, pts_a, pts_b, K_avg, mask=inlier_mask)
    conf = min(1.0, n_inliers / 50.0)

    return R, t.flatten(), conf


def estimate_relative_pose_homography(
    gray_a: np.ndarray,
    gray_b: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """Fallback rotation-only estimate using homography (for mostly-planar scenes).

    Returns:
        R    : approximate 3×3 rotation matrix
        conf : confidence in [0, 1]
    """
    kps_a, desc_a = _extract_orb(gray_a)
    kps_b, desc_b = _extract_orb(gray_b)
    matches = _match_features(desc_a, desc_b)

    if len(matches) < 8:
        return np.eye(3), 0.0

    pts_a = np.float64([kps_a[m.queryIdx].pt for m in matches])
    pts_b = np.float64([kps_b[m.trainIdx].pt for m in matches])

    try:
        H, mask = cv2.findHomography(pts_a, pts_b, cv2.RANSAC, 3.0)
    except Exception:
        return np.eye(3), 0.0

    if H is None:
        return np.eye(3), 0.0

    # Extract rotation from homography (rough approximation)
    # Normalise the first two columns to get a rotation-like matrix
    r1 = H[:, 0] / (np.linalg.norm(H[:, 0]) + 1e-9)
    r2 = H[:, 1] / (np.linalg.norm(H[:, 1]) + 1e-9)
    r3 = np.cross(r1, r2)
    R_raw = np.column_stack([r1, r2, r3])

    # Project onto SO(3) via SVD
    U, _, Vt = np.linalg.svd(R_raw)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        R = -R

    n_inliers = int(mask.sum()) if mask is not None else 0
    conf = min(0.5, n_inliers / 40.0)  # homography gives less confident rotation
    return R, conf


# ---------------------------------------------------------------------------
# Scale recovery from depth cloud pairing
# ---------------------------------------------------------------------------

def recover_scale_from_architecture(
    pts_3d: np.ndarray,
    ceiling_height_prior: float = 2.50,
    floor_z_prior: float = 0.0,
) -> float:
    """Estimate a global metric scale factor by comparing detected ceiling height
    to the architectural prior.

    If the ceiling is clearly identifiable (top 5th-percentile of z), the ratio
    of prior_height / detected_height gives the scale.

    Returns:
        scale : float >= 0.5 clamped to [0.5, 2.0] to prevent runaway corrections.
    """
    if len(pts_3d) < 50:
        return 1.0

    z_lo = float(np.percentile(pts_3d[:, 2], 2))
    z_hi = float(np.percentile(pts_3d[:, 2], 97))
    detected_height = z_hi - z_lo

    if detected_height < 0.5:
        return 1.0

    scale = ceiling_height_prior / detected_height
    return float(np.clip(scale, 0.5, 2.0))


# ---------------------------------------------------------------------------
# ICP fine-alignment
# ---------------------------------------------------------------------------

def _nearest_neighbour_dists(src: np.ndarray, dst: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Nearest-neighbour search using cKDTree with fallback.
    Returns (distances_squared, indices_into_dst).
    """
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(dst)
        dists, idx = tree.query(src)
        return dists ** 2, idx
    except Exception:
        best_dist = np.full(len(src), np.inf)
        best_idx = np.zeros(len(src), dtype=np.int64)
        chunk = 256
        for i in range(0, len(src), chunk):
            s_chunk = src[i:i + chunk]
            diff = dst[np.newaxis, :, :] - s_chunk[:, np.newaxis, :]  # (C, N, 3)
            d2 = np.sum(diff ** 2, axis=2)  # (C, N)
            idx = np.argmin(d2, axis=1)
            dist = d2[np.arange(len(s_chunk)), idx]
            best_dist[i:i + chunk] = dist
            best_idx[i:i + chunk] = idx
        return best_dist, best_idx


def icp_align(
    source: np.ndarray,
    target: np.ndarray,
    max_iterations: int = ICP_MAX_ITERATIONS,
    tolerance: float = ICP_TOLERANCE,
    max_correspondence_dist: float = ICP_MAX_CORR_DIST,
    subsample: int = 500,
) -> Tuple[np.ndarray, float]:
    """Point-to-point ICP.

    Args:
        source: (N, 3) point cloud to align.
        target: (M, 3) reference point cloud.
        subsample: randomly sample this many source points per iteration.

    Returns:
        T : 4×4 rigid transformation (source → target frame)
        rmse: final root-mean-square residual (metres)
    """
    T_total = np.eye(4)
    src = source.copy()

    rng = np.random.default_rng(42)

    # Subsample target reference if huge to keep tree construction and search instantaneous
    if len(target) > 5000:
        t_sub_idx = rng.choice(len(target), 5000, replace=False)
        target_ref = target[t_sub_idx]
    else:
        target_ref = target

    tree = None
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(target_ref)
    except Exception:
        pass

    prev_rmse = np.inf
    for _ in range(max_iterations):
        # Subsample for speed
        idx = rng.choice(len(src), min(subsample, len(src)), replace=False)
        src_sub = src[idx]

        if tree is not None:
            dists_raw, nn_idx = tree.query(src_sub)
            dists = dists_raw ** 2
        else:
            dists, nn_idx = _nearest_neighbour_dists(src_sub, target_ref)

        valid = dists < max_correspondence_dist ** 2
        if valid.sum() < 6:
            break

        s_pts = src_sub[valid]
        t_pts = target_ref[nn_idx[valid]]

        # Solve for R, t via SVD (Procrustes)
        mu_s = s_pts.mean(axis=0)
        mu_t = t_pts.mean(axis=0)
        H = (s_pts - mu_s).T @ (t_pts - mu_t)
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1] *= -1
            R = Vt.T @ U.T
        t_vec = mu_t - R @ mu_s

        # Build 4×4 delta transform
        dT = np.eye(4)
        dT[:3, :3] = R
        dT[:3, 3] = t_vec

        T_total = dT @ T_total
        src = (R @ src.T).T + t_vec

        rmse = float(np.sqrt(np.mean(dists[valid])))
        if abs(prev_rmse - rmse) < tolerance:
            break
        prev_rmse = rmse

    return T_total, prev_rmse


# ---------------------------------------------------------------------------
# Main photo-tier registration chain
# ---------------------------------------------------------------------------

def register_photo_sequence(
    image_paths: List[Path | str],
    per_image_intrinsics: List[Dict[str, float]],
    per_image_point_clouds: List[np.ndarray],
    ceiling_height_prior: float = 2.50,
    use_icp: bool = True,
) -> Tuple[np.ndarray, List[Dict]]:
    """Register a sequence of per-image point clouds into a common coordinate frame.

    Algorithm:
        1. For each consecutive image pair (i-1, i), estimate R_i_from_{i-1}
           and translation direction t (scale-ambiguous).
        2. Recover scale by matching architectural ceiling height to prior.
        3. Compose transforms into the world frame.
        4. Optionally run ICP to fine-align adjacent clouds.
        5. Return a single unified (N, 3) point cloud.

    Args:
        image_paths         : ordered list of image file paths
        per_image_intrinsics: corresponding intrinsics dicts {fx, fy, cx, cy, width, height}
        per_image_point_clouds: corresponding (Ni, 3) arrays in camera-local coords
        ceiling_height_prior: used for scale recovery
        use_icp             : whether to refine with point-to-point ICP

    Returns:
        unified_cloud : (N_total, 3) registered point cloud in world frame
        pose_log      : list of per-image dicts {R, t_world, conf, scale}
    """
    n = len(image_paths)
    if n == 0:
        return np.empty((0, 3)), []

    pose_log: List[Dict] = []

    # Anchor: first camera is at world origin, pointing in +Z
    T_world_from = [np.eye(4)]  # 4×4 homogeneous transforms, T_world_from_cam_i

    # Pre-load grayscale images
    grays: List[Optional[np.ndarray]] = [_load_gray(p) for p in image_paths]

    # Step 1: Pairwise relative pose estimation
    for i in range(1, n):
        g_prev = grays[i - 1]
        g_curr = grays[i]
        intr_prev = per_image_intrinsics[i - 1]
        intr_curr = per_image_intrinsics[i]

        if g_prev is not None and g_curr is not None:
            R_rel, t_unit, conf = estimate_relative_pose_essential(
                g_prev, g_curr,
                intr_prev, intr_curr,
                intr_prev.get("width", 1920.0),
                intr_curr.get("width", 1920.0),
            )

            if conf < 0.15:
                # Try homography fallback
                R_rel, conf_hom = estimate_relative_pose_homography(g_prev, g_curr)
                t_unit = np.array([0.0, 0.0, 0.0])
                conf = conf_hom * 0.5
        else:
            R_rel, t_unit, conf = np.eye(3), np.zeros(3), 0.0

        # Step 2: recover scale for the current cloud
        pts_curr = per_image_point_clouds[i]
        scale = recover_scale_from_architecture(pts_curr, ceiling_height_prior)

        # Translation magnitude from scale (camera moves ~0.4m between typical stills)
        typical_baseline_m = 0.4
        t_world = R_rel @ t_unit * (typical_baseline_m * scale)

        # Build 4×4 relative transform
        T_rel = np.eye(4)
        T_rel[:3, :3] = R_rel
        T_rel[:3, 3] = t_world

        T_world_i = T_world_from[i - 1] @ T_rel
        T_world_from.append(T_world_i)

        pose_log.append({
            "image": str(image_paths[i]),
            "R": R_rel.tolist(),
            "t_world": t_world.tolist(),
            "conf": round(conf, 3),
            "scale": round(scale, 4),
        })

    # Step 3: transform all clouds to world frame
    cloud_chunks: List[np.ndarray] = []

    for i, (T, pts) in enumerate(zip(T_world_from, per_image_point_clouds)):
        if len(pts) == 0:
            continue

        R_w = T[:3, :3]
        t_w = T[:3, 3]

        # Apply architectural scale correction to the cloud
        scale = (
            recover_scale_from_architecture(pts, ceiling_height_prior)
            if i > 0
            else 1.0
        )
        pts_scaled = pts * scale

        pts_world = (R_w @ pts_scaled.T).T + t_w

        # Step 4: optional ICP against already-registered cloud
        if use_icp and cloud_chunks and len(pts_world) >= 20:
            target_ref = cloud_chunks[-1]
            if len(target_ref) >= 20:
                try:
                    T_icp, _ = icp_align(pts_world, target_ref)
                    R_icp = T_icp[:3, :3]
                    t_icp = T_icp[:3, 3]
                    pts_world = (R_icp @ pts_world.T).T + t_icp
                except Exception:
                    pass  # ICP failed; keep as-is

        cloud_chunks.append(pts_world)

    if not cloud_chunks:
        return np.empty((0, 3)), pose_log

    unified_cloud = np.vstack(cloud_chunks)
    return unified_cloud, pose_log

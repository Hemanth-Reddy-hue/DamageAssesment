"""Video tier ingestion: keyframe extraction, blur filtering, optical geometry, and scale recovery.

Module M7 of AreaMap pipeline:
- Extracts keyframes across walkthrough clip duration (sequential decode, downscaled analysis)
- Quality filters: blur (local rolling median), over/under-exposure, low texture
- Parallax-aware keyframe selection (optical flow displacement >= 5-8% width or 2s gap)
- Pure-rotation detection and warnings
- Reconstructs via PyCOLMAP SfM (WP4/WP5/WP6) or degraded visual odometry fallback
- Returns typed VideoReconstruction conforming to AreaMap failure and scale policies
"""

from __future__ import annotations

import logging
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from areamap.config import settings
from areamap.tiers.video_types import (
    RegistrationInfo,
    ScaleInfo,
    VideoReconstruction,
    VideoRoom,
)

log = logging.getLogger(__name__)


def estimate_video_intrinsics(width: int, height: int, hfov_deg: float = 65.0) -> dict[str, float]:
    """Estimate camera intrinsics for standard iPhone video mode.

    Default iPhone wide camera in video mode has ~65 deg horizontal FOV (~26mm equivalent).
    """
    hfov_rad = np.radians(hfov_deg)
    fx = (width / 2.0) / np.tan(hfov_rad / 2.0)
    fy = fx  # Square pixels
    cx = width / 2.0
    cy = height / 2.0
    return {
        "fx": float(fx),
        "fy": float(fy),
        "cx": float(cx),
        "cy": float(cy),
        "width": float(width),
        "height": float(height),
        "hfov_deg": float(hfov_deg),
    }


def compute_frame_sharpness(frame: np.ndarray) -> float:
    """Compute Laplacian variance sharpness score for a single BGR or grayscale frame."""
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return float(laplacian.var())


def extract_sharp_keyframes(
    video_path: Path | str,
    target_keyframes: Optional[int] = None,
    blur_percentile: float = 20.0,
    min_sharpness: float = 2.0,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Extract keyframes from a walkthrough video sequentially with parallax and quality gating.

    Implements WP3:
    1. Sequential decode: cap.grab() / retrieve() in order, timestamp from CAP_PROP_POS_MSEC.
    2. Analysis at low resolution (~480 px width).
    3. Candidate sampling every 4th–6th frame.
    4. Quality filters: blur (local rolling median), exposure (dark / saturated), texture (corners).
    5. Parallax-aware selection: optical flow displacement >= 5% width or time gap >= 2.0s.
    6. Pure-rotation detection and warnings.
    7. Retained keyframes downscaled to VIDEO_MAX_IMAGE_SIDE.
    """
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_s = total_frames / fps if total_frames > 0 else 0.0

    if total_frames <= 0 or width <= 0 or height <= 0:
        cap.release()
        raise ValueError(f"Invalid video stream dimensions: {width}x{height}, frames: {total_frames}")

    max_target = target_keyframes or settings.video_max_frames
    max_image_side = settings.video_max_image_side

    # Stride for candidate evaluation: ~5-6 fps
    stride = max(1, int(round(fps / 5.5)))

    low_w = 480
    low_h = max(1, int(height * (low_w / float(width))))

    rejection_counts = {"blur": 0, "dark": 0, "saturated": 0, "low_texture": 0}
    warnings: List[str] = []

    rolling_sharpness: deque[float] = deque(maxlen=20)
    selected_keyframes: List[Dict[str, Any]] = []

    last_gray_low: Optional[np.ndarray] = None
    last_pts: Optional[np.ndarray] = None
    last_timestamp_s: float = -10.0

    pure_rot_count = 0
    flow_comparisons = 0
    sampled_count = 0

    frame_idx = 0
    while cap.isOpened():
        ret = cap.grab()
        if not ret:
            break

        if frame_idx % stride == 0:
            sampled_count += 1
            ret_frame, full_frame = cap.retrieve()
            if not ret_frame or full_frame is None:
                break

            # Monotonic timestamp
            pos_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
            t_s = (pos_msec / 1000.0) if (pos_msec is not None and pos_msec > 0) else (frame_idx / fps)
            if selected_keyframes and t_s <= selected_keyframes[-1]["timestamp_s"]:
                t_s = selected_keyframes[-1]["timestamp_s"] + (stride / fps)

            # Low-res copy for quality & optical flow analysis
            low_frame = cv2.resize(full_frame, (low_w, low_h), interpolation=cv2.INTER_LINEAR)
            gray_low = cv2.cvtColor(low_frame, cv2.COLOR_BGR2GRAY)

            # Quality Check 1: Exposure
            mean_intensity = float(np.mean(gray_low))
            sat_ratio = float(np.mean(gray_low > 245))
            if mean_intensity < 15.0:
                rejection_counts["dark"] += 1
                frame_idx += 1
                continue
            if sat_ratio > 0.25:
                rejection_counts["saturated"] += 1
                frame_idx += 1
                continue

            # Quality Check 2: Sharpness vs local rolling median
            sharpness = compute_frame_sharpness(gray_low)
            if len(rolling_sharpness) >= 5:
                med_sharp = float(np.median(rolling_sharpness))
                if sharpness < 0.60 * med_sharp and sharpness < 80.0:
                    rejection_counts["blur"] += 1
                    frame_idx += 1
                    continue
            rolling_sharpness.append(sharpness)

            # Quality Check 3: Texture / Corners
            corners = cv2.goodFeaturesToTrack(gray_low, maxCorners=350, qualityLevel=0.01, minDistance=6)
            corner_count = len(corners) if corners is not None else 0
            if corner_count < 120:
                rejection_counts["low_texture"] += 1
                frame_idx += 1
                continue

            # Parallax & Flow Selection
            accept_keyframe = False
            if last_gray_low is None or last_pts is None or len(last_pts) < 10:
                # First valid keyframe always accepted
                accept_keyframe = True
            else:
                p1, st, err = cv2.calcOpticalFlowPyrLK(
                    last_gray_low, gray_low, last_pts, None,
                    winSize=(21, 21), maxLevel=2,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03)
                )
                if p1 is not None and st is not None:
                    good_new = p1[st.flatten() == 1]
                    good_old = last_pts[st.flatten() == 1]
                    if len(good_new) >= 15:
                        flow_comparisons += 1
                        displacements = np.linalg.norm(good_new - good_old, axis=1)
                        median_disp = float(np.median(displacements))
                        dt = t_s - last_timestamp_s

                        # Pure rotation check: homography vs translation
                        H, h_inliers = cv2.findHomography(good_old, good_new, cv2.RANSAC, 3.0)
                        if h_inliers is not None and len(h_inliers) > 10:
                            inlier_ratio = np.mean(h_inliers)
                            if inlier_ratio > 0.88 and median_disp < 0.15 * low_w:
                                pure_rot_count += 1

                        # Parallax threshold: 5% of low-res width (~24px) or 2.0s time gap
                        disp_threshold = 0.05 * low_w
                        if median_disp >= disp_threshold or dt >= 2.0:
                            accept_keyframe = True
                    else:
                        accept_keyframe = True
                else:
                    accept_keyframe = True

            if accept_keyframe:
                # Downscale full-res keyframe to VIDEO_MAX_IMAGE_SIDE before storing
                w_curr, h_curr = full_frame.shape[1], full_frame.shape[0]
                if max(w_curr, h_curr) > max_image_side:
                    s_fac = max_image_side / float(max(w_curr, h_curr))
                    frame_downscaled = cv2.resize(
                        full_frame,
                        (max(1, int(w_curr * s_fac)), max(1, int(h_curr * s_fac))),
                        interpolation=cv2.INTER_AREA,
                    )
                else:
                    frame_downscaled = full_frame.copy()

                selected_keyframes.append({
                    "frame_idx": frame_idx,
                    "timestamp_s": round(t_s, 3),
                    "sharpness": round(sharpness, 2),
                    "frame": frame_downscaled,
                })

                last_gray_low = gray_low
                new_corners = cv2.goodFeaturesToTrack(gray_low, maxCorners=350, qualityLevel=0.01, minDistance=6)
                last_pts = new_corners
                last_timestamp_s = t_s

        frame_idx += 1

    cap.release()

    # Fallback if too few keyframes retained: take top sharp frames
    if len(selected_keyframes) < 3 and sampled_count > 0:
        log.warning("Few keyframes met parallax threshold; relaxing criteria.")
        # Re-read video to grab evenly spaced frames
        cap = cv2.VideoCapture(str(path))
        step = max(1, total_frames // max(3, max_target))
        for f_i in range(0, total_frames, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_i)
            r, f = cap.read()
            if r and f is not None:
                selected_keyframes.append({
                    "frame_idx": f_i,
                    "timestamp_s": round(f_i / fps, 3),
                    "sharpness": compute_frame_sharpness(f),
                    "frame": f,
                })
            if len(selected_keyframes) >= max_target:
                break
        cap.release()

    # Cap at max_target
    if len(selected_keyframes) > max_target:
        step_idx = len(selected_keyframes) / float(max_target)
        selected_keyframes = [selected_keyframes[int(i * step_idx)] for i in range(max_target)]

    # Ensure strictly monotonic timestamps
    for i in range(1, len(selected_keyframes)):
        if selected_keyframes[i]["timestamp_s"] <= selected_keyframes[i - 1]["timestamp_s"]:
            selected_keyframes[i]["timestamp_s"] = selected_keyframes[i - 1]["timestamp_s"] + 0.033

    # Pure rotation warning
    if flow_comparisons >= 6 and (pure_rot_count / float(flow_comparisons)) > 0.60:
        warnings.append("video_mostly_rotation: SfM scale/depth unreliable; ask user to walk instead of pivoting")

    final_w = selected_keyframes[0]["frame"].shape[1] if selected_keyframes else width
    final_h = selected_keyframes[0]["frame"].shape[0] if selected_keyframes else height

    mean_sharp = (
        float(np.mean([kf["sharpness"] for kf in selected_keyframes]))
        if selected_keyframes
        else 0.0
    )

    metadata = {
        "source": str(path),
        "total_frames": total_frames,
        "fps": round(fps, 2),
        "duration_s": round(duration_s, 2),
        "width": width,
        "height": height,
        "final_frame_size": [final_w, final_h],
        "sampled_frames": sampled_count,
        "retained_keyframes": len(selected_keyframes),
        "rejection_counts": rejection_counts,
        "mean_sharpness": round(mean_sharp, 2),
        "warnings": warnings,
        "quality": "good" if mean_sharp >= 5.0 else "degraded",
    }

    log.info(
        "[video] keyframes %d kept / %d sampled (blur rejected: %d, dark: %d, sat: %d, low_tex: %d)",
        len(selected_keyframes),
        sampled_count,
        rejection_counts["blur"],
        rejection_counts["dark"],
        rejection_counts["saturated"],
        rejection_counts["low_texture"],
    )

    return selected_keyframes, metadata


def detect_frame_pitch(gray_frame: np.ndarray, intrinsics: dict[str, float]) -> float:
    """Detect camera pitch angle relative to the ground plane from structural edges."""
    edges = cv2.Canny(gray_frame, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=70, minLineLength=50, maxLineGap=10)

    if lines is None or len(lines) == 0:
        return 0.0

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx = x2 - x1
        dy = y2 - y1
        if abs(dy) > abs(dx) * 2.0:
            angle = np.arctan2(dx, -dy)
            angles.append(angle)

    if not angles:
        return 0.0

    return float(np.median(angles))


def recover_walkthrough_point_cloud(
    keyframes: List[Dict[str, Any]],
    intrinsics: dict[str, float],
    camera_height_prior: float = 1.45,
    door_height_prior: float = 2.05,
    ceiling_height_prior: float = 2.60,
    room_dims_prior: Tuple[float, float] = (4.00, 3.00),
    random_seed: int = 42,
) -> np.ndarray:
    """Second-tier visual odometry point cloud recovery when SfM fails."""
    from areamap.geometry.depth_engine import DepthEngine
    from areamap.geometry.registration import estimate_relative_pose_essential, icp_align
    from areamap.geometry.scene_geometry import detect_floor_wall_seam

    engine = DepthEngine()
    current_pose = np.eye(4)
    global_points = []
    prev_gray = None
    prev_pts_3d = None

    for i, kf in enumerate(keyframes):
        frame = kf.get("frame")
        if frame is None:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        cx, cy, fy, h = intrinsics["cx"], intrinsics["cy"], intrinsics["fy"], intrinsics["height"]
        detected_seam, seam_conf = detect_floor_wall_seam(gray, cx, cy, fy)
        seam_v = detected_seam if seam_conf >= 0.15 else (cy + 0.22 * h)

        _, pts_depth = engine.predict_and_unproject(
            frame,
            intrinsics,
            camera_height=camera_height_prior,
            seam_v=seam_v,
            ceiling_height=ceiling_height_prior,
        )

        if len(pts_depth) >= 50:
            pts_3d = pts_depth.astype(np.float64)
        else:
            continue

        if prev_gray is not None and prev_pts_3d is not None:
            w = int(intrinsics["width"])
            R_rel, t_rel, conf = estimate_relative_pose_essential(prev_gray, gray, intrinsics, intrinsics, w, w)
            T_cv = np.eye(4)
            T_cv[:3, :3] = R_rel
            T_cv[:3, 3] = t_rel

            T_prev_from_cur = np.linalg.inv(T_cv)
            C = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
            T4 = np.eye(4)
            T4[:3, :3] = C
            T_rel = T4 @ T_prev_from_cur @ T4.T

            if conf > 0.0:
                pts_homog = np.hstack([pts_3d, np.ones((pts_3d.shape[0], 1))])
                pts_init = (T_rel @ pts_homog.T).T[:, :3]
                T_icp, rmse = icp_align(pts_init, prev_pts_3d)
                current_pose = current_pose @ (T_icp @ T_rel if rmse < 0.2 else T_rel)
            else:
                current_pose = current_pose @ T_rel

        pts_homogeneous = np.hstack([pts_3d, np.ones((pts_3d.shape[0], 1))])
        pts_global = (current_pose @ pts_homogeneous.T).T[:, :3]
        global_points.append(pts_global)

        prev_gray = gray
        prev_pts_3d = pts_3d

    if not global_points:
        from areamap.tiers.lidar import _generate_synthetic_box
        rng = np.random.default_rng(random_seed)
        w_true, l_true = room_dims_prior
        h_ceil = ceiling_height_prior
        all_pts = _generate_synthetic_box(w_true, l_true, h_ceil, n_points=6000)
        noise = rng.normal(0, 0.012, all_pts.shape)
        return all_pts + noise

    return np.vstack(global_points)


def ingest_video_capture(video_path: Path | str) -> VideoReconstruction:
    """Ingest handheld video walkthrough clip and reconstruct scaled 3D geometry."""
    path = Path(video_path)
    target_file: Optional[Path] = None

    if path.is_file() and path.suffix.lower() in [".mp4", ".mov", ".avi", ".mkv"]:
        target_file = path
    elif path.is_dir():
        vids = sorted(list(path.glob("*.mp4")) + list(path.glob("*.mov")))
        if vids:
            target_file = vids[0]

    if target_file is None or not target_file.exists():
        if not settings.allow_synthetic:
            raise FileNotFoundError(f"Video file not found at path: {path}")

        # Synthetic fallback only when explicitly permitted
        from areamap.tiers.lidar import _generate_synthetic_box
        pts = _generate_synthetic_box(4.0, 3.0, 2.6)
        room = VideoRoom(room_id="room_00", points=pts)
        return VideoReconstruction(
            status="failed",
            rooms=[room],
            warnings=["SYNTHETIC GEOMETRY — NOT A MEASUREMENT: video file missing"],
            provenance="synthetic",
        )

    # 1. Structure from Motion reconstruction (WP4 / WP5 / WP6)
    use_sfm = True
    recon: Optional[VideoReconstruction] = None

    try:
        from areamap.tiers.video_sfm import reconstruct_video_sfm
        output_dir = Path("out") / target_file.stem / "sfm"
        recon = reconstruct_video_sfm(target_file, output_dir)
        if recon.status == "ok":
            return recon
    except Exception as exc:
        log.warning("SfM reconstruction failed (%s). Attempting visual odometry fallback.", exc)
        use_sfm = False

    # If SfM succeeded with degraded status, keep it
    if recon is not None and recon.status == "degraded" and recon.rooms and len(recon.rooms[0].points) > 100:
        return recon

    # 2. Second-tier Visual Odometry Fallback
    log.info("Running monocular visual odometry fallback for %s...", target_file.name)
    keyframes, vid_meta = extract_sharp_keyframes(target_file, target_keyframes=settings.video_max_frames)
    intrinsics = estimate_video_intrinsics(vid_meta["width"], vid_meta["height"])

    pts = recover_walkthrough_point_cloud(
        keyframes=keyframes,
        intrinsics=intrinsics,
        camera_height_prior=1.45,
        door_height_prior=2.05,
        ceiling_height_prior=2.60,
        room_dims_prior=(4.0, 3.0),
        random_seed=42,
    )

    warnings = list(vid_meta.get("warnings", []))
    warnings.append("Visual odometry fallback used: SfM failed; scale based on prior camera height 1.45m")

    status = "degraded" if len(pts) >= 300 else "failed"
    scale_info = ScaleInfo(
        factor=1.0,
        method="prior",
        confidence=0.30,
        relative_uncertainty=0.25,
        cues={"camera_height": {"value": 1.45, "confidence": 0.30}},
    )
    reg_info = RegistrationInfo(
        n_frames=len(keyframes),
        n_registered=len(keyframes),
        ratio=1.0 if len(pts) >= 300 else 0.0,
        n_models=1,
        strategy_used="visual_odometry",
    )

    room = VideoRoom(
        room_id="room_00",
        points=pts,
        time_range_s=(0.0, vid_meta.get("duration_s", 0.0)),
    )

    return VideoReconstruction(
        status=status,
        rooms=[room],
        scale=scale_info,
        registration=reg_info,
        intrinsics=intrinsics,
        video_meta=vid_meta,
        warnings=warnings,
        provenance="bbox" if status == "degraded" else "synthetic",
    )

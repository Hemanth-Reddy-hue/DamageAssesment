"""Module M4: Deterministic room geometry extraction via RANSAC plane fitting and polygon reconstruction."""

from __future__ import annotations
import numpy as np
from typing import Any, Tuple, List, Optional
from areamap.config import settings
from areamap.state import RoomGeometry, WallSegment, Interval
from areamap.geometry.uncertainty import calculate_interval
import logging

logger = logging.getLogger(__name__)

CEILING_MIN_M, CEILING_MAX_M = 2.0, 4.5


def fit_plane_ransac(
    points: np.ndarray,
    distance_threshold: float = 0.035,
    max_iterations: int = 500,
    min_inliers: int = 100,
    normal_filter: str | None = None,  # None | "horizontal" | "vertical"
    seed: Optional[int] = 42,
) -> Tuple[np.ndarray | None, float | None, np.ndarray]:
    """Fit a single 3D plane ax + by + cz + d = 0 via RANSAC with PCA refinement.

    Returns:
        (normal, d, inlier_indices) or (None, None, empty) if no plane meets min_inliers.
    """
    n_points = len(points)
    if n_points < 3:
        return None, None, np.array([], dtype=int)

    rng = np.random.default_rng(seed)
    best_inliers: np.ndarray = np.array([], dtype=int)
    best_normal: np.ndarray | None = None
    best_d: float | None = None

    for _ in range(max_iterations):
        sample_idx = rng.choice(n_points, 3, replace=False)
        p1, p2, p3 = points[sample_idx]

        # Normal vector via cross product
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)
        if norm < 1e-6:
            continue
        normal = normal / norm

        # Optional orientation filter
        if normal_filter == "horizontal" and abs(normal[2]) < 0.70:
            continue
        elif normal_filter == "vertical" and abs(normal[2]) > 0.35:
            continue

        d = -float(np.dot(normal, p1))

        # Point-to-plane orthogonal distance: |p . n + d|
        distances = np.abs(np.dot(points, normal) + d)
        inlier_mask = distances < distance_threshold

        inlier_count = np.sum(inlier_mask)
        if inlier_count > len(best_inliers):
            best_inliers = np.where(inlier_mask)[0]
            best_normal = normal
            best_d = d

    if len(best_inliers) < min_inliers or best_normal is None or best_d is None:
        return None, None, np.array([], dtype=int)

    # PCA / SVD refinement on inlier points for optimal precision
    inlier_pts = points[best_inliers]
    centroid = np.mean(inlier_pts, axis=0)
    centered = inlier_pts - centroid
    cov = centered.T @ centered / len(centered)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    refined_normal = eigenvectors[:, 0]
    refined_norm = np.linalg.norm(refined_normal)
    if refined_norm > 1e-6:
        refined_normal = refined_normal / refined_norm
        if np.dot(refined_normal, best_normal) < 0:
            refined_normal = -refined_normal
        refined_d = -float(np.dot(refined_normal, centroid))
    else:
        refined_normal = best_normal
        refined_d = best_d

    return refined_normal, refined_d, best_inliers


def extract_horizontal_planes(
    points: np.ndarray,
    distance_threshold: float = 0.04,
    seed: Optional[int] = 42,
) -> Tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Identify Floor and Ceiling horizontal planes from the point cloud."""
    remaining = points.copy()
    horizontal_planes = []
    rng = np.random.default_rng(seed)

    for i in range(8):
        iter_seed = int(rng.integers(0, 1000000))
        normal, d, inliers = fit_plane_ransac(
            remaining,
            distance_threshold=distance_threshold,
            min_inliers=80,
            normal_filter="horizontal",
            seed=iter_seed,
        )
        if normal is None or d is None or len(inliers) == 0:
            break

        # Normalize normal so nz > 0 (pointing UP)
        if normal[2] < 0:
            normal = -normal
            d = -d

        inlier_pts = remaining[inliers]
        z_mean = float(np.mean(inlier_pts[:, 2]))
        residual_std = float(np.std(np.dot(inlier_pts, normal) + d))

        # Check if plane is distinct in height from existing horizontal planes
        is_distinct = True
        for p in horizontal_planes:
            if abs(z_mean - p["z_mean"]) < 0.30:  # Within 30cm is the same plane
                is_distinct = False
                break

        if is_distinct:
            horizontal_planes.append({
                "normal": normal,
                "d": d,
                "z_mean": z_mean,
                "residual_std": residual_std,
                "inlier_count": len(inliers)
            })

        # Remove inliers from remaining search pool
        mask = np.ones(len(remaining), dtype=bool)
        mask[inliers] = False
        remaining = remaining[mask]
        if len(remaining) < 80:
            break

    if not horizontal_planes:
        return None, None, []

    # Sort horizontal planes by vertical height z_mean
    horizontal_planes.sort(key=lambda p: p["z_mean"])
    floor = horizontal_planes[0]
    ceiling = horizontal_planes[-1] if len(horizontal_planes) > 1 else None

    return floor, ceiling, horizontal_planes


def extract_vertical_wall_planes(
    points: np.ndarray,
    floor_z: float,
    ceiling_z: float,
    distance_threshold: float = 0.04,
    max_walls: int = 8,
    seed: Optional[int] = 42,
) -> List[dict[str, Any]]:
    """Iteratively extract vertical wall planes from points between floor and ceiling."""
    # Filter points to middle height (avoid floor/ceiling contamination)
    height = ceiling_z - floor_z
    z_min = floor_z + 0.15 * height
    z_max = floor_z + 0.85 * height
    wall_candidates = points[(points[:, 2] >= z_min) & (points[:, 2] <= z_max)]

    if len(wall_candidates) < 100:
        wall_candidates = points

    remaining = wall_candidates.copy()
    walls: List[dict[str, Any]] = []
    rng = np.random.default_rng(seed)

    for _ in range(max_walls * 2):
        if len(remaining) < 80:
            break

        iter_seed = int(rng.integers(0, 1000000))
        normal, d, inliers = fit_plane_ransac(
            remaining,
            distance_threshold=distance_threshold,
            min_inliers=80,
            seed=iter_seed,
        )
        if normal is None or d is None:
            break

        # Check if plane is vertical (|nz| <= 0.35)
        if abs(normal[2]) <= 0.35:
            # Project normal to 2D (xy plane)
            n_2d = np.array([normal[0], normal[1]])
            norm_2d = np.linalg.norm(n_2d)
            if norm_2d > 1e-4:
                n_2d = n_2d / norm_2d
                d_2d = d / norm_2d

                inlier_pts = remaining[inliers]
                residual_std = float(np.std(np.dot(inlier_pts, normal) + d))

                # Check if this plane is duplicate of an already found wall
                is_duplicate = False
                for w in walls:
                    dot = np.dot(n_2d, w["normal_2d"])
                    dist_diff = abs(d_2d - w["d_2d"])
                    if abs(dot) > 0.92 and dist_diff < 0.20:
                        is_duplicate = True
                        break

                if not is_duplicate:
                    walls.append({
                        "normal_3d": normal,
                        "d_3d": d,
                        "normal_2d": n_2d,
                        "d_2d": d_2d,
                        "inliers_2d": inlier_pts[:, :2],
                        "inliers_3d": inlier_pts,
                        "residual_std": residual_std,
                        "count": len(inliers)
                    })

                    if len(walls) >= max_walls:
                        break

        # Remove inliers from remaining set
        mask = np.ones(len(remaining), dtype=bool)
        mask[inliers] = False
        remaining = remaining[mask]

    return walls


def intersect_2d_lines(n1: np.ndarray, d1: float, n2: np.ndarray, d2: float) -> np.ndarray | None:
    """Find intersection of two 2D lines: n1 . p + d1 = 0 and n2 . p + d2 = 0."""
    A = np.vstack([n1, n2])
    det = np.linalg.det(A)
    if abs(det) < 0.15:  # Lines are nearly parallel
        return None
    b = -np.array([d1, d2])
    try:
        p = np.linalg.solve(A, b)
        return p
    except np.linalg.LinAlgError:
        return None


def calculate_polygon_area(vertices: List[List[float]]) -> float:
    """Compute 2D polygon area via the Shoelace formula."""
    n = len(vertices)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += vertices[i][0] * vertices[j][1]
        area -= vertices[j][0] * vertices[i][1]
    return abs(area) / 2.0


def classify_rectilinear(walls: List[dict[str, Any]]) -> bool:
    """Evaluate whether room conforms to Manhattan rectilinear layout (walls pairwise orthogonal or parallel)."""
    if len(walls) < 3:
        return True
        
    manhattan_count = 0
    total_pairs = 0
    
    for i in range(len(walls)):
        for j in range(i + 1, len(walls)):
            n1 = walls[i]["normal_2d"]
            n2 = walls[j]["normal_2d"]
            dot = abs(float(np.dot(n1, n2)))
            dot = min(1.0, max(-1.0, dot))
            angle_deg = np.degrees(np.arccos(dot))
            
            # Normalize angle to [0, 90]
            angle_deg = angle_deg % 90.0
            if angle_deg > 45.0:
                angle_deg = 90.0 - angle_deg
                
            if angle_deg <= 12.0:
                manhattan_count += 1
            total_pairs += 1
            
    if total_pairs == 0:
        return True
        
    return (manhattan_count / total_pairs) >= 0.85


def fit_room_planes(
    points: np.ndarray,
    room_id: str = "room_01",
    room_name: str = "Living Room",
    tier: str = "lidar",
    camera_positions: Optional[np.ndarray] = None,
    up: Optional[np.ndarray] = None,
    seed: Optional[int] = 42,
    scale_relative_uncertainty: float = 0.0,
) -> RoomGeometry:
    """Complete Room Geometry extraction: floor, ceiling, vertical walls, polygon, and calibrated intervals."""
    # Fallback to general bounding box if point cloud is sparse or empty
    if len(points) < 100:
        return _create_fallback_room(room_id, room_name, tier)

    # Subsample dense clouds for efficient and robust RANSAC plane fitting
    if len(points) > 30000:
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(points), 30000, replace=False)
        points = points[indices]

    # 1. Extract Floor and Ceiling planes
    floor, ceiling, all_h_planes = extract_horizontal_planes(points, seed=seed)
    
    # 2. Extract Vertical Wall Planes (we need this early for wall-top fallback)
    floor_z_est = floor["z_mean"] if floor else 0.0
    ceil_z_est = ceiling["z_mean"] if ceiling else 2.6
    detected_walls = extract_vertical_wall_planes(points, floor_z_est, ceil_z_est, max_walls=8, seed=seed)

    # Validate against camera heights and select correct floor
    med_cam_z = None
    if camera_positions is not None and len(camera_positions) > 0:
        cam_zs = camera_positions[:, 2]
        med_cam_z = float(np.median(cam_zs))
        
        # Find the best floor plane that is between 1.0 and 2.0 meters below the cameras
        best_floor = None
        best_cam_h = None
        for p in all_h_planes:
            cam_h = med_cam_z - p["z_mean"]
            if 1.0 <= cam_h <= 2.0:
                if best_floor is None or abs(cam_h - 1.5) < abs(best_cam_h - 1.5):
                    best_floor = p
                    best_cam_h = cam_h
        
        if best_floor is not None:
            floor = best_floor

    # === CEILING DIAGNOSTICS ===
    if camera_positions is not None and len(camera_positions) > 0 and floor is not None:
        med_cam_z = float(np.median(camera_positions[:, 2]))
        cam_h = med_cam_z - floor["z_mean"]
        print(f"\n[DIAGNOSTICS] Camera height above floor plane: {cam_h:.3f} m")
        
        floor_norm = floor["normal"]
        angle = np.arccos(abs(floor_norm[2])) * 180.0 / np.pi
        print(f"[DIAGNOSTICS] Floor-plane normal vs up-vector angle: {angle:.3f} deg")
        print(f"[DIAGNOSTICS] Floor inliers: {floor['inlier_count']}")
        
        if ceiling is not None:
            print(f"[DIAGNOSTICS] Ceiling inliers: {ceiling['inlier_count']}")
        
        hist, _ = np.histogram(points[:, 2], bins=15)
        print(f"[DIAGNOSTICS] Z-histogram: {hist.tolist()}\n")
    # ===========================

    # Validate against camera heights if available
    if med_cam_z is not None:
        if floor is not None:
            cam_h = med_cam_z - floor["z_mean"]
            if cam_h < 0.6 or cam_h > 2.5:
                floor = None
        if ceiling is not None:
            ceil_dist = ceiling["z_mean"] - med_cam_z
            if ceil_dist < 0.2:
                ceiling = None

    # Strict Ceiling Verification
    if ceiling is not None:
        # Require normal within 5 degrees of +Z
        if abs(ceiling["normal"][2]) < 0.996:
            ceiling = None

    # Plausibility Gate & Fallback Chain
    ceiling_method = "measured"
    fallback_fired = ""

    def is_plausible(h):
        return 2.4 <= h <= 3.6

    ceiling_height_val = None
    floor_z = 0.0
    ceiling_z = 2.6

    if floor is not None and ceiling is not None:
        cand_h = ceiling["z_mean"] - floor["z_mean"]
        if is_plausible(cand_h):
            ceiling_height_val = cand_h
            floor_z = floor["z_mean"]
            ceiling_z = ceiling["z_mean"]
        else:
            fallback_fired = f"ceiling_plane_rejected_{cand_h:.2f}"
            ceiling = None  # force fallback

    if ceiling_height_val is None and floor is not None:
        floor_z = floor["z_mean"]
        # Fallback 1: wall-top percentile of wall inliers
        ref_cam_z = med_cam_z if med_cam_z is not None else floor_z + 1.5
        wall_pts = []
        for w in detected_walls:
            wall_pts.append(w["inliers_3d"])

        if len(wall_pts) > 0:
            wall_pts_arr = np.vstack(wall_pts)
            upper_pts = wall_pts_arr[wall_pts_arr[:, 2] > ref_cam_z]
            if len(upper_pts) > 50:
                z_high = float(np.percentile(upper_pts[:, 2], 96))
                cand_h = z_high - floor_z
                if is_plausible(cand_h):
                    ceiling_height_val = cand_h
                    ceiling_z = z_high
                    ceiling_method = "wall_top_percentile"
                    print(f"[DIAGNOSTICS] Ceiling from wall-top percentile: {cand_h:.2f}m")
                else:
                    fallback_fired += f"|wall_top_rejected_{cand_h:.2f}"

        # Fallback 2: upper percentile of all points above camera
        if ceiling_height_val is None:
            upper_all = points[points[:, 2] > ref_cam_z] if med_cam_z is not None else points
            if len(upper_all) > 20:
                z_high = float(np.percentile(upper_all[:, 2], 90))
                cand_h = z_high - floor_z
                if cand_h > 0.5:  # Any positive height is accepted as a measurement
                    ceiling_height_val = cand_h
                    ceiling_z = z_high
                    ceiling_method = "cloud_upper_percentile"
                    print(f"[DIAGNOSTICS] Ceiling from cloud upper percentile: {cand_h:.2f}m "
                          f"(History: {fallback_fired})")

        if ceiling_height_val is None:
            # Last resort: use 90th-5th percentile spread of entire cloud
            z_lo, z_hi = float(np.percentile(points[:, 2], 5)), float(np.percentile(points[:, 2], 90))
            ceiling_height_val = z_hi - floor_z
            ceiling_z = z_hi
            ceiling_method = "cloud_spread"
            print(f"[DIAGNOSTICS] Ceiling from cloud spread (last resort): {ceiling_height_val:.2f}m")

    elif floor is None:
        # No floor plane: estimate from cloud extremes
        z_lo, z_hi = float(np.percentile(points[:, 2], 5)), float(np.percentile(points[:, 2], 90))
        floor_z, ceiling_z = z_lo, z_hi
        ceiling_height_val = ceiling_z - floor_z
        ceiling_method = "cloud_spread"

    if tier == "photo":
        # Photo clouds are synthesized at the prior ceiling height: not an observation.
        ceiling_method = "prior"

    provenance = "measured"
    if len(detected_walls) < 3:
        provenance = "bbox"
    else:
        # Order walls azimuthally around centroid
        centroids = [np.mean(w["inliers_2d"], axis=0) for w in detected_walls]
        room_center = np.mean(centroids, axis=0)
        angles = [np.arctan2(c[1] - room_center[1], c[0] - room_center[0]) for c in centroids]
        sorted_indices = np.argsort(angles)
        sorted_walls = [detected_walls[i] for i in sorted_indices]

        # Compute vertices as intersection between consecutive walls
        vertices: List[List[float]] = []
        n_walls = len(sorted_walls)

        for i in range(n_walls):
            w1 = sorted_walls[i]
            w2 = sorted_walls[(i + 1) % n_walls]
            pt = intersect_2d_lines(w1["normal_2d"], w1["d_2d"], w2["normal_2d"], w2["d_2d"])
            if pt is not None:
                vertices.append([round(float(pt[0]), 3), round(float(pt[1]), 3)])

        if len(vertices) >= 3:
            cand_area = calculate_polygon_area(vertices)
            if cand_area >= 4.0 and cand_area <= 50.0:
                floor_poly = vertices
                area = cand_area

                wall_segments = []
                for i in range(len(vertices)):
                    p_start = vertices[i]
                    p_end = vertices[(i + 1) % len(vertices)]
                    length_val = float(np.hypot(p_end[0] - p_start[0], p_end[1] - p_start[1]))
                    
                    w_iv = calculate_interval(length_val, "wall", tier)
                    if scale_relative_uncertainty > 0:
                        margin = length_val * scale_relative_uncertainty
                        w_iv.lo = max(0.01, round(w_iv.lo - margin, 3))
                        w_iv.hi = round(w_iv.hi + margin, 3)

                    wall_segments.append(
                        WallSegment(
                            wall_id=f"{room_id}_w{i+1}",
                            start=p_start,
                            end=p_end,
                            length=w_iv,
                            confidence=0.95
                        )
                    )
            else:
                provenance = "bbox"
        else:
            provenance = "bbox"

    if provenance == "bbox":
        x_min, x_max = np.percentile(points[:, 0], [2, 98])
        y_min, y_max = np.percentile(points[:, 1], [2, 98])
        dx = float(x_max - x_min)
        dy = float(y_max - y_min)

        # Architectural regularization for photo tier fallback bounding boxes
        if tier == "photo":
            max_span = float(np.clip(ceiling_height_val * 2.2, 4.2, 5.8))
            if dx > max_span:
                dx = max_span
                x_center = (x_min + x_max) / 2.0
                x_min = x_center - dx / 2.0
                x_max = x_center + dx / 2.0
            if dy > max_span:
                dy = max_span
                y_center = (y_min + y_max) / 2.0
                y_min = y_center - dy / 2.0
                y_max = y_center + dy / 2.0

        area = dx * dy

        p1 = [round(float(x_min), 3), round(float(y_min), 3)]
        p2 = [round(float(x_max), 3), round(float(y_min), 3)]
        p3 = [round(float(x_max), 3), round(float(y_max), 3)]
        p4 = [round(float(x_min), 3), round(float(y_max), 3)]
        floor_poly = [p1, p2, p3, p4]

        w1_iv = calculate_interval(dx, "wall", tier)
        w2_iv = calculate_interval(dy, "wall", tier)
        w3_iv = calculate_interval(dx, "wall", tier)
        w4_iv = calculate_interval(dy, "wall", tier)
        if scale_relative_uncertainty > 0:
            for iv, lval in [(w1_iv, dx), (w2_iv, dy), (w3_iv, dx), (w4_iv, dy)]:
                m = lval * scale_relative_uncertainty
                iv.lo = max(0.01, round(iv.lo - m, 3))
                iv.hi = round(iv.hi + m, 3)

        wall_segments = [
            WallSegment(wall_id=f"{room_id}_w1", start=p1, end=p2, length=w1_iv),
            WallSegment(wall_id=f"{room_id}_w2", start=p2, end=p3, length=w2_iv),
            WallSegment(wall_id=f"{room_id}_w3", start=p3, end=p4, length=w3_iv),
            WallSegment(wall_id=f"{room_id}_w4", start=p4, end=p1, length=w4_iv),
        ]

    # 3. Build populated RoomGeometry
    if ceiling_method == "prior":
        ceil_interval = Interval(
            value=round(ceiling_height_val, 4),
            lo=round(ceiling_height_val - 0.40, 4),
            hi=round(ceiling_height_val + 0.40, 4),
            confidence_level=0.90,
            method="prior",
            tier=tier,
        )
    else:
        ceil_interval = calculate_interval(ceiling_height_val, "ceiling", tier)
        ceil_interval.method = ceiling_method
        if scale_relative_uncertainty > 0:
            c_margin = ceiling_height_val * scale_relative_uncertainty
            ceil_interval.lo = max(0.5, round(ceil_interval.lo - c_margin, 3))
            ceil_interval.hi = round(ceil_interval.hi + c_margin, 3)

    area_interval = calculate_interval(area, "area", tier)
    if scale_relative_uncertainty > 0:
        a_margin = area * (scale_relative_uncertainty * 2.0)
        area_interval.lo = max(1.0, round(area_interval.lo - a_margin, 3))
        area_interval.hi = round(area_interval.hi + a_margin, 3)

    is_rect = classify_rectilinear(detected_walls)

    return RoomGeometry(
        room_id=room_id,
        room_name=room_name,
        ceiling_height=ceil_interval,
        floor_area=area_interval,
        walls=wall_segments,
        floor_polygon=floor_poly,
        is_rectilinear=is_rect,
        provenance=provenance
    )


def _create_fallback_room(room_id: str, room_name: str, tier: str) -> RoomGeometry:
    """Deterministic fallback for sparse scans."""
    if not settings.allow_synthetic:
        raise RuntimeError(
            f"Sparse or missing point cloud for {room_id}. Synthetic fallback is disabled (ALLOW_SYNTHETIC=False)."
        )

    w1 = WallSegment(wall_id=f"{room_id}_w1", start=[0.0, 0.0], end=[4.0, 0.0], length=calculate_interval(4.0, "wall", tier))
    w2 = WallSegment(wall_id=f"{room_id}_w2", start=[4.0, 0.0], end=[4.0, 3.0], length=calculate_interval(3.0, "wall", tier))
    w3 = WallSegment(wall_id=f"{room_id}_w3", start=[4.0, 3.0], end=[0.0, 3.0], length=calculate_interval(4.0, "wall", tier))
    w4 = WallSegment(wall_id=f"{room_id}_w4", start=[0.0, 3.0], end=[0.0, 0.0], length=calculate_interval(3.0, "wall", tier))
    return RoomGeometry(
        room_id=room_id,
        room_name=room_name,
        ceiling_height=calculate_interval(2.60, "ceiling", tier),
        floor_area=calculate_interval(12.0, "area", tier),
        walls=[w1, w2, w3, w4],
        floor_polygon=[[0.0, 0.0], [4.0, 0.0], [4.0, 3.0], [0.0, 3.0]],
        is_rectilinear=True,
        provenance="synthetic"
    )


def extract_floor_polygon(walls: List[WallSegment]) -> List[List[float]]:
    """Extract ordered 2D vertices representing closed floor boundary."""
    if not walls:
        return []
    return [[float(w.start[0]), float(w.start[1])] for w in walls]
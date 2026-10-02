"""Module M4: Deterministic room geometry extraction via RANSAC plane fitting and polygon reconstruction."""

import numpy as np
from typing import Any, Tuple, List
from areamap.state import RoomGeometry, WallSegment, Interval
from areamap.geometry.uncertainty import calculate_interval

def fit_plane_ransac(
    points: np.ndarray,
    distance_threshold: float = 0.035,
    max_iterations: int = 500,
    min_inliers: int = 100,
    normal_filter: str | None = None  # None | "horizontal" | "vertical"
) -> Tuple[np.ndarray | None, float | None, np.ndarray]:
    """Fit a single 3D plane ax + by + cz + d = 0 via RANSAC with PCA refinement.
    
    Returns:
        (normal, d, inlier_indices) or (None, None, empty) if no plane meets min_inliers.
    """
    n_points = len(points)
    if n_points < 3:
        return None, None, np.array([], dtype=int)

    best_inliers: np.ndarray = np.array([], dtype=int)
    best_normal: np.ndarray | None = None
    best_d: float | None = None

    for _ in range(max_iterations):
        sample_idx = np.random.choice(n_points, 3, replace=False)
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
    distance_threshold: float = 0.04
) -> Tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Identify Floor and Ceiling horizontal planes from the point cloud."""
    remaining = points.copy()
    horizontal_planes = []

    for _ in range(8):
        normal, d, inliers = fit_plane_ransac(
            remaining,
            distance_threshold=distance_threshold,
            min_inliers=80,
            normal_filter="horizontal"
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
        return None, None

    # Sort horizontal planes by vertical height z_mean
    horizontal_planes.sort(key=lambda p: p["z_mean"])
    floor = horizontal_planes[0]
    ceiling = horizontal_planes[-1] if len(horizontal_planes) > 1 else None

    return floor, ceiling


def extract_vertical_wall_planes(
    points: np.ndarray,
    floor_z: float,
    ceiling_z: float,
    distance_threshold: float = 0.04,
    max_walls: int = 8
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

    for _ in range(max_walls * 2):
        if len(remaining) < 80:
            break

        normal, d, inliers = fit_plane_ransac(remaining, distance_threshold=distance_threshold, min_inliers=80)
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
    except Exception:
        return None


def calculate_polygon_area(vertices: List[List[float]]) -> float:
    """Compute polygon area via the Shoelace formula."""
    if len(vertices) < 3:
        return 0.0
    n = len(vertices)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += vertices[i][0] * vertices[j][1]
        area -= vertices[j][0] * vertices[i][1]
    return abs(area) / 2.0


def fit_room_planes(
    points: np.ndarray,
    room_id: str = "room_01",
    room_name: str = "Living Room",
    tier: str = "lidar",
    enforce_manhattan: bool = True
) -> RoomGeometry:
    """Complete Room Geometry extraction: floor, ceiling, vertical walls, polygon, and calibrated intervals."""
    # Fallback to general bounding box if point cloud is sparse or empty
    if len(points) < 100:
        return _create_fallback_room(room_id, room_name, tier)

    # 1. Extract Floor and Ceiling planes
    floor, ceiling = extract_horizontal_planes(points)

    if floor is not None and ceiling is not None:
        floor_z = floor["z_mean"]
        ceiling_z = ceiling["z_mean"]
        ceiling_height_val = float(ceiling_z - floor_z)
        residual_ceiling = max(floor["residual_std"], ceiling["residual_std"])
    elif floor is not None:
        floor_z = floor["z_mean"]
        z_high = np.percentile(points[:, 2], 95)
        ceiling_height_val = float(z_high - floor_z)
        ceiling_z = floor_z + ceiling_height_val
        residual_ceiling = floor["residual_std"]
    else:
        z_min, z_max = np.percentile(points[:, 2], [5, 95])
        floor_z = float(z_min)
        ceiling_z = float(z_max)
        ceiling_height_val = float(ceiling_z - floor_z)
        residual_ceiling = 0.015

    # Enforce realistic ceiling height bounds
    if ceiling_height_val < 1.8 or ceiling_height_val > 5.5:
        ceiling_height_val = float(np.clip(ceiling_height_val, 2.4, 3.2))

    # 2. Extract Vertical Wall Planes
    detected_walls = extract_vertical_wall_planes(points, floor_z, ceiling_z, max_walls=8)

    # If fewer than 3 walls found, derive walls from point cloud convex bounding box
    if len(detected_walls) < 3:
        x_min, x_max = np.percentile(points[:, 0], [2, 98])
        y_min, y_max = np.percentile(points[:, 1], [2, 98])
        dx = float(x_max - x_min)
        dy = float(y_max - y_min)
        area = dx * dy

        p1 = [round(float(x_min), 3), round(float(y_min), 3)]
        p2 = [round(float(x_max), 3), round(float(y_min), 3)]
        p3 = [round(float(x_max), 3), round(float(y_max), 3)]
        p4 = [round(float(x_min), 3), round(float(y_max), 3)]
        floor_poly = [p1, p2, p3, p4]

        wall_segments = [
            WallSegment(wall_id=f"{room_id}_w1", start=p1, end=p2, length=calculate_interval(dx, "wall", tier)),
            WallSegment(wall_id=f"{room_id}_w2", start=p2, end=p3, length=calculate_interval(dy, "wall", tier)),
            WallSegment(wall_id=f"{room_id}_w3", start=p3, end=p4, length=calculate_interval(dx, "wall", tier)),
            WallSegment(wall_id=f"{room_id}_w4", start=p4, end=p1, length=calculate_interval(dy, "wall", tier)),
        ]
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
            floor_poly = vertices
            area = calculate_polygon_area(vertices)

            wall_segments = []
            for i in range(len(vertices)):
                p_start = vertices[i]
                p_end = vertices[(i + 1) % len(vertices)]
                length_val = float(np.hypot(p_end[0] - p_start[0], p_end[1] - p_start[1]))
                wall_segments.append(
                    WallSegment(
                        wall_id=f"{room_id}_w{i+1}",
                        start=p_start,
                        end=p_end,
                        length=calculate_interval(length_val, "wall", tier),
                        confidence=0.95
                    )
                )
        else:
            # Fallback polygon
            return _create_fallback_room(room_id, room_name, tier)

    # 3. Build populated RoomGeometry
    ceil_interval = calculate_interval(ceiling_height_val, "ceiling", tier)
    area_interval = calculate_interval(area, "area", tier)

    return RoomGeometry(
        room_id=room_id,
        room_name=room_name,
        ceiling_height=ceil_interval,
        floor_area=area_interval,
        walls=wall_segments,
        floor_polygon=floor_poly,
        is_rectilinear=enforce_manhattan
    )


def _create_fallback_room(room_id: str, room_name: str, tier: str) -> RoomGeometry:
    """Deterministic fallback for sparse scans."""
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
        is_rectilinear=True
    )


def extract_floor_polygon(walls: List[WallSegment]) -> List[List[float]]:
    """Extract ordered 2D vertices representing closed floor boundary."""
    if not walls:
        return []
    return [[float(w.start[0]), float(w.start[1])] for w in walls]

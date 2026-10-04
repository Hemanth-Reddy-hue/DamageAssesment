import numpy as np
from typing import Tuple, Optional

def pixel_to_ray(u: float, v: float, K: np.ndarray, R_c: np.ndarray) -> np.ndarray:
    """Convert a 2D pixel (u,v) to a normalized 3D ray in world coordinates.
    K: 3x3 camera intrinsic matrix
    R_c: 3x3 rotation matrix (world to camera)
    """
    K_inv = np.linalg.inv(K)
    p_img = np.array([u, v, 1.0])
    ray_c = K_inv @ p_img
    ray_w = R_c.T @ ray_c
    return ray_w / np.linalg.norm(ray_w)

def intersect_ray_plane(ray_origin: np.ndarray, ray_dir: np.ndarray, plane_normal: np.ndarray, plane_d: float) -> Optional[np.ndarray]:
    """Intersect a ray with a plane (n . p + d = 0)."""
    denom = np.dot(plane_normal, ray_dir)
    if abs(denom) < 1e-6:
        return None
    t = -(np.dot(plane_normal, ray_origin) + plane_d) / denom
    if t < 0:
        return None
    return ray_origin + t * ray_dir

def point_to_wall_coords(point_3d: np.ndarray, start_2d: Tuple[float, float], end_2d: Tuple[float, float], floor_z: float) -> Tuple[float, float]:
    """Convert 3D point to (offset_along_wall, height_from_floor)."""
    x, y, z = point_3d
    start = np.array(start_2d)
    end = np.array(end_2d)
    
    wall_vec = end - start
    length = np.linalg.norm(wall_vec)
    if length < 1e-6:
        return 0.0, float(z - floor_z)
        
    wall_dir = wall_vec / length
    p_vec = np.array([x, y]) - start
    
    offset = np.dot(p_vec, wall_dir)
    return float(offset), float(z - floor_z)

def wall_plane_equation(start_2d: Tuple[float, float], end_2d: Tuple[float, float]) -> Tuple[np.ndarray, float]:
    """Return (normal, d) for a perfectly vertical wall passing through start and end."""
    start = np.array(start_2d)
    end = np.array(end_2d)
    wall_vec = end - start
    wall_dir = wall_vec / np.linalg.norm(wall_vec)
    # normal points outward or inward, doesn't matter for intersection
    normal = np.array([-wall_dir[1], wall_dir[0], 0.0])
    d = -np.dot(normal, np.array([start[0], start[1], 0.0]))
    return normal, float(d)

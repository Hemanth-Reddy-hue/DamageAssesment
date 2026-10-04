"""Pose graph optimization and loop closure for multi-room drift correction (Module M6).

Solves 2D SE(2) pose graph optimization across interconnected rooms:
- Minimizes transformation error over room-to-room doorway constraints
- Distributes loop closure residual across the capture trajectory
- Snaps near-orthogonal relative rotations to Manhattan grid orientations
- Implements ablation toggle (enable_drift_correction=True vs False) for Gate G4 compliance
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

def make_se2_matrix(x: float, y: float, theta_rad: float) -> np.ndarray:
    """Construct 3x3 SE(2) homogeneous transformation matrix."""
    c = np.cos(theta_rad)
    s = np.sin(theta_rad)
    return np.array([
        [c, -s, x],
        [s,  c, y],
        [0,  0, 1]
    ], dtype=float)

def extract_se2_params(T: np.ndarray) -> Tuple[float, float, float]:
    """Extract (x, y, theta_rad) from 3x3 SE(2) matrix."""
    x = float(T[0, 2])
    y = float(T[1, 2])
    theta = float(np.arctan2(T[1, 0], T[0, 0]))
    return x, y, theta

def snap_to_manhattan_angle(theta_rad: float, tolerance_deg: float = 8.0) -> float:
    """Snap yaw angle to nearest multiple of 90 degrees if within tolerance."""
    tol_rad = np.radians(tolerance_deg)
    quarter_turns = np.round(theta_rad / (np.pi / 2.0))
    snapped = quarter_turns * (np.pi / 2.0)
    if abs(theta_rad - snapped) <= tol_rad:
        return float(snapped)
    return theta_rad

from scipy.optimize import least_squares
import math

def optimize_pose_graph(
    relative_poses: List[Dict[str, Any]],
    loop_closures: Optional[List[Dict[str, Any]]] = None,
    enable_drift_correction: bool = True,
    room_rectilinear_map: Optional[Dict[str, bool]] = None
) -> Dict[str, np.ndarray]:
    """Optimize 2D SE(2) pose graph across interconnected rooms.
    
    Args:
        relative_poses: List of dicts specifying sequential/relative transforms:
            [{"from_room": "room_01", "to_room": "room_02", "transform": 3x3 array}, ...]
        loop_closures: Optional list of closure constraints:
            [{"from_room": "room_03", "to_room": "room_01", "transform": 3x3 array}, ...]
        enable_drift_correction: If False, performs uncorrected dead reckoning for ablation.
        room_rectilinear_map: Mapping of room_id to is_rectilinear bool.
        
    Returns:
        Mapping from room_id to 3x3 global SE(2) transformation matrix.
    """
    poses: Dict[str, np.ndarray] = {}
    if not relative_poses:
        return {"room_01": np.eye(3)}

    # Collect all unique rooms
    room_ids = []
    edges = []
    for edge in relative_poses:
        u = edge.get("from_room", "room_01")
        v = edge.get("to_room", "room_02")
        if u not in room_ids: room_ids.append(u)
        if v not in room_ids: room_ids.append(v)
        edges.append((u, v, edge["transform"]))
        
    if loop_closures and enable_drift_correction:
        for edge in loop_closures:
            u = edge["from_room"]
            v = edge["to_room"]
            if u not in room_ids: room_ids.append(u)
            if v not in room_ids: room_ids.append(v)
            edges.append((u, v, edge["transform"]))
            
    num_rooms = len(room_ids)
    room_to_idx = {r: i for i, r in enumerate(room_ids)}
    
    # Initial guess via naive dead-reckoning
    init_poses = {room_ids[0]: np.eye(3)}
    visited = {room_ids[0]}
    
    queue = [room_ids[0]]
    # BFS to populate initial poses
    while queue:
        curr = queue.pop(0)
        for u, v, T in edges:
            if u == curr and v not in visited:
                init_poses[v] = init_poses[u] @ T
                visited.add(v)
                queue.append(v)
            elif v == curr and u not in visited:
                init_poses[u] = init_poses[v] @ np.linalg.inv(T)
                visited.add(u)
                queue.append(u)
                
    if not enable_drift_correction:
        return init_poses
        
    # State vector: [x1, y1, t1, x2, y2, t2, ...] for all rooms EXCEPT room_0 (pinned to origin)
    def wrap_angle(angle: float) -> float:
        return (angle + np.pi) % (2 * np.pi) - np.pi
        
    def extract_state(pose_dict):
        state = []
        for i in range(1, num_rooms):
            r = room_ids[i]
            T = pose_dict[r]
            x, y, t = extract_se2_params(T)
            state.extend([x, y, t])
        return np.array(state)
        
    def apply_state(state):
        pose_dict = {room_ids[0]: np.eye(3)}
        for i in range(1, num_rooms):
            x = state[(i-1)*3 + 0]
            y = state[(i-1)*3 + 1]
            t = state[(i-1)*3 + 2]
            pose_dict[room_ids[i]] = make_se2_matrix(x, y, t)
        return pose_dict
        
    def residuals(state):
        pose_dict = apply_state(state)
        res = []
        for u, v, T_meas in edges:
            T_u = pose_dict[u]
            T_v = pose_dict[v]
            
            # T_meas is relative pose of v in u's frame: T_u^-1 * T_v
            T_pred = np.linalg.inv(T_u) @ T_v
            
            x_p, y_p, t_p = extract_se2_params(T_pred)
            x_m, y_m, t_m = extract_se2_params(T_meas)
            
            # Error vector
            ex = x_p - x_m
            ey = y_p - y_m
            et = wrap_angle(t_p - t_m)
            
            # Weighting: translations are in meters, angles in radians. 
            # Give angular errors slightly more weight to keep rooms orthogonal if possible
            res.extend([ex, ey, et * 2.0])
            
        return np.array(res)

    x0 = extract_state(init_poses)
    if len(x0) == 0:
        return init_poses # Only 1 room
        
    # Optimize with Huber loss (soft_l1) to resist outlier loop closures
    result = least_squares(residuals, x0, loss='soft_l1', f_scale=0.1, method='trf')
    
    optimized_poses = apply_state(result.x)
    
    # Manhattan snapping
    for r, T in optimized_poses.items():
        x, y, t = extract_se2_params(T)
        is_rect = True
        if room_rectilinear_map is not None:
            is_rect = room_rectilinear_map.get(r, True)
            
        if is_rect:
            t_snapped = snap_to_manhattan_angle(t)
            optimized_poses[r] = make_se2_matrix(x, y, t_snapped)
        else:
            optimized_poses[r] = T
            
    return optimized_poses

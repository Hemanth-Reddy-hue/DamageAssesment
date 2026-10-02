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

def optimize_pose_graph(
    relative_poses: List[Dict[str, Any]],
    loop_closures: Optional[List[Dict[str, Any]]] = None,
    enable_drift_correction: bool = True
) -> Dict[str, np.ndarray]:
    """Optimize 2D SE(2) pose graph across interconnected rooms.
    
    Args:
        relative_poses: List of dicts specifying sequential/relative transforms:
            [{"from_room": "room_01", "to_room": "room_02", "transform": 3x3 array}, ...]
        loop_closures: Optional list of closure constraints:
            [{"from_room": "room_03", "to_room": "room_01", "transform": 3x3 array}, ...]
        enable_drift_correction: If False, performs uncorrected dead reckoning for ablation.
        
    Returns:
        Mapping from room_id to 3x3 global SE(2) transformation matrix.
    """
    poses: Dict[str, np.ndarray] = {}
    if not relative_poses:
        poses["room_01"] = np.eye(3)
        return poses

    # Base anchor: First room is at global origin (0, 0, 0)
    root_room = relative_poses[0].get("from_room", "room_01")
    poses[root_room] = np.eye(3)

    # 1. Forward traversal (dead reckoning accumulation)
    for edge in relative_poses:
        u = edge["from_room"]
        v = edge["to_room"]
        rel_T = edge["transform"]
        if rel_T.shape == (4, 4):
            # Convert 4x4 to 3x3 SE(2) if 3D matrix passed
            rel_T = np.array([
                [rel_T[0, 0], rel_T[0, 1], rel_T[0, 3]],
                [rel_T[1, 0], rel_T[1, 1], rel_T[1, 3]],
                [0, 0, 1]
            ], dtype=float)

        parent_pose = poses.get(u, np.eye(3))
        accumulated_pose = parent_pose @ rel_T

        if enable_drift_correction:
            # Snap rotation to Manhattan grid if nearly orthogonal
            x, y, theta = extract_se2_params(accumulated_pose)
            snapped_theta = snap_to_manhattan_angle(theta)
            accumulated_pose = make_se2_matrix(x, y, snapped_theta)

        poses[v] = accumulated_pose

    # 2. Loop closure optimization if drift correction is enabled
    if enable_drift_correction and loop_closures:
        for closure in loop_closures:
            u = closure["from_room"]
            v = closure["to_room"]
            meas_T = closure["transform"]

            if u in poses and v in poses:
                # Expected pose of v from u vs current pose of v
                predicted_v = poses[u] @ meas_T
                current_v = poses[v]

                err_x = predicted_v[0, 2] - current_v[0, 2]
                err_y = predicted_v[1, 2] - current_v[1, 2]

                # Distribute error proportionally across intermediate nodes
                room_keys = list(poses.keys())
                num_nodes = len(room_keys)
                if num_nodes > 1:
                    for i, r_id in enumerate(room_keys):
                        weight = float(i) / float(num_nodes - 1)
                        poses[r_id][0, 2] -= err_x * weight
                        poses[r_id][1, 2] -= err_y * weight

    return poses

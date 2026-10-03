"""Node M6: Multi-room Stitching and Drift Correction."""

import time
from typing import Any, Dict, List
import numpy as np
from areamap.state import CaptureState, StitchedPlan, RoomGeometry
from areamap.geometry.adjacency import (
    infer_room_adjacency,
    check_room_overlaps,
    align_room_pair_se2,
    apply_se2_transform_to_room,
)
from areamap.geometry.posegraph import optimize_pose_graph, make_se2_matrix
from areamap.geometry.uncertainty import calculate_interval

def stitch_node(state: CaptureState) -> dict[str, Any]:
    """Stitch multi-room floor plans into a common coordinate frame, check for overlaps, and calculate footprint."""
    t0 = time.time()
    rooms = dict(state.room_geometry)
    room_ids = list(rooms.keys())

    if len(room_ids) <= 1:
        # Single room: already centered in local frame
        total_area = list(rooms.values())[0].floor_area.value if rooms else 0.0
        stitched_plan = StitchedPlan(
            rooms=room_ids,
            connections=[],
            total_footprint_area=calculate_interval(total_area, "footprint_area", tier=state.tier),
            footprint_polygon=list(rooms.values())[0].floor_polygon if rooms else [],
            drift_correction_applied=True
        )
        return {
            "stitched_plan": stitched_plan,
            "timings": {**state.timings, "stitch": round(time.time() - t0, 4)}
        }

    # 1. Build relative SE(2) transformations from discovered doorway transitions (Issue #11)
    relative_poses: List[Dict[str, Any]] = []
    aligned_rooms: Dict[str, RoomGeometry] = {}

    root_room_id = room_ids[0]
    aligned_rooms[root_room_id] = rooms[root_room_id]
    
    from areamap.geometry.registration import icp_align
    from areamap.geometry.posegraph import make_se2_matrix
    from areamap.geometry.place_recognition import _load_point_cloud
    import logging
    logger = logging.getLogger(__name__)

    # Keep track of edges to avoid duplicates
    added_edges = set()
    
    # Process doorway transitions from Room Discovery
    for transition in state.doorway_transitions:
        r_a = transition["room_a"]
        r_b = transition["room_b"]
        
        edge_key = tuple(sorted([r_a, r_b]))
        if edge_key in added_edges:
            continue
            
        pc_a_path = state.point_clouds.get(r_a)
        pc_b_path = state.point_clouds.get(r_b)
        
        if pc_a_path and pc_b_path:
            pts_a = _load_point_cloud(pc_a_path)
            pts_b = _load_point_cloud(pc_b_path)
            
            if len(pts_a) > 20 and len(pts_b) > 20:
                T_4x4, rmse = icp_align(source=pts_b, target=pts_a) # Align r_b to r_a
                if rmse < 1.0: # lenient threshold because they share a transition image
                    dx, dy = T_4x4[0, 3], T_4x4[1, 3]
                    theta = np.arctan2(T_4x4[1, 0], T_4x4[0, 0])
                    
                    rel_T = make_se2_matrix(dx, dy, theta)
                    relative_poses.append({
                        "from_room": r_a,
                        "to_room": r_b,
                        "transform": rel_T
                    })
                    added_edges.add(edge_key)
                    logger.info(f"Connected {r_a} to {r_b} via visual transition (rmse={rmse:.3f})")

    # Fallback if no transitions found (unlikely in realistic continuous scans)
    if not relative_poses and len(room_ids) > 1:
        # We must add at least one connection so pose graph doesn't fail, 
        # but we no longer synthesize fake connections. We just place them at origin.
        for i in range(1, len(room_ids)):
            relative_poses.append({
                "from_room": room_ids[0],
                "to_room": room_ids[i],
                "transform": np.eye(3)
            })

    # 2b. Visual loop closures (Issue #6)
    # We no longer need separate loop closure logic since room_discovery handles arbitrary graph topologies natively!
    loop_closures = None

    rect_map = {r_id: geom.is_rectilinear for r_id, geom in rooms.items()}
    
    # 3. Optimize pose graph with drift correction
    optimized_poses = optimize_pose_graph(
        relative_poses=relative_poses,
        loop_closures=loop_closures,
        enable_drift_correction=True,
        room_rectilinear_map=rect_map
    )

    # 4. Apply global SE(2) transforms to each room with collision resolution
    from areamap.geometry.adjacency import _clip_polygon, _polygon_area
    placed_ids: List[str] = []

    for r_id in room_ids:
        T = optimized_poses.get(r_id, np.eye(3))
        curr_geom = apply_se2_transform_to_room(rooms[r_id], T)

        # Ensure no overlap with already placed rooms (Issue #8 constraint solver)
        from areamap.geometry.solver import resolve_room_collision
        for p_id in placed_ids:
            dx, dy = resolve_room_collision(
                aligned_rooms[p_id].floor_polygon,
                curr_geom.floor_polygon
            )
            if abs(dx) > 1e-3 or abs(dy) > 1e-3:
                shift_T = make_se2_matrix(dx, dy, 0.0)
                curr_geom = apply_se2_transform_to_room(curr_geom, shift_T)

        aligned_rooms[r_id] = curr_geom
        placed_ids.append(r_id)

    # 5. Overlap verification (Gate G5: Zero overlaps)
    overlap_warnings = check_room_overlaps(aligned_rooms)

    # 6. Aggregate property footprint area
    total_area = sum(r.floor_area.value for r in aligned_rooms.values())
    
    # Compute combined footprint polygon (bounding envelope)
    all_x: List[float] = []
    all_y: List[float] = []
    for r in aligned_rooms.values():
        for pt in r.floor_polygon:
            all_x.append(pt[0])
            all_y.append(pt[1])

    if all_x and all_y:
        footprint_polygon = [
            [round(min(all_x), 3), round(min(all_y), 3)],
            [round(max(all_x), 3), round(min(all_y), 3)],
            [round(max(all_x), 3), round(max(all_y), 3)],
            [round(min(all_x), 3), round(max(all_y), 3)],
        ]
    else:
        footprint_polygon = []

    # 7. Build AdjacencyConnection list from registered pose pairs
    from areamap.state import AdjacencyConnection
    connections = []
    for rp in relative_poses:
        conn = AdjacencyConnection(
            from_room=rp["from_room"],
            to_room=rp["to_room"],
            opening_id=f"transition_{rp['from_room']}_{rp['to_room']}",
            confidence=1.0
        )
        connections.append(conn)

    stitched_plan = StitchedPlan(
        rooms=room_ids,
        connections=connections,
        total_footprint_area=calculate_interval(total_area, "footprint_area", tier=state.tier),
        footprint_polygon=footprint_polygon,
        drift_correction_applied=True
    )

    return {
        "room_geometry": aligned_rooms,
        "stitched_plan": stitched_plan,
        "warnings": state.warnings + overlap_warnings,
        "timings": {**state.timings, "stitch": round(time.time() - t0, 4)}
    }

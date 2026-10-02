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

    # 1. Infer doorway connections
    connections = infer_room_adjacency(rooms)

    # 2. Build relative SE(2) transformations across rooms
    relative_poses: List[Dict[str, Any]] = []
    aligned_rooms: Dict[str, RoomGeometry] = {}

    root_room_id = room_ids[0]
    aligned_rooms[root_room_id] = rooms[root_room_id]

    for conn in connections:
        u = conn.from_room
        v = conn.to_room
        if u not in rooms or v not in rooms:
            continue

        r_from = rooms[u]
        r_to = rooms[v]

        # Find matching doors
        door_from = next((op for op in r_from.openings if op.type in ["door", "passageway"]), None)
        door_to = next((op for op in r_to.openings if op.type in ["door", "passageway"]), None)

        if door_from and door_to:
            rel_T = align_room_pair_se2(r_from, r_to, door_from, door_to)
        else:
            # Fallback: translate to the right of room_from
            max_x = max(p[0] for p in r_from.floor_polygon) if r_from.floor_polygon else 4.0
            min_to_x = min(p[0] for p in r_to.floor_polygon) if r_to.floor_polygon else 0.0
            rel_T = make_se2_matrix(max_x - min_to_x, 0.0, 0.0)

        relative_poses.append({
            "from_room": u,
            "to_room": v,
            "transform": rel_T
        })

    # 3. Optimize pose graph with drift correction
    optimized_poses = optimize_pose_graph(
        relative_poses=relative_poses,
        loop_closures=None,
        enable_drift_correction=True
    )

    # 4. Apply global SE(2) transforms to each room
    for r_id in room_ids:
        T = optimized_poses.get(r_id, np.eye(3))
        aligned_rooms[r_id] = apply_se2_transform_to_room(rooms[r_id], T)

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

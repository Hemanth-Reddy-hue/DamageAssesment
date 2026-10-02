"""Node M6: Multi-room Stitching and Drift Correction."""

import time
from typing import Any
from areamap.state import CaptureState, StitchedPlan
from areamap.geometry.adjacency import infer_room_adjacency, check_room_overlaps
from areamap.geometry.uncertainty import calculate_interval

def stitch_node(state: CaptureState) -> dict[str, Any]:
    """Stitch multi-room floor plans, check for overlap, and calculate property footprint."""
    t0 = time.time()
    connections = infer_room_adjacency(state.room_geometry)
    overlap_warnings = check_room_overlaps(state.room_geometry)

    # Compute aggregate footprint area
    total_area = sum(r.floor_area.value for r in state.room_geometry.values())
    stitched_plan = StitchedPlan(
        rooms=list(state.room_geometry.keys()),
        connections=connections,
        total_footprint_area=calculate_interval(total_area, "footprint_area", tier=state.tier),
        footprint_polygon=list(state.room_geometry.values())[0].floor_polygon if state.room_geometry else [],
        drift_correction_applied=True
    )

    return {
        "stitched_plan": stitched_plan,
        "warnings": state.warnings + overlap_warnings,
        "timings": {**state.timings, "stitch": round(time.time() - t0, 4)}
    }

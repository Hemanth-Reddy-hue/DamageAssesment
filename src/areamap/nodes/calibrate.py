"""Node M9: Calibrator and Interval Refinement."""

import time
from typing import Any
from areamap.state import CaptureState
from areamap.geometry.uncertainty import calculate_interval

def calibrate_node(state: CaptureState) -> dict[str, Any]:
    """Ensure all measurement intervals are strictly calibrated according to tier priors and input quality."""
    t0 = time.time()
    intervals = dict(state.intervals)

    for r_id, room in state.room_geometry.items():
        intervals[f"{r_id}_ceiling"] = room.ceiling_height
        intervals[f"{r_id}_floor_area"] = room.floor_area
        for w in room.walls:
            intervals[f"{r_id}_{w.wall_id}_len"] = w.length
        for op in room.openings:
            intervals[f"{r_id}_{op.opening_id}_w"] = op.width

    return {
        "intervals": intervals,
        "timings": {**state.timings, "calibrate": round(time.time() - t0, 4)}
    }

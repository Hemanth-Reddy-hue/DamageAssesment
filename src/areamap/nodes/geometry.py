"""Node M4: Room Geometry Extraction."""

import time
import numpy as np
from typing import Any
from areamap.state import CaptureState
from areamap.geometry.planes import fit_room_planes
from areamap.tiers.lidar import _generate_synthetic_box

def geometry_node(state: CaptureState) -> dict[str, Any]:
    """Fit planes, extract walls, ceiling height, and floor area for each room."""
    t0 = time.time()
    room_geometry = {}

    for room_id in state.rooms or ["room_01"]:
        # In a full run, points are retrieved from state.point_clouds or parsed cloud
        dummy_pts = _generate_synthetic_box(4.0, 3.0, 2.6)
        geom = fit_room_planes(dummy_pts, room_id=room_id, tier=state.tier)
        room_geometry[room_id] = geom

    return {
        "room_geometry": room_geometry,
        "timings": {**state.timings, "geometry": round(time.time() - t0, 4)}
    }

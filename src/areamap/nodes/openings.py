"""Node M5: Openings and Portals Detection."""

import time
from typing import Any
from areamap.state import CaptureState, Opening
from areamap.geometry.openings import detect_openings_from_cutouts

def openings_node(state: CaptureState) -> dict[str, Any]:
    """Detect architectural openings (doors and windows) per room."""
    t0 = time.time()
    all_openings: dict[str, list[Opening]] = {}
    updated_geom = dict(state.room_geometry)

    for room_id, geom in updated_geom.items():
        detected = detect_openings_from_cutouts(geom.walls, tier=state.tier)
        all_openings[room_id] = detected
        geom.openings = detected

    return {
        "openings": all_openings,
        "room_geometry": updated_geom,
        "timings": {**state.timings, "openings": round(time.time() - t0, 4)}
    }

"""Node M5: Openings and Portals Detection."""

import time
from pathlib import Path
from typing import Any
import numpy as np
from areamap.state import CaptureState, Opening
from areamap.geometry.openings import detect_openings_from_cutouts

def openings_node(state: CaptureState) -> dict[str, Any]:
    """Detect architectural openings (doors and windows) per room."""
    t0 = time.time()
    all_openings: dict[str, list[Opening]] = {}
    updated_geom = dict(state.room_geometry)

    for room_id, geom in updated_geom.items():
        pts = None
        if room_id in state.point_clouds:
            cloud_file = Path(state.point_clouds[room_id])
            if cloud_file.exists():
                try:
                    pts = np.load(cloud_file)
                except Exception:
                    pts = None

        ceil_h = geom.ceiling_height.value if geom.ceiling_height else 2.60
        detected = detect_openings_from_cutouts(
            walls=geom.walls,
            tier=state.tier,
            point_cloud=pts,
            ceiling_height=ceil_h
        )
        all_openings[room_id] = detected
        geom.openings = detected

    return {
        "openings": all_openings,
        "room_geometry": updated_geom,
        "timings": {**state.timings, "openings": round(time.time() - t0, 4)}
    }

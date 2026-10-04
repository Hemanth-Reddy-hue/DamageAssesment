import time
from pathlib import Path
import numpy as np
from typing import Any
from areamap.config import settings
from areamap.state import CaptureState
from areamap.geometry.planes import fit_room_planes
from areamap.tiers.lidar import _generate_synthetic_box


def geometry_node(state: CaptureState) -> dict[str, Any]:
    """Fit planes, extract walls, ceiling height, and floor area for each room."""
    t0 = time.time()
    room_geometry = {}
    warnings = state.warnings.copy()

    # Extract scale uncertainty if available from video/registration metadata
    scale_meta = state.device_meta.get("scale", {}) if isinstance(state.device_meta, dict) else {}
    scale_uncertainty = scale_meta.get("relative_uncertainty", 0.0) if isinstance(scale_meta, dict) else 0.0

    for room_id in state.rooms or ["room_01"]:
        pts = None
        cloud_ref = state.point_clouds.get(room_id)
        if cloud_ref and Path(cloud_ref).exists():
            try:
                pts = np.load(cloud_ref)
            except Exception:
                pts = None

        synthetic_used = False
        if pts is None or len(pts) == 0:
            if settings.allow_synthetic:
                pts = _generate_synthetic_box(4.0, 3.0, 2.6)
                synthetic_used = True
                warnings.append(f"SYNTHETIC GEOMETRY — NOT A MEASUREMENT for {room_id}")
            else:
                raise RuntimeError(
                    f"No point cloud data found for room {room_id}. Synthetic fallback is disabled (ALLOW_SYNTHETIC=False)."
                )

        r_name = room_id.replace("_", " ").title()
        geom = fit_room_planes(
            pts,
            room_id=room_id,
            room_name=r_name,
            tier=state.tier or "lidar",
            scale_relative_uncertainty=scale_uncertainty,
        )
        if synthetic_used:
            geom.provenance = "synthetic"
        room_geometry[room_id] = geom

    for rid, geom in room_geometry.items():
        if geom.provenance in ("fallback", "synthetic"):
            msg = f"Room {rid} used synthetic geometry — not a measurement."
            if msg not in warnings:
                warnings.append(msg)

    return {
        "room_geometry": room_geometry,
        "warnings": warnings,
        "timings": {**state.timings, "geometry": round(time.time() - t0, 4)}
    }

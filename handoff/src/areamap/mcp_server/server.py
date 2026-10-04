"""MCP server exposing AreaMap geometry and rendering tools over stdio."""

import sys
import json
from typing import Any
import numpy as np

from areamap.geometry.planes import fit_room_planes
from areamap.geometry.openings import detect_openings_from_cutouts
from areamap.geometry.uncertainty import calculate_interval
from areamap.render.plan_svg import render_plan_svg

def handle_rpc_call(method: str, params: dict[str, Any]) -> dict[str, Any]:
    """Handle standard RPC invocations for tools."""
    if method == "fit_room_geometry":
        room_id = params.get("room_id", "room_01")
        tier = params.get("tier", "lidar")
        dummy_pts = np.empty((0, 3))
        geom = fit_room_planes(dummy_pts, room_id=room_id, tier=tier)
        return geom.model_dump()

    elif method == "calculate_interval":
        val = params.get("value", 1.0)
        mtype = params.get("type", "wall")
        tier = params.get("tier", "lidar")
        res = calculate_interval(val, mtype, tier=tier)
        return res.model_dump()

    return {"error": f"Unknown method: {method}"}

def run_stdio_server():
    """Run MCP stdio listener loop."""
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            req = json.loads(line)
            method = req.get("method")
            params = req.get("params", {})
            req_id = req.get("id")
            result = handle_rpc_call(method, params)
            resp = {"jsonrpc": "2.0", "id": req_id, "result": result}
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()
        except Exception as e:
            sys.stderr.write(f"Error handling MCP call: {e}\n")

if __name__ == "__main__":
    run_stdio_server()

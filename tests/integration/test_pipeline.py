"""Integration test executing the full AreaMap pipeline."""

import os
from pathlib import Path
from areamap.state import CaptureState
from areamap.graph import build_areamap_graph

def test_full_pipeline_run(tmp_path):
    out_dir = tmp_path / "out"
    state = CaptureState(
        capture_path="tests/fixtures/synthetic_room.json",
        tier="lidar"
    )

    graph = build_areamap_graph()
    res = graph.invoke(state)

    # Check state updates
    assert "room_01" in res.get("room_geometry", {})
    assert len(res.get("openings", {})) > 0
    assert len(res.get("damage", [])) > 0
    assert len(res.get("scope_items", [])) > 0
    assert res.get("qa_report", {}).get("passed") is True

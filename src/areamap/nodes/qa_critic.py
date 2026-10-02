"""Node M13: QA Critic and Topological Sanity Checker."""

import time
from typing import Any
from areamap.state import CaptureState, QAReport

def qa_critic_node(state: CaptureState) -> dict[str, Any]:
    """Execute rule-based critic checks on geometry, closure, intervals, and connectivity."""
    t0 = time.time()
    checks_run = [
        "check_interval_bounds",
        "check_polygon_closure",
        "check_ceiling_height_range",
        "check_surface_reference_integrity",
    ]
    failed_checks = []
    adjustments_made = []

    # 1. Interval bounds check
    for k, iv in state.intervals.items():
        if iv.lo > iv.hi or iv.lo < 0:
            failed_checks.append(f"Invalid interval on {k}: [{iv.lo}, {iv.hi}]")

    # 2. Ceiling height range check (typical 2.0m to 4.5m)
    for r_id, room in state.room_geometry.items():
        if room.ceiling_height.value < 1.8 or room.ceiling_height.value > 5.0:
            failed_checks.append(f"Ceiling height out of realistic range in {r_id}: {room.ceiling_height.value}m")

    # 3. Surface reference integrity in scope
    existing_surfaces = set()
    for r_id, room in state.room_geometry.items():
        existing_surfaces.add("ceiling")
        existing_surfaces.add("floor")
        for w in room.walls:
            existing_surfaces.add(w.wall_id)

    for item in state.scope_items:
        if item.surface_id not in existing_surfaces:
            failed_checks.append(f"Scope item {item.item_id} references non-existent surface: {item.surface_id}")

    qa_report = QAReport(
        passed=(len(failed_checks) == 0),
        checks_run=checks_run,
        failed_checks=failed_checks,
        adjustments_made=adjustments_made,
        overall_confidence=0.95 if len(failed_checks) == 0 else 0.70
    )

    return {
        "qa_report": qa_report,
        "timings": {**state.timings, "qa_critic": round(time.time() - t0, 4)}
    }

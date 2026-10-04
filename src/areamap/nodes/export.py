"""Node M14: Export and Plan Rendering."""

import json
import time
from pathlib import Path
from typing import Any
from areamap.state import CaptureState
from areamap.render.plan_svg import render_plan_svg

def export_node(state: CaptureState, output_dir: Path | str | None = None) -> dict[str, Any]:
    """Export finalized results to plan.json, plan.svg, and run_log.json."""
    t0 = time.time()
    target_dir = output_dir or getattr(state, "output_dir", None)
    if not target_dir or target_dir == "out" or Path(target_dir) == Path("out"):
        capture_path = getattr(state, "capture_path", None)
        if capture_path:
            clean_path = str(capture_path).strip().strip("\"'").rstrip("/\\")
            p = Path(clean_path)
            if p.is_file() or (p.suffix and not p.is_dir()):
                parent_name = p.parent.name
                if parent_name and parent_name.lower() not in ["", ".", "data", "raw"]:
                    folder_name = parent_name
                else:
                    folder_name = p.stem or "capture"
            else:
                folder_name = p.name or "capture"
            target_dir = Path("out") / folder_name
        else:
            target_dir = Path("out")

    state.output_dir = str(target_dir)
    out = Path(target_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. Output plan.json
    state_dict = state.model_dump(mode="json")
    plan_json_path = out / "plan.json"
    with open(plan_json_path, "w", encoding="utf-8") as f:
        json.dump(state_dict, f, indent=2)

    # 2. Render plan.svg
    plan_svg_path = out / "plan.svg"
    render_plan_svg(state, plan_svg_path)

    # 3. Output run_log.json
    run_log = {
        "capture_path": state.capture_path,
        "tier": state.tier,
        "timings": {**state.timings, "export": round(time.time() - t0, 4)},
        "warnings": state.warnings,
        "qa_passed": state.qa_report.passed if state.qa_report else True
    }
    with open(out / "run_log.json", "w", encoding="utf-8") as f:
        json.dump(run_log, f, indent=2)

    return {
        "timings": {**state.timings, "export": round(time.time() - t0, 4)}
    }

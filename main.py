"""AreaMap: iPhone Capture to Dimensioned Floor Plan, Damage Findings & Scope.

Primary CLI entry point. Supports single-room and multi-room captures
across all three tiers (LiDAR, Video Walkthrough, and Photo Stills).
"""

import sys
import os
import argparse
import subprocess
from pathlib import Path
from typing import Optional

# Ensure src is on sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parent
SRC_DIR = WORKSPACE_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from areamap.config import settings
from areamap.state import CaptureState
from areamap.nodes import (
    ingest_node,
    geometry_node,
    openings_node,
    stitch_node,
    calibrate_node,
    damage_node,
    concealed_node,
    scope_node,
    qa_critic_node,
    export_node,
)

BANNER = r"""
================================================================================
     _                          __  __             
    / \   _ __ ___  __ _  /\/\  \ \/ /   AreaMap Pipeline v1.0
   / _ \ | '__/ _ \/ _` |/    \  \  /    iPhone Capture -> Dimensioned Floor Plan
  / ___ \| | |  __/ (_| / /\/\ \ /  \    Damage Assessment & Calibrated Scope
 /_/   \_\_|  \___|\__,_\/    \//_/\_\   Conformal Intervals | Schema Compliant
================================================================================
"""

INSTRUCTIONS = """
QUICK-START INSTRUCTIONS:
-------------------------
1. Run on a LiDAR scan directory:
   python main.py Data/SingleRoom --out out/lidar

2. Run on a handheld Video walkthrough:
   python main.py Data/1BHKRoom/1bhKRoom.mp4 --out out/1bhk

3. Run with a real measured ceiling height (recommended):
   python main.py Data/1BHKRoom/1bhKRoom.mp4 --reference-height 2.7

4. Run fully offline / no LLMs:
   python main.py Data/1BHKRoom/1bhKRoom.mp4 --no-llm

5. Run geometry-only (no Hugging Face model downloads):
   python main.py Data/1BHKRoom/1bhKRoom.mp4 --no-local-models

6. Run automated test suite:
   python main.py --test
"""


def get_folder_name(capture_path: str) -> str:
    """Extract semantic folder name from capture path for output directory grouping."""
    clean_path = str(capture_path).strip().strip("\"'").rstrip("/\\")
    p = Path(clean_path)
    if p.is_file() or (p.suffix and not p.is_dir()):
        parent_name = p.parent.name
        if parent_name and parent_name.lower() not in ["", ".", "data", "raw"]:
            return parent_name
        return p.stem or "capture"
    return p.name or "capture"


def resolve_output_dir(capture_path: str, output_dir: Optional[str] = None) -> str:
    """Resolve output directory to out/{FolderName} unless a custom directory was specified."""
    folder_name = get_folder_name(capture_path)
    if not output_dir or Path(output_dir) == Path("out"):
        return str(Path("out") / folder_name)
    return str(Path(output_dir))


def _run_node(label, node_fn, state, failures):
    try:
        updates = node_fn(state)
    except Exception as exc:  # partial-result policy (M15)
        msg = f"NODE_FAILED {label}: {type(exc).__name__}: {str(exc)[:200]}"
        state.warnings.append(msg)
        failures.append(msg)
        return False
    if updates and isinstance(updates, dict):
        for k, v in updates.items():
            setattr(state, k, v)
    return True


def _write_emergency_plan(state, out_dir):
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    (p / "plan.json").write_text(state.model_dump_json(indent=2), encoding="utf-8")


def run_pipeline(
    capture_path: str,
    tier: str = "auto",
    output_dir: Optional[str] = None,
    verbose: bool = False,
    no_llm: bool = False,
    no_local_models: bool = False,
    reference_height: Optional[float] = None,
    rooms_json: Optional[str] = None,
    allow_synthetic: bool = False,
) -> dict:
    """Execute the full AreaMap pipeline with step-by-step progress logging."""
    resolved_output_dir = resolve_output_dir(capture_path, output_dir)
    print(f"\n[AreaMap] Input Target    : {capture_path}")
    print(f"[AreaMap] Selected Tier   : {tier.upper()}")
    print(f"[AreaMap] Output Directory: {resolved_output_dir}")
    if reference_height:
        print(f"[AreaMap] Known Reference : {reference_height:.2f} m (ceiling_height)")
    if no_llm:
        print("[AreaMap] Cloud LLMs     : DISABLED (--no-llm)")
    if no_local_models:
        print("[AreaMap] Local Models   : DISABLED (--no-local-models)")
    print("-" * 80)

    # 1. Apply global configuration flags
    if no_llm:
        settings.llm_enabled = False
    if allow_synthetic:
        settings.allow_synthetic = True
    if reference_height is not None:
        settings.known_reference_m = reference_height
        settings.known_reference_kind = "ceiling_height"

    # 2. Local Models Startup (PLAN2.md)
    from areamap.models.manager import get_model_manager
    model_mgr = get_model_manager(enabled=not no_local_models)
    if not no_local_models and settings.models_preload:
        print("  -> Initializing local models (soft-failing)...", end="", flush=True)
        model_mgr.load_room_classifier()
        print(" [READY]")

    # 3. Handle manual rooms_json override if supplied
    if rooms_json and Path(rooms_json).exists():
        target_override = Path(capture_path).parent / "rooms.json" if Path(capture_path).is_file() else Path(capture_path) / "rooms.json"
        if not target_override.exists() or target_override != Path(rooms_json):
            try:
                target_override.write_text(Path(rooms_json).read_text(encoding="utf-8"), encoding="utf-8")
                print(f"[AreaMap] Loaded rooms configuration: {rooms_json}")
            except Exception:
                pass

    # 4. Initialize State
    tier_arg = tier if tier in ["lidar", "video", "photo"] else None
    state = CaptureState(
        capture_path=str(capture_path),
        tier=tier_arg,
        output_dir=str(resolved_output_dir),
    )

    steps = [
        ("[1/9] Ingestion & Tier Router", ingest_node),
        ("[2/9] RANSAC Room Geometry & Planes", geometry_node),
        ("[3/9] Opening Cutouts & Phantom Suppression", openings_node),
        ("[4/9] Multi-Room Alignment & Stitching", stitch_node),
        ("[5/9] Interval Calibration", calibrate_node),
        ("[6/9] Damage Proposals", damage_node),
        ("[7/9] Forensic Rules", concealed_node),
        ("[8/9] Repair Scope", scope_node),
        ("[9/9] QA Critic", qa_critic_node),
    ]
    failures: list[str] = []
    for label, node_fn in steps:
        print(f"  -> {label}...", end="", flush=True)
        ok = _run_node(label, node_fn, state, failures)
        print(" [DONE]" if ok else " [FAILED]")

    if failures:
        from areamap.state import QAReport
        prior = state.qa_report
        state.qa_report = QAReport(
            passed=False,
            checks_run=prior.checks_run if prior else [],
            failed_checks=failures + (prior.failed_checks if prior else []),
            overall_confidence=0.0,
        )

    def _export(s):
        return export_node(s, output_dir=resolved_output_dir)
    if not _run_node("export", _export, state, failures):
        _write_emergency_plan(state, resolved_output_dir)

    try:
        _print_summary(state, resolved_output_dir)
    except Exception as exc:
        print(f"[AreaMap] summary skipped: {exc}")
    return state.model_dump()


def _print_summary(state: CaptureState, output_dir: str):
    """Print clean formatted summary of pipeline outputs."""
    out_path = Path(output_dir)
    plan_json = out_path / "plan.json"
    plan_svg = out_path / "plan.svg"
    run_log = out_path / "run_log.json"

    print("=" * 80)
    print("                      CAPTURE ASSESSMENT SUMMARY")
    print("=" * 80)
    print(f"Tier Used          : {state.tier.upper() if state.tier else 'UNKNOWN'}")
    print(f"Rooms Processed    : {len(state.room_geometry)} room(s) -> {list(state.room_geometry.keys())}")

    # Scale & Registration telemetry (WP7 / PLAN2.md)
    dev_meta = state.device_meta if isinstance(state.device_meta, dict) else {}
    if "scale" in dev_meta:
        sc = dev_meta["scale"]
        mth = sc.get("method", "unknown")
        conf = sc.get("confidence", 0.0)
        unc = sc.get("relative_uncertainty", 0.0)
        fac = sc.get("factor")
        print(f"Scale Recovery     : {fac} via {mth} (conf: {conf:.2f}, uncertainty: ±{int(unc*100)}%)")
    if "registration" in dev_meta:
        rg = dev_meta["registration"]
        rat = rg.get("ratio", 0.0)
        n_reg = rg.get("n_registered", 0)
        n_tot = rg.get("n_frames", 0)
        print(f"SfM Registration   : {n_reg}/{n_tot} frames ({rat*100:.1f}%) across {rg.get('n_models', 1)} model(s)")

    # Footprint
    if state.stitched_plan:
        fp = state.stitched_plan.total_footprint_area
        print(f"Total Footprint    : {fp.value:.2f} m^2 [{fp.lo:.2f}, {fp.hi:.2f}] (confidence: {int(fp.confidence_level*100)}%)")
        print(f"Inter-Room Doors   : {len(state.stitched_plan.connections)} connection(s)")
        print(f"Drift Correction   : {'APPLIED' if state.stitched_plan.drift_correction_applied else 'OFF'}")

    # Per-room details
    print("\nRoom Dimensions:")
    for r_id, room in state.room_geometry.items():
        ceil = room.ceiling_height
        area = room.floor_area
        print(f"  * {room.room_name} ({r_id}) [provenance: {room.provenance}]:")
        print(f"      Floor Area     : {area.value:.2f} m^2 [{area.lo:.2f}, {area.hi:.2f}]")
        print(f"      Ceiling Height : {ceil.value:.2f} m [{ceil.lo:.2f}, {ceil.hi:.2f}] (method: {ceil.method})")
        print(f"      Perimeter Walls: {len(room.walls)} wall segments")
        print(f"      Openings       : {len(room.openings)} opening(s)")
        for op in room.openings:
            print(f"        - {op.type.upper()}: width={op.width.value:.2f}m [{op.width.lo:.2f}, {op.width.hi:.2f}], pos={op.position[:2]}")

    # Damage & Scope
    if state.damage:
        print(f"\nDamage Findings ({len(state.damage)}):")
        for dmg in state.damage:
            print(f"  * {dmg.damage_id}: {dmg.damage_class.upper()} on {dmg.surface_id} (severity: {dmg.severity})")

    if state.concealed_flags:
        print(f"\nConcealed Damage Flags ({len(state.concealed_flags)}):")
        for flag in state.concealed_flags:
            print(f"  * [{flag.rule_id}] {flag.description}")

    if state.scope_items:
        print(f"\nRepair Scope Items ({len(state.scope_items)}):")
        for item in state.scope_items:
            print(f"  * {item.item_id}: {item.item_description} ({item.quantity.value:.1f} {item.unit})")

    if state.warnings:
        print(f"\nPipeline Warnings ({len(state.warnings)}):")
        for w in state.warnings:
            print(f"  ! {w}")

    print("\nGenerated Artifacts:")
    print(f"  [1] JSON Schema State : {plan_json.resolve()}")
    print(f"  [2] Vector Floor Plan : {plan_svg.resolve()}")
    print(f"  [3] Execution Audit   : {run_log.resolve()}")
    print("=" * 80)


def interactive_prompt() -> Optional[str]:
    """Provide a user-friendly interactive selector when run with no arguments."""
    print(INSTRUCTIONS)
    print("AVAILABLE SAMPLE DATA IN WORKSPACE:")
    samples = [
        ("1", "Data/1BHKRoom/1bhKRoom.mp4", "Two-Room Video Walkthrough (1BHK clip)"),
        ("2", "Data/SingleRoom", "Real iPhone LiDAR Scan (1,715 depth frames + odometry)"),
        ("3", "Data/raw/sample_living_room_photos", "Photo Stills Directory (multi-photo)"),
    ]
    for key, path, desc in samples:
        exists = " [AVAILABLE]" if Path(path).exists() else " [MISSING]"
        print(f"  [{key}] {path:<36} : {desc}{exists}")

    print("\nPress 1, 2, or 3 to run a sample, enter a custom file/folder path, or 'q' to quit:")
    try:
        choice = input("Enter choice or path: ").strip().strip("\"'")
    except (EOFError, KeyboardInterrupt):
        return None

    if not choice or choice.lower() in ["q", "quit", "exit"]:
        return None

    choice_map = {
        "1": "Data/1BHKRoom/1bhKRoom.mp4",
        "2": "Data/SingleRoom",
        "3": "Data/raw/sample_living_room_photos"
    }
    return choice_map.get(choice, choice)


def main():
    parser = argparse.ArgumentParser(
        description="AreaMap: Turn smartphone photos, video, or LiDAR into a dimensioned floor plan with calibrated scope.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=INSTRUCTIONS
    )
    parser.add_argument("capture", nargs="?", default=None, help="Path to capture directory, video file, or photo folder")
    parser.add_argument("--tier", choices=["auto", "lidar", "video", "photo"], default="auto", help="Sensor tier (default: auto-detect)")
    parser.add_argument("--out", default=None, help="Output directory for plan.json and plan.svg (default: out/<folder_name>)")
    parser.add_argument("--offline", action="store_true", default=True, help="Force strict offline local execution (default: True)")
    parser.add_argument("--verbose", action="store_true", help="Print verbose intermediate execution details")
    parser.add_argument("--no-llm", action="store_true", help="Disable all cloud LLM calls; video tier geometry runs without LLM")
    parser.add_argument("--no-local-models", action="store_true", help="Skip loading Hugging Face models; rooms use numbered names")
    parser.add_argument("--reference-height", type=float, default=None, help="Known reference ceiling height in metres (e.g. 2.7)")
    parser.add_argument("--rooms-json", type=str, default=None, help="Path to manual room boundaries rooms.json")
    parser.add_argument("--allow-synthetic", action="store_true", help="Allow synthetic fallback geometry (demo/test only; QA fails)")
    parser.add_argument("--test", action="store_true", help="Run the automated test suite (pytest)")
    parser.add_argument("--drift-ablation", action="store_true", help="Run Gate G4 drift correction ablation study")
    parser.add_argument("--instructions", action="store_true", help="Print detailed usage instructions and exit")

    args = parser.parse_args()

    print(BANNER)

    if args.instructions:
        print(INSTRUCTIONS)
        return

    if args.test:
        print("[AreaMap] Launching Pytest Test Suite...")
        cmd = [sys.executable, "-m", "pytest", "tests/", "-v"]
        subprocess.run(cmd, cwd=str(WORKSPACE_ROOT))
        return

    if args.drift_ablation:
        print("[AreaMap] Running Gate G4 Drift Correction Ablation...")
        from bench.ablation_drift import run_drift_ablation
        run_drift_ablation()
        return

    target_path = args.capture
    if not target_path:
        target_path = interactive_prompt()
        if not target_path:
            print("\nExiting. Use 'python main.py --help' for usage options.")
            return

    res = run_pipeline(
        capture_path=target_path,
        tier=args.tier,
        output_dir=args.out,
        verbose=args.verbose,
        no_llm=args.no_llm,
        no_local_models=args.no_local_models,
        reference_height=args.reference_height,
        rooms_json=args.rooms_json,
        allow_synthetic=args.allow_synthetic,
    )
    if any(w.startswith("NODE_FAILED") for w in res.get("warnings", [])):
        sys.exit(2)


if __name__ == "__main__":
    main()

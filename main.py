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
   python main.py Data/SingleRoom/rgb.mp4 --tier video --out out/video

3. Run on a folder of room photos:
   python main.py Data/raw/sample_living_room_photos --tier photo --out out/photo

4. Run on a multi-room house folder (subdirectories per room):
   python main.py path/to/house_capture --out out/multiroom

5. Run automated test suite (37 tests):
   python main.py --test

6. Run Gate G4 drift correction ablation study:
   python main.py --drift-ablation
"""

def run_pipeline(
    capture_path: str,
    tier: str = "auto",
    output_dir: str = "out",
    verbose: bool = False
) -> dict:
    """Execute the full AreaMap pipeline with step-by-step progress logging."""
    print(f"\n[AreaMap] Input Target   : {capture_path}")
    print(f"[AreaMap] Selected Tier  : {tier.upper()}")
    print(f"[AreaMap] Output Directory: {output_dir}")
    print("-" * 80)

    # Initialize State
    # "auto" is a valid sentinel: ingest_node will call detect_tier() when tier is not
    # explicitly lidar/video/photo. We default CaptureState to "lidar" only as a type
    # placeholder; ingest_node updates it correctly via auto-detection.
    tier_arg = tier if tier in ["lidar", "video", "photo"] else "lidar"
    state = CaptureState(
        capture_path=str(capture_path),
        # Pass "lidar" as placeholder; ingest_node overrides when tier is auto
        tier=tier_arg,
        output_dir=str(output_dir),
    )
    # Store whether we're in auto mode so ingest_node can detect properly
    state_tier_is_auto = tier not in ["lidar", "video", "photo"]

    steps = [
        ("[1/8] Ingestion & Tier Router", ingest_node),
        ("[2/8] RANSAC Room Geometry & Planes", geometry_node),
        ("[3/8] Opening Cutouts & Phantom Suppression", openings_node),
        ("[4/8] Multi-Room Alignment & Stitching", stitch_node),
        ("[5/8] Calibrated Conformal Intervals", calibrate_node),
        ("[6/8] Damage Proposals & Semantic Overlay", damage_node),
        ("[7/8] Forensic Rules & Repair Scope", concealed_node),
        ("[8/8] QA Critic & Vector Plan Export", export_node),
    ]

    # Additional intermediate node execution
    for label, node_fn in steps:
        print(f"  -> {label}...", end="", flush=True)
        updates = node_fn(state)
        if updates and isinstance(updates, dict):
            for k, v in updates.items():
                setattr(state, k, v)
        print(" [DONE]")

    # Run remaining scope and QA nodes if not in step list
    scope_updates = scope_node(state)
    if scope_updates:
        for k, v in scope_updates.items():
            setattr(state, k, v)

    qa_updates = qa_critic_node(state)
    if qa_updates:
        for k, v in qa_updates.items():
            setattr(state, k, v)

    # Re-export with finalized scope & QA report
    export_node(state, output_dir=output_dir)

    print("-" * 80)
    print("[AreaMap] Pipeline execution completed successfully!\n")
    _print_summary(state, output_dir)
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
    print(f"Tier Used          : {state.tier.upper()}")
    print(f"Rooms Processed    : {len(state.room_geometry)} room(s) -> {list(state.room_geometry.keys())}")

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
        print(f"  * {room.room_name} ({r_id}):")
        print(f"      Floor Area     : {area.value:.2f} m^2 [{area.lo:.2f}, {area.hi:.2f}]")
        print(f"      Ceiling Height : {ceil.value:.2f} m [{ceil.lo:.2f}, {ceil.hi:.2f}]")
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
        ("1", "Data/SingleRoom", "Real iPhone LiDAR Scan (1,715 depth frames + odometry)"),
        ("2", "Data/SingleRoom/rgb.mp4", "Handheld Video Walkthrough (1920x1440 clip)"),
        ("3", "Data/raw/sample_living_room_photos", "Photo Stills Directory (multi-photo)"),
    ]
    for key, path, desc in samples:
        exists = " [AVAILABLE]" if Path(path).exists() else " [MISSING]"
        print(f"  [{key}] {path:<36} : {desc}{exists}")

    print("\nPress 1, 2, or 3 to run a sample, enter a custom file/folder path, or 'q' to quit:")
    try:
        choice = input("Enter choice or path: ").strip()
    except (EOFError, KeyboardInterrupt):
        return None

    if not choice or choice.lower() in ["q", "quit", "exit"]:
        return None

    choice_map = {
        "1": "Data/SingleRoom",
        "2": "Data/SingleRoom/rgb.mp4",
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
    parser.add_argument("--out", default="out", help="Output directory for plan.json and plan.svg (default: out)")
    parser.add_argument("--offline", action="store_true", default=True, help="Force strict offline local execution (default: True)")
    parser.add_argument("--verbose", action="store_true", help="Print verbose intermediate execution details")
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

    run_pipeline(
        capture_path=target_path,
        tier=args.tier,
        output_dir=args.out,
        verbose=args.verbose
    )

if __name__ == "__main__":
    main()

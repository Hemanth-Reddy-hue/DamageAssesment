"""Primary CLI entry point for AreaMap capture execution."""

import sys
import argparse
from pathlib import Path

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from areamap.state import CaptureState
from areamap.graph import build_areamap_graph

def main():
    parser = argparse.ArgumentParser(description="AreaMap: iPhone Capture to Dimensioned Floor Plan")
    parser.add_argument("capture", help="Path to input capture directory or file")
    parser.add_argument("--tier", choices=["photo", "video", "lidar"], default="lidar", help="Force specific tier")
    parser.add_argument("--out", default="out", help="Output directory (default: out)")
    args = parser.parse_args()

    print(f"[AreaMap] Initializing capture pipeline for: {args.capture}")
    print(f"[AreaMap] Output directory: {args.out}")

    init_state = CaptureState(
        capture_path=str(args.capture),
        tier=args.tier,
        output_dir=str(args.out),
    )

    graph = build_areamap_graph()
    result = graph.invoke(init_state)

    print(f"[AreaMap] Completed successfully. Outputs saved in '{args.out}/plan.json' and '{args.out}/plan.svg'")

if __name__ == "__main__":
    main()

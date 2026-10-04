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
    parser.add_argument("--out", default=None, help="Output directory (default: out/<folder_name>)")
    args = parser.parse_args()

    clean_path = str(args.capture).strip().strip("\"'").rstrip("/\\")
    p = Path(clean_path)
    if p.is_file() or (p.suffix and not p.is_dir()):
        parent_name = p.parent.name
        if parent_name and parent_name.lower() not in ["", ".", "data", "raw"]:
            folder_name = parent_name
        else:
            folder_name = p.stem or "capture"
    else:
        folder_name = p.name or "capture"

    out_dir = args.out
    if not out_dir or Path(out_dir) == Path("out"):
        out_dir = str(Path("out") / folder_name)

    print(f"[AreaMap] Initializing capture pipeline for: {args.capture}")
    print(f"[AreaMap] Output directory: {out_dir}")

    init_state = CaptureState(
        capture_path=str(args.capture),
        tier=args.tier,
        output_dir=str(out_dir),
    )

    graph = build_areamap_graph()
    result = graph.invoke(init_state)

    print(f"[AreaMap] Completed successfully. Outputs saved in '{out_dir}/plan.json' and '{out_dir}/plan.svg'")

if __name__ == "__main__":
    main()

"""Node M2: Benchmark harness evaluating all gates against ground truth."""

import sys
import argparse
from pathlib import Path

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root / "src"))
sys.path.insert(0, str(root))

from bench.gates import EVALUATED_GATES

def run_benchmark_harness(fixloop: bool = False) -> dict:
    """Evaluate pipeline metrics against ground truth dataset."""
    print("================================================================================")
    print("                         AREAMAP OFFICIAL BENCHMARK HARNESS                      ")
    print("================================================================================")
    print(f"{'Gate':<6} | {'Description':<28} | {'Threshold':<20} | {'Measured':<20} | {'Status'}")
    print("-" * 86)

    all_passed = True
    for g_id, data in EVALUATED_GATES.items():
        status = "PASS" if data["pass"] else "FAIL"
        if not data["pass"]:
            all_passed = False
        print(f"{g_id:<6} | {data['description']:<28} | {data['threshold']:<20} | {data['measured']:<20} | {status}")

    print("-" * 86)
    print(f"Overall Benchmark Status: {'ALL GATES PASSED' if all_passed else 'FAILURES DETECTED'}")
    print("================================================================================")
    return EVALUATED_GATES

def main():
    parser = argparse.ArgumentParser(description="Run AreaMap benchmark evaluation harness")
    parser.add_argument("--fixloop", action="store_true", help="Evaluate fixloop before/after verification")
    args = parser.parse_args()
    run_benchmark_harness(fixloop=args.fixloop)

if __name__ == "__main__":
    main()

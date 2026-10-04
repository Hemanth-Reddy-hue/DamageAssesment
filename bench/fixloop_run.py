import argparse, json, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CASES = {"floor_only": "Data/single_scan_floor_only", "with_ceiling": "Data/single_scan_with_ceiling"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, choices=["before", "after"])
    a = ap.parse_args()
    summary = {}
    for name, cap in CASES.items():
        out = f"fixloop/{a.label}/{name}"
        subprocess.run([sys.executable, "main.py", cap, "--tier", "lidar", "--out", out,
                        "--no-llm", "--no-local-models"], cwd=ROOT, check=False)
        plan = json.loads((ROOT / out / "plan.json").read_text(encoding="utf-8"))
        g = next(iter(plan["room_geometry"].values()))
        c = g["ceiling_height"]
        summary[name] = dict(value=c["value"], lo=c["lo"], hi=c["hi"], width=round(c["hi"] - c["lo"], 4),
                             method=c["method"], warned=any("ceiling" in w.lower() and "prior" in w.lower()
                                                            for w in plan.get("warnings", [])))
    p = ROOT / f"fixloop/{a.label}/summary.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

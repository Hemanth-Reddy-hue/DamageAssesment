"""Regenerate every reported number from raw inputs. Run from the repo root (works inside an extracted bundle).
   python scripts/reproduce.py            # regenerate and compare with reports/expected_numbers.json
   python scripts/reproduce.py --snapshot # (author only) write expected_numbers.json from current outputs"""
import argparse, json, os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run(cmd, env=None):
    print(">>", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=ROOT, env=env).returncode


def key(r):
    return f"{r['case']}|{r['type']}|{r['feature']}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", action="store_true")
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    env = dict(os.environ, OFFLINE="1")   # LLM answers replay from data/cache; no network is used
    skipped = []
    for c in json.loads((ROOT / "bench/cases.json").read_text(encoding="utf-8")):
        if a.only and c["case_id"] not in a.only:
            continue
        if not (ROOT / c["capture"]).exists():
            skipped.append(c["case_id"])
            print(f"SKIP {c['case_id']}: {c['capture']} missing (unpack the raw archive)")
            continue
        cmd = [sys.executable, "main.py", c["capture"], "--out", c["out"], "--no-local-models"]
        if c["tier"] != "auto":
            cmd += ["--tier", c["tier"]]
        run(cmd, env)
    run([sys.executable, "bench/headtohead.py"], env)
    run([sys.executable, "bench/ablation_drift.py", "Data/RealHouse", "--tier", "photo"], env)
    run([sys.executable, "bench/harness.py"], env)
    run([sys.executable, "scripts/make_benchmark_report.py"], env)

    res = json.loads((ROOT / "out/bench_results.json").read_text(encoding="utf-8"))
    got = {key(r): r["pred"] for r in res["rows"]}
    gates = {g["gate"]: g["status"] for g in res["gates"]}
    exp_path = ROOT / "reports/expected_numbers.json"
    if a.snapshot:
        exp_path.write_text(json.dumps({"preds": got, "gates": gates}, indent=2, sort_keys=True), encoding="utf-8")
        print("snapshot written")
        return 0

    exp = json.loads(exp_path.read_text(encoding="utf-8"))
    bad = []
    for k, v in exp["preds"].items():
        g = got.get(k)
        if g is None:
            bad.append(f"{k}: missing")
        elif abs(g - v) > max(0.002, 0.001 * abs(v)):
            bad.append(f"{k}: expected {v} got {g}")
    for k, v in exp["gates"].items():
        if gates.get(k) != v:
            bad.append(f"gate {k}: expected {v} got {gates.get(k)}")
    print("\nREPRODUCTION:", "MATCH" if not bad and not skipped else "DIFFERENCES")
    for b in bad:
        print("  -", b)
    if skipped:
        print("  skipped cases:", ", ".join(skipped))
    return 1 if bad or skipped else 0


if __name__ == "__main__":
    sys.exit(main())

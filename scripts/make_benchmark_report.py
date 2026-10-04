"""Compile reports/benchmark_report.md from computed artifacts only."""
from __future__ import annotations
import csv, glob, json, os, platform, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def jload(rel, default=None):
    p = ROOT / rel
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else default


def cload(rel):
    p = ROOT / rel
    if not p.exists():
        return []
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def table(head, rows):
    if not rows:
        return "_no data_\n"
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out) + "\n"


def pct(x):
    return "-" if x is None else f"{x * 100:.1f}%"


def mean(v):
    return sum(v) / len(v) if v else None


def git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def gates_section(res):
    rows = [[g["gate"], g["name"], g["status"], g["evidence"], g["detail"]] for g in res["gates"]]
    return table(["Gate", "Name", "Status", "Evidence", "Detail"], rows)


def tier_section(rows):
    out = []
    for t in ("lidar", "video", "photo"):
        rs = [r for r in rows if r["tier"] == t]
        walls = [r for r in rs if r["type"] == "wall_length" and r["gt"]]
        rel = [r["err"] / r["gt"] for r in walls]
        cm = {}
        for r in rs:
            if r["type"] == "ceiling_height":
                cm[r["method"]] = cm.get(r["method"], 0) + 1
        prec = [r for r in rs if r["sigma"] <= ((r["hi"] - r["lo"]) / 2) / 2]
        cov = mean([1.0 if r["covered"] else 0.0 for r in prec])
        out.append([t, len(rs), len(walls), pct(mean(rel)), pct(max(rel) if rel else None),
                    ", ".join(f"{k}:{v}" for k, v in sorted(cm.items())) or "-",
                    ", ".join(sorted({r["source"] for r in rs})) or "-", len(prec), pct(cov)])
    return table(["Tier", "Reference rows", "Wall rows", "Mean wall error", "Max wall error",
                  "Ceiling methods", "Reference sources", "Precise refs (n)", "Interval coverage"], out)


def repeat_section():
    out = []
    for pr in cload("Data/ground_truth/repeat_pairs.csv"):
        a, b = jload(pr["plan_a"]), jload(pr["plan_b"])
        if not a or not b:
            continue
        ga, gb = a["room_geometry"].get(pr["room_a"]), b["room_geometry"].get(pr["room_b"])
        if not ga or not gb:
            continue
        label = f"{Path(pr['plan_a']).parent.name} vs {Path(pr['plan_b']).parent.name}"
        wa = sorted(w["length"]["value"] for w in ga["walls"])
        wb = sorted(w["length"]["value"] for w in gb["walls"])
        for i, (x, y) in enumerate(zip(wa, wb)):
            tol = max(0.01, 0.005 * x)
            out.append([label, f"wall_rank_{i+1}", f"{x:.3f}", f"{y:.3f}", f"{abs(x-y)*100:.1f} cm",
                        f"{tol*100:.1f} cm", "hit" if abs(x - y) <= tol else "miss"])
        ca, cb = ga["ceiling_height"], gb["ceiling_height"]
        if ca["method"] == "measured" and cb["method"] == "measured":
            d = abs(ca["value"] - cb["value"])
            out.append([label, "ceiling", f"{ca['value']:.3f}", f"{cb['value']:.3f}", f"{d*100:.1f} cm",
                        "1.0 cm", "hit" if d <= 0.01 else "miss"])
        else:
            out.append([label, "ceiling", f"{ca['value']:.3f} ({ca['method']})",
                        f"{cb['value']:.3f} ({cb['method']})", "-", "1.0 cm",
                        "not a measurement in at least one run"])
    return table(["Pair", "Item", "Run A (m)", "Run B (m)", "Delta", "Tolerance", "Verdict"], out)


def h2h_section():
    h = jload("out/headtohead.json")
    if not h or not h.get("rows"):
        return ("**G9 NOT MEASURED.** No consumer-app export was available, so no head-to-head table "
                "can be produced. Nothing in this report estimates one.\n\n"
                "Columns the table would have: room, feature, reference, ours, app, error ours, error app, verdict.\n")
    rows = [[r["room"], r["feature"], f"{r['truth']:.3f} ({r['truth_source']})", f"{r['ours']:.3f}", f"{r['app']:.3f}",
             f"{r['err_ours']*100:.1f} cm", f"{r['err_app']*100:.1f} cm", r["verdict"]] for r in h["rows"]]
    return f"App: {', '.join(h['app'])}. Result: {h['status']} ({h['evidence']}): {h['detail']}\n\n" + \
        table(["Room", "Feature", "Reference", "Ours (m)", "App (m)", "Err ours", "Err app", "Verdict"], rows)


def timing_section():
    out = []
    for p in sorted(glob.glob(str(ROOT / "out" / "*" / "plan.json"))):
        name = Path(p).parent.name
        if name.startswith(("det_run", "ablation_")):
            continue
        d = json.loads(Path(p).read_text(encoding="utf-8"))
        t = d.get("timings") or {}
        if not t:
            continue
        slow = max(t, key=t.get)
        out.append([name, d.get("tier"), len(d.get("rooms", [])), f"{sum(t.values()):.1f}", f"{slow} ({t[slow]:.1f} s)"])
    return table(["Capture", "Tier", "Rooms", "Total pipeline (s)", "Slowest node"], out)


def hard_section():
    out = []
    for h in cload("Data/ground_truth/hard_cases.csv"):
        d = jload(h["plan_json"])
        if not d:
            out.append([h["case_id"], h["kind"], "no output", "-", "-", "-", "-", h.get("notes", "")])
            continue
        geoms = list(d["room_geometry"].values())
        ops = sum(len(g.get("openings", [])) for g in geoms)
        wid = [(w["length"]["hi"] - w["length"]["lo"]) / w["length"]["value"]
               for g in geoms for w in g["walls"] if w["length"]["value"]]
        qa = d.get("qa_report") or {}
        out.append([h["case_id"], h["kind"], d.get("tier"), ops, pct(mean(wid)), len(d.get("warnings", [])),
                    "yes" if qa.get("passed") else "no", h.get("notes", "")])
    return table(["Case", "Kind", "Tier", "Openings reported", "Mean rel. interval width", "Warnings",
                  "QA passed", "Notes"], out)


def fix_section():
    b, a = jload("fixloop/before/summary.json"), jload("fixloop/after/summary.json")
    v = jload("fixloop/verify_report.json")
    if not (b and a):
        return "_fix loop artifacts missing_\n"
    rows = []
    for k in sorted(b):
        rows.append([k, f"{b[k]['value']} / {b[k]['method']} / width {b[k]['width']} m",
                     f"{a[k]['value']} / {a[k]['method']} / width {a[k]['width']} m"])
    s = table(["Capture", "Before (value / method / width)", "After (value / method / width)"], rows)
    return s + (f"\nVerification: {v['detail']}\n" if v else "")


def evidence_section(rows):
    cnt = {}
    for r in rows:
        cnt[r["source"]] = cnt.get(r["source"], 0) + 1
    return ("Reference rows by source: " + (", ".join(f"{k}={v}" for k, v in sorted(cnt.items())) or "none") + ".\n\n"
            "No tape or laser measurements were available. `tile` = floor-tile count, `lidar_ref` = value taken from our own "
            "LiDAR run (a consistency check between tiers, not accuracy), `estimate` = rough estimate (about 5% uncertain). "
            "A gate is reported PASS only when the reference is precise enough to resolve it and FAIL only when the error "
            "exceeds tolerance even after allowing for reference error; otherwise INCONCLUSIVE.\n")


def main():
    res = jload("out/bench_results.json")
    if not res:
        raise SystemExit("run bench/harness.py first")
    rows = res["rows"]
    parts = [
        "# AreaMap Benchmark Report\n",
        f"Generated by `scripts/make_benchmark_report.py` from computed artifacts at commit `{git_hash()}`. "
        "Do not edit by hand: regenerate with `python scripts/reproduce.py`.\n",
        "## 1. Evidence base\n", evidence_section(rows),
        "## 2. Gates (all tiers)\n", gates_section(res),
        "## 3. Accuracy and calibration by tier\n", tier_section(rows),
        "## 4. Repeatability\n", repeat_section(),
        "Repeatable-but-biased versus unrepeatable cannot be separated without a precise reference; this report only "
        "states whether two runs agree.\n",
        "## 5. Head-to-head (G9)\n", h2h_section(),
        "## 6. Timing\n", timing_section(),
        f"Hardware: {platform.platform()}, {platform.processor() or 'cpu n/a'}, {os.cpu_count()} logical CPUs, "
        f"Python {platform.python_version()}. Timings are the per-node values saved in each `plan.json` and exclude model "
        "downloads. The cold-run setup time is in `reports/verification_log.md` (V9).\n",
        "## 7. Hard cases (mirror/glass, low light)\n", hard_section(),
        "No reference exists for these; the table shows what the pipeline claimed (openings, interval width, warnings), "
        "so a phantom opening or an overconfident interval is visible.\n",
        "## 8. Fix loop\n", fix_section(),
        "## 9. Limits of this report\n",
        "- G1 and G2 need tape or laser and are not measured.\n"
        "- Intervals are tier-based relative bands plus noise floors, not fitted conformal intervals; no hold-out split exists "
        "because nothing is fitted.\n"
        "- Damage extents are VLM estimates (`vlm_estimate`), not measured areas.\n",
    ]
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports/benchmark_report.md").write_text("\n".join(parts), encoding="utf-8")
    print("wrote reports/benchmark_report.md")


if __name__ == "__main__":
    main()

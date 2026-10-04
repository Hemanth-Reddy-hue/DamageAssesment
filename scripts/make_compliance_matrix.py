"""Generate compliance_matrix.md. Usage:
   python -m pytest -q --junitxml=out/junit.xml ; python scripts/make_compliance_matrix.py [--junit out/junit.xml]"""
from __future__ import annotations
import argparse, json, subprocess, sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
S = "src/areamap/"

MODULES = [
    ("M0", "Contracts and schema", [S + "state.py", "schema/capture_v1.json"], ["schema/capture_v1.json"], ["tests/unit/test_state.py"]),
    ("M1", "Ingest and router", [S + "nodes/ingest.py"], [], ["tests/video/test_failure_policy.py", "tests/unit/test_partial_result.py"]),
    ("M2", "Benchmark harness", ["bench/harness.py", "bench/gates.py"], ["out/bench_results.json"], ["tests/unit/test_harness_synthetic.py"]),
    ("M3", "LiDAR tier ingest", [S + "tiers/lidar.py"], [], ["tests/unit/test_lidar_ingest.py", "tests/integration/test_lidar_real_data.py"]),
    ("M4", "Room geometry", [S + "geometry/planes.py", S + "nodes/geometry.py"], [],
     ["tests/unit/test_geometry_planes.py", "tests/unit/test_manhattan_classifier.py", "tests/unit/test_ceiling_provenance.py", "tests/unit/test_geometry_fixes_123_45.py"]),
    ("M5", "Openings", [S + "geometry/openings.py", S + "nodes/openings.py"], [], ["tests/unit/test_openings.py", "tests/unit/test_door_detector.py"]),
    ("M6", "Stitcher and drift", [S + "nodes/stitch.py", S + "geometry/posegraph.py", "bench/ablation_drift.py"], ["out/ablation_drift.json"],
     ["tests/unit/test_stitch_multiroom.py", "tests/unit/test_collision_solver.py", "tests/unit/test_place_recognition.py", "tests/integration/test_room_discovery.py"]),
    ("M7", "Video tier", [S + "tiers/video.py", S + "tiers/video_sfm.py"], [],
     ["tests/unit/test_video_tier.py", "tests/video/test_keyframes.py", "tests/video/test_gravity.py", "tests/video/test_scale_fusion.py", "tests/video/test_determinism.py"]),
    ("M8", "Photo tier", [S + "tiers/photo.py", S + "geometry/depth_engine.py"], [], ["tests/unit/test_photo_tier.py", "tests/unit/test_depth_engine.py"]),
    ("M9", "Calibrator", [S + "nodes/calibrate.py", S + "geometry/uncertainty.py", "bench/calibration_report.py"], [], ["tests/unit/test_calibrate.py"]),
    ("M10", "Damage detection (VLM; extent is an estimate)", [S + "nodes/damage.py", S + "llm/client.py"], [],
     ["tests/unit/test_damage_surface_binding.py", "tests/video/test_llm_errors.py", "tests/video/test_llm_no_fabrication.py"]),
    ("M11", "Concealed-damage rules", [S + "nodes/concealed.py", S + "rules/concealed_rules.yaml"], [], ["tests/unit/test_rules.py"]),
    ("M12", "Scope generation", [S + "nodes/scope.py", S + "catalog/scope_items.yaml"], [], ["tests/unit/test_damage_surface_binding.py"]),
    ("M13", "QA critic", [S + "nodes/qa_critic.py"], [], ["tests/unit/test_qa_critic_checks.py"]),
    ("M14", "Export and render", [S + "nodes/export.py", S + "render/plan_svg.py"], [], ["tests/integration/test_pipeline.py"]),
    ("M15", "Orchestration", [S + "graph.py", "main.py", "scripts/run_capture.py"], [], ["tests/integration/test_pipeline.py", "tests/unit/test_partial_result.py"]),
    ("M16", "Head-to-head", ["bench/headtohead.py"], ["out/headtohead.json"], []),
    ("M17", "Fix loop", ["fixloop/declaration.md", "fixloop/diff.patch", "bench/verify_fixloop.py"],
     ["fixloop/before/summary.json", "fixloop/after/summary.json", "fixloop/verify_report.json", "fixloop/postmortem.md"], []),
    ("M18", "Capture protocol and device matrix", ["protocol/capture_protocol.md", "reports/device_matrix.md"], [], []),
    ("M19", "Technical report", ["reports/technical_report.md", "scripts/make_report_tables.py"], ["reports/gate_table.md"], []),
]

GATES = [
    ("G1", "Opening widths <= 2 cm on >= 85%", [S + "geometry/openings.py", "bench/harness.py"]),
    ("G2", "Ceiling <= 1.5 cm; spread <= 1 cm", [S + "geometry/planes.py", "bench/harness.py"]),
    ("G3", "Repeatability 1 cm or 0.5%", ["bench/harness.py", "Data/ground_truth/repeat_pairs.csv"]),
    ("G4", "Drift accountability (on/off ablation)", [S + "geometry/posegraph.py", "bench/ablation_drift.py"]),
    ("G5", "Photo whole-property stitch +/-8%", [S + "tiers/photo.py", S + "nodes/stitch.py"]),
    ("G6", "Photo wall lengths +/-8%", [S + "tiers/photo.py"]),
    ("G7", "Video wall lengths +/-3%", [S + "tiers/video.py"]),
    ("G8", "Calibration: 90% intervals cover 85-95%", [S + "geometry/uncertainty.py", "bench/calibration_report.py"]),
    ("G9", "Head-to-head >= 70% win/tie", ["bench/headtohead.py"]),
    ("G10", "Fix loop", ["fixloop/declaration.md", "fixloop/diff.patch"]),
]

DELIVERABLES = [
    ("D1", "Compliance matrix", ["scripts/make_compliance_matrix.py"], ["compliance_matrix.md"]),
    ("D2", "Capture route + device matrix", ["protocol/capture_protocol.md", "reports/device_matrix.md"], []),
    ("D3", "Repo: README, one command per capture", ["README.md", "main.py", "Makefile", "requirements.txt"], ["reports/verification_log.md"]),
    ("D4", "Reproduction bundle (built last: scripts/build_bundle.py)",
     ["scripts/build_bundle.py", "scripts/verify_bundle.py", "scripts/reproduce.py", "bench/cases.json"], ["reports/expected_numbers.json"]),
    ("D5", "Benchmark report", ["scripts/make_benchmark_report.py"], ["reports/benchmark_report.md"]),
    ("D6", "Fix loop bundle", ["fixloop/declaration.md", "fixloop/diff.patch", "fixloop/prediction.json"],
     ["fixloop/before/summary.json", "fixloop/after/summary.json", "fixloop/verify_report.json"]),
    ("D7", "Technical report (<= 6 pages)", ["reports/technical_report.md"], []),
    ("D8", "Raw benchmark data", ["Data/DATA_INDEX.md", "Data/ground_truth/manifest.csv", "Data/bench_set_coverage.csv"], []),
]


def junit_status(path):
    status = {}
    if not path or not Path(path).exists():
        return status
    for tc in ET.parse(path).getroot().iter("testcase"):
        parts = tc.get("classname", "").split(".")
        f = None
        for i in range(len(parts), 0, -1):
            cand = ROOT.joinpath(*parts[:i]).with_suffix(".py")
            if cand.exists():
                f = cand.relative_to(ROOT).as_posix()
                break
        if f is None:
            continue
        s = status.setdefault(f, {"pass": 0, "fail": 0, "skip": 0})
        if tc.find("failure") is not None or tc.find("error") is not None:
            s["fail"] += 1
        elif tc.find("skipped") is not None:
            s["skip"] += 1
        else:
            s["pass"] += 1
    return status


def exists(p):
    return (ROOT / p).exists()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--junit", default="out/junit.xml")
    a = ap.parse_args()
    jt = junit_status(ROOT / a.junit)
    bench = {}
    bp = ROOT / "out/bench_results.json"
    if bp.exists():
        bench = {g["gate"]: g for g in json.loads(bp.read_text(encoding="utf-8"))["gates"]}

    lines = ["# AreaMap Compliance Matrix", "",
             "Generated by `scripts/make_compliance_matrix.py`. Status is computed from file existence, pytest results "
             "(`out/junit.xml`) and `out/bench_results.json`; it is not edited by hand.", "",
             "| ID | Requirement | File path(s) | Artifact(s) | Status | Evidence |", "|---|---|---|---|---|---|"]
    missing_any, notes = False, []

    def row(i, req, files, arts, status, ev):
        lines.append(f"| {i} | {req} | {', '.join(f'`{f}`' for f in files) or '-'} | "
                     f"{', '.join(f'`{f}`' for f in arts) or '-'} | {status} | {ev} |")

    for mid, req, files, arts, tests in MODULES:
        miss = [p for p in files + arts if not exists(p)]
        if miss:
            missing_any = True
            row(mid, req, files, arts, "MISSING", "missing: " + ", ".join(miss))
            continue
        if not tests:
            row(mid, req, files, arts, "Complete — present (no unit tests)", "artifact/file check only")
            continue
        found = [t for t in tests if t in jt]
        for t in tests:
            if t not in jt:
                notes.append(f"{mid}: no junit result for {t}")
        fail = sum(jt[t]["fail"] for t in found)
        passed = sum(jt[t]["pass"] for t in found)
        if not found:
            status = "Implemented, no test results"
        elif fail:
            status = "Failing tests"
        else:
            status = "Complete — tested"
        row(mid, req, files, arts, status, f"{passed} pass / {fail} fail in {len(found)} test file(s)")

    for gid, req, files in GATES:
        miss = [p for p in files if not exists(p)]
        g = bench.get(gid)
        if miss:
            missing_any = True
            row(gid, req, files, ["out/bench_results.json"], "MISSING", "missing: " + ", ".join(miss))
        elif not g:
            row(gid, req, files, ["out/bench_results.json"], "Not measured", "no harness result")
        else:
            row(gid, req, files, ["out/bench_results.json", "reports/benchmark_report.md"],
                f"{g['status']} · {g['evidence']}", g["detail"])

    for did, req, files, arts in DELIVERABLES:
        miss = [p for p in files + arts if not exists(p)]
        if miss:
            missing_any = True
            row(did, req, files, arts, "MISSING", "missing: " + ", ".join(miss))
        else:
            row(did, req, files, arts, "Complete — present", "all listed files exist")

    if notes:
        lines += ["", "Notes:"] + [f"- {n}" for n in notes]
    (ROOT / "compliance_matrix.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote compliance_matrix.md" + ("  (MISSING entries present)" if missing_any else ""))
    sys.exit(1 if missing_any else 0)


if __name__ == "__main__":
    main()

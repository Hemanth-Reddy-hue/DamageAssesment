import json, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def git(*a):
    return subprocess.check_output(["git", *a], cwd=ROOT, text=True).strip()


def first(marker, path=None):
    args = ["log", "--reverse", "--format=%H", f"-S{marker}"] + (["--", path] if path else [])
    out = git(*args).splitlines()
    return out[0] if out else None


decl = first("CEILING-CLAMP-DECLARATION", "fixloop/declaration.md")
fix = first("CEILING_PRIOR_M", "src/areamap/geometry/planes.py")
order_ok = False
if decl and fix:
    order_ok = subprocess.run(["git", "merge-base", "--is-ancestor", decl, fix], cwd=ROOT).returncode == 0 and decl != fix

pred = json.loads((ROOT / "fixloop/prediction.json").read_text())
before = json.loads((ROOT / "fixloop/before/summary.json").read_text())
after = json.loads((ROOT / "fixloop/after/summary.json").read_text())
checks = []
fo, wc = after["floor_only"], after["with_ceiling"]
checks.append(("floor_only prior", fo["method"] == pred["after"]["floor_only"]["method"]))
checks.append(("floor_only width", fo["width"] >= pred["after"]["floor_only"]["min_width_m"]))
checks.append(("with_ceiling method", wc["method"] == pred["after"]["with_ceiling"]["method"]))
checks.append(("with_ceiling width", wc["width"] <= pred["after"]["with_ceiling"]["max_width_m"]))
checks.append(("before had conformal 2.4", all(v["method"] == "conformal" and abs(v["value"] - 2.4) < 1e-6 for v in before.values())))
diff_nonempty = (ROOT / "fixloop/diff.patch").exists() and (ROOT / "fixloop/diff.patch").stat().st_size > 0
ok = order_ok and diff_nonempty
detail = (f"declaration {str(decl)[:7]} precedes fix {str(fix)[:7]}: {order_ok}; diff.patch non-empty: {diff_nonempty}; "
          + ", ".join(f"{n}={'ok' if v else 'MISSED'}" for n, v in checks))
rep = dict(status="PASS" if ok else "FAIL", detail=detail, n=len(checks) + 2, order_ok=order_ok, checks=dict(checks))
(ROOT / "fixloop/verify_report.json").write_text(json.dumps(rep, indent=2, sort_keys=True), encoding="utf-8")
print(detail)

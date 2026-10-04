"""Build dist/areamap_repo_bundle.zip, dist/areamap_raw_lidar.zip, dist/areamap.gitbundle and manifests."""
import hashlib, json, subprocess, sys, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "dist"
BIG_DIR = 100 * 1024 * 1024
SKIP_PARTS = {"venv", ".venv", "__pycache__", ".pytest_cache", ".git", "dist"}
TEXT = {".py", ".md", ".txt", ".json", ".csv", ".yaml", ".yml", ".toml", ".cfg", ".ini", ".sh", ".ps1", ".patch", ".html", ".svg"}

REPRODUCE_MD = """# Reproducing AreaMap results

1. `git clone areamap.gitbundle areamap && cd areamap`   (or unzip areamap_repo_bundle.zip)
2. Unzip areamap_raw_lidar.zip into the same folder (adds the large LiDAR captures under Data/).
3. `python -m venv venv` then activate it, then `pip install -r requirements.txt`
4. `bash scripts/fetch_models.sh`  (needs HF_TOKEN once; skip with --no-local-models, already set in cases)
5. `python scripts/reproduce.py`   (OFFLINE=1; damage answers replay from data/cache)
6. One capture: `python main.py <capture> --out out/<name>`

Compare with reports/benchmark_report.md. Differences are printed by reproduce.py.
"""


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ok_path(p: Path):
    return not (set(p.parts) & SKIP_PARTS) and p.name != ".env"


def env_secrets():
    vals = []
    e = ROOT / ".env"
    if e.exists():
        for line in e.read_text(encoding="utf-8", errors="ignore").splitlines():
            if "=" in line and not line.strip().startswith("#"):
                v = line.split("=", 1)[1].strip().strip("'\"")
                if len(v) >= 8:
                    vals.append(v)
    return vals


def main():
    DIST.mkdir(exist_ok=True)
    tracked = [ROOT / x for x in subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0") if x]
    A, B = {}, {}
    for p in tracked:
        rel = p.relative_to(ROOT)
        if p.exists() and ok_path(rel) and rel.parts[0] != "Data" and p.stat().st_size < 50 * 1024 * 1024:
            A[rel.as_posix()] = p
    for p in (ROOT / "data/cache").glob("*.json"):
        A[p.relative_to(ROOT).as_posix()] = p
    for extra in ("out/bench_results.json", "out/ablation_drift.json", "out/headtohead.json"):
        if (ROOT / extra).exists():
            A[extra] = ROOT / extra
    data = ROOT / "Data"
    for top in sorted(data.iterdir()) if data.exists() else []:
        files = [f for f in top.rglob("*") if f.is_file()] if top.is_dir() else [top]
        size = sum(f.stat().st_size for f in files)
        dest = B if size >= BIG_DIR else A
        for f in files:
            if ok_path(f.relative_to(ROOT)):
                dest[f.relative_to(ROOT).as_posix()] = f

    secrets = env_secrets()
    for arc, p in {**A, **B}.items():
        if p.suffix.lower() in TEXT and p.stat().st_size < 5 * 1024 * 1024:
            txt = p.read_text(encoding="utf-8", errors="ignore")
            for s in secrets:
                if s in txt:
                    sys.exit(f"ABORT: a secret from .env appears in {arc}")

    def write(zip_path, files, manifest_name, extra=None):
        manifest = {arc: {"size": p.stat().st_size, "sha256": sha(p)} for arc, p in sorted(files.items())}
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
            for arc, p in sorted(files.items()):
                print(f"Adding to {zip_path.name}: {arc}")
                z.write(p, arc)
            z.writestr(manifest_name, json.dumps(manifest, indent=1, sort_keys=True))
            for name, text in (extra or {}).items():
                z.writestr(name, text)
        (DIST / manifest_name).write_text(json.dumps(manifest, indent=1, sort_keys=True), encoding="utf-8")
        print(zip_path.name, len(manifest), "files", round(zip_path.stat().st_size / 1e6, 1), "MB")

    print("Writing main zip...")
    write(DIST / "areamap_repo_bundle.zip", A, "MANIFEST.json", {"REPRODUCE.md": REPRODUCE_MD})
    if B:
        print("Writing raw lidar zip...")
        write(DIST / "areamap_raw_lidar.zip", B, "MANIFEST_RAW.json")
    print("Writing git bundle...")
    subprocess.run(["git", "bundle", "create", str(DIST / "areamap.gitbundle"), "--all"], cwd=ROOT, check=True)
    print("bundle built; large raw archive is", "present" if B else "not needed")


if __name__ == "__main__":
    main()

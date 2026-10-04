"""Unpack the bundle into a temp dir, verify hashes, optionally install in a fresh venv, run reproduce.py."""
import argparse, hashlib, json, os, subprocess, sys, tempfile, time, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh-venv", action="store_true")
    a = ap.parse_args()
    tmp = Path(tempfile.mkdtemp(prefix="areamap_verify_"))
    repo = tmp / "repo"
    for z in ("areamap_repo_bundle.zip", "areamap_raw_lidar.zip"):
        zp = ROOT / "dist" / z
        if zp.exists():
            zipfile.ZipFile(zp).extractall(repo)
    bad = []
    for mf in ("MANIFEST.json", "MANIFEST_RAW.json"):
        mp = repo / mf
        if mp.exists():
            for arc, m in json.loads(mp.read_text()).items():
                p = repo / arc
                if not p.exists() or sha(p) != m["sha256"]:
                    bad.append(arc)
    py, t_setup = sys.executable, 0.0
    if a.fresh_venv:
        t0 = time.time()
        subprocess.run([sys.executable, "-m", "venv", str(tmp / "venv")], check=True)
        py = str(tmp / "venv" / ("Scripts" if os.name == "nt" else "bin") / "python")
        subprocess.run([py, "-m", "pip", "install", "-q", "-r", "requirements.txt"], cwd=repo, check=True)
        t_setup = time.time() - t0
    t1 = time.time()
    rc = subprocess.run([py, "scripts/reproduce.py"], cwd=repo, env=dict(os.environ, OFFLINE="1")).returncode
    rep = {"hash_mismatches": bad, "reproduce_exit": rc, "setup_seconds": round(t_setup, 1),
           "reproduce_seconds": round(time.time() - t1, 1), "fresh_venv": a.fresh_venv}
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports/bundle_verified.json").write_text(json.dumps(rep, indent=2, sort_keys=True))
    print(rep)
    sys.exit(1 if bad or rc else 0)


if __name__ == "__main__":
    main()

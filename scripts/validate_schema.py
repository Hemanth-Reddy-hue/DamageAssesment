import glob, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from areamap.state import CaptureState

schema = json.loads(Path("schema/capture_v1.json").read_text(encoding="utf-8"))
try:
    import jsonschema
except ImportError:
    jsonschema = None
bad = 0
for p in sorted(glob.glob("out/*/plan.json")):
    d = json.loads(Path(p).read_text(encoding="utf-8"))
    try:
        CaptureState.model_validate(d)
        if jsonschema:
            jsonschema.validate(d, schema)
        print("OK  ", p)
    except Exception as e:
        bad += 1
        print("FAIL", p, str(e)[:200])
sys.exit(1 if bad else 0)

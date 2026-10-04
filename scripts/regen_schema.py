import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from areamap.state import CaptureState

Path("schema/capture_v1.json").write_text(json.dumps(CaptureState.model_json_schema(), indent=2), encoding="utf-8")
print("schema regenerated")

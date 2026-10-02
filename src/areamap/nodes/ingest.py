import time
import numpy as np
from pathlib import Path
from typing import Any
from areamap.state import CaptureState
from areamap.tiers.lidar import ingest_lidar_capture
from areamap.tiers.video import ingest_video_capture
from areamap.tiers.photo import ingest_photo_capture

def detect_tier(path: Path) -> str:
    """Detect capture tier from folder structure and contents."""
    if path.is_file() and path.suffix.lower() in [".mp4", ".mov", ".avi"]:
        return "video"

    if path.is_dir():
        # Check for LiDAR indicators
        if (path / "camera_matrix.csv").exists() or (path / "odometry.csv").exists() or (path / "depth").exists():
            return "lidar"

        # Check for single video file inside dir
        video_files = list(path.glob("*.mp4")) + list(path.glob("*.mov"))
        if video_files and not (path / "depth").exists():
            return "video"

        # Check for photo folder
        img_files = list(path.glob("*.jpg")) + list(path.glob("*.jpeg")) + list(path.glob("*.png"))
        if img_files or any(d.is_dir() for d in path.iterdir()):
            return "photo"

    return "lidar"  # Default fallback

def ingest_node(state: CaptureState) -> dict[str, Any]:
    """Ingest input directory, route to appropriate tier, and load points."""
    t0 = time.time()
    capture_path = Path(state.capture_path)

    if not capture_path.exists():
        state.warnings.append(f"Input path does not exist: {capture_path}. Using synthetic room.")
        tier = state.tier or "lidar"
    else:
        tier = detect_tier(capture_path)

    updates: dict[str, Any] = {"tier": tier}

    if tier == "lidar":
        pts, meta = ingest_lidar_capture(capture_path)
    elif tier == "video":
        pts, meta = ingest_video_capture(capture_path)
    else:
        pts, meta = ingest_photo_capture(capture_path)

    # Cache point cloud artifact
    cache_dir = Path("data/cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cloud_path = cache_dir / "cloud_room_01.npy"
    np.save(cloud_path, pts)

    updates["device_meta"] = meta
    updates["rooms"] = ["room_01"]
    updates["point_clouds"] = {"room_01": str(cloud_path)}
    updates["timings"] = {**state.timings, "ingest": round(time.time() - t0, 4)}

    return updates

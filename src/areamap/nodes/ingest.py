"""Node M1: Ingest and Tier Router."""

import time
import numpy as np
from pathlib import Path
from typing import Any, List
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
        tier = state.tier if state.tier in ["lidar", "video", "photo"] else detect_tier(capture_path)

    # 1. Check for multi-room directory structure
    subdirs: List[Path] = []
    if capture_path.exists() and capture_path.is_dir():
        ignore_names = {"depth", "confidence", "cache", "__pycache__", ".git", ".pytest_cache"}
        subdirs = sorted([d for d in capture_path.iterdir() if d.is_dir() and d.name.lower() not in ignore_names])

    cache_dir = Path("data/cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    if len(subdirs) >= 2:
        # Multi-room capture dataset
        rooms_list: List[str] = []
        point_clouds_map: dict[str, str] = {}
        device_meta = {"tier": tier, "multi_room": True, "room_count": len(subdirs), "source": str(capture_path)}

        for idx, s_dir in enumerate(subdirs):
            r_id = f"room_{idx+1:02d}"
            r_tier = state.tier if state.tier in ["lidar", "video", "photo"] else detect_tier(s_dir)
            if r_tier == "lidar":
                pts, meta = ingest_lidar_capture(s_dir)
            elif r_tier == "video":
                pts, meta = ingest_video_capture(s_dir)
            else:
                pts, meta = ingest_photo_capture(s_dir)

            cloud_path = cache_dir / f"cloud_{r_id}.npy"
            np.save(cloud_path, pts)
            rooms_list.append(r_id)
            point_clouds_map[r_id] = str(cloud_path)

        return {
            "tier": tier,
            "device_meta": device_meta,
            "rooms": rooms_list,
            "point_clouds": point_clouds_map,
            "timings": {**state.timings, "ingest": round(time.time() - t0, 4)}
        }

    # 2. Single-room capture
    updates: dict[str, Any] = {"tier": tier}

    if tier == "lidar":
        pts, meta = ingest_lidar_capture(capture_path)
    elif tier == "video":
        pts, meta = ingest_video_capture(capture_path)
    else:
        pts, meta = ingest_photo_capture(capture_path)

    cloud_path = cache_dir / "cloud_room_01.npy"
    np.save(cloud_path, pts)

    updates["device_meta"] = meta
    updates["rooms"] = ["room_01"]
    updates["point_clouds"] = {"room_01": str(cloud_path)}
    updates["timings"] = {**state.timings, "ingest": round(time.time() - t0, 4)}

    return updates

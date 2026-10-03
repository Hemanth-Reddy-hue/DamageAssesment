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
        # Always auto-detect from path content first
        detected = detect_tier(capture_path)
        # Only use state.tier if it was *explicitly* forced by the user via CLI flag.
        # The state always carries a default of "lidar" even when --tier was not passed,
        # so we rely on detect_tier() unless the caller explicitly chose a non-matching tier.
        if state.tier in ["video", "photo"] and state.tier != detected:
            # User explicitly overrode: respect it
            tier = state.tier
        else:
            tier = detected

    if tier == "photo":
        # 1. Gather all photos recursively, ignoring directory names
        valid_exts = {".jpg", ".jpeg", ".png"}
        image_files = []
        if capture_path.is_file():
            image_files = [capture_path]
        else:
            image_files = [p for p in capture_path.rglob("*") if p.is_file() and p.suffix.lower() in valid_exts]
            
        # 2. Run Automatic Room Discovery
        from areamap.geometry.room_discovery import discover_rooms
        clusters, transitions = discover_rooms(image_files)
        
        rooms_list: List[str] = []
        point_clouds_map: dict[str, str] = {}
        device_meta = {"tier": tier, "multi_room": len(clusters) > 1, "room_count": len(clusters), "source": str(capture_path)}
        
        cache_dir = Path("data/cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 3. Process each discovered room cluster
        for cluster in clusters:
            pts, meta = ingest_photo_capture(cluster.images)
            cloud_path = cache_dir / f"cloud_{cluster.room_id}.npy"
            np.save(cloud_path, pts)
            
            rooms_list.append(cluster.room_id)
            point_clouds_map[cluster.room_id] = str(cloud_path)
            
        return {
            "tier": tier,
            "device_meta": device_meta,
            "rooms": rooms_list,
            "point_clouds": point_clouds_map,
            "doorway_transitions": transitions,  # Pass to posegraph/stitch later!
            "timings": {**state.timings, "ingest": round(time.time() - t0, 4)}
        }
    else:
        # Fallback for LiDAR/Video (keep existing structure-based logic for now)
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
                r_id = s_dir.name.lower()
                r_tier = state.tier if state.tier in ["lidar", "video", "photo"] else detect_tier(s_dir)
                if r_tier == "lidar":
                    pts, meta = ingest_lidar_capture(s_dir)
                elif r_tier == "video":
                    pts, meta = ingest_video_capture(s_dir)

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

        # Single-room capture fallback
        updates: dict[str, Any] = {"tier": tier}
        if tier == "lidar":
            pts, meta = ingest_lidar_capture(capture_path)
        elif tier == "video":
            pts, meta = ingest_video_capture(capture_path)
            
        cloud_path = cache_dir / "cloud_room_01.npy"
        np.save(cloud_path, pts)

        updates["device_meta"] = meta
        updates["rooms"] = ["room_01"]
        updates["point_clouds"] = {"room_01": str(cloud_path)}
        updates["timings"] = {**state.timings, "ingest": round(time.time() - t0, 4)}
        return updates

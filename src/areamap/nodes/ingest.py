"""Tier routing, sensor ingest, and point cloud registration.

Implements WP7:
- New return contract: ingest_video_capture(path) -> VideoReconstruction
- Writes VideoRoom.points to data/cache/cloud_<room_id>.npy
- Sets state.rooms, state.point_clouds, state.device_meta with scale, registration, llm, warnings
- Strict failure policy: no synthetic geometry unless ALLOW_SYNTHETIC=True
"""

import time
import logging
from pathlib import Path
from typing import Any, List, Optional
import numpy as np

from areamap.config import settings
from areamap.state import CaptureState
from areamap.tiers.lidar import ingest_lidar_capture
from areamap.tiers.photo import ingest_photo_capture
from areamap.tiers.video import ingest_video_capture

logger = logging.getLogger(__name__)


def detect_tier(path: Path) -> str:
    """Classify capture directory or file into LiDAR, Video, or Photo Stills tier."""
    if path.is_file():
        suffix = path.suffix.lower()
        if suffix in [".mp4", ".mov", ".avi", ".mkv"]:
            return "video"
        elif suffix in [".ply", ".las", ".xyz", ".pcd"]:
            return "lidar"
        elif suffix in [".jpg", ".jpeg", ".png", ".heic"]:
            return "photo"

    # Directory checks
    if (path / "depth").is_dir() or (path / "confidence").is_dir():
        return "lidar"

    vids = list(path.glob("*.mp4")) + list(path.glob("*.mov"))
    if vids:
        return "video"

    return "photo"


def ingest_node(state: CaptureState) -> dict[str, Any]:
    """Ingest input directory or file, route to appropriate tier, and load points."""
    t0 = time.time()
    capture_path = Path(state.capture_path)

    if not capture_path.exists():
        if not settings.allow_synthetic:
            raise FileNotFoundError(f"Input path does not exist: {capture_path}")
        state.warnings.append(f"Input path does not exist: {capture_path}. Using synthetic room.")
        tier = state.tier or "lidar"
    else:
        # Always auto-detect from path content first
        detected = detect_tier(capture_path)
        if state.tier is not None and state.tier != detected:
            tier = state.tier
        else:
            tier = detected

    cache_dir = Path("data/cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Check for multi-room directory structure with subdirectories
    subdirs: List[Path] = []
    if capture_path.exists() and capture_path.is_dir():
        ignore_names = {"depth", "confidence", "cache", "__pycache__", ".git", ".pytest_cache", "images", "models", "sfm"}
        subdirs = sorted([d for d in capture_path.iterdir() if d.is_dir() and d.name.lower() not in ignore_names])

    if len(subdirs) >= 2:
        # Multi-room capture dataset with subdirectories per room
        rooms_list: List[str] = []
        point_clouds_map: dict[str, str] = {}
        device_meta = {"tier": tier, "multi_room": True, "room_count": len(subdirs), "source": str(capture_path)}

        for idx, s_dir in enumerate(subdirs):
            r_id = s_dir.name.lower()
            r_tier = state.tier if state.tier is not None else detect_tier(s_dir)
            if r_tier == "lidar":
                pts, meta = ingest_lidar_capture(s_dir)
                cams = None
            elif r_tier == "video":
                recon = ingest_video_capture(s_dir)
                pts = recon.rooms[0].points if recon.rooms else np.zeros((0, 3), dtype=np.float32)
                cams = recon.rooms[0].camera_positions if recon.rooms else None
                meta = recon.to_metadata_dict()
            else:  # photo
                img_files = [p for p in s_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
                pts, meta = ingest_photo_capture(img_files)
                cams = None

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

    # Photo tier single directory / discovery
    if tier == "photo":
        valid_exts = {".jpg", ".jpeg", ".png"}
        if capture_path.is_file():
            image_files = [capture_path]
        else:
            image_files = [p for p in capture_path.rglob("*") if p.is_file() and p.suffix.lower() in valid_exts]

        from areamap.geometry.room_discovery import discover_rooms
        clusters, transitions = discover_rooms(image_files, output_dir=state.output_dir)

        rooms_list = []
        point_clouds_map = {}
        device_meta = {"tier": tier, "multi_room": len(clusters) > 1, "room_count": len(clusters), "source": str(capture_path)}

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
            "doorway_transitions": transitions,
            "timings": {**state.timings, "ingest": round(time.time() - t0, 4)}
        }

    # LiDAR or Video Tier
    updates: dict[str, Any] = {"tier": tier}
    rooms_dict = {}
    meta = {"tier": tier, "source": str(capture_path)}

    camera_positions_dict = {}

    if tier == "lidar":
        pts, meta = ingest_lidar_capture(capture_path)
        rooms_dict["room_00"] = pts

    elif tier == "video":
        recon = ingest_video_capture(capture_path)
        if hasattr(recon, "status"):
            if recon.status == "failed" and not settings.allow_synthetic:
                raise RuntimeError(
                    f"Video reconstruction failed: {recon.warnings}. Synthetic geometry is disabled (ALLOW_SYNTHETIC=False)."
                )

            if hasattr(recon, "rooms") and recon.rooms:
                for v_room in recon.rooms:
                    rooms_dict[v_room.room_id] = v_room.points
                    if v_room.camera_positions is not None and len(v_room.camera_positions) > 0:
                        camera_positions_dict[v_room.room_id] = v_room.camera_positions
                meta = recon.to_metadata_dict()
                if recon.transitions:
                    updates["doorway_transitions"] = recon.transitions
                for w in recon.warnings:
                    if w not in state.warnings:
                        state.warnings.append(w)
            elif isinstance(recon, tuple):
                pts_out, meta = recon
                if isinstance(pts_out, dict):
                    rooms_dict = pts_out
                else:
                    rooms_dict["room_00"] = pts_out
            elif isinstance(recon, dict):
                rooms_dict = recon
            else:
                rooms_dict["room_00"] = recon
        else:
            # Tuple or dict fallback
            if isinstance(recon, tuple):
                pts_out, meta = recon
                if isinstance(pts_out, dict):
                    rooms_dict = pts_out
                else:
                    rooms_dict["room_00"] = pts_out
            elif isinstance(recon, dict):
                rooms_dict = recon
            else:
                rooms_dict["room_00"] = recon

    updates["device_meta"] = meta
    updates["rooms"] = list(rooms_dict.keys())
    updates["point_clouds"] = {}

    for r_id, r_pts in rooms_dict.items():
        cloud_path = cache_dir / f"cloud_{r_id}.npy"
        np.save(cloud_path, r_pts)
        updates["point_clouds"][r_id] = str(cloud_path)

    import json
    updates["camera_positions"] = {}
    for r_id, cam_poses in camera_positions_dict.items():
        cam_path = cache_dir / f"cams_{r_id}.json"
        with open(cam_path, "w") as f:
            json.dump(cam_poses, f)
        updates["camera_positions"][r_id] = str(cam_path)

    updates["timings"] = {**state.timings, "ingest": round(time.time() - t0, 4)}
    return updates

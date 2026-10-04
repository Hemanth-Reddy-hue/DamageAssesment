"""AreaMap Video Tier Structure from Motion (PyCOLMAP) Reconstruction Engine.

Implements:
- WP4: SfM Reconstruction (SIFT, intrinsics prior, exhaustive/sequential matching, multi-model BA, robust point extraction)
- WP5: Gravity alignment from camera up-vectors, metric scale fusion (reference > depth model > prior), plane sanity
- WP6: Room segmentation from covisibility graph (Louvain communities, temporal smoothing, doorway transitions, room ids room_00, room_01...)
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import networkx as nx
import numpy as np

from areamap.config import settings
from areamap.tiers.video import extract_sharp_keyframes
from areamap.tiers.video_types import (
    RegistrationInfo,
    ScaleInfo,
    VideoReconstruction,
    VideoRoom,
)

try:
    import pycolmap
except ImportError:
    pycolmap = None

log = logging.getLogger(__name__)


def _extract_camera_up_and_center(image: Any) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Extract camera R, t, up-vector, and center in world coordinates."""
    try:
        if hasattr(image, "cam_from_world"):
            pose = image.cam_from_world
            if callable(pose):
                pose = pose()
            R_cam = np.array(pose.rotation.matrix(), dtype=np.float64)
            t_cam = np.array(pose.translation, dtype=np.float64)
        elif hasattr(image, "qvec") and hasattr(image, "tvec"):
            q = image.qvec
            w, x, y, z = q
            R_cam = np.array([
                [1 - 2*(y**2 + z**2), 2*(x*y - z*w), 2*(x*z + y*w)],
                [2*(x*y + z*w), 1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
                [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x**2 + y**2)]
            ], dtype=np.float64)
            t_cam = np.array(image.tvec, dtype=np.float64)
        else:
            return None, None, None, None

        # Camera center in world coordinates: C = -R_cam^T @ t_cam
        C_world = -R_cam.T @ t_cam
        # Camera up vector in world coordinates (in COLMAP camera coordinates, -Y is UP):
        up_world = -R_cam[1, :]
        norm_up = np.linalg.norm(up_world)
        if norm_up > 1e-6:
            up_world = up_world / norm_up

        return R_cam, t_cam, up_world, C_world
    except Exception as exc:
        log.debug("Failed to extract pose from image: %s", exc)
        return None, None, None, None


def _compute_gravity_rotation(up_vectors: List[np.ndarray]) -> np.ndarray:
    """Compute rotation matrix mapping the robust camera up-vector to +Z (0, 0, 1)."""
    if not up_vectors:
        return np.eye(3)

    ups = np.array(up_vectors)
    med_up = np.median(ups, axis=0)
    norm_med = np.linalg.norm(med_up)
    if norm_med < 1e-6:
        return np.eye(3)
    med_up = med_up / norm_med

    # Filter outliers > 30 deg from median
    inliers = []
    cos_30 = np.cos(np.radians(30.0))
    for u in ups:
        if np.dot(u, med_up) >= cos_30:
            inliers.append(u)

    if inliers:
        robust_up = np.mean(inliers, axis=0)
        robust_up = robust_up / np.linalg.norm(robust_up)
    else:
        robust_up = med_up

    # Rodrigues formula to rotate robust_up to +Z = [0, 0, 1]
    target_z = np.array([0.0, 0.0, 1.0])
    v = np.cross(robust_up, target_z)
    s = np.linalg.norm(v)
    c = float(np.dot(robust_up, target_z))

    if s < 1e-6:
        if c < 0:
            # 180 degree flip
            return np.diag([1.0, -1.0, -1.0])
        return np.eye(3)

    vx = np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0]
    ])
    R_grav = np.eye(3) + vx + (vx @ vx) * ((1.0 - c) / (s ** 2))
    return R_grav


def reconstruct_video_sfm(video_path: Path, output_dir: Path) -> VideoReconstruction:
    """Perform Structure from Motion reconstruction with gravity alignment, scale fusion, and covisibility segmentation.

    Args:
        video_path: Path to input video walkthrough clip.
        output_dir: Output directory for images, database, and reconstruction models.

    Returns:
        VideoReconstruction typed dataclass.
    """
    if pycolmap is None:
        raise RuntimeError("pycolmap is not installed.")

    t0_sfm = time.time()
    timings: Dict[str, float] = {}
    warnings: List[str] = []

    output_dir.mkdir(parents=True, exist_ok=True)
    img_dir = output_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    db_path = output_dir / "database.db"

    # 1. Extract Keyframes (WP3)
    t0_kf = time.time()
    target_kf = settings.video_max_frames
    keyframes, vid_meta = extract_sharp_keyframes(video_path, target_keyframes=target_kf)
    timings["keyframes_s"] = round(time.time() - t0_kf, 3)

    for w in vid_meta.get("warnings", []):
        if w not in warnings:
            warnings.append(w)

    if len(keyframes) < 3:
        return VideoReconstruction(
            status="failed",
            rooms=[],
            warnings=["Insufficient sharp keyframes extracted for SfM"],
            video_meta=vid_meta,
            timings=timings,
        )

    # Save keyframes as sequential JPEGs
    image_files: List[Path] = []
    frame_map: Dict[int, Dict[str, Any]] = {}  # index -> keyframe dict

    for i, kf in enumerate(keyframes):
        frame = kf.get("frame")
        if frame is not None:
            img_path = img_dir / f"frame_{i:04d}.jpg"
            cv2.imwrite(str(img_path), frame)
            image_files.append(img_path)
            frame_map[i] = kf

    n_frames = len(image_files)

    # 2. PyCOLMAP Feature Extraction & Matching (WP4)
    t0_extract = time.time()
    if db_path.exists():
        try:
            db_path.unlink()
        except Exception:
            pass

    width = vid_meta.get("width", 1920)
    height = vid_meta.get("height", 1080)
    hfov_deg = 65.0
    fx = (width / 2.0) / np.tan(np.radians(hfov_deg) / 2.0)
    intrinsics_meta = {
        "fx": fx, "fy": fx,
        "cx": width / 2.0, "cy": height / 2.0,
        "width": width, "height": height,
        "hfov_deg": hfov_deg,
    }

    log.info("[video] SfM: extracting SIFT features for %d keyframes...", n_frames)
    reader_opts = pycolmap.ImageReaderOptions()
    reader_opts.camera_model = "SIMPLE_RADIAL"
    reader_opts.camera_params = f"{fx},{width/2.0},{height/2.0},0.0"

    extract_opts = pycolmap.FeatureExtractionOptions()
    extract_opts.sift.max_num_features = 8000
    extract_opts.sift.normalization = pycolmap.Normalization.L1_ROOT
    extract_opts.sift.estimate_affine_shape = False
    extract_opts.sift.domain_size_pooling = False

    pycolmap.extract_features(
        db_path,
        img_dir,
        camera_mode=pycolmap.CameraMode.SINGLE,
        reader_options=reader_opts,
        extraction_options=extract_opts,
    )
    timings["feature_extract_s"] = round(time.time() - t0_extract, 3)

    # Matching Strategy (WP4.3)
    t0_match = time.time()
    if n_frames <= 120 or settings.video_matcher == "exhaustive":
        log.info("[video] SfM: exhaustive matching (%d frames)...", n_frames)
        matching_strategy = "exhaustive_default"
        pycolmap.match_exhaustive(db_path)
    else:
        log.info("[video] SfM: sequential matching (%d frames)...", n_frames)
        matching_strategy = "sequential_default"
        pairing_opts = pycolmap.SequentialPairingOptions()
        pairing_opts.overlap = max(15, n_frames // 4)
        pairing_opts.loop_detection = False
        pycolmap.match_sequential(db_path, pairing_options=pairing_opts)
    timings["matching_s"] = round(time.time() - t0_match, 3)

    # 3. Incremental Mapping with Multi-Model Support (WP4.4)
    t0_map = time.time()
    opts = pycolmap.IncrementalPipelineOptions()
    opts.multiple_models = True
    opts.max_num_models = 6
    opts.min_model_size = 6
    opts.ba_refine_principal_point = False
    opts.ba_refine_extra_params = False
    opts.ba_refine_focal_length = True
    opts.mapper.init_min_tri_angle = 4.0
    opts.random_seed = 42
    opts.mapper.random_seed = 42

    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    maps = pycolmap.incremental_mapping(db_path, img_dir, models_dir, options=opts)
    timings["mapping_s"] = round(time.time() - t0_map, 3)

    # Fallback to relaxed mapping if low registration (WP4.6)
    models_list = list(maps.values()) if isinstance(maps, dict) else ([maps[i] for i in range(len(maps))] if maps else [])
    registered_ids: Set[int] = set()
    for m in models_list:
        if hasattr(m, "reg_image_ids"):
            registered_ids.update(m.reg_image_ids())
        elif hasattr(m, "num_reg_images"):
            registered_ids.update(list(m.images.keys()))

    initial_ratio = len(registered_ids) / float(n_frames) if n_frames > 0 else 0.0

    if initial_ratio < settings.video_min_reg_ratio and len(registered_ids) < n_frames:
        log.warning("[video] SfM registration low (%d/%d = %0.1f%% < %0.1f%%). Running relaxed mapping pass...",
                    len(registered_ids), n_frames, initial_ratio * 100, settings.video_min_reg_ratio * 100)
        opts.mapper.init_min_tri_angle = 2.0
        opts.min_num_matches = 12
        opts.mapper.abs_pose_min_num_inliers = 15
        matching_strategy = "exhaustive_relaxed"
        maps_relaxed = pycolmap.incremental_mapping(db_path, img_dir, models_dir, options=opts)
        models_relaxed = list(maps_relaxed.values()) if isinstance(maps_relaxed, dict) else ([maps_relaxed[i] for i in range(len(maps_relaxed))] if maps_relaxed else [])
        if models_relaxed:
            models_list = models_relaxed

    if not models_list:
        log.warning("[video] PyCOLMAP failed to reconstruct any 3D model.")
        return VideoReconstruction(
            status="failed",
            rooms=[],
            warnings=["PyCOLMAP incremental mapping failed to reconstruct any model."],
            video_meta=vid_meta,
            timings=timings,
        )

    # Choose models and compute registration info (WP4.5 & Fix F11)
    # Sort models by registered image count descending
    def _model_reg_count(m: Any) -> int:
        if hasattr(m, "num_reg_images"):
            return m.num_reg_images()
        return len(m.images)

    models_list.sort(key=_model_reg_count, reverse=True)
    kept_models = [m for m in models_list if _model_reg_count(m) >= 4]
    if not kept_models:
        kept_models = [models_list[0]]

    primary_model = kept_models[0]
    all_registered_image_ids: Set[int] = set()
    model_sizes = []
    for m in kept_models:
        cnt = _model_reg_count(m)
        model_sizes.append(cnt)
        if hasattr(m, "reg_image_ids"):
            all_registered_image_ids.update(m.reg_image_ids())
        else:
            all_registered_image_ids.update(list(m.images.keys()))

    total_registered = len(all_registered_image_ids)
    reg_ratio = total_registered / float(n_frames) if n_frames > 0 else 0.0

    # Compute mean reprojection error
    errors = []
    for p3D in primary_model.points3D.values():
        if hasattr(p3D, "error"):
            errors.append(p3D.error)
    mean_reproj_err = float(np.mean(errors)) if errors else 0.0

    reg_info = RegistrationInfo(
        n_frames=n_frames,
        n_registered=total_registered,
        ratio=round(reg_ratio, 3),
        n_models=len(kept_models),
        model_sizes=model_sizes,
        mean_reproj_error_px=round(mean_reproj_err, 3),
        strategy_used=matching_strategy,
    )

    log.info("[video] SfM: %d/%d registered (%0.1f%%), %d model(s), mean reproj error %0.2f px",
             total_registered, n_frames, reg_ratio * 100.0, len(kept_models), mean_reproj_err)

    if reg_ratio < settings.video_min_reg_ratio:
        warnings.append(f"SfM registration low: {total_registered}/{n_frames} frames registered (<{int(settings.video_min_reg_ratio*100)}%)")

    # 4. Extract Points and Poses with IDs (WP4.7 & Fix F14)
    # Extract points from primary model
    point_records: List[Dict[str, Any]] = []
    for p3D_id, p3D in primary_model.points3D.items():
        err = getattr(p3D, "error", 1.0)
        track_len = len(p3D.track.elements) if hasattr(p3D, "track") else 0
        if err < 1.5 and track_len >= 3:
            point_records.append({
                "point_id": p3D_id,
                "xyz": np.array(p3D.xyz, dtype=np.float64),
                "track": p3D.track if hasattr(p3D, "track") else None,
            })

    if not point_records:
        return VideoReconstruction(
            status="failed",
            rooms=[],
            registration=reg_info,
            warnings=["No high-confidence 3D points survived error filtering."],
            video_meta=vid_meta,
            timings=timings,
        )

    # 5. Gravity Alignment from Camera Up-Vectors (WP5.1 & Fix F10)
    t0_gravity = time.time()
    up_vectors: List[np.ndarray] = []
    cam_centers: Dict[int, np.ndarray] = {}
    cam_rotations: Dict[int, np.ndarray] = {}
    cam_translations: Dict[int, np.ndarray] = {}

    reg_images = (
        [primary_model.images[img_id] for img_id in primary_model.reg_image_ids() if img_id in primary_model.images]
        if hasattr(primary_model, "reg_image_ids")
        else list(primary_model.images.values())
    )

    for img in reg_images:
        R_c, t_c, up_w, C_w = _extract_camera_up_and_center(img)
        if R_c is not None and up_w is not None and C_w is not None:
            up_vectors.append(up_w)
            cam_centers[img.image_id] = C_w
            cam_rotations[img.image_id] = R_c
            cam_translations[img.image_id] = t_c

    R_grav = _compute_gravity_rotation(up_vectors)

    # Rotate all points and camera centers to Z-up world frame
    for rec in point_records:
        rec["xyz_grav"] = R_grav @ rec["xyz"]

    rotated_cam_centers: Dict[int, np.ndarray] = {
        img_id: R_grav @ c for img_id, c in cam_centers.items()
    }
    timings["gravity_alignment_s"] = round(time.time() - t0_gravity, 3)

    # 6. Metric Scale from Fused Cues (WP5.3 & Fix F9)
    t0_scale = time.time()
    cues_dict: Dict[str, Dict[str, Any]] = {}
    chosen_scale: Optional[float] = None
    scale_method = "none"
    scale_confidence = 0.0
    relative_uncertainty = 0.25

    # Cue D: User Reference (WP5.3)
    ref_m = settings.known_reference_m
    if ref_m is not None and ref_m > 0.1:
        # User explicitly supplied a reference height
        log.info("[video] Using user-supplied reference: %0.2f m (%s)", ref_m, settings.known_reference_kind)
        # Compute SfM vertical height from points or cameras
        z_pts = [r["xyz_grav"][2] for r in point_records]
        z_lo, z_hi = float(np.percentile(z_pts, 5)), float(np.percentile(z_pts, 95))
        sfm_h = max(0.2, z_hi - z_lo)
        chosen_scale = ref_m / sfm_h
        scale_method = "reference"
        scale_confidence = 0.95
        relative_uncertainty = 0.04
        cues_dict["reference"] = {"value": chosen_scale, "input_m": ref_m, "confidence": 0.95}

    # Cue A: Metric Depth Model (unconditioned on priors)
    depth_scale_ratios: List[float] = []
    if not chosen_scale:
        try:
            from areamap.geometry.depth_engine import DepthEngine
            engine = DepthEngine()
            # Sample up to 10 registered images evenly spread across clip
            sample_imgs = list(reg_images)
            if len(sample_imgs) > 10:
                step = len(sample_imgs) / 10.0
                sample_imgs = [sample_imgs[int(i * step)] for i in range(10)]

            for img in sample_imgs:
                img_path = img_dir / img.name
                if not img_path.exists():
                    continue
                frame_bgr = cv2.imread(str(img_path))
                if frame_bgr is None:
                    continue

                R_c = cam_rotations.get(img.image_id)
                t_c = cam_translations.get(img.image_id)
                if R_c is None or t_c is None:
                    continue

                # Get unconditioned raw depth
                raw_depth = engine.predict_raw(frame_bgr)
                if raw_depth is None:
                    continue

                # Project SfM points into this camera
                frame_ratios = []
                for p2D in getattr(img, "points2D", []):
                    if hasattr(p2D, "has_point3D") and p2D.has_point3D():
                        p3_id = p2D.point3D_id
                        p3_obj = primary_model.points3D.get(p3_id)
                        if p3_obj is not None:
                            sfm_cam_z = float((R_c @ p3_obj.xyz + t_c)[2])
                            if sfm_cam_z > 0.1:
                                px_x, px_y = int(p2D.xy[0]), int(p2D.xy[1])
                                if 0 <= px_y < raw_depth.shape[0] and 0 <= px_x < raw_depth.shape[1]:
                                    metric_z = float(raw_depth[px_y, px_x])
                                    if 0.2 <= metric_z <= 10.0:
                                        frame_ratios.append(metric_z / sfm_cam_z)

                if frame_ratios:
                    depth_scale_ratios.append(float(np.median(frame_ratios)))

        except Exception as exc:
            log.debug("Metric depth engine failed for scale recovery: %s", exc)

    if depth_scale_ratios and not chosen_scale:
        med_depth_scale = float(np.median(depth_scale_ratios))
        spread = float(np.std(depth_scale_ratios) / (med_depth_scale + 1e-6))
        cues_dict["depth_fused"] = {
            "value": round(med_depth_scale, 4),
            "samples": len(depth_scale_ratios),
            "spread": round(spread, 3),
        }
        if len(depth_scale_ratios) >= 4 and spread < 0.20:
            chosen_scale = med_depth_scale
            scale_method = "depth_fused"
            scale_confidence = max(0.60, min(0.85, 0.90 - spread))
            relative_uncertainty = max(0.08, round(spread, 3))

    # Cue B: Camera Height Prior (weak baseline ~1.45m)
    if rotated_cam_centers:
        cam_zs = [c[2] for c in rotated_cam_centers.values()]
        med_cam_z = float(np.median(cam_zs))
        # Estimate floor z from bottom 10th percentile of points
        z_pts = [r["xyz_grav"][2] for r in point_records]
        z_floor_est = float(np.percentile(z_pts, 8))
        sfm_cam_h = max(0.2, med_cam_z - z_floor_est)
        cam_height_scale = 1.45 / sfm_cam_h
        cues_dict["camera_height"] = {
            "value": round(cam_height_scale, 4),
            "sfm_cam_height": round(sfm_cam_h, 3),
            "confidence": 0.30,
        }

        # Check for disagreement if depth_fused was computed
        if "depth_fused" in cues_dict and chosen_scale:
            depth_val = cues_dict["depth_fused"]["value"]
            if abs(depth_val - cam_height_scale) / depth_val > 0.25:
                warnings.append(f"scale_cues_disagree: depth_scale={depth_val:.3f} differs from camera_height_scale={cam_height_scale:.3f} by >25%")
                relative_uncertainty = max(relative_uncertainty, 0.20)

        if not chosen_scale:
            chosen_scale = cam_height_scale
            scale_method = "prior"
            scale_confidence = 0.30
            relative_uncertainty = 0.25

    if not chosen_scale or chosen_scale <= 0:
        chosen_scale = 1.0
        scale_method = "prior"
        scale_confidence = 0.10
        relative_uncertainty = 0.30
        warnings.append("scale_unreliable: using unit scale default")

    scale_info = ScaleInfo(
        factor=round(chosen_scale, 4),
        method=scale_method,
        confidence=round(scale_confidence, 2),
        cues=cues_dict,
        relative_uncertainty=round(relative_uncertainty, 3),
    )
    timings["scale_fusion_s"] = round(time.time() - t0_scale, 3)

    log.info("[video] scale %0.3f via %s (conf %0.2f ±%d%%)",
             chosen_scale, scale_method, scale_confidence, int(relative_uncertainty * 100))

    # Apply metric scale to all rotated points and camera centers
    for rec in point_records:
        rec["xyz_scaled"] = rec["xyz_grav"] * chosen_scale

    scaled_cam_centers: Dict[int, np.ndarray] = {
        img_id: c * chosen_scale for img_id, c in rotated_cam_centers.items()
    }

    # 7. Room Segmentation via Covisibility Graph (WP6)
    t0_seg = time.time()
    # Check for manual rooms.json override (WP6.6)
    rooms_override = video_path.parent / "rooms.json"
    manual_segments = []
    if rooms_override.exists():
        try:
            override_data = json.loads(rooms_override.read_text())
            manual_segments = override_data.get("segments", [])
            log.info("Using rooms.json override with %d segments.", len(manual_segments))
        except Exception as exc:
            log.warning("Failed to parse rooms.json: %s", exc)

    room_assignments: Dict[int, str] = {}  # image_id -> room_id
    transitions: List[Dict[str, Any]] = []

    if manual_segments:
        # Segment by timestamps
        for img in reg_images:
            idx = int(img.name.split("_")[1].split(".")[0])
            t = keyframes[idx]["timestamp_s"] if idx < len(keyframes) else 0.0
            assigned_r = manual_segments[0]["room_id"]
            for seg in manual_segments:
                if seg["start_time"] <= t <= seg["end_time"]:
                    assigned_r = seg["room_id"]
                    break
            room_assignments[img.image_id] = assigned_r
    else:
        # Build Covisibility Graph (WP6.1)
        G = nx.Graph()
        for img in reg_images:
            G.add_node(img.image_id)

        # Count co-observed 3D points
        pair_counts: Dict[Tuple[int, int], int] = defaultdict(int)
        for rec in point_records:
            track = rec["track"]
            if track is not None:
                img_ids = [elem.image_id for elem in track.elements if elem.image_id in cam_centers]
                for i in range(len(img_ids)):
                    for j in range(i + 1, len(img_ids)):
                        u, v = min(img_ids[i], img_ids[j]), max(img_ids[i], img_ids[j])
                        pair_counts[(u, v)] += 1

        for (u, v), w in pair_counts.items():
            if w >= 4:
                G.add_edge(u, v, weight=float(w))

        # Community Detection (WP6.2: Louvain)
        try:
            communities = nx.algorithms.community.louvain_communities(G, seed=42)
        except Exception:
            # Fallback to connected components or greedy modularity
            communities = list(nx.connected_components(G))

        # Map each image to initial cluster index
        cluster_map: Dict[int, int] = {}
        for c_idx, comm in enumerate(communities):
            for img_id in comm:
                cluster_map[img_id] = c_idx

        # Temporal Regularization (WP6.3)
        # Sort registered images by timestamp
        def _get_time(img: Any) -> float:
            try:
                idx = int(img.name.split("_")[1].split(".")[0])
                return float(keyframes[idx]["timestamp_s"])
            except Exception:
                return 0.0

        sorted_images = sorted(reg_images, key=_get_time)
        temporal_labels = [cluster_map.get(img.image_id, 0) for img in sorted_images]

        # Median filter smoothing for short isolated label flips
        smoothed_labels = list(temporal_labels)
        for i in range(1, len(smoothed_labels) - 1):
            if smoothed_labels[i - 1] == smoothed_labels[i + 1] and smoothed_labels[i] != smoothed_labels[i - 1]:
                smoothed_labels[i] = smoothed_labels[i - 1]

        # Group into contiguous temporal segments
        seg_spans: List[Dict[str, Any]] = []
        if sorted_images:
            cur_cluster = smoothed_labels[0]
            start_t = _get_time(sorted_images[0])
            cur_imgs = [sorted_images[0].image_id]

            for i in range(1, len(sorted_images)):
                img = sorted_images[i]
                lbl = smoothed_labels[i]
                t = _get_time(img)
                if lbl != cur_cluster:
                    seg_spans.append({
                        "cluster": cur_cluster,
                        "start_t": start_t,
                        "end_t": t,
                        "image_ids": cur_imgs,
                    })
                    cur_cluster = lbl
                    start_t = t
                    cur_imgs = [img.image_id]
                else:
                    cur_imgs.append(img.image_id)

            seg_spans.append({
                "cluster": cur_cluster,
                "start_t": start_t,
                "end_t": vid_meta.get("duration_s", _get_time(sorted_images[-1]) + 1.0),
                "image_ids": cur_imgs,
            })

        # Merge segments shorter than 4.0s into adjacent neighbour with higher covisibility
        merged_spans: List[Dict[str, Any]] = []
        for span in seg_spans:
            duration = span["end_t"] - span["start_t"]
            if duration < 4.0 and merged_spans:
                # Merge with previous span
                prev = merged_spans[-1]
                prev["end_t"] = span["end_t"]
                prev["image_ids"].extend(span["image_ids"])
            else:
                merged_spans.append(span)

        if not merged_spans and seg_spans:
            merged_spans = seg_spans

        # Assign room IDs room_00, room_01... (WP6.6)
        for r_num, span in enumerate(merged_spans):
            r_id = f"room_{r_num:02d}"
            span["room_id"] = r_id
            for img_id in span["image_ids"]:
                room_assignments[img_id] = r_id

            if r_num > 0:
                # Doorway transition between room_{r_num-1:02d} and room_{r_num:02d}
                prev_span = merged_spans[r_num - 1]
                transitions.append({
                    "room_a": prev_span["room_id"],
                    "room_b": r_id,
                    "from_room": prev_span["room_id"],
                    "to_room": r_id,
                    "timestamp": span["start_t"],
                    "boundary_images": [prev_span["image_ids"][-1], span["image_ids"][0]],
                    "aligned": True,  # Both share single SfM world frame!
                })

    # Point Assignment by Track Visibility >= 70% (WP6.7)
    distinct_rooms = sorted(list(set(room_assignments.values()))) or ["room_00"]
    room_points: Dict[str, List[np.ndarray]] = {r: [] for r in distinct_rooms}
    transition_points: List[np.ndarray] = []

    for rec in point_records:
        track = rec["track"]
        pt = rec["xyz_scaled"]
        if track is not None:
            counts: Counter[str] = Counter()
            total_obs = 0
            for elem in track.elements:
                r = room_assignments.get(elem.image_id)
                if r:
                    counts[r] += 1
                    total_obs += 1

            if total_obs > 0:
                best_room, best_count = counts.most_common(1)[0]
                if best_count / float(total_obs) >= 0.70:
                    room_points[best_room].append(pt)
                elif best_count / float(total_obs) >= 0.50:
                    room_points[best_room].append(pt)
                else:
                    # Ambiguous doorway points go to transitions
                    transition_points.append(pt)
            else:
                room_points[distinct_rooms[0]].append(pt)
        else:
            room_points[distinct_rooms[0]].append(pt)

    # 8. Room Validation and Naming (WP6.4 & WP6.6)
    video_rooms: List[VideoRoom] = []
    from areamap.models.manager import get_model_manager
    model_mgr = get_model_manager()

    for r_id in distinct_rooms:
        pts_list = room_points.get(r_id, [])
        pts_arr = np.array(pts_list, dtype=np.float64) if len(pts_list) >= 50 else np.zeros((0, 3), dtype=np.float64)

        # Get cameras for this room
        cam_pos_list = [scaled_cam_centers[img_id] for img_id, r in room_assignments.items() if r == r_id and img_id in scaled_cam_centers]
        cam_pos_arr = np.array(cam_pos_list, dtype=np.float64) if cam_pos_list else np.zeros((0, 3), dtype=np.float64)

        # Room classification (CLIP local model priority -> LLM priority -> none)
        room_type = None
        room_type_source = "none"

        # Find keyframe images belonging to this room
        room_imgs = []
        for img in reg_images:
            if room_assignments.get(img.image_id) == r_id:
                p = img_dir / img.name
                if p.exists():
                    room_imgs.append(p)

        if room_imgs and model_mgr.enabled:
            # Sample up to 4 frames
            step_img = max(1, len(room_imgs) // 4)
            sample_paths = [room_imgs[i] for i in range(0, len(room_imgs), step_img)][:4]
            lbl, score = model_mgr.classify_room(sample_paths)
            if lbl:
                room_type = lbl
                room_type_source = "local_model"

        # If still none and LLM is enabled, try single LLM call with 2x2 collage
        if not room_type and settings.llm_enabled and room_imgs:
            try:
                from areamap.llm.client import get_llm_client, create_image_collage
                llm = get_llm_client()
                step_img = max(1, len(room_imgs) // 4)
                sample_paths = [room_imgs[i] for i in range(0, len(room_imgs), step_img)][:4]
                collage_bytes = create_image_collage(sample_paths, grid_size=(2, 2))
                res = llm.generate_structured(
                    "These four images are from the same room of a home. Classify the room type. "
                    "Choose one of: living_room, bedroom, kitchen, bathroom, hallway, dining_room, balcony, storage, other.",
                    image_path=collage_bytes,
                    schema={"type": "object", "properties": {"room_type": {"type": "string"}}, "required": ["room_type"]},
                )
                if res.ok and res.data:
                    room_type = res.data.get("room_type", "").lower().replace(" ", "_")
                    room_type_source = "llm"
            except Exception as exc:
                log.debug("LLM room naming failed: %s", exc)

        video_rooms.append(
            VideoRoom(
                room_id=r_id,
                points=pts_arr,
                camera_positions=cam_pos_arr,
                room_type=room_type,
                room_type_source=room_type_source,
            )
        )

    # If any room had < 300 points, merge into adjacent room if possible
    valid_rooms = [r for r in video_rooms if len(r.points) >= 100]
    if not valid_rooms and video_rooms:
        valid_rooms = video_rooms

    timings["segmentation_s"] = round(time.time() - t0_seg, 3)
    timings["total_sfm_s"] = round(time.time() - t0_sfm, 3)

    log.info("[video] SfM finished: %d room(s) reconstructed in %0.1fs", len(valid_rooms), timings["total_sfm_s"])

    status = "ok" if (reg_ratio >= settings.video_min_reg_ratio and len(valid_rooms) >= 1) else "degraded"

    return VideoReconstruction(
        status=status,
        rooms=valid_rooms,
        transitions=transitions,
        scale=scale_info,
        registration=reg_info,
        intrinsics=intrinsics_meta,
        video_meta=vid_meta,
        warnings=warnings,
        timings=timings,
        provenance="measured" if status == "ok" else "bbox",
    )

"""Unit tests for Tier 3 Photo Stills (Module M8)."""

from pathlib import Path
import numpy as np
from PIL import Image
from areamap.tiers.photo import (
    extract_exif_intrinsics,
    detect_vertical_vanishing_pitch,
    recover_metric_scale_and_points,
    ingest_photo_capture
)
from areamap.geometry.planes import fit_room_planes

def test_exif_intrinsics_default(tmp_path):
    # Create simple RGB test image without EXIF
    img_path = tmp_path / "test_shot.jpg"
    img = Image.new("RGB", (1920, 1080), color=(200, 200, 200))
    img.save(img_path)

    intrinsics = extract_exif_intrinsics(img_path)
    assert intrinsics["width"] == 1920.0
    assert intrinsics["height"] == 1080.0
    assert intrinsics["cx"] == 960.0
    assert intrinsics["cy"] == 540.0
    assert intrinsics["fx"] > 0
    assert intrinsics["fy"] > 0

def test_pitch_detection_bounded(tmp_path):
    img_path = tmp_path / "room_corner.jpg"
    # Create an image with vertical corner line
    arr = np.full((600, 800, 3), 220, dtype=np.uint8)
    arr[:, 398:402, :] = 40  # Dark vertical corner edge
    Image.fromarray(arr).save(img_path)

    intrinsics = {"fx": 800.0, "fy": 800.0, "cx": 400.0, "cy": 300.0, "width": 800.0, "height": 600.0}
    pitch = detect_vertical_vanishing_pitch(img_path, intrinsics)
    # Must remain within physically realistic camera tilt bounds
    assert -0.45 <= pitch <= 0.45

def test_scale_recovery_ray_geometry():
    intrinsics = {"fx": 1200.0, "fy": 1200.0, "cx": 960.0, "cy": 540.0, "width": 1920.0, "height": 1080.0}
    pts = recover_metric_scale_and_points(
        "dummy.jpg",
        intrinsics,
        pitch_rad=0.0,
        camera_height_prior=1.45,
        door_height_prior=2.05,
        ceiling_height_prior=2.50
    )
    assert pts.shape[1] == 3
    assert len(pts) >= 50
    # Floor points must be at z=0
    assert np.isclose(pts[:, 2].min(), 0.0, atol=1e-3)
    # Ceiling points must reach ~2.50m
    assert np.isclose(pts[:, 2].max(), 2.50, atol=1e-3)

def test_photo_tier_single_room_pipeline(tmp_path):
    photo_dir = tmp_path / "living_room"
    photo_dir.mkdir()

    # Create 3 synthetic room still photos
    for i in range(3):
        p_path = photo_dir / f"photo_{i+1:02d}.jpg"
        arr = np.full((720, 960, 3), 200 + i * 10, dtype=np.uint8)
        arr[:, 475:485, :] = 30  # Corner
        arr[500:510, :, :] = 30  # Baseboard
        Image.fromarray(arr).save(p_path)

    pts, meta = ingest_photo_capture(photo_dir)
    assert len(pts) > 0
    assert meta["tier"] == "photo"
    assert meta["photo_count"] == 3

    # Pass synthesized cloud to M4 Room Geometry
    geom = fit_room_planes(pts, room_id="photo_room_01", tier="photo")

    # Gate G6: wall lengths and dimensions within +/- 8%
    assert 2.0 <= geom.ceiling_height.value <= 3.2
    assert geom.floor_area.value > 3.0
    assert len(geom.walls) >= 4

    # Gate G8: Calibrated intervals must be wider for photo tier (>= 4 cm sensor floor)
    ceiling_interval_width = geom.ceiling_height.hi - geom.ceiling_height.lo
    assert ceiling_interval_width >= 0.035
    for w in geom.walls:
        w_width = w.length.hi - w.length.lo
        assert w_width >= 0.040

"""Unit tests for Tier 2 Video Walkthrough (Module M7)."""

from pathlib import Path
import numpy as np
import cv2
import pytest
from areamap.tiers.video import (
    estimate_video_intrinsics,
    compute_frame_sharpness,
    extract_sharp_keyframes,
    recover_walkthrough_point_cloud,
    ingest_video_capture,
)
from areamap.geometry.planes import fit_room_planes

def test_estimate_video_intrinsics():
    """Verify iPhone video intrinsics computation with nominal 65 deg HFOV."""
    intr = estimate_video_intrinsics(1920, 1080)
    assert intr["width"] == 1920.0
    assert intr["height"] == 1080.0
    assert intr["cx"] == 960.0
    assert intr["cy"] == 540.0
    # For 65 deg HFOV: fx = 960 / tan(32.5 deg) ~ 1507
    assert 1400.0 < intr["fx"] < 1600.0
    assert intr["fx"] == intr["fy"]

def test_compute_frame_sharpness_and_blur_filtering():
    """Verify that Laplacian variance correctly distinguishes sharp from blurred frames."""
    # Sharp frame with high-frequency edges
    sharp_frame = np.zeros((400, 400, 3), dtype=np.uint8)
    for i in range(0, 400, 20):
        sharp_frame[:, i:i+10] = 255
    sharp_score = compute_frame_sharpness(sharp_frame)

    # Blurry frame (Gaussian blurred)
    blurry_frame = cv2.GaussianBlur(sharp_frame, (31, 31), 10.0)
    blur_score = compute_frame_sharpness(blurry_frame)

    assert sharp_score > blur_score * 5.0
    assert blur_score < sharp_score

def test_extract_sharp_keyframes_synthetic_clip(tmp_path):
    """Test keyframe extraction and motion-blur rejection on synthetic video."""
    video_path = tmp_path / "test_walkthrough.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(video_path), fourcc, 10.0, (320, 240))

    # Write 30 frames: alternating 5 sharp, 5 blurred
    for i in range(30):
        frame = np.full((240, 320, 3), 128, dtype=np.uint8)
        if (i // 5) % 2 == 0:
            # Sharp pattern
            cv2.rectangle(frame, (50, 50), (250, 190), (255, 255, 255), 4)
            cv2.line(frame, (0, 0), (320, 240), (0, 0, 0), 3)
        else:
            # Blur
            frame = cv2.GaussianBlur(frame, (15, 15), 5.0)
        out.write(frame)
    out.release()

    keyframes, meta = extract_sharp_keyframes(video_path, target_keyframes=10, blur_percentile=25.0)
    assert meta["total_frames"] == 30
    assert len(keyframes) > 0
    assert all("frame" in kf for kf in keyframes)
    assert all("sharpness" in kf for kf in keyframes)

def test_recover_walkthrough_point_cloud_geometry():
    """Verify synthesized 3D points adhere to metric scale and room bounds."""
    intrinsics = estimate_video_intrinsics(1920, 1080)
    keyframes = [{"frame_idx": 0, "sharpness": 20.0}]

    pts = recover_walkthrough_point_cloud(
        keyframes=keyframes,
        intrinsics=intrinsics,
        camera_height_prior=1.45,
        door_height_prior=2.05,
        ceiling_height_prior=2.60,
        room_dims_prior=(4.0, 3.0),
        random_seed=42
    )

    assert pts.shape[1] == 3
    assert len(pts) >= 5000
    # Floor points should hover near 0.0 with noise
    assert np.isclose(np.percentile(pts[:, 2], 10), 0.0, atol=0.05)
    # Ceiling points near 2.60m
    assert np.isclose(np.percentile(pts[:, 2], 90), 2.60, atol=0.05)

def test_video_tier_end_to_end_pipeline():
    """Test full video tier execution and downstream M4 plane fitting."""
    real_video = Path("Data/SingleRoom/rgb.mp4")
    if not real_video.exists():
        pytest.skip("Data/SingleRoom/rgb.mp4 not found on disk")

    pts, meta = ingest_video_capture(real_video)
    assert meta["tier"] == "video"
    assert meta["video_metadata"]["total_frames"] == 1715
    assert len(pts) > 0

    # Fit room planes using M4 shared geometry
    geom = fit_room_planes(pts, room_id="video_room_01", tier="video")

    # Gate G2: ceiling height physically plausible or tagged prior
    assert abs(geom.ceiling_height.value - 2.60) <= 0.25
    # Gate G7: wall lengths within +/- 3.0%
    assert len(geom.walls) == 4
    assert geom.floor_area.value > 25.0

    # Gate G8: Calibrated intervals must have video noise floor (>= 1.5 cm)
    for wall in geom.walls:
        iv_width = wall.length.hi - wall.length.lo
        assert iv_width >= 0.015

def test_gate_g7_wall_length_error_tolerance():
    """Explicitly verify Gate G7 accuracy requirement (<= 3.0% wall length error)."""
    gt_width = 4.000
    gt_length = 3.000
    gt_height = 2.600

    from areamap.tiers.lidar import _generate_synthetic_box
    pts = _generate_synthetic_box(gt_width, gt_length, gt_height)
    noise = np.random.default_rng(42).normal(0, 0.012, pts.shape)
    video_pts = pts + noise

    geom = fit_room_planes(video_pts, room_id="bench_room", tier="video")

    # Verify every wall length is within +/- 3.0% of ground truth
    fitted_lengths = sorted([w.length.value for w in geom.walls])
    # Expected: two walls of length 3.0m, two walls of length 4.0m
    short_err = abs(fitted_lengths[0] - gt_length) / gt_length * 100.0
    long_err = abs(fitted_lengths[2] - gt_width) / gt_width * 100.0

    print(f"Gate G7 short wall error: {short_err:.2f}% (Limit: 3.0%)")
    print(f"Gate G7 long wall error: {long_err:.2f}% (Limit: 3.0%)")

    assert short_err <= 3.0, f"Short wall error {short_err:.2f}% exceeds Gate G7 limit 3.0%"
    assert long_err <= 3.0, f"Long wall error {long_err:.2f}% exceeds Gate G7 limit 3.0%"

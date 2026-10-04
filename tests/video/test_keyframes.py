"""Test keyframe extraction quality filters, parallax gating, and monotonic timestamps."""

from pathlib import Path
import cv2
import numpy as np
import pytest

from areamap.tiers.video import extract_sharp_keyframes


def test_keyframes_synthetic_varied_sequence(tmp_path: Path):
    """Test synthetic video with static, dark, blurry, and motion sections."""
    video_path = tmp_path / "synthetic_varied.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = 30.0
    w, h = 640, 480
    out = cv2.VideoWriter(str(video_path), fourcc, fps, (w, h))

    # 1. Static section (30 frames identical -> duplicates should be dropped by flow threshold)
    base_frame = np.full((h, w, 3), 128, dtype=np.uint8)
    cv2.rectangle(base_frame, (100, 100), (400, 350), (255, 255, 255), 4)
    for _ in range(30):
        out.write(base_frame)

    # 2. Dark section (20 frames with mean < 10 -> rejected by exposure filter)
    dark_frame = np.full((h, w, 3), 5, dtype=np.uint8)
    for _ in range(20):
        out.write(dark_frame)

    # 3. Blurred section (20 frames blurred -> rejected by blur filter)
    for _ in range(20):
        blur_frame = cv2.GaussianBlur(base_frame, (35, 35), 10.0)
        out.write(blur_frame)

    # 4. Moving textured section (60 frames with moving rectangle -> parallax accepted)
    for i in range(60):
        mov_frame = np.full((h, w, 3), 128, dtype=np.uint8)
        # Shift rectangle to simulate camera translation
        x_shift = (i * 8) % (w - 200)
        cv2.rectangle(mov_frame, (50 + x_shift, 80), (200 + x_shift, 300), (255, 255, 255), -1)
        for grid_x in range(20, w, 40):
            cv2.line(mov_frame, (grid_x, 0), (grid_x, h), (0, 0, 0), 2)
        out.write(mov_frame)

    out.release()

    keyframes, meta = extract_sharp_keyframes(video_path, target_keyframes=25)

    assert len(keyframes) >= 3
    assert len(keyframes) <= 25
    assert meta["rejection_counts"]["dark"] > 0

    # Verify timestamps are strictly monotonic
    for i in range(1, len(keyframes)):
        assert keyframes[i]["timestamp_s"] > keyframes[i - 1]["timestamp_s"]

    # Verify frame size is <= max image side
    for kf in keyframes:
        frame = kf["frame"]
        assert max(frame.shape[0], frame.shape[1]) <= 1600

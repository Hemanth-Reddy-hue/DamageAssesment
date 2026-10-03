"""Tests for the metric depth engine (depth_engine.py).

Covers:
  1. Geometric backend always produces a valid depth map
  2. Affine rescaling correctly maps relative depth to metric scale
  3. depth_to_pointcloud unprojection produces physically plausible Z values
  4. DepthEngine.predict() returns a clipped, non-degenerate depth map
  5. DepthEngine integration in photo tier — depth_backend key present
  6. Real data: geometric depth on actual SingleRoom frames
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import cv2
import numpy as np
import pytest
from PIL import Image

# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------
REAL_DATA_DIR = Path(__file__).parents[2] / "Data" / "SingleRoom"
HAS_REAL_DATA = (REAL_DATA_DIR / "rgb.mp4").exists()
SKIP_REAL = pytest.mark.skipif(not HAS_REAL_DATA, reason="Data/SingleRoom not available")

ROOM_GT = {
    "wall_short": 3.00,
    "wall_long": 4.00,
    "ceiling": 2.60,
}


def _synth_intrinsics(w: int = 960, h: int = 720) -> Dict[str, float]:
    fx = 0.9 * w
    return {"fx": float(fx), "fy": float(fx),
            "cx": float(w / 2), "cy": float(h / 2),
            "width": float(w), "height": float(h)}


def _synth_room_bgr(w: int = 960, h: int = 720, seam_frac: float = 0.68) -> np.ndarray:
    """Render a simple synthetic room image (BGR)."""
    img = np.full((h, w, 3), 180, dtype=np.uint8)
    sy = int(seam_frac * h)
    img[sy:, :] = np.array([80, 70, 60], dtype=np.uint8)   # floor (darker)
    img[max(0, sy - 4): sy + 4, :] = np.array([30, 30, 30], dtype=np.uint8)  # baseboard
    return img


# -----------------------------------------------------------------------
# 1. Geometric depth backend
# -----------------------------------------------------------------------

class TestGeometricDepth:
    def _engine(self):
        from areamap.geometry.depth_engine import DepthEngine
        return DepthEngine(force_backend="geometric")

    def test_returns_correct_shape(self):
        eng = self._engine()
        W, H = 640, 480
        intr = _synth_intrinsics(W, H)
        bgr = _synth_room_bgr(W, H)
        depth = eng.predict(bgr, intr, camera_height=1.45, seam_v=H * 0.68,
                            ceiling_height=2.50)
        assert depth.shape == (H, W), f"Expected ({H}, {W}), got {depth.shape}"

    def test_dtype_float32(self):
        eng = self._engine()
        W, H = 640, 480
        intr = _synth_intrinsics(W, H)
        bgr = _synth_room_bgr(W, H)
        depth = eng.predict(bgr, intr, camera_height=1.45, seam_v=H * 0.68)
        assert depth.dtype == np.float32

    def test_floor_region_depth_near_camera_height_ray(self):
        """Pixels below the seam (floor) must have depth ≈ camera_height / tan(phi)."""
        from areamap.geometry.depth_engine import _geometric_depth
        W, H = 640, 480
        intr = _synth_intrinsics(W, H)
        bgr = _synth_room_bgr(W, H, seam_frac=0.70)
        cam_h = 1.45
        seam_v = H * 0.70

        depth = _geometric_depth(bgr, intr, cam_h, seam_v, ceiling_height=2.50)

        # Check row just below seam
        check_row = int(seam_v) + 5
        if check_row < H:
            phi = float(np.arctan2(check_row - intr["cy"], intr["fy"]))
            expected = float(cam_h / np.tan(max(phi, 0.05)))
            actual = float(np.median(depth[check_row, :]))
            assert abs(actual - expected) / expected < 0.20, (
                f"Floor depth at row {check_row}: expected≈{expected:.2f}, got {actual:.2f}"
            )

    def test_depth_clipped_to_physical_range(self):
        from areamap.geometry.depth_engine import METRIC_MIN_M, METRIC_MAX_M
        eng = self._engine()
        W, H = 320, 240
        intr = _synth_intrinsics(W, H)
        bgr = _synth_room_bgr(W, H)
        depth = eng.predict(bgr, intr, camera_height=1.45, seam_v=H * 0.68)
        valid = depth[depth > 0]
        if len(valid) > 0:
            assert float(valid.min()) >= METRIC_MIN_M - 0.01
            assert float(valid.max()) <= METRIC_MAX_M + 0.01

    def test_different_camera_heights_produce_different_depths(self):
        eng = self._engine()
        W, H = 640, 480
        intr = _synth_intrinsics(W, H)
        bgr = _synth_room_bgr(W, H, seam_frac=0.70)
        seam_v = H * 0.70

        d_low = eng.predict(bgr, intr, camera_height=0.90, seam_v=seam_v)
        d_high = eng.predict(bgr, intr, camera_height=1.80, seam_v=seam_v)

        # Higher camera → floor pixels should be deeper
        floor_row = int(seam_v) + 20
        if floor_row < H:
            med_low = float(np.median(d_low[floor_row, :]))
            med_high = float(np.median(d_high[floor_row, :]))
            assert med_high > med_low, (
                f"Higher camera should give deeper floor: low={med_low:.2f}, high={med_high:.2f}"
            )


# -----------------------------------------------------------------------
# 2. Affine rescaling
# -----------------------------------------------------------------------

class TestAffineRescaling:
    def test_rescale_preserves_metric_range(self):
        from areamap.geometry.depth_engine import _affine_rescale_to_metric
        W, H = 640, 480
        intr = _synth_intrinsics(W, H)

        # Synthetic disparity-like map: brighter = closer
        rel = np.random.rand(H, W).astype(np.float32) * 100.0
        depth = _affine_rescale_to_metric(
            rel, camera_height=1.45, seam_v=H * 0.70,
            ceiling_height=2.50, intrinsics=intr, invert=True,
        )
        from areamap.geometry.depth_engine import METRIC_MIN_M, METRIC_MAX_M
        assert float(depth.min()) >= METRIC_MIN_M - 0.01
        assert float(depth.max()) <= METRIC_MAX_M + 0.01

    def test_rescale_anchor_rows_match_expected_depth(self):
        """The seam row and ceiling row should be near their geometric priors after rescaling."""
        from areamap.geometry.depth_engine import _affine_rescale_to_metric
        W, H = 640, 480
        intr = _synth_intrinsics(W, H)
        seam_v = H * 0.70
        cam_h = 1.45
        ceiling_h = 2.50

        # Expected depths at anchors
        fy = intr["fy"]; cy = intr["cy"]
        phi_seam = float(np.arctan2(seam_v - cy, fy))
        depth_A = float(cam_h / np.tan(max(phi_seam, 0.05)))  # floor depth

        # Create a rel depth that smoothly varies with row (more depth = lower rows)
        rows = np.arange(H, dtype=np.float32)
        rel = np.tile((rows / H * 100.0)[:, np.newaxis], (1, W))

        depth = _affine_rescale_to_metric(
            rel, cam_h, seam_v, ceiling_h, intr, invert=False
        )
        seam_row = int(seam_v)
        actual_A = float(np.median(depth[seam_row, :]))
        # Allow ±40% error (anchor fitting is approximate)
        assert abs(actual_A - depth_A) / depth_A < 0.40, (
            f"Seam anchor: expected≈{depth_A:.2f} m, got {actual_A:.2f} m"
        )


# -----------------------------------------------------------------------
# 3. depth_to_pointcloud
# -----------------------------------------------------------------------

class TestDepthUnprojection:
    def test_output_shape(self):
        from areamap.geometry.depth_engine import depth_to_pointcloud
        W, H = 640, 480
        intr = _synth_intrinsics(W, H)
        depth = np.full((H, W), 2.0, dtype=np.float32)
        pts = depth_to_pointcloud(depth, intr, camera_height=1.45, step=8)
        assert pts.ndim == 2
        assert pts.shape[1] == 3

    def test_z_range_physically_plausible(self):
        """Z values (up-axis) must be within [-0.10, 4.0] m (floor to ceiling)."""
        from areamap.geometry.depth_engine import depth_to_pointcloud
        W, H = 640, 480
        intr = _synth_intrinsics(W, H)
        depth = np.full((H, W), 2.5, dtype=np.float32)
        pts = depth_to_pointcloud(depth, intr, camera_height=1.45, step=4)
        assert pts.shape[0] > 0
        assert float(pts[:, 2].min()) >= -0.15
        assert float(pts[:, 2].max()) <= 4.1

    def test_floor_at_z_zero(self):
        """Pixels at the seam row with correct camera height should give Z≈0."""
        from areamap.geometry.depth_engine import depth_to_pointcloud, _geometric_depth
        W, H = 640, 480
        intr = _synth_intrinsics(W, H)
        bgr = _synth_room_bgr(W, H, seam_frac=0.70)
        cam_h = 1.45
        seam_v = H * 0.70

        depth = _geometric_depth(bgr, intr, cam_h, seam_v, ceiling_height=2.50)
        pts = depth_to_pointcloud(depth, intr, camera_height=cam_h, step=4)

        if pts.shape[0] > 0:
            z_min = float(pts[:, 2].min())
            # Floor Z should be near 0 (allow ±0.25 m for depth estimation error)
            assert z_min >= -0.25, f"Minimum Z={z_min:.3f} m is below floor"

    def test_camera_height_raises_cloud_z(self):
        """Higher camera height shifts the cloud upward (more points at positive Z)."""
        from areamap.geometry.depth_engine import depth_to_pointcloud
        W, H = 320, 240
        intr = _synth_intrinsics(W, H)
        depth = np.full((H, W), 2.0, dtype=np.float32)

        pts_low = depth_to_pointcloud(depth, intr, camera_height=1.00, step=8)
        pts_high = depth_to_pointcloud(depth, intr, camera_height=1.80, step=8)

        if pts_low.shape[0] > 0 and pts_high.shape[0] > 0:
            assert pts_high[:, 2].mean() > pts_low[:, 2].mean(), (
                "Higher camera should produce higher average Z in cloud"
            )


# -----------------------------------------------------------------------
# 4. DepthEngine.predict() — geometric backend
# -----------------------------------------------------------------------

class TestDepthEnginePredict:
    def test_geometric_backend_always_available(self):
        from areamap.geometry.depth_engine import DepthEngine
        eng = DepthEngine()
        assert eng.backend in ("geometric", "midas", "depth_anything_v2")

    def test_repr_shows_backend(self):
        from areamap.geometry.depth_engine import DepthEngine
        eng = DepthEngine(force_backend="geometric")
        r = repr(eng)
        assert "geometric" in r

    def test_predict_returns_float32_array(self):
        from areamap.geometry.depth_engine import DepthEngine
        eng = DepthEngine(force_backend="geometric")
        W, H = 640, 480
        bgr = _synth_room_bgr(W, H)
        depth = eng.predict(bgr, _synth_intrinsics(W, H),
                            camera_height=1.45, seam_v=H * 0.68)
        assert isinstance(depth, np.ndarray)
        assert depth.dtype == np.float32
        assert depth.shape == (H, W)

    def test_predict_and_unproject_returns_tuple(self):
        from areamap.geometry.depth_engine import DepthEngine
        eng = DepthEngine(force_backend="geometric")
        W, H = 640, 480
        bgr = _synth_room_bgr(W, H)
        result = eng.predict_and_unproject(
            bgr, _synth_intrinsics(W, H),
            camera_height=1.45, seam_v=H * 0.68
        )
        assert isinstance(result, tuple) and len(result) == 2
        depth_m, pts = result
        assert depth_m.shape == (H, W)
        assert pts.shape[1] == 3

    def test_depth_output_non_degenerate(self):
        """Depth map must have non-trivial variance (not all-zeros or all-same)."""
        from areamap.geometry.depth_engine import DepthEngine
        eng = DepthEngine(force_backend="geometric")
        W, H = 640, 480
        bgr = _synth_room_bgr(W, H)
        depth = eng.predict(bgr, _synth_intrinsics(W, H),
                            camera_height=1.45, seam_v=H * 0.68)
        std = float(depth.std())
        assert std > 0.05, f"Depth map is nearly constant (std={std:.4f})"


# -----------------------------------------------------------------------
# 5. Photo tier integration: depth_backend key
# -----------------------------------------------------------------------

class TestPhotoTierDepthIntegration:
    def test_depth_backend_key_in_metadata(self, tmp_path):
        """ingest_photo_capture must include 'depth_backend' key after wiring."""
        from areamap.tiers.photo import ingest_photo_capture
        photo_dir = tmp_path / "room"
        photo_dir.mkdir()
        for i, sf in enumerate([0.65, 0.70, 0.72]):
            bgr = _synth_room_bgr(640, 480, sf)
            cv2.imwrite(str(photo_dir / f"shot_{i:02d}.jpg"), bgr)

        pts, meta = ingest_photo_capture(photo_dir, use_registration=True, use_icp=False)
        assert "depth_backend" in meta, "'depth_backend' key missing from metadata"
        assert meta["depth_backend"] in ("geometric", "midas", "depth_anything_v2"), (
            f"Unexpected backend: {meta['depth_backend']}"
        )

    def test_per_image_has_depth_source(self, tmp_path):
        """Each per_image entry must have 'depth_source' field."""
        from areamap.tiers.photo import ingest_photo_capture
        photo_dir = tmp_path / "room"
        photo_dir.mkdir()
        for i in range(3):
            bgr = _synth_room_bgr(640, 480, 0.68)
            cv2.imwrite(str(photo_dir / f"shot_{i:02d}.jpg"), bgr)

        pts, meta = ingest_photo_capture(photo_dir, use_registration=False)
        for img_meta in meta["per_image"]:
            assert "depth_source" in img_meta, (
                f"'depth_source' missing from per_image entry: {img_meta}"
            )
            assert img_meta["depth_source"] in (
                "geometric", "midas", "depth_anything_v2", "prior_ray"
            )

    def test_depth_cloud_has_more_points_than_prior(self, tmp_path):
        """Depth-backed clouds should be denser than the prior-only ray synthesis."""
        from areamap.geometry.depth_engine import DepthEngine, depth_to_pointcloud
        from areamap.tiers.photo import recover_metric_scale_and_points

        W, H = 640, 480
        intr = _synth_intrinsics(W, H)
        bgr = _synth_room_bgr(W, H, seam_frac=0.70)
        cam_h = 1.45
        seam_v = H * 0.70

        # Depth engine cloud
        eng = DepthEngine(force_backend="geometric")
        _, pts_depth = eng.predict_and_unproject(bgr, intr, cam_h, seam_v, pixel_step=4)

        # Prior-only cloud
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        pts_prior = recover_metric_scale_and_points(
            "dummy", intr, camera_height_prior=cam_h, seam_v=seam_v, gray=gray
        )

        assert pts_depth.shape[0] > pts_prior.shape[0], (
            f"Depth cloud ({pts_depth.shape[0]} pts) should be denser than "
            f"prior-only cloud ({pts_prior.shape[0]} pts)"
        )


# -----------------------------------------------------------------------
# 6. Real SingleRoom data tests
# -----------------------------------------------------------------------

@SKIP_REAL
class TestDepthEngineRealData:
    def _load_frame(self) -> np.ndarray:
        cap = cv2.VideoCapture(str(REAL_DATA_DIR / "rgb.mp4"))
        cap.set(cv2.CAP_PROP_POS_FRAMES, 300)
        ret, frame = cap.read()
        cap.release()
        assert ret, "Could not read frame from rgb.mp4"
        return frame  # BGR

    def _real_intrinsics(self) -> Dict[str, float]:
        mat = np.loadtxt(str(REAL_DATA_DIR / "camera_matrix.csv"), delimiter=",").reshape(3, 3)
        H, W = 1440, 1920
        return {"fx": float(mat[0, 0]), "fy": float(mat[1, 1]),
                "cx": float(mat[0, 2]), "cy": float(mat[1, 2]),
                "width": float(W), "height": float(H)}

    def test_geometric_depth_on_real_frame(self):
        from areamap.geometry.depth_engine import DepthEngine
        eng = DepthEngine(force_backend="geometric")
        frame = self._load_frame()
        intr = self._real_intrinsics()
        H, W = frame.shape[:2]
        seam_v = H * 0.68

        depth = eng.predict(frame, intr, camera_height=1.45, seam_v=seam_v)
        assert depth.shape == (H, W)
        assert float(depth.max()) > 0.5, "All depths are near zero on real frame"

    def test_real_frame_pointcloud_z_span(self):
        """Z-span of the point cloud from a real frame must be > 1 m."""
        from areamap.geometry.depth_engine import DepthEngine
        eng = DepthEngine(force_backend="geometric")
        frame = self._load_frame()
        intr = self._real_intrinsics()
        H, W = frame.shape[:2]
        seam_v = H * 0.68

        _, pts = eng.predict_and_unproject(
            frame, intr, camera_height=1.45, seam_v=seam_v,
            ceiling_height=ROOM_GT["ceiling"], pixel_step=8
        )
        assert pts.shape[0] > 100, "Too few points from real frame"
        z_span = float(pts[:, 2].max() - pts[:, 2].min())
        assert z_span >= 1.0, (
            f"Z-span={z_span:.3f} m is less than 1 m on a real room frame"
        )

    def test_ceiling_height_estimate_within_30pct(self):
        """95th-percentile Z should be within 30% of GT ceiling height."""
        from areamap.geometry.depth_engine import DepthEngine
        eng = DepthEngine(force_backend="geometric")
        frame = self._load_frame()
        intr = self._real_intrinsics()
        H = frame.shape[0]
        seam_v = H * 0.68
        cam_h = 1.45
        gt_ceil = ROOM_GT["ceiling"]

        _, pts = eng.predict_and_unproject(
            frame, intr, camera_height=cam_h, seam_v=seam_v,
            ceiling_height=gt_ceil, pixel_step=8
        )
        z95 = float(np.percentile(pts[:, 2], 95))
        # With geometric backend, Z95 ≈ ceiling_height (prior-anchored)
        error_pct = abs(z95 - gt_ceil) / gt_ceil
        assert error_pct <= 0.30, (
            f"Z95={z95:.3f} m vs GT ceiling {gt_ceil} m: error={error_pct*100:.1f}%"
        )

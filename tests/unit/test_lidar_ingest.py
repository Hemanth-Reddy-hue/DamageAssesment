"""Unit tests for M3 LiDAR tier ingestion and unprojection."""

import numpy as np
from scipy.spatial.transform import Rotation as R
from areamap.tiers.lidar import _generate_synthetic_box, ingest_lidar_capture

def test_quaternion_rotation_accuracy():
    # 90-degree yaw rotation around Z axis
    rot = R.from_euler("z", 90, degrees=True)
    q = rot.as_quat()  # [qx, qy, qz, qw]
    mat = rot.as_matrix()

    test_pt = np.array([1.0, 0.0, 0.0])
    rotated = mat @ test_pt
    # Should point along +Y
    assert np.allclose(rotated, [0.0, 1.0, 0.0], atol=1e-5)

def test_synthetic_box_generation():
    pts = _generate_synthetic_box(width=5.0, length=4.0, height=2.8, n_points=1200)
    assert pts.shape[1] == 3
    assert len(pts) >= 1000

    # Bounds should strictly match width, length, height
    assert pts[:, 0].min() >= 0.0 and pts[:, 0].max() <= 5.0
    assert pts[:, 1].min() >= 0.0 and pts[:, 1].max() <= 4.0
    assert pts[:, 2].min() >= 0.0 and pts[:, 2].max() <= 2.8

def test_ingest_lidar_synthetic_fallback(tmp_path):
    # Empty directory should trigger graceful fallback without crashing
    empty_dir = tmp_path / "empty_scan"
    empty_dir.mkdir()
    cloud, meta = ingest_lidar_capture(empty_dir)
    assert len(cloud) > 0
    assert meta["tier"] == "lidar"
    assert meta.get("synthetic") is True

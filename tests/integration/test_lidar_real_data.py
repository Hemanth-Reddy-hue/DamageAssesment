"""Integration test verifying M3 LiDAR ingestion and M4 Room Geometry on real sensor data."""

from pathlib import Path
from areamap.tiers.lidar import ingest_lidar_capture
from areamap.geometry.planes import fit_room_planes

def test_lidar_real_data_pipeline():
    scan_dir = Path("Data/SingleRoom")
    if not scan_dir.exists():
        return

    # Ingest real sensor files
    pts, meta = ingest_lidar_capture(scan_dir, keyframe_stride=30)
    assert len(pts) >= 1000
    assert meta["tier"] == "lidar"
    assert meta["downsampled_point_count"] > 0

    # Fit room geometry
    geom = fit_room_planes(pts, room_id="single_room_01", room_name="Living Room", tier="lidar")

    # Assertions on real room geometry
    assert geom.ceiling_height.value >= 2.0 and geom.ceiling_height.value <= 3.5
    assert geom.floor_area.value >= 10.0 and geom.floor_area.value <= 80.0
    assert len(geom.walls) >= 4
    assert len(geom.floor_polygon) >= 4

    # Confidence interval validity
    assert geom.ceiling_height.lo < geom.ceiling_height.hi
    assert geom.floor_area.lo < geom.floor_area.hi
    for wall in geom.walls:
        assert wall.length.lo < wall.length.hi
        assert wall.length.value > 0.5

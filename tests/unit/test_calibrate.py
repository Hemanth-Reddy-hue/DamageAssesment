"""Unit tests for interval calculation and calibration math."""

from areamap.geometry.uncertainty import calculate_interval, PHYSICAL_NOISE_FLOORS

def test_interval_tier_ordering():
    val = 4.0
    iv_lidar = calculate_interval(val, "wall", tier="lidar")
    iv_video = calculate_interval(val, "wall", tier="video")
    iv_photo = calculate_interval(val, "wall", tier="photo")

    width_lidar = iv_lidar.hi - iv_lidar.lo
    width_video = iv_video.hi - iv_video.lo
    width_photo = iv_photo.hi - iv_photo.lo

    assert width_photo > width_video > width_lidar

def test_physical_noise_floor():
    # Very small nominal value should still adhere to physical sensor noise floor
    tiny_val = 0.001
    iv = calculate_interval(tiny_val, "wall", tier="lidar")
    width = iv.hi - iv.lo
    # Sensor physical floor is at least 5mm
    assert width >= PHYSICAL_NOISE_FLOORS["lidar"]

"""Unit tests for Pydantic state models and schema validation."""

import pytest
from pydantic import ValidationError
from areamap.state import Interval, WallSegment, CaptureState, RoomGeometry

def test_interval_validation():
    # Valid interval
    iv = Interval(value=4.0, lo=3.95, hi=4.05, method="conformal", tier="lidar")
    assert iv.value == 4.0
    assert iv.lo == 3.95
    assert iv.hi == 4.05

    # Missing required field must raise ValidationError
    with pytest.raises(ValidationError):
        Interval(value=4.0, lo=3.95)

def test_wall_segment_validation():
    iv = Interval(value=4.0, lo=3.95, hi=4.05, method="conformal", tier="lidar")
    wall = WallSegment(wall_id="w1", start=[0.0, 0.0], end=[4.0, 0.0], length=iv)
    assert wall.wall_id == "w1"
    assert wall.length.value == 4.0

def test_capture_state_minimal():
    state = CaptureState(capture_path="data/raw/room_01/lidar/run1", tier="lidar")
    assert state.tier == "lidar"
    assert len(state.rooms) == 0
    assert state.damage == []

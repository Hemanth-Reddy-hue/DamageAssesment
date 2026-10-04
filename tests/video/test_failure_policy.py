"""Test failure policy: blank/missing video fails cleanly and no synthetic box is created unless permitted."""

from pathlib import Path
import pytest

from areamap.config import settings
from areamap.tiers.video import ingest_video_capture
from areamap.geometry.planes import _create_fallback_room


def test_missing_video_raises_or_fails():
    """Missing video path raises FileNotFoundError when ALLOW_SYNTHETIC=False."""
    settings.allow_synthetic = False
    with pytest.raises(FileNotFoundError):
        ingest_video_capture("non_existent_video_path.mp4")


def test_synthetic_fallback_disabled_by_default():
    """_create_fallback_room raises RuntimeError when ALLOW_SYNTHETIC=False."""
    settings.allow_synthetic = False
    with pytest.raises(RuntimeError):
        _create_fallback_room("room_01", "Living Room", tier="video")


def test_synthetic_fallback_tagged_when_explicitly_allowed():
    """When ALLOW_SYNTHETIC=True, fallback room is explicitly tagged with provenance='synthetic'."""
    settings.allow_synthetic = True
    room = _create_fallback_room("room_01", "Living Room", tier="video")
    assert room.provenance == "synthetic"

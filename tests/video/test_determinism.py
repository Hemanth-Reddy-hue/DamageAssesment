"""Test determinism of plane fitting with fixed RNG seed."""

import numpy as np
import pytest

from areamap.geometry.planes import fit_room_planes
from areamap.tiers.lidar import _generate_synthetic_box


def test_fit_room_planes_deterministic():
    """Two runs on the identical cloud with the same seed yield identical plane dimensions and walls."""
    pts = _generate_synthetic_box(4.5, 3.5, 2.6)

    geom1 = fit_room_planes(pts, room_id="room_01", tier="video", seed=42)
    geom2 = fit_room_planes(pts, room_id="room_01", tier="video", seed=42)

    assert geom1.ceiling_height.value == geom2.ceiling_height.value
    assert geom1.floor_area.value == geom2.floor_area.value
    assert len(geom1.walls) == len(geom2.walls)

    for w1, w2 in zip(geom1.walls, geom2.walls):
        assert np.isclose(w1.length.value, w2.length.value, atol=1e-5)
        assert w1.start == w2.start
        assert w1.end == w2.end

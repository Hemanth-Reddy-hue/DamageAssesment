import numpy as np
from areamap.geometry.planes import fit_room_planes
from areamap.tiers.lidar import _generate_synthetic_box


def _noisy(box, seed=0, sigma=0.004):
    rng = np.random.default_rng(seed)
    return box + rng.normal(0, sigma, box.shape)


def test_full_box_lidar_is_measured_and_tight():
    pts = _noisy(_generate_synthetic_box(4.0, 3.0, 2.6, n_points=8000))
    g = fit_room_planes(pts, tier="lidar")
    assert g.ceiling_height.method == "measured"
    assert abs(g.ceiling_height.value - 2.6) < 0.05
    assert (g.ceiling_height.hi - g.ceiling_height.lo) <= 0.06


def test_floor_only_is_prior_with_wide_band():
    box = _generate_synthetic_box(4.0, 3.0, 2.6, n_points=8000)
    pts = _noisy(box[box[:, 2] <= 1.4])
    g = fit_room_planes(pts, tier="lidar")
    assert g.ceiling_height.method == "prior"
    assert (g.ceiling_height.hi - g.ceiling_height.lo) >= 0.79


def test_implausible_ceiling_is_not_clamped_to_2_4():
    pts = _noisy(_generate_synthetic_box(4.0, 3.0, 1.5, n_points=8000))
    g = fit_room_planes(pts, tier="lidar")
    assert g.ceiling_height.method == "prior"
    assert g.ceiling_height.value != 2.4


def test_photo_tier_ceiling_is_always_prior():
    pts = _noisy(_generate_synthetic_box(4.0, 3.2, 2.5, n_points=4000), sigma=0.03)
    g = fit_room_planes(pts, tier="photo")
    assert g.ceiling_height.method == "prior"

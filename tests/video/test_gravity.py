"""Test camera-based gravity vector estimation and world alignment."""

import numpy as np
import pytest

from areamap.tiers.video_sfm import _compute_gravity_rotation


def test_gravity_recovery_known_tilt():
    """Synthetic cameras tilted by 25 degrees around X-axis should be correctly rotated back to +Z."""
    tilt_rad = np.radians(25.0)
    cos_t = np.cos(tilt_rad)
    sin_t = np.sin(tilt_rad)

    # In world coordinates, up is tilted along Y by 25 degrees: [0, sin_t, cos_t]
    known_up = np.array([0.0, sin_t, cos_t])

    # Generate 10 cameras around this mean up with small jitter
    rng = np.random.default_rng(42)
    synthetic_ups = []
    for _ in range(10):
        noise = rng.normal(0, 0.02, 3)
        u = known_up + noise
        u = u / np.linalg.norm(u)
        synthetic_ups.append(u)

    R_grav = _compute_gravity_rotation(synthetic_ups)

    # Apply R_grav to known_up: should become [0, 0, 1]
    aligned = R_grav @ known_up
    target_z = np.array([0.0, 0.0, 1.0])

    angle_err_deg = np.degrees(np.arccos(np.clip(np.dot(aligned, target_z), -1.0, 1.0)))
    assert angle_err_deg < 2.0, f"Gravity alignment error {angle_err_deg:.2f}° exceeds 2.0°"

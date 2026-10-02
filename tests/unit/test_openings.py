"""Unit tests for Module M5: Openings (Doors, Windows, Passageways) & Gate G1 Compliance."""

import numpy as np
import pytest
from areamap.state import WallSegment, Interval, CaptureState, RoomGeometry
from areamap.geometry.openings import (
    unroll_wall_points,
    detect_cutouts_on_wall,
    detect_openings_from_cutouts,
)
from areamap.nodes.openings import openings_node

def _make_wall(wall_id: str, start: list[float], end: list[float]) -> WallSegment:
    length = float(np.hypot(end[0] - start[0], end[1] - start[1]))
    iv = Interval(value=length, lo=length - 0.02, hi=length + 0.02, confidence_level=0.9, method="conformal", tier="lidar")
    return WallSegment(wall_id=wall_id, start=start, end=end, length=iv, plane_equation=[0, 1, 0, 0])

def test_unroll_wall_points():
    """Verify 3D to 2D projection along a wall baseline."""
    wall = _make_wall("w1", [0.0, 0.0], [4.0, 0.0])
    
    # Points along the wall at X in [0, 4], Y near 0, Z in [0, 2.5]
    pts = np.array([
        [1.0, 0.05, 1.2],
        [2.5, -0.05, 1.8],
        [3.8, 0.02, 0.5],
        [2.0, 0.80, 1.0],  # Too far perpendicularly (should be filtered)
        [5.0, 0.00, 1.0],  # Beyond wall end (should be filtered)
    ])
    
    u, v = unroll_wall_points(wall, pts, wall_thickness_tolerance=0.20, ceiling_height=2.60)
    assert len(u) == 3
    assert np.allclose(u, [1.0, 2.5, 3.8], atol=1e-3)
    assert np.allclose(v, [1.2, 1.8, 0.5], atol=1e-3)

def test_detect_cutouts_door_exact_dimensions():
    """Verify door cutout detection and Gate G1 opening width error <= 2 cm."""
    wall = _make_wall("w_door", [0.0, 0.0], [4.0, 0.0])
    gt_door_width = 0.900  # 90 cm standard door
    gt_door_height = 2.050

    # Generate synthetic wall points with a door cutout at u in [1.50, 2.40]
    rng = np.random.default_rng(42)
    u_all = rng.uniform(0.0, 4.0, 10000)
    z_all = rng.uniform(0.0, 2.6, 10000)
    y_all = rng.normal(0.0, 0.01, 10000)

    # Cutout: u in [1.50, 2.40] and z in [0.0, 2.05]
    door_mask = (u_all >= 1.50) & (u_all <= 2.40) & (z_all <= 2.05)
    pts = np.column_stack([u_all[~door_mask], y_all[~door_mask], z_all[~door_mask]])

    cutouts = detect_cutouts_on_wall(wall, pts, ceiling_height=2.60)
    assert len(cutouts) == 1

    door = cutouts[0]
    assert door["type"] == "door"
    assert door["sill_height"] == 0.0
    assert door["height"] == 2.05

    # Gate G1 Accuracy Test: Width error must be <= 2 cm (0.02 m)
    width_err_cm = abs(door["width"] - gt_door_width) * 100.0
    print(f"Door width error: {width_err_cm:.2f} cm (Gate G1 limit <= 2.0 cm)")
    assert width_err_cm <= 2.0, f"Door width error {width_err_cm:.2f} cm exceeds Gate G1 limit of 2 cm"

def test_detect_cutouts_window():
    """Verify window cutout detection with sill height above floor level."""
    wall = _make_wall("w_win", [0.0, 0.0], [4.0, 0.0])
    gt_win_width = 1.200  # 120 cm window
    gt_win_sill = 0.900

    # Generate synthetic wall points with window cutout at u in [1.00, 2.20], z in [0.90, 2.00]
    rng = np.random.default_rng(42)
    u_all = rng.uniform(0.0, 4.0, 12000)
    z_all = rng.uniform(0.0, 2.6, 12000)
    y_all = rng.normal(0.0, 0.01, 12000)

    win_mask = (u_all >= 1.00) & (u_all <= 2.20) & (z_all >= 0.90) & (z_all <= 2.00)
    pts = np.column_stack([u_all[~win_mask], y_all[~win_mask], z_all[~win_mask]])

    cutouts = detect_cutouts_on_wall(wall, pts, ceiling_height=2.60)
    assert len(cutouts) == 1

    win = cutouts[0]
    assert win["type"] == "window"
    assert win["sill_height"] >= 0.80

    # Gate G1 Accuracy Test
    width_err_cm = abs(win["width"] - gt_win_width) * 100.0
    print(f"Window width error: {width_err_cm:.2f} cm (Gate G1 limit <= 2.0 cm)")
    assert width_err_cm <= 2.0

def test_phantom_suppression():
    """Verify that narrow shadows and un-framed corner voids are suppressed as phantoms."""
    wall = _make_wall("w_solid", [0.0, 0.0], [4.0, 0.0])

    # 1. Narrow void (width 0.30m < 0.65m minimum threshold)
    rng = np.random.default_rng(42)
    u_all = rng.uniform(0.0, 4.0, 8000)
    z_all = rng.uniform(0.0, 2.6, 8000)
    y_all = rng.normal(0.0, 0.01, 8000)

    narrow_shadow = (u_all >= 1.50) & (u_all <= 1.80) & (z_all <= 2.05)
    pts_narrow = np.column_stack([u_all[~narrow_shadow], y_all[~narrow_shadow], z_all[~narrow_shadow]])

    cutouts_narrow = detect_cutouts_on_wall(wall, pts_narrow, ceiling_height=2.60)
    assert len(cutouts_narrow) == 0, "Phantom narrow occlusion was not suppressed!"

    # 2. Corner boundary void (u < 0.15m from corner)
    corner_dropout = (u_all >= 0.0) & (u_all <= 0.80) & (z_all <= 2.05)
    pts_corner = np.column_stack([u_all[~corner_dropout], y_all[~corner_dropout], z_all[~corner_dropout]])

    cutouts_corner = detect_cutouts_on_wall(wall, pts_corner, ceiling_height=2.60)
    assert len(cutouts_corner) == 0, "Phantom corner edge dropout was not suppressed!"

def test_gate_g1_multi_opening_compliance():
    """Verify Gate G1: Opening width error <= 2 cm on >= 85% of openings."""
    wall1 = _make_wall("w1", [0.0, 0.0], [4.0, 0.0])
    wall2 = _make_wall("w2", [4.0, 0.0], [4.0, 3.0])

    rng = np.random.default_rng(42)
    
    # Wall 1 has door: width 0.90m at [1.5, 2.4]
    u1 = rng.uniform(0.0, 4.0, 8000)
    z1 = rng.uniform(0.0, 2.6, 8000)
    y1 = rng.normal(0.0, 0.01, 8000)
    d_mask = (u1 >= 1.50) & (u1 <= 2.40) & (z1 <= 2.05)
    pts1 = np.column_stack([u1[~d_mask], y1[~d_mask], z1[~d_mask]])

    # Wall 2 has window: width 1.20m at [0.9, 2.1]
    y2 = rng.uniform(0.0, 3.0, 8000)
    z2 = rng.uniform(0.0, 2.6, 8000)
    x2 = rng.normal(4.0, 0.01, 8000)
    w_mask = (y2 >= 0.90) & (y2 <= 2.10) & (z2 >= 0.90) & (z2 <= 2.00)
    pts2 = np.column_stack([x2[~w_mask], y2[~w_mask], z2[~w_mask]])

    all_pts = np.vstack([pts1, pts2])
    openings = detect_openings_from_cutouts([wall1, wall2], tier="lidar", point_cloud=all_pts, ceiling_height=2.60)
    
    assert len(openings) == 2
    ground_truth_widths = [0.900, 1.200]

    passes = 0
    for op, gt_w in zip(openings, ground_truth_widths):
        err_cm = abs(op.width.value - gt_w) * 100.0
        if err_cm <= 2.0:
            passes += 1
        # Also verify calibrated interval bounds bracket the ground truth
        assert op.width.lo <= gt_w <= op.width.hi

    pass_rate = (passes / len(openings)) * 100.0
    print(f"Gate G1 Opening Widths Pass Rate: {pass_rate:.1f}% (Required >= 85.0%)")
    assert pass_rate >= 85.0

def test_openings_node_end_to_end(tmp_path):
    """Test openings_node integration in full state pipeline."""
    w1 = _make_wall("w1", [0.0, 0.0], [4.0, 0.0])
    w2 = _make_wall("w2", [4.0, 0.0], [4.0, 3.0])
    w3 = _make_wall("w3", [4.0, 3.0], [0.0, 3.0])
    w4 = _make_wall("w4", [0.0, 3.0], [0.0, 0.0])

    ceil_iv = Interval(value=2.60, lo=2.58, hi=2.62, confidence_level=0.9, method="conformal", tier="lidar")
    area_iv = Interval(value=12.0, lo=11.8, hi=12.2, confidence_level=0.9, method="conformal", tier="lidar")

    geom = RoomGeometry(
        room_id="room_01",
        room_name="Living Room",
        ceiling_height=ceil_iv,
        floor_area=area_iv,
        walls=[w1, w2, w3, w4],
        floor_polygon=[[0, 0], [4, 0], [4, 3], [0, 3]]
    )

    state = CaptureState(
        capture_path="dummy",
        tier="lidar",
        room_geometry={"room_01": geom}
    )

    out = openings_node(state)
    assert "openings" in out
    assert "room_01" in out["openings"]
    assert len(out["openings"]["room_01"]) >= 1
    for op in out["openings"]["room_01"]:
        assert op.type in ["door", "window", "passageway"]
        assert op.width.value > 0.5
        assert len(op.position) == 3

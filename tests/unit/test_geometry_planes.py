"""Unit tests for M4 Room Geometry and RANSAC plane fitting."""

import numpy as np
from areamap.geometry.planes import (
    fit_plane_ransac,
    calculate_polygon_area,
    intersect_2d_lines,
    fit_room_planes
)
from areamap.tiers.lidar import _generate_synthetic_box

def test_plane_ransac_known_plane():
    # Generate points on z = 2.5 plane with small Gaussian noise
    x = np.random.uniform(-3, 3, 500)
    y = np.random.uniform(-3, 3, 500)
    z = 2.5 + np.random.normal(0, 0.005, 500)
    pts = np.column_stack([x, y, z])

    normal, d, inliers = fit_plane_ransac(pts, distance_threshold=0.02)
    assert normal is not None
    # Normal should point along Z [0, 0, 1] or [0, 0, -1]
    assert abs(abs(normal[2]) - 1.0) < 0.02
    assert len(inliers) >= 450

def test_shoelace_polygon_area():
    # 4m x 3m rectangle
    rect = [[0.0, 0.0], [4.0, 0.0], [4.0, 3.0], [0.0, 3.0]]
    area = calculate_polygon_area(rect)
    assert abs(area - 12.0) < 1e-5

    # Triangle 3m base, 4m height -> area 6.0
    tri = [[0.0, 0.0], [3.0, 0.0], [0.0, 4.0]]
    assert abs(calculate_polygon_area(tri) - 6.0) < 1e-5

def test_intersect_2d_lines():
    # Line 1: y = 0  ->  n1=[0, 1], d1=0
    # Line 2: x = 4  ->  n2=[1, 0], d2=-4
    n1 = np.array([0.0, 1.0])
    d1 = 0.0
    n2 = np.array([1.0, 0.0])
    d2 = -4.0
    pt = intersect_2d_lines(n1, d1, n2, d2)
    assert pt is not None
    assert np.allclose(pt, [4.0, 0.0], atol=1e-5)

def test_fit_room_planes_synthetic_box():
    pts = _generate_synthetic_box(width=4.0, length=3.0, height=2.6, n_points=3000)
    geom = fit_room_planes(pts, room_id="test_room", tier="lidar")

    # Gate G2: ceiling error <= 1.5 cm (0.015m)
    assert abs(geom.ceiling_height.value - 2.60) <= 0.015
    # Floor area within 2%
    assert abs(geom.floor_area.value - 12.0) <= 0.25
    # Intervals must be populated and valid
    assert geom.ceiling_height.lo <= geom.ceiling_height.value <= geom.ceiling_height.hi
    assert len(geom.walls) >= 4

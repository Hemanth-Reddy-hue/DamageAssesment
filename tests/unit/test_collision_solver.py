"""Tests for the constrained room collision solver (Fix #8)."""

import pytest
from areamap.geometry.solver import resolve_room_collision

def test_no_overlap_returns_zero():
    # Rooms placed side by side with gap
    placed = [[0, 0], [4, 0], [4, 4], [0, 4]]
    new_rm = [[5, 0], [9, 0], [9, 4], [5, 4]]
    dx, dy = resolve_room_collision(placed, new_rm)
    assert abs(dx) < 1e-3 and abs(dy) < 1e-3, "Should not shift if no overlap"

def test_identical_overlap_shifts_apart():
    # Complete overlap
    placed = [[0, 0], [4, 0], [4, 4], [0, 4]]
    new_rm = [[0, 0], [4, 0], [4, 4], [0, 4]]
    dx, dy = resolve_room_collision(placed, new_rm)
    assert abs(dx) > 0.5 or abs(dy) > 0.5, "Must shift significantly for total overlap"

def test_partial_overlap_shifts_minimum_distance():
    # Overlapping by 1 unit on X axis
    placed = [[0, 0], [4, 0], [4, 4], [0, 4]]
    new_rm = [[3, 0], [7, 0], [7, 4], [3, 4]]
    dx, dy = resolve_room_collision(placed, new_rm)
    
    # Ideally should shift by ~1.0 in X and 0 in Y, since that resolves it
    assert abs(abs(dx) - 1.0) < 0.2 or abs(abs(dy) - 4.0) < 0.2, f"Shift was dx={dx}, dy={dy}"

def test_invalid_polygon_handled_gracefully():
    placed = [[0, 0], [4, 0], [4, 4], [0, 4]]
    # Bowtie / self-intersecting
    new_rm = [[0, 0], [4, 4], [4, 0], [0, 4]]
    dx, dy = resolve_room_collision(placed, new_rm)
    assert isinstance(dx, float) and isinstance(dy, float)

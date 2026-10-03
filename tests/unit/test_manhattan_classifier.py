"""Unit tests for the dynamic Manhattan room classifier (Issue #10)."""

import pytest
import numpy as np
from areamap.geometry.planes import classify_rectilinear

def test_square_room_is_rectilinear():
    # Walls at 0, 90, 180, 270 degrees
    walls = [
        {"normal_2d": np.array([1.0, 0.0])},
        {"normal_2d": np.array([0.0, 1.0])},
        {"normal_2d": np.array([-1.0, 0.0])},
        {"normal_2d": np.array([0.0, -1.0])}
    ]
    
    assert classify_rectilinear(walls) == True, "Square room should be rectilinear"

def test_hexagonal_room_is_not_rectilinear():
    # Hexagonal room walls at 60 degree increments
    angles = [0, 60, 120, 180, 240, 300]
    walls = []
    for a in angles:
        rad = np.radians(a)
        walls.append({"normal_2d": np.array([np.cos(rad), np.sin(rad)])})
        
    assert classify_rectilinear(walls) == False, "Hexagonal room should not be rectilinear"

def test_room_with_bay_window_is_not_rectilinear():
    # Normal square room, but one wall is at 45 degrees
    walls = [
        {"normal_2d": np.array([1.0, 0.0])},
        {"normal_2d": np.array([0.0, 1.0])},
        {"normal_2d": np.array([-1.0, 0.0])},
        {"normal_2d": np.array([0.707, -0.707])} # 45 degree bay window wall
    ]
    
    assert classify_rectilinear(walls) == False, "Room with 45-degree wall should not be rectilinear"

def test_noisy_manhattan_room_is_rectilinear():
    # Walls at 0, 85, 182, 275 (all within ~8 degree tolerance of 90 deg increments)
    angles = [0, 85, 182, 275]
    walls = []
    for a in angles:
        rad = np.radians(a)
        walls.append({"normal_2d": np.array([np.cos(rad), np.sin(rad)])})
        
    assert classify_rectilinear(walls) == True, "Noisy Manhattan room should be rectilinear"

def test_few_walls():
    # Fallback to True if less than 3 walls found (too little data to disqualify)
    walls = [
        {"normal_2d": np.array([1.0, 0.0])},
    ]
    assert classify_rectilinear(walls) == True

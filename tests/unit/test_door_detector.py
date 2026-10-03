"""Unit test for door_detector.py"""
import pytest
import numpy as np
from areamap.geometry.door_detector import detect_doors_in_image

def test_detect_doors_in_image_synthetic():
    h, w = 1080, 1920
    # Create fake RGB with a black door frame
    img = np.ones((h, w, 3), dtype=np.uint8) * 200
    # Door from x=800 to x=1100, reaching the floor (seam_v = 800)
    img[200:800, 800:1100] = 50 # Darker door
    
    # Create fake depth map
    depth = np.ones((h, w), dtype=np.float32) * 2.0 # Wall is 2m away
    # Doorway is 4m away (hole)
    depth[200:800, 800:1100] = 4.0
    
    seam_v = 800.0
    
    doors = detect_doors_in_image(img, depth, seam_v)
    
    assert len(doors) > 0, "Failed to detect synthetic door"
    best_door = doors[0]
    
    assert 940 <= best_door["u_center"] <= 960, f"Expected center ~950, got {best_door['u_center']}"
    assert 280 <= best_door["width_pixels"] <= 320, f"Expected width ~300, got {best_door['width_pixels']}"
    assert best_door["confidence"] > 0.8

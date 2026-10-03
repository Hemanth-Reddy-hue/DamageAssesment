"""Unit tests for the visual place recognition engine (Fix #6)."""

import pytest
import numpy as np
from areamap.geometry.place_recognition import PlaceRecognizer, detect_visual_loop_closures
from areamap.state import CaptureState, RoomGeometry, Interval

def _make_iv(val: float, err: float = 0.02) -> Interval:
    return Interval(value=val, lo=val - err, hi=val + err, confidence_level=0.9, method="conformal", tier="photo")

@pytest.fixture
def dummy_state():
    room_A = RoomGeometry(
        room_id="room_A", room_name="Living",
        ceiling_height=_make_iv(2.5), floor_area=_make_iv(10),
        walls=[], floor_polygon=[[0,0], [3,0], [3,3], [0,3]]
    )
    room_B = RoomGeometry(
        room_id="room_B", room_name="Bed",
        ceiling_height=_make_iv(2.5), floor_area=_make_iv(10),
        walls=[], floor_polygon=[[3,0], [6,0], [6,3], [3,3]]
    )
    room_C = RoomGeometry(
        room_id="room_C", room_name="Bath",
        ceiling_height=_make_iv(2.5), floor_area=_make_iv(10),
        walls=[], floor_polygon=[[6,0], [9,0], [9,3], [6,3]]
    )
    return CaptureState(
        capture_path="dummy",
        tier="photo",
        room_geometry={"room_A": room_A, "room_B": room_B, "room_C": room_C},
        rooms=["room_A", "room_B", "room_C"],
        point_clouds={}
    )

def test_place_recognizer_init():
    pr = PlaceRecognizer()
    # It shouldn't crash. If HAS_ML is true, it loads mobilenetv3_small_050
    assert True

def test_detect_loop_closures_empty(dummy_state, monkeypatch):
    # Mock no images found
    monkeypatch.setattr("areamap.geometry.place_recognition.PlaceRecognizer._get_images_for_room", lambda self, p, r: [])
    closures = detect_visual_loop_closures(dummy_state)
    assert len(closures) == 0

def test_detect_loop_closures_finds_match(dummy_state, monkeypatch):
    def mock_compute(self, paths):
        # Fake identical embeddings for room A and C
        if "room_B" in paths:
            return np.array([1.0, 0.0])
        return np.array([0.0, 1.0])
        
    def mock_get(self, path, r_id):
        return [f"dummy_{r_id}.jpg"]

    monkeypatch.setattr("areamap.geometry.place_recognition.PlaceRecognizer._get_images_for_room", mock_get)
    monkeypatch.setattr("areamap.geometry.place_recognition.PlaceRecognizer.compute_room_embedding", mock_compute)
    
    # Mock ICP to return a valid transform
    monkeypatch.setattr("areamap.geometry.place_recognition._load_point_cloud", lambda p: np.zeros((30, 3)))
    monkeypatch.setattr("areamap.geometry.place_recognition.icp_align", lambda source, target: (np.eye(4), 0.1))
    
    # Needs 3+ rooms. dummy_state has 3 rooms.
    dummy_state.point_clouds = {"room_A": "a.ply", "room_C": "c.ply"}
    
    closures = detect_visual_loop_closures(dummy_state, similarity_threshold=0.9)
    assert len(closures) == 1
    assert closures[0]["from_room"] == "room_C"
    assert closures[0]["to_room"] == "room_A"
    assert closures[0]["confidence"] > 0.9

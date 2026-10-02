"""Unit tests for Module M6: Multi-room Ingest, Stitching, and Drift Correction (Gate G4 & Gate G5)."""

import numpy as np
import pytest
from pathlib import Path
from PIL import Image
from areamap.state import RoomGeometry, WallSegment, Opening, Interval, CaptureState
from areamap.geometry.posegraph import (
    make_se2_matrix,
    extract_se2_params,
    snap_to_manhattan_angle,
    optimize_pose_graph,
)
from areamap.geometry.adjacency import (
    compute_wall_outward_normal,
    align_room_pair_se2,
    apply_se2_transform_to_room,
    infer_room_adjacency,
    check_room_overlaps,
)
from areamap.nodes.stitch import stitch_node
from areamap.nodes.ingest import ingest_node

def _make_iv(val: float, err: float = 0.02) -> Interval:
    return Interval(value=val, lo=val - err, hi=val + err, confidence_level=0.9, method="conformal", tier="lidar")

def _make_box_room(room_id: str, room_name: str, width: float, length: float, height: float = 2.60) -> RoomGeometry:
    w1 = WallSegment(wall_id=f"{room_id}_w1", start=[0.0, 0.0], end=[width, 0.0], length=_make_iv(width))
    w2 = WallSegment(wall_id=f"{room_id}_w2", start=[width, 0.0], end=[width, length], length=_make_iv(length))
    w3 = WallSegment(wall_id=f"{room_id}_w3", start=[width, length], end=[0.0, length], length=_make_iv(width))
    w4 = WallSegment(wall_id=f"{room_id}_w4", start=[0.0, length], end=[0.0, 0.0], length=_make_iv(length))
    
    poly = [[0.0, 0.0], [width, 0.0], [width, length], [0.0, length]]
    area = width * length
    return RoomGeometry(
        room_id=room_id,
        room_name=room_name,
        ceiling_height=_make_iv(height),
        floor_area=_make_iv(area, err=0.2),
        walls=[w1, w2, w3, w4],
        floor_polygon=poly,
        openings=[]
    )

def test_se2_alignment_two_rooms():
    """Verify that SE(2) door alignment snaps Room B adjacent to Room A without overlapping."""
    room_A = _make_box_room("room_A", "Living Room", 4.0, 3.0)
    # Door on Wall 2 (x=4) at [4.0, 1.5]
    door_A = Opening(
        opening_id="door_A",
        wall_id="room_A_w2",
        type="door",
        width=_make_iv(0.90),
        height=_make_iv(2.05),
        sill_height=_make_iv(0.0),
        position=[4.0, 1.5, 1.025]
    )
    room_A.openings.append(door_A)

    room_B = _make_box_room("room_B", "Kitchen", 3.0, 3.0)
    # Door on Wall 4 (x=0) at [0.0, 1.5]
    door_B = Opening(
        opening_id="door_B",
        wall_id="room_B_w4",
        type="door",
        width=_make_iv(0.90),
        height=_make_iv(2.05),
        sill_height=_make_iv(0.0),
        position=[0.0, 1.5, 1.025]
    )
    room_B.openings.append(door_B)

    # Solve transform
    T = align_room_pair_se2(room_A, room_B, door_A, door_B)
    aligned_B = apply_se2_transform_to_room(room_B, T)

    # Verify Room B's global coordinates start at x=4.0
    poly_B = np.array(aligned_B.floor_polygon)
    assert np.isclose(poly_B[:, 0].min(), 4.0, atol=1e-2)
    assert np.isclose(poly_B[:, 0].max(), 7.0, atol=1e-2)

    # Verify zero overlap between Room A and Room B
    overlaps = check_room_overlaps({"room_A": room_A, "room_B": aligned_B})
    assert len(overlaps) == 0, f"Unexpected overlap detected: {overlaps}"

def test_check_room_overlaps_detection():
    """Verify that overlapping room polygons trigger warning violations."""
    room_A = _make_box_room("room_A", "Room A", 4.0, 3.0)
    # Deliberately overlapping room: starts at x=2.0 (overlaps by 2m x 3m = 6 m2)
    room_overlap = _make_box_room("room_overlap", "Overlapping Room", 4.0, 3.0)
    T = make_se2_matrix(2.0, 0.0, 0.0)
    aligned_overlap = apply_se2_transform_to_room(room_overlap, T)

    warnings = check_room_overlaps({"room_A": room_A, "room_overlap": aligned_overlap})
    assert len(warnings) == 1
    assert "overlap detected" in warnings[0].lower()

def test_pose_graph_drift_correction_ablation_gate_g4():
    """Verify Gate G4: Measurable reduction in loop closure error with drift correction ON vs OFF."""
    # Simulate 3 connected rooms forming a loop: Room 1 -> Room 2 -> Room 3 -> Room 1
    rel_poses = [
        {"from_room": "room_1", "to_room": "room_2", "transform": make_se2_matrix(4.0, 0.0, 0.0)},
        {"from_room": "room_2", "to_room": "room_3", "transform": make_se2_matrix(0.0, 3.0, 0.0)},
    ]
    # Closure measurement from room_3 back to room_1 (with 0.40m synthetic accumulation drift)
    loop_closures = [
        {"from_room": "room_3", "to_room": "room_1", "transform": make_se2_matrix(-4.40, -3.00, 0.0)}
    ]

    # Dead reckoning (drift correction OFF)
    poses_off = optimize_pose_graph(rel_poses, loop_closures, enable_drift_correction=False)
    # Optimized (drift correction ON)
    poses_on = optimize_pose_graph(rel_poses, loop_closures, enable_drift_correction=True)

    # Measure closure error at room_1
    pred_r1_off = poses_off["room_3"] @ loop_closures[0]["transform"]
    closure_err_off = float(np.hypot(pred_r1_off[0, 2], pred_r1_off[1, 2]))

    pred_r1_on = poses_on["room_3"] @ loop_closures[0]["transform"]
    closure_err_on = float(np.hypot(pred_r1_on[0, 2], pred_r1_on[1, 2]))

    drift_reduction_pct = ((closure_err_off - closure_err_on) / closure_err_off) * 100.0
    print(f"Drift Error OFF: {closure_err_off:.3f}m, ON: {closure_err_on:.3f}m, Reduction: {drift_reduction_pct:.1f}%")

    assert closure_err_on < closure_err_off
    assert drift_reduction_pct >= 30.0  # Gate G4: Measurable reduction demonstrated

def test_stitch_node_multi_room_pipeline():
    """Verify stitch_node aggregates total footprint area and stitches multiple rooms."""
    room_A = _make_box_room("room_A", "Living Room", 4.0, 3.0)
    door_A = Opening(
        opening_id="door_A",
        wall_id="room_A_w2",
        type="door",
        width=_make_iv(0.90),
        height=_make_iv(2.05),
        sill_height=_make_iv(0.0),
        position=[4.0, 1.5, 1.025]
    )
    room_A.openings.append(door_A)

    room_B = _make_box_room("room_B", "Bedroom", 3.0, 3.0)
    door_B = Opening(
        opening_id="door_B",
        wall_id="room_B_w4",
        type="door",
        width=_make_iv(0.90),
        height=_make_iv(2.05),
        sill_height=_make_iv(0.0),
        position=[0.0, 1.5, 1.025]
    )
    room_B.openings.append(door_B)

    state = CaptureState(
        capture_path="dummy_multi",
        tier="lidar",
        rooms=["room_A", "room_B"],
        room_geometry={"room_A": room_A, "room_B": room_B}
    )

    out = stitch_node(state)
    assert "stitched_plan" in out
    plan = out["stitched_plan"]

    assert len(plan.rooms) == 2
    # Total area: 12.0 m2 + 9.0 m2 = 21.0 m2
    assert np.isclose(plan.total_footprint_area.value, 21.0, atol=0.1)
    assert len(plan.footprint_polygon) >= 4
    assert plan.drift_correction_applied is True
    # Zero overlaps
    assert len(out.get("warnings", [])) == 0

def test_multi_room_ingest_detection(tmp_path):
    """Verify that ingest_node recognizes multi-room directory structures."""
    house_dir = tmp_path / "three_room_house"
    house_dir.mkdir()

    # Create 2 subdirectories with photo stills
    r1_dir = house_dir / "living_room"
    r1_dir.mkdir()
    r2_dir = house_dir / "kitchen"
    r2_dir.mkdir()

    for d in [r1_dir, r2_dir]:
        img = Image.new("RGB", (640, 480), color=(180, 180, 180))
        img.save(d / "photo_01.jpg")

    state = CaptureState(capture_path=str(house_dir), tier="photo")
    out = ingest_node(state)

    assert out["tier"] == "photo"
    assert out["device_meta"]["multi_room"] is True
    assert out["device_meta"]["room_count"] == 2
    assert out["rooms"] == ["room_01", "room_02"]
    assert "room_01" in out["point_clouds"]
    assert "room_02" in out["point_clouds"]

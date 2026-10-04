from areamap.state import CaptureState, RoomGeometry, WallSegment
from areamap.geometry.uncertainty import calculate_interval
from areamap.nodes.qa_critic import qa_critic_node

def _room():
    iv = lambda v: calculate_interval(v, "wall", "lidar")
    pts = [[0, 0], [4, 0], [4, 3], [0, 3]]
    walls = [
        WallSegment(wall_id="w1", start=[0, 0], end=[4, 0], length=iv(4.0)),
        WallSegment(wall_id="w2", start=[4, 0], end=[4, 3], length=iv(3.0)),
        WallSegment(wall_id="w3", start=[4, 3], end=[0, 3], length=iv(4.0)),
        WallSegment(wall_id="w4", start=[0, 3], end=[0, 0], length=iv(3.0)),
    ]
    return RoomGeometry(room_id="r1", room_name="r1", ceiling_height=calculate_interval(2.6, "ceiling", "lidar"),
                        floor_area=calculate_interval(12.0, "area", "lidar"), walls=walls, floor_polygon=pts)

def test_wall_closure_failure():
    st = CaptureState(capture_path="x", tier="lidar", rooms=["r1"], room_geometry={"r1": _room()})
    st.room_geometry["r1"].walls[0].end[0] += 0.3
    qa = qa_critic_node(st)["qa_report"]
    assert any("does not close to next wall" in f for f in qa.failed_checks)

def test_polygon_area_inconsistency_failure():
    st = CaptureState(capture_path="x", tier="lidar", rooms=["r1"], room_geometry={"r1": _room()})
    st.room_geometry["r1"].floor_area.value = 12.0 * 1.25 # 25% error
    qa = qa_critic_node(st)["qa_report"]
    assert any("differ >5%" in f for f in qa.failed_checks)

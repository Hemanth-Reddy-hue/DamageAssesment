import pytest
from areamap.state import CaptureState, RoomGeometry, WallSegment
from areamap.geometry.uncertainty import calculate_interval
from areamap.llm.client import LLMResult
import areamap.nodes.damage as dmg
from areamap.nodes.scope import scope_node
from areamap.nodes.qa_critic import qa_critic_node


def _room(rid):
    iv = lambda v: calculate_interval(v, "wall", "lidar")
    pts = [[0, 0], [4, 0], [4, 3], [0, 3]]
    walls = [WallSegment(wall_id=f"{rid}_w{i+1}", start=pts[i], end=pts[(i+1) % 4], length=iv(3.0)) for i in range(4)]
    return RoomGeometry(room_id=rid, room_name=rid, ceiling_height=calculate_interval(2.6, "ceiling", "lidar"),
                        floor_area=calculate_interval(12.0, "area", "lidar"), walls=walls, floor_polygon=pts)


class _Fake:
    def __init__(self, items): self.items = items
    def generate_structured(self, **kw):
        return LLMResult(ok=True, data={"damage_findings": self.items}, status="ok")


@pytest.mark.parametrize("rid", ["room_00", "room_A", "hallway"])
def test_wall_damage_binds_to_existing_wall(monkeypatch, rid):
    items = [{"damage_class": "crack", "severity": "moderate", "surface": "wall",
              "bounding_box_2d": [0.2, 0.2, 0.5, 0.6], "estimated_extent_m2": 0.5}]
    monkeypatch.setattr(dmg, "get_llm_client", lambda *a, **k: _Fake(items))
    monkeypatch.setattr(dmg, "_find_representative_image", lambda p: None)
    st = CaptureState(capture_path="x", tier="lidar", rooms=[rid], room_geometry={rid: _room(rid)})
    st.damage = dmg.damage_node(st)["damage"]
    wall_ids = {w.wall_id for w in st.room_geometry[rid].walls}
    assert st.damage and st.damage[0].surface_id in wall_ids
    st.scope_items = scope_node(st)["scope_items"]
    qa = qa_critic_node(st)["qa_report"]
    assert not [f for f in qa.failed_checks if "non-existent surface" in f]


def test_whole_image_bbox_flagged_and_extent_is_wide(monkeypatch):
    items = [{"damage_class": "crack", "severity": "minor", "surface": "wall",
              "bounding_box_2d": [0.0, 0.0, 1.0, 1.0], "estimated_extent_m2": 1.0}]
    monkeypatch.setattr(dmg, "get_llm_client", lambda *a, **k: _Fake(items))
    monkeypatch.setattr(dmg, "_find_representative_image", lambda p: None)
    st = CaptureState(capture_path="x", tier="lidar", rooms=["room_00"], room_geometry={"room_00": _room("room_00")})
    d = dmg.damage_node(st)["damage"][0]
    assert "whole_image_fallback" in d.notes
    assert d.extent_metric.method == "vlm_estimate"
    assert (d.extent_metric.hi - d.extent_metric.lo) / d.extent_metric.value >= 0.9


def test_pixel_bbox_is_rejected(monkeypatch):
    items = [{"damage_class": "crack", "severity": "minor", "surface": "ceiling",
              "bounding_box_2d": [0, 0, 1000, 800], "estimated_extent_m2": 0.4}]
    monkeypatch.setattr(dmg, "get_llm_client", lambda *a, **k: _Fake(items))
    monkeypatch.setattr(dmg, "_find_representative_image", lambda p: None)
    st = CaptureState(capture_path="x", tier="lidar", rooms=["room_00"], room_geometry={"room_00": _room("room_00")})
    assert dmg.damage_node(st)["damage"][0].bounding_box_2d is None

"""Unit tests for deterministic concealed damage rule engine."""

from areamap.state import CaptureState, DamageRegion, Interval
from areamap.nodes.concealed import concealed_node

def test_concealed_rule_trigger():
    # Ceiling water stain should trigger RULE_CEIL_WET_01
    iv = Interval(value=0.5, lo=0.45, hi=0.55, method="conformal", tier="lidar")
    dmg = DamageRegion(
        damage_id="dmg_water_01",
        surface_id="ceiling",
        damage_class="water_stain",
        extent_metric=iv
    )
    state = CaptureState(
        capture_path="test",
        tier="lidar",
        damage=[dmg]
    )

    updates = concealed_node(state)
    flags = updates["concealed_flags"]
    assert len(flags) >= 1
    assert any(f.rule_id == "RULE_CEIL_WET_01" for f in flags)
    assert "RULE_CEIL_WET_01" in flags[0].flag_id

def test_concealed_rule_non_trigger():
    # Minor impact on wall should not trigger ceiling wet-area rule
    iv = Interval(value=0.1, lo=0.08, hi=0.12, method="conformal", tier="lidar")
    dmg = DamageRegion(
        damage_id="dmg_impact_01",
        surface_id="wall_w1",
        damage_class="impact",
        extent_metric=iv
    )
    state = CaptureState(
        capture_path="test",
        tier="lidar",
        damage=[dmg]
    )

    updates = concealed_node(state)
    flags = updates["concealed_flags"]
    # No matching rule for impact on wall without water
    assert not any(f.rule_id == "RULE_CEIL_WET_01" for f in flags)

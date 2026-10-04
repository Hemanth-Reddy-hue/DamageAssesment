"""Node M11: Deterministic Concealed Damage Rule Engine."""

import time
from pathlib import Path
from typing import Any
import yaml
from areamap.state import CaptureState, ConcealedFlag

def load_rules() -> list[dict[str, Any]]:
    rules_file = Path(__file__).resolve().parent.parent / "rules" / "concealed_rules.yaml"
    if not rules_file.exists():
        return []
    with open(rules_file, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("rules", [])

def concealed_node(state: CaptureState) -> dict[str, Any]:
    """Evaluate deterministic concealed damage rules against detected damage findings."""
    t0 = time.time()
    rules = load_rules()
    flags: list[ConcealedFlag] = []

    for dmg in state.damage:
        for rule in rules:
            conds = rule.get("conditions", {})
            req_surface = conds.get("surface_type")
            req_class = conds.get("damage_class")

            # Check matching conditions
            surface_match = (req_surface is None) or (req_surface in dmg.surface_id.lower())
            class_match = (req_class is None) or (req_class == dmg.damage_class)

            if surface_match and class_match:
                flags.append(
                    ConcealedFlag(
                        flag_id=f"flag_{rule['id']}_{dmg.damage_id}",
                        rule_id=rule["id"],
                        description=rule["description"],
                        trigger_evidence=rule.get("trigger_evidence", []) + [f"Triggered by {dmg.damage_id} ({dmg.damage_class})"],
                        recommended_action=rule.get("recommended_action", "Physical cavity inspection")
                    )
                )

    return {
        "concealed_flags": flags,
        "timings": {**state.timings, "concealed": round(time.time() - t0, 4)}
    }

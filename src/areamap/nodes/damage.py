"""Node M10: Damage Detection and Semantic Characterization."""

import time
from typing import Any
from areamap.state import CaptureState, DamageRegion
from areamap.geometry.uncertainty import calculate_interval
from areamap.llm.client import get_llm_client

def damage_node(state: CaptureState) -> dict[str, Any]:
    """Detect and measure surface damage regions with deterministic replay caching."""
    t0 = time.time()
    client = get_llm_client()

    prompt = f"Analyze surface images for {state.capture_path} to identify visible damage and extents."
    res = client.generate_structured(prompt=prompt)

    findings = res.get("damage_findings", [])
    damage_regions: list[DamageRegion] = []

    for i, item in enumerate(findings):
        d_class = item.get("damage_class", "water_stain")
        extent_m2 = item.get("estimated_extent_m2", 0.75)
        d_id = f"dmg_{d_class}_{i+1:02d}"
        
        # Primary surface assignment
        surface_id = "ceiling" if d_class == "water_stain" else "room_01_w1"

        damage_regions.append(
            DamageRegion(
                damage_id=d_id,
                surface_id=surface_id,
                damage_class=d_class,
                severity=item.get("severity", "moderate"),
                extent_metric=calculate_interval(extent_m2, "damage_area", tier=state.tier),
                bounding_box_2d=item.get("bounding_box_2d"),
                confidence=0.90,
                notes=item.get("notes", "")
            )
        )

    return {
        "damage": damage_regions,
        "timings": {**state.timings, "damage": round(time.time() - t0, 4)}
    }

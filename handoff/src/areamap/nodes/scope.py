"""Node M12: Repair Scope Line Item Generation."""

import time
from pathlib import Path
from typing import Any
import yaml
from areamap.state import CaptureState, ScopeItem, Interval
from areamap.geometry.uncertainty import calculate_interval

def load_catalog() -> dict[str, Any]:
    catalog_file = Path(__file__).resolve().parent.parent / "catalog" / "scope_items.yaml"
    if not catalog_file.exists():
        return {}
    with open(catalog_file, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("catalog", {})

def scope_node(state: CaptureState) -> dict[str, Any]:
    """Generate structured repair line items strictly keyed to affected surfaces and measured extents."""
    t0 = time.time()
    catalog = load_catalog()
    scope_items: list[ScopeItem] = []

    for dmg in state.damage:
        cat_entry = catalog.get(dmg.damage_class, {})
        items = cat_entry.get("items", [])

        for idx, item in enumerate(items):
            multiplier = item.get("multiplier", 1.0)
            nom_qty = dmg.extent_metric.value * multiplier
            lo_qty = dmg.extent_metric.lo * multiplier
            hi_qty = dmg.extent_metric.hi * multiplier

            qty_interval = Interval(
                value=round(nom_qty, 3),
                lo=round(lo_qty, 3),
                hi=round(hi_qty, 3),
                confidence_level=dmg.extent_metric.confidence_level,
                method="propagated",
                tier=state.tier
            )

            scope_items.append(
                ScopeItem(
                    item_id=f"scope_{dmg.damage_id}_{idx+1:02d}",
                    surface_id=dmg.surface_id,
                    damage_id=dmg.damage_id,
                    item_description=item["description"],
                    unit=item["unit"],
                    quantity=qty_interval,
                    unit_cost_est=item.get("base_cost"),
                    rationale=f"Derived from {dmg.damage_id} ({dmg.damage_class}) extent with {multiplier:.2f}x trade allowance."
                )
            )

    return {
        "scope_items": scope_items,
        "timings": {**state.timings, "scope": round(time.time() - t0, 4)}
    }

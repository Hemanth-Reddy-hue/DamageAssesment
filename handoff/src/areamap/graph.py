"""LangGraph orchestration graph for AreaMap pipeline."""

from typing import Any
from areamap.state import CaptureState
from areamap.nodes import (
    ingest_node,
    geometry_node,
    openings_node,
    stitch_node,
    calibrate_node,
    damage_node,
    concealed_node,
    scope_node,
    qa_critic_node,
    export_node,
)

def build_areamap_graph():
    """Build LangGraph StateGraph instance or return callable runner."""
    try:
        from langgraph.graph import StateGraph, END
        builder = StateGraph(CaptureState)

        # Register nodes
        builder.add_node("ingest", ingest_node)
        builder.add_node("geometry", geometry_node)
        builder.add_node("openings", openings_node)
        builder.add_node("stitch", stitch_node)
        builder.add_node("calibrate", calibrate_node)
        builder.add_node("damage", damage_node)
        builder.add_node("concealed", concealed_node)
        builder.add_node("scope", scope_node)
        builder.add_node("qa_critic", qa_critic_node)
        builder.add_node("export", export_node)

        # Set entry point
        builder.set_entry_point("ingest")

        # Linear & parallel edges
        builder.add_edge("ingest", "geometry")
        builder.add_edge("geometry", "openings")
        builder.add_edge("openings", "stitch")
        builder.add_edge("stitch", "calibrate")
        builder.add_edge("calibrate", "damage")
        builder.add_edge("damage", "concealed")
        builder.add_edge("concealed", "scope")
        builder.add_edge("scope", "qa_critic")
        builder.add_edge("qa_critic", "export")
        builder.add_edge("export", END)

        return builder.compile()
    except ImportError:
        # Graceful deterministic fallback runner if langgraph is not yet installed
        return SimpleGraphRunner()

class SimpleGraphRunner:
    """In-process deterministic linear graph runner mirroring LangGraph pipeline."""
    def invoke(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        state = CaptureState(**state_dict) if isinstance(state_dict, dict) else state_dict

        # Sequential node execution
        for node_fn in [
            ingest_node,
            geometry_node,
            openings_node,
            stitch_node,
            calibrate_node,
            damage_node,
            concealed_node,
            scope_node,
            qa_critic_node,
            export_node,
        ]:
            updates = node_fn(state)
            if updates and isinstance(updates, dict):
                for k, v in updates.items():
                    setattr(state, k, v)

        return state.model_dump()

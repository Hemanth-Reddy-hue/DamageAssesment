"""Pydantic state models and data contract for AreaMap."""

from __future__ import annotations
from typing import Any, Literal
from pydantic import BaseModel, Field

class Interval(BaseModel):
    """Calibrated confidence interval for a metric measurement."""
    value: float = Field(..., description="Estimated nominal value in metric units (meters or m^2)")
    lo: float = Field(..., description="Lower bound of confidence interval")
    hi: float = Field(..., description="Upper bound of confidence interval")
    confidence_level: float = Field(default=0.90, description="Nominal confidence level, e.g. 0.90 for 90%")
    method: str = Field(default="conformal", description="Interval estimation method")
    tier: str = Field(..., description="Input tier used: photo, video, or lidar")

class WallSegment(BaseModel):
    """A planar wall segment in room-local or plan coordinates."""
    wall_id: str
    start: list[float] = Field(..., min_length=2, max_length=3, description="Wall start point [x, y] or [x, y, z]")
    end: list[float] = Field(..., min_length=2, max_length=3, description="Wall end point [x, y] or [x, y, z]")
    length: Interval = Field(..., description="Wall length with confidence interval")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

class Opening(BaseModel):
    """An architectural opening: door, window, or passageway."""
    opening_id: str
    wall_id: str
    type: Literal["door", "window", "passageway"] = "door"
    width: Interval = Field(..., description="Width with calibrated interval")
    height: Interval = Field(..., description="Height with calibrated interval")
    sill_height: Interval | None = Field(default=None, description="Sill height from floor")
    position: list[float] = Field(default_factory=list, description="3D center coordinate [x, y, z]")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

class RoomGeometry(BaseModel):
    """Complete geometrical reconstruction of a single room."""
    room_id: str
    room_name: str
    ceiling_height: Interval = Field(..., description="Room ceiling height with interval")
    floor_area: Interval = Field(..., description="Floor area in m^2 with interval")
    walls: list[WallSegment] = Field(default_factory=list)
    floor_polygon: list[list[float]] = Field(default_factory=list, description="Ordered boundary vertices [[x, y], ...]")
    openings: list[Opening] = Field(default_factory=list)
    is_rectilinear: bool = Field(default=True, description="Whether Manhattan assumption holds")
    transform_to_plan: list[list[float]] | None = Field(default=None, description="4x4 transform to global plan coordinates")

class DamageRegion(BaseModel):
    """Identified surface damage region."""
    damage_id: str
    surface_id: str = Field(..., description="ID of wall, floor, or ceiling surface")
    damage_class: str = Field(..., description="e.g. water_stain, crack, mold, impact")
    severity: Literal["minor", "moderate", "severe"] = "minor"
    extent_metric: Interval = Field(..., description="Metric extent (area m^2 or crack length m) with interval")
    bounding_box_2d: list[float] | None = Field(default=None, description="2D bbox [ymin, xmin, ymax, xmax] in source image")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    notes: str = ""

class ConcealedFlag(BaseModel):
    """Deterministic concealed-damage flag."""
    flag_id: str
    rule_id: str = Field(..., description="Rule ID that fired (e.g. RULE_CEIL_WET_01)")
    description: str = Field(..., description="Explanation of suspected concealed damage")
    trigger_evidence: list[str] = Field(default_factory=list, description="List of observed evidence items that triggered rule")
    recommended_action: str = Field(default="Inspect cavity / moisture meter probe")

class ScopeItem(BaseModel):
    """Repair scope line item keyed to a surface."""
    item_id: str
    surface_id: str
    damage_id: str | None = None
    item_description: str
    unit: str = Field(..., description="Measurement unit, e.g. m^2, m, ea")
    quantity: Interval = Field(..., description="Quantity with interval propagated from damage extent")
    unit_cost_est: float | None = None
    rationale: str = ""

class AdjacencyConnection(BaseModel):
    """Doorway connection linking two rooms."""
    from_room: str
    to_room: str
    opening_id: str
    confidence: float = 1.0

class StitchedPlan(BaseModel):
    """Whole-property stitched floor plan."""
    rooms: list[str] = Field(default_factory=list)
    connections: list[AdjacencyConnection] = Field(default_factory=list)
    total_footprint_area: Interval = Field(..., description="Total footprint area with interval")
    footprint_polygon: list[list[float]] = Field(default_factory=list)
    drift_correction_applied: bool = True

class QAReport(BaseModel):
    """Critic and validation report."""
    passed: bool = True
    checks_run: list[str] = Field(default_factory=list)
    failed_checks: list[str] = Field(default_factory=list)
    adjustments_made: list[str] = Field(default_factory=list)
    overall_confidence: float = 1.0

class CaptureState(BaseModel):
    """Complete shared state across all nodes in AreaMap LangGraph."""
    capture_path: str = Field(..., description="Path to capture data")
    tier: Literal["photo", "video", "lidar"] = "lidar"
    device_meta: dict[str, Any] = Field(default_factory=dict)
    rooms: list[str] = Field(default_factory=list)
    point_clouds: dict[str, str] = Field(default_factory=dict, description="room_id -> point cloud artifact path")
    room_geometry: dict[str, RoomGeometry] = Field(default_factory=dict)
    openings: dict[str, list[Opening]] = Field(default_factory=dict)
    doorway_transitions: list[dict[str, Any]] = Field(default_factory=list, description="Transition edges between rooms")
    stitched_plan: StitchedPlan | None = None
    intervals: dict[str, Interval] = Field(default_factory=dict)
    damage: list[DamageRegion] = Field(default_factory=list)
    concealed_flags: list[ConcealedFlag] = Field(default_factory=list)
    scope_items: list[ScopeItem] = Field(default_factory=list)
    qa_report: QAReport | None = None
    timings: dict[str, float] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    output_dir: str = "out"

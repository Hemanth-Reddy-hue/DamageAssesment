"""Typed data structures for the Video Tier reconstruction pipeline."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


@dataclass
class ScaleInfo:
    factor: float | None = None                 # SfM units -> metres
    method: str = "none"                        # "depth_fused" | "reference" | "door" | "prior" | "none"
    confidence: float = 0.0                     # 0..1
    cues: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # per-cue estimate, n_samples, spread
    relative_uncertainty: float = 0.25          # e.g. 0.08 for ±8 %


@dataclass
class RegistrationInfo:
    n_frames: int = 0
    n_registered: int = 0
    ratio: float = 0.0
    n_models: int = 0
    model_sizes: List[int] = field(default_factory=list)
    mean_reproj_error_px: float = 0.0
    strategy_used: str = "none"                 # "exhaustive_default" | "exhaustive_relaxed" | ...


@dataclass
class VideoRoom:
    room_id: str                                # "room_00", "room_01", ...
    points: np.ndarray                          # (N,3) metres, z-up, shared world frame
    camera_positions: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    time_range_s: Tuple[float, float] = (0.0, 0.0)
    room_type: Optional[str] = None             # optional name, from LLM/classifier if available
    room_type_source: str = "none"              # "local_model" | "llm" | "override" | "none"


@dataclass
class VideoReconstruction:
    status: str                                 # "ok" | "degraded" | "failed"
    rooms: List[VideoRoom]
    transitions: List[Dict[str, Any]] = field(default_factory=list)  # doorway candidates between rooms
    scale: ScaleInfo = field(default_factory=ScaleInfo)
    registration: RegistrationInfo = field(default_factory=RegistrationInfo)
    intrinsics: Dict[str, Any] = field(default_factory=dict)
    video_meta: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    timings: Dict[str, float] = field(default_factory=dict)
    llm: Dict[str, Any] = field(default_factory=dict)  # {"provider": ..., "calls": n, "disabled_reason": ...}
    provenance: str = "measured"

    def to_metadata_dict(self) -> Dict[str, Any]:
        """Convert to metadata dictionary for state integration and legacy callers."""
        return {
            "tier": "video",
            "status": self.status,
            "source": self.video_meta.get("source", ""),
            "video_metadata": self.video_meta,
            "intrinsics": self.intrinsics,
            "scale": {
                "factor": self.scale.factor,
                "method": self.scale.method,
                "confidence": self.scale.confidence,
                "relative_uncertainty": self.scale.relative_uncertainty,
                "cues": self.scale.cues,
            },
            "registration": {
                "n_frames": self.registration.n_frames,
                "n_registered": self.registration.n_registered,
                "ratio": self.registration.ratio,
                "n_models": self.registration.n_models,
                "model_sizes": self.registration.model_sizes,
                "mean_reproj_error_px": self.registration.mean_reproj_error_px,
                "strategy_used": self.registration.strategy_used,
            },
            "warnings": self.warnings,
            "timings": self.timings,
            "llm": self.llm,
            "doorway_transitions": self.transitions,
            "provenance": self.provenance,
            "shared_frame": True,
        }

    def __iter__(self):
        """Allow backwards-compatible tuple unpacking: `pts, meta = ingest_video_capture(path)`."""
        valid_pts = [r.points for r in self.rooms if len(r.points) > 0]
        if len(self.rooms) == 1:
            pts_out = self.rooms[0].points
        elif valid_pts:
            pts_out = np.vstack(valid_pts)
        else:
            pts_out = np.zeros((0, 3), dtype=np.float64)
        yield pts_out
        meta = self.to_metadata_dict()
        meta["rooms"] = [
            {
                "room_id": r.room_id,
                "room_type": r.room_type,
                "room_type_source": r.room_type_source,
                "points": r.points,
                "camera_positions": r.camera_positions,
                "time_range_s": r.time_range_s,
            }
            for r in self.rooms
        ]
        yield meta

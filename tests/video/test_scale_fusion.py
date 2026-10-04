"""Test metric scale fusion logic: agreeing cues, disagreeing cues, and reference override."""

from typing import Tuple
import numpy as np
import pytest

from areamap.config import settings
from areamap.tiers.video_types import ScaleInfo


def fuse_scale_cues(
    reference_m: float | None = None,
    depth_ratios: list[float] | None = None,
    camera_height_est: float | None = None,
    ceiling_prior: float = 2.60,
    sfm_height: float = 1.0,
) -> Tuple[ScaleInfo, list[str]]:
    """Isolated scale fusion logic mirroring video_sfm.py."""
    warnings: list[str] = []
    cues_dict = {}
    chosen_scale = None
    scale_method = "none"
    scale_confidence = 0.0
    relative_uncertainty = 0.25

    # 1. Reference
    if reference_m is not None and reference_m > 0.1:
        chosen_scale = reference_m / sfm_height
        scale_method = "reference"
        scale_confidence = 0.95
        relative_uncertainty = 0.04
        cues_dict["reference"] = {"value": chosen_scale, "confidence": 0.95}

    # 2. Depth fused
    depth_val = None
    if depth_ratios and not chosen_scale:
        med_depth = float(np.median(depth_ratios))
        spread = float(np.std(depth_ratios) / (med_depth + 1e-6))
        cues_dict["depth_fused"] = {"value": med_depth, "samples": len(depth_ratios), "spread": spread}
        if len(depth_ratios) >= 4 and spread < 0.20:
            chosen_scale = med_depth
            scale_method = "depth_fused"
            scale_confidence = max(0.60, min(0.85, 0.90 - spread))
            relative_uncertainty = max(0.08, spread)
            depth_val = med_depth

    # 3. Camera height prior
    if camera_height_est and camera_height_est > 0.1:
        cam_scale = 1.45 / camera_height_est
        cues_dict["camera_height"] = {"value": cam_scale, "confidence": 0.30}

        # Check disagreement
        if depth_val is not None:
            if abs(depth_val - cam_scale) / depth_val > 0.20:
                warnings.append("scale_cues_disagree")
                relative_uncertainty = max(relative_uncertainty, 0.20)

        if not chosen_scale:
            chosen_scale = cam_scale
            scale_method = "prior"
            scale_confidence = 0.30
            relative_uncertainty = 0.25

    if not chosen_scale:
        chosen_scale = 1.0
        scale_method = "prior"

    info = ScaleInfo(
        factor=round(chosen_scale, 4),
        method=scale_method,
        confidence=round(scale_confidence, 2),
        cues=cues_dict,
        relative_uncertainty=round(relative_uncertainty, 3),
    )
    return info, warnings


def test_agreeing_cues_fused_value():
    """When depth model and camera height agree, depth_fused is chosen with high confidence."""
    # 6 depth samples around 1.80
    depth_samples = [1.78, 1.82, 1.80, 1.79, 1.81, 1.83]
    # cam height in sfm units: 1.45 / 1.80 ~ 0.805
    info, warnings = fuse_scale_cues(depth_ratios=depth_samples, camera_height_est=0.805)

    assert info.method == "depth_fused"
    assert np.isclose(info.factor, 1.80, atol=0.05)
    assert info.confidence >= 0.70
    assert "scale_cues_disagree" not in warnings


def test_disagreeing_cues_never_overwritten_by_prior():
    """When depth model (1.80) and camera height (1.20) disagree, primary depth cue is KEPT, not overwritten."""
    depth_samples = [1.80, 1.82, 1.79, 1.81, 1.80]
    # cam height in sfm units giving scale 1.20: 1.45 / 1.208 = 1.20 (differs by 33%)
    info, warnings = fuse_scale_cues(depth_ratios=depth_samples, camera_height_est=1.208)

    assert info.method == "depth_fused"
    # Never overwritten with 2.6 / sfm_height!
    assert np.isclose(info.factor, 1.80, atol=0.05)
    assert "scale_cues_disagree" in warnings
    assert info.relative_uncertainty >= 0.20


def test_reference_overrides_all():
    """User-supplied reference height overrides depth and prior cues with 0.95 confidence."""
    depth_samples = [1.50, 1.55, 1.48, 1.52]
    info, warnings = fuse_scale_cues(reference_m=2.70, depth_ratios=depth_samples, sfm_height=1.0)

    assert info.method == "reference"
    assert info.factor == 2.70
    assert info.confidence >= 0.90
    assert info.relative_uncertainty <= 0.05

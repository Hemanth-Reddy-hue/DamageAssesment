"""Node M10: Damage Detection and Semantic Characterization."""

import time
import logging
from pathlib import Path
from typing import Any
from areamap.state import CaptureState, DamageRegion, Interval
from areamap.geometry.uncertainty import calculate_interval
from areamap.llm.client import get_llm_client

logger = logging.getLogger(__name__)

# Image extensions to search for representative keyframes
_IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

def _find_representative_image(capture_path: str) -> Path | None:
    """Find the best representative image from the capture to send to the VLM.
    For video files/folders, extracts a middle frame and saves it to cache.
    """
    import tempfile
    p = Path(capture_path)
    candidates: list[Path] = []

    if p.is_file() and p.suffix.lower() in _IMAGE_EXTS:
        return p

    # Check for video files (file or inside folder)
    video_paths: list[Path] = []
    if p.is_file() and p.suffix.lower() in {".mp4", ".mov", ".avi"}:
        video_paths = [p]
    elif p.is_dir():
        video_paths = list(p.glob("*.mp4")) + list(p.glob("*.mov")) + list(p.glob("*.avi"))

    if video_paths:
        try:
            import cv2
            video_file = str(video_paths[0])
            cap = cv2.VideoCapture(video_file)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            # Seek to 1/3 of the video (representative, not black-frame start)
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total_frames // 3))
            ret, frame = cap.read()
            cap.release()
            if ret and frame is not None:
                cache_dir = Path("data/cache")
                cache_dir.mkdir(parents=True, exist_ok=True)
                frame_path = cache_dir / "damage_keyframe.jpg"
                cv2.imwrite(str(frame_path), frame)
                logger.info(f"damage_node: Extracted video frame {total_frames//3}/{total_frames} to {frame_path}")
                return frame_path
        except Exception as exc:
            logger.warning(f"damage_node: Failed to extract video frame: {exc}")

    if p.is_dir():
        # Prefer images from an 'rgb' subdir (LiDAR captures)
        rgb_dir = p / "rgb"
        if rgb_dir.exists():
            candidates = sorted(rgb_dir.glob("*.jpg")) + sorted(rgb_dir.glob("*.png"))
        if not candidates:
            candidates = sorted(p.rglob("*.jpg")) + sorted(p.rglob("*.jpeg")) + sorted(p.rglob("*.png"))
        
        if candidates:
            # Pick the frame from the middle of the sequence (most representative)
            return candidates[len(candidates) // 2]

    return None


def _resolve_surface_id(surface: str, state: CaptureState) -> tuple[str | None, str]:
    """Return (surface_id, note). Wall damage cannot be tied to a specific wall from one image."""
    s = (surface or "").lower()
    if "ceil" in s:
        return "ceiling", ""
    if "floor" in s:
        return "floor", ""
    geom = None
    for rid in state.rooms:
        geom = state.room_geometry.get(rid)
        if geom:
            break
    if geom is None and state.room_geometry:
        geom = next(iter(state.room_geometry.values()))
    if geom and geom.walls:
        return geom.walls[0].wall_id, (
            f"wall not localized from a single image; assigned to {geom.walls[0].wall_id}")
    return None, "no wall geometry available to bind damage to"


def _clean_bbox(bb) -> tuple[list[float] | None, bool, str]:
    """Return (bbox, whole_image_fallback, note). bbox must be [ymin,xmin,ymax,xmax] in [0,1]."""
    try:
        v = [float(x) for x in bb]
    except (TypeError, ValueError):
        return None, False, "bbox missing"
    if len(v) != 4 or any(x < 0.0 or x > 1.0 for x in v):
        return None, False, "bbox invalid (not normalized [0,1])"
    ymin, xmin, ymax, xmax = v
    if ymax <= ymin or xmax <= xmin:
        return None, False, "bbox degenerate"
    if (ymax - ymin) * (xmax - xmin) > 0.95:
        return v, True, "whole_image_fallback: bbox covers >95% of image"
    return v, False, ""


def damage_node(state: CaptureState) -> dict[str, Any]:
    """Detect and measure surface damage regions using the VLM on real image data."""
    t0 = time.time()
    client = get_llm_client()

    # --- FIX #4: Find a real representative image to send to the VLM ---
    rep_image = _find_representative_image(state.capture_path)
    if rep_image:
        logger.info(f"damage_node: Sending image to VLM: {rep_image}")
    else:
        logger.warning(f"damage_node: No image found at {state.capture_path}. VLM will work text-only.")

    schema = {
        "type": "object",
        "properties": {
            "damage_findings": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "damage_class": {"type": "string", "enum": ["water_stain", "mold", "crack", "spalling", "efflorescence", "none"]},
                        "severity": {"type": "string", "enum": ["none", "minor", "moderate", "severe"]},
                        "bounding_box_2d": {"type": "array", "items": {"type": "number", "minimum": 0, "maximum": 1},
                                            "minItems": 4, "maxItems": 4,
                                            "description": "[ymin, xmin, ymax, xmax] normalized to 0..1"},
                        "estimated_extent_m2": {"type": "number"},
                        "surface": {"type": "string", "description": "e.g. ceiling, wall, floor"},
                        "notes": {"type": "string"}
                    },
                    "required": ["damage_class", "severity"]
                }
            }
        }
    }

    prompt = (
        "You are a professional property damage assessor. "
        "Inspect the image for visible surface damage: water stains, mold, cracks, spalling, efflorescence. "
        "If there is no damage, return an empty damage_findings array. Do NOT fabricate findings. "
        "For each real finding give: damage_class, severity (minor|moderate|severe), "
        "surface (ceiling|wall|floor), and bounding_box_2d as [ymin, xmin, ymax, xmax] with every value "
        "normalized to the range 0..1 relative to the image (never pixels, never the whole image unless "
        "the damage truly fills it). Also give estimated_extent_m2 only if you can justify it; otherwise omit it."
    )

    res = client.generate_structured(prompt=prompt, image_path=rep_image, schema=schema)

    if hasattr(res, "ok"):
        if not res.ok or not res.data:
            state.warnings.append(f"Damage inspection skipped: LLM unavailable ({res.status}: {res.detail or ''})")
            findings = []
        else:
            findings = res.data.get("damage_findings", [])
    elif isinstance(res, dict):
        findings = res.get("damage_findings", [])
    else:
        findings = []
    damage_regions: list[DamageRegion] = []

    for i, item in enumerate(findings):
        d_class = item.get("damage_class", "water_stain")
        if d_class == "none":
            continue
        severity = item.get("severity")
        if severity not in ("minor", "moderate", "severe"):
            severity = "minor"

        surface = item.get("surface") or ("ceiling" if d_class == "water_stain" else "wall")
        surface_id, sid_note = _resolve_surface_id(surface, state)
        if surface_id is None:
            state.warnings.append(f"Damage {d_class} #{i+1} dropped: {sid_note}")
            continue

        bbox, whole, bbox_note = _clean_bbox(item.get("bounding_box_2d"))

        # The extent is a VLM estimate, not a measurement: wide band, labeled as such.
        raw_extent = item.get("estimated_extent_m2")
        notes = [n for n in (item.get("notes", ""), sid_note, bbox_note) if n]
        confidence = 0.6
        if isinstance(raw_extent, (int, float)) and raw_extent > 0:
            v = float(raw_extent)
            half = 0.5 if not whole else 1.0
            lo, hi = max(0.0, v * (1 - min(half, 0.9))), v * (1 + half)
        else:
            v, lo, hi = 0.5, 0.1, 2.0
            confidence = 0.3
            notes.append("extent not provided by VLM; wide placeholder band, not a measurement")
        if whole:
            confidence = min(confidence, 0.3)

        damage_regions.append(DamageRegion(
            damage_id=f"dmg_{d_class}_{i+1:02d}",
            surface_id=surface_id,
            damage_class=d_class,
            severity=severity,
            extent_metric=Interval(value=round(v, 4), lo=round(lo, 4), hi=round(hi, 4),
                                   confidence_level=0.90, method="vlm_estimate", tier=state.tier or "lidar"),
            bounding_box_2d=bbox,
            confidence=confidence,
            notes="; ".join(notes),
        ))

    return {
        "damage": damage_regions,
        "timings": {**state.timings, "damage": round(time.time() - t0, 4)}
    }

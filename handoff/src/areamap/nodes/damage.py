"""Node M10: Damage Detection and Semantic Characterization."""

import time
import logging
from pathlib import Path
from typing import Any
from areamap.state import CaptureState, DamageRegion
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
                        "bounding_box_2d": {"type": "array", "items": {"type": "number"}},
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
        "Carefully inspect the provided image for any visible surface damage including "
        "water stains, mold, cracks, spalling, or structural issues. "
        "If the image shows no damage, return an empty damage_findings array. "
        "Be accurate — do NOT fabricate findings. "
        "For each real finding, estimate its area in m^2 and the affected surface (ceiling/wall/floor)."
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
        extent_m2 = item.get("estimated_extent_m2", 0.75)
        d_id = f"dmg_{d_class}_{i+1:02d}"
        
        surface = item.get("surface", "ceiling" if d_class == "water_stain" else "wall")
        # Map to room wall ID if applicable
        if "wall" in surface.lower():
            surface_id = "room_01_w1"
        else:
            surface_id = surface

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

"""Local Hugging Face model manager (PLAN2.md).

Loads and runs local models for:
- Zero-shot room classification (OpenAI CLIP / SigLIP)
- Metric depth estimation (DepthEngine)
- Soft-failing for optional models: continues gracefully with warnings if models cannot be loaded.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

from areamap.config import settings

logger = logging.getLogger(__name__)

CANDIDATE_ROOM_LABELS = [
    "living room",
    "bedroom",
    "kitchen",
    "bathroom",
    "hallway",
    "dining room",
    "balcony",
    "storage",
    "office",
]


class ModelManager:
    """Manages local PyTorch / Hugging Face models."""

    def __init__(
        self,
        enabled: bool = True,
        device: Optional[str] = None,
        model_name: Optional[str] = None,
        min_score: Optional[float] = None,
    ):
        self.enabled = enabled
        self.device_str = device or settings.models_device
        if self.device_str == "auto":
            try:
                import torch
                self.device_str = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                self.device_str = "cpu"

        self.model_name = model_name or settings.room_classifier_model
        self.min_score = min_score if min_score is not None else settings.room_classifier_min_score

        self._classifier_pipeline = None
        self._classifier_loaded = False
        self._depth_engine = None
        self.load_warnings: List[str] = []

    def load_room_classifier(self) -> bool:
        """Load zero-shot image classification pipeline (soft-failing)."""
        if not self.enabled:
            return False
        if self._classifier_loaded:
            return self._classifier_pipeline is not None

        self._classifier_loaded = True
        try:
            from transformers import pipeline

            device_arg = 0 if self.device_str == "cuda" else -1
            logger.info("Loading local room classifier (%s) on %s...", self.model_name, self.device_str)
            self._classifier_pipeline = pipeline(
                "zero-shot-image-classification",
                model=self.model_name,
                device=device_arg,
            )
            logger.info("Room classifier loaded successfully.")
            return True
        except Exception as exc:
            msg = f"Optional room classifier ({self.model_name}) could not be loaded: {exc}. Rooms will use numbered names."
            logger.warning(msg)
            self.load_warnings.append(msg)
            self._classifier_pipeline = None
            return False

    def classify_room(
        self,
        images: List[Union[np.ndarray, Image.Image, Path, str]],
        candidate_labels: Optional[List[str]] = None,
    ) -> Tuple[Optional[str], float]:
        """Classify room type from a set of representative images using zero-shot CLIP.

        Returns:
            (room_type_normalized, confidence_score) or (None, score) if below min_score.
        """
        if not self.enabled:
            return None, 0.0

        if not self._classifier_loaded:
            self.load_room_classifier()

        if self._classifier_pipeline is None or not images:
            return None, 0.0

        labels = candidate_labels or CANDIDATE_ROOM_LABELS
        pil_images = []

        for item in images:
            try:
                if isinstance(item, (str, Path)):
                    p = Path(item)
                    if p.exists():
                        im = Image.open(str(p)).convert("RGB")
                        pil_images.append(im)
                elif isinstance(item, Image.Image):
                    pil_images.append(item.convert("RGB"))
                elif isinstance(item, np.ndarray):
                    # OpenCV BGR to PIL RGB
                    if len(item.shape) == 3 and item.shape[2] == 3:
                        im = Image.fromarray(item[:, :, ::-1])
                        pil_images.append(im)
                    elif len(item.shape) == 2:
                        im = Image.fromarray(item).convert("RGB")
                        pil_images.append(im)
            except Exception as exc:
                logger.debug("Failed to convert image for classification: %s", exc)

        if not pil_images:
            return None, 0.0

        try:
            # Score each candidate label across all provided images
            scores_by_label: Dict[str, List[float]] = {lbl: [] for lbl in labels}

            # Batch or per-image prediction
            for img in pil_images:
                res = self._classifier_pipeline(img, candidate_labels=labels)
                for entry in res:
                    lbl = entry.get("label")
                    sc = entry.get("score", 0.0)
                    if lbl in scores_by_label:
                        scores_by_label[lbl].append(sc)

            # Average scores per label
            avg_scores = {lbl: float(np.mean(vals)) if vals else 0.0 for lbl, vals in scores_by_label.items()}
            best_label = max(avg_scores.keys(), key=lambda k: avg_scores[k])
            best_score = avg_scores[best_label]

            if best_score >= self.min_score:
                norm_label = best_label.lower().strip().replace(" ", "_")
                return norm_label, best_score
            else:
                logger.info("Room classification score %0.2f < threshold %0.2f. Leaving as unnamed room.", best_score, self.min_score)
                return None, best_score

        except Exception as exc:
            logger.warning("Zero-shot room classification failed: %s", exc)
            return None, 0.0

    def get_depth_engine(self):
        """Lazy load or return the metric depth engine."""
        if self._depth_engine is None:
            from areamap.geometry.depth_engine import DepthEngine
            self._depth_engine = DepthEngine(device=self.device_str)
        return self._depth_engine


_MODEL_MANAGER: Optional[ModelManager] = None


def get_model_manager(enabled: bool = True) -> ModelManager:
    """Return model manager singleton."""
    global _MODEL_MANAGER
    if _MODEL_MANAGER is None:
        _MODEL_MANAGER = ModelManager(enabled=enabled)
    elif not enabled:
        _MODEL_MANAGER.enabled = False
    return _MODEL_MANAGER

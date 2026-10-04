"""Prefetch and cache local Hugging Face models for offline use (PLAN2.md).

Run with:
    python -m areamap.models.prefetch
"""

import sys
import logging
from areamap.config import settings
from areamap.models.manager import get_model_manager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("prefetch")


def prefetch_models():
    logger.info("=== AreaMap Local Model Prefetch ===")
    logger.info("Room classifier target: %s", settings.room_classifier_model)
    logger.info("Device: %s", settings.models_device)

    # 1. Room classifier
    mgr = get_model_manager(enabled=True)
    ok = mgr.load_room_classifier()
    if ok:
        logger.info("Room classifier successfully cached!")
    else:
        logger.warning("Room classifier prefetch failed or soft-failed.")

    # 2. Depth engine
    logger.info("Initializing DepthEngine...")
    try:
        from areamap.geometry.depth_engine import DepthEngine
        engine = DepthEngine()
        logger.info("DepthEngine ready (backend: %s)", engine.backend)
    except Exception as exc:
        logger.warning("DepthEngine initialization warning: %s", exc)

    logger.info("Prefetch completed.")


if __name__ == "__main__":
    prefetch_models()

"""Deterministic hash-based response replay cache for LLM and VLM outputs."""

import json
import hashlib
from pathlib import Path
from typing import Any
from areamap.config import settings

class LLMCache:
    def __init__(self, cache_dir: Path | str | None = None):
        self.cache_dir = Path(cache_dir or settings.llm_cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _compute_key(self, prompt: str, image_bytes: bytes | None = None, model: str = "") -> str:
        h = hashlib.sha256()
        h.update(prompt.encode("utf-8"))
        h.update(model.encode("utf-8"))
        if image_bytes:
            h.update(image_bytes)
        return h.hexdigest()

    def get(self, prompt: str, image_bytes: bytes | None = None, model: str = "") -> dict[str, Any] | None:
        key = self._compute_key(prompt, image_bytes, model)
        cache_file = self.cache_dir / f"{key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return None
        return None

    def set(self, prompt: str, response: dict[str, Any], image_bytes: bytes | None = None, model: str = "") -> str:
        key = self._compute_key(prompt, image_bytes, model)
        cache_file = self.cache_dir / f"{key}.json"
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(response, f, indent=2)
        return key

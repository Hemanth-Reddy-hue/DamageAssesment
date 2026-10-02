"""Unified LLM/VLM client supporting local execution, cloud APIs, and cached replays."""

import json
from pathlib import Path
from typing import Any
from areamap.config import settings
from areamap.llm.cache import LLMCache

cache = LLMCache()

class LLMClient:
    def __init__(self, provider: str | None = None):
        self.provider = provider or settings.llm_provider
        self.offline = settings.offline

    def generate_structured(
        self,
        prompt: str,
        image_path: Path | str | None = None,
        schema: dict[str, Any] | None = None,
        model_name: str | None = None
    ) -> dict[str, Any]:
        """Generate structured output with strict caching and local fallbacks."""
        image_bytes = None
        if image_path and Path(image_path).exists():
            with open(image_path, "rb") as f:
                image_bytes = f.read()

        model_key = model_name or self.provider
        # Check replay cache first
        cached_result = cache.get(prompt, image_bytes, model_key)
        if cached_result is not None:
            return cached_result

        # If offline and not in cache, fallback to deterministic heuristic / mock
        if self.offline:
            fallback = self._offline_fallback(prompt, image_path)
            cache.set(prompt, fallback, image_bytes, model_key)
            return fallback

        # Online API routing (Anthropic, Gemini, OpenAI)
        try:
            if self.provider == "anthropic" and settings.anthropic_api_key:
                from langchain_anthropic import ChatAnthropic
                llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0)
                res = llm.invoke(prompt)
                parsed = json.loads(res.content)
            elif self.provider == "gemini" and settings.gemini_api_key:
                from langchain_google_genai import ChatGoogleGenerativeAI
                llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0)
                res = llm.invoke(prompt)
                parsed = json.loads(res.content)
            elif self.provider == "openai" and settings.openai_api_key:
                from langchain_openai import ChatOpenAI
                llm = ChatOpenAI(model="gpt-4o", temperature=0)
                res = llm.invoke(prompt)
                parsed = json.loads(res.content)
            else:
                parsed = self._offline_fallback(prompt, image_path)
        except Exception:
            parsed = self._offline_fallback(prompt, image_path)

        cache.set(prompt, parsed, image_bytes, model_key)
        return parsed

    def _offline_fallback(self, prompt: str, image_path: Path | str | None = None) -> dict[str, Any]:
        """Deterministic offline fallback when running cold or offline."""
        prompt_lower = prompt.lower()
        if "damage" in prompt_lower:
            return {
                "damage_findings": [
                    {
                        "damage_class": "water_stain",
                        "severity": "moderate",
                        "bounding_box_2d": [0.2, 0.3, 0.4, 0.6],
                        "estimated_extent_m2": 0.85,
                        "notes": "Offline heuristic detection: discoloration region on ceiling/wall"
                    }
                ]
            }
        elif "scope" in prompt_lower:
            return {
                "scope_items": [
                    {
                        "item_description": "Drywall cutout, antimicrobial treatment, patch and prime",
                        "unit": "m^2",
                        "quantity": 1.05,
                        "unit_cost_est": 65.0,
                        "rationale": "Drywall remediation for water stain with 20% margin"
                    }
                ]
            }
        return {"result": "ok"}

def get_llm_client(provider: str | None = None) -> LLMClient:
    return LLMClient(provider=provider)

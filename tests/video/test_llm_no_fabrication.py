"""Test that LLM client NEVER fabricates data when APIs fail or are unavailable."""

import pytest
from areamap.config import settings
from areamap.llm.client import LLMClient, LLMResult


def test_no_provider_returns_ok_false_without_fabrication():
    """When no API keys are configured and offline=True, client returns ok=False and data=None."""
    settings.offline = True
    settings.llm_enabled = True
    client = LLMClient(provider="none")

    res = client.generate_structured("Identify surface damage")

    assert isinstance(res, LLMResult)
    assert res.ok is False
    assert res.data is None
    assert res.status in ("unavailable", "disabled")


def test_no_fabricated_keys_ever_produced():
    """No simulated damage_findings or default living_room are ever returned by LLMClient."""
    settings.offline = False
    settings.llm_enabled = True
    settings.mistral_api_key = None
    settings.groq_api_key = None
    settings.gemini_api_key = None
    settings.openai_api_key = None

    client = LLMClient()
    for prompt in [
        "Classify the room type",
        "Inspect for water stains and mold",
        "Generate repair scope items",
        "Arbitrary query",
    ]:
        res = client.generate_structured(prompt)
        assert res.ok is False
        assert res.data is None
        # Check that no dictionary with fake keys is returned
        if res.data is not None:
            assert "damage_findings" not in res.data
            assert "scope_items" not in res.data
            assert res.data.get("room_type") != "living_room"

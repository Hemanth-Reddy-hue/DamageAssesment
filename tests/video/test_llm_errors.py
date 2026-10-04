"""Test LLM error classification, circuit breaker, budget, and retries with mock HTTP."""

import time
from unittest.mock import MagicMock, patch
import pytest

from areamap.config import settings
from areamap.llm.client import LLMClient, LLMResult


class MockResponse:
    def __init__(self, status_code: int, text: str, headers: dict | None = None):
        self.status_code = status_code
        self.text = text
        self.headers = headers or {}

    def json(self):
        import json
        return json.loads(self.text)


def test_long_reset_429_single_request_no_sleep():
    """A 429 with a daily quota error (retryDelay: 77553s) must NOT be retried, must trip breaker, and execute in < 1s."""
    client = LLMClient(provider="mistral")
    client.cache = None
    settings.llm_provider_chain = ["mistral"]
    settings.mistral_api_key = "test_key"
    settings.offline = False
    settings.llm_enabled = True

    quota_body = '{"error": {"message": "Resource exhausted", "details": [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "77553s"}]}}'
    mock_post = MagicMock(return_value=MockResponse(429, quota_body))

    t0 = time.time()
    with patch("httpx.Client.post", mock_post):
        res = client.generate_structured("prompt_long_reset_test")
    elapsed = time.time() - t0

    assert mock_post.call_count == 1
    assert elapsed < 5.0, f"Took {elapsed}s, should not sleep on long reset 429"
    assert res.status == "quota"
    assert "mistral" in client.disabled_providers

    # Second call should be blocked by circuit breaker without making an HTTP request
    with patch("httpx.Client.post", mock_post):
        res2 = client.generate_structured("prompt_long_reset_second_call")
    assert mock_post.call_count == 1  # Still 1!


def test_short_reset_429_retries_once():
    """A short rate limit (retryDelay: 1s) is retried once."""
    client = LLMClient(provider="mistral")
    client.cache = None
    settings.llm_provider_chain = ["mistral"]
    settings.mistral_api_key = "test_key"
    settings.offline = False
    settings.llm_enabled = True

    short_body = '{"error": {"message": "Rate limit exceeded", "retryDelay": "1s"}}'
    ok_body = '{"choices": [{"message": {"content": "{\\"room_type\\": \\"bedroom\\"}"}}]}'

    mock_post = MagicMock(side_effect=[
        MockResponse(429, short_body),
        MockResponse(200, ok_body)
    ])

    with patch("time.sleep", return_value=None):
        with patch("httpx.Client.post", mock_post):
            res = client.generate_structured("prompt_short_reset_test")

    assert mock_post.call_count == 2
    assert res.ok is True
    assert res.data == {"room_type": "bedroom"}


def test_server_error_5xx_retries_up_to_two():
    """A 500 error retries up to 2 times (3 calls total) before failing."""
    client = LLMClient(provider="mistral")
    client.cache = None
    settings.llm_provider_chain = ["mistral"]
    settings.mistral_api_key = "test_key"
    settings.offline = False
    settings.llm_enabled = True

    mock_post = MagicMock(return_value=MockResponse(500, "Internal Server Error"))

    with patch("time.sleep", return_value=None):  # Fast forward sleep
        with patch("httpx.Client.post", mock_post):
            res = client.generate_structured("prompt_5xx_test")

    assert mock_post.call_count == 3
    assert res.ok is False
    assert res.status == "unavailable"


def test_client_error_401_no_retry():
    """An auth error 401 must fail immediately with NO retry and open breaker."""
    client = LLMClient(provider="mistral")
    client.cache = None
    settings.llm_provider_chain = ["mistral"]
    settings.mistral_api_key = "bad_key"
    settings.offline = False
    settings.llm_enabled = True

    mock_post = MagicMock(return_value=MockResponse(401, '{"error": "Unauthorized"}'))

    with patch("httpx.Client.post", mock_post):
        res = client.generate_structured("prompt_401_test")

    assert mock_post.call_count == 1
    assert res.ok is False
    assert "mistral" in client.disabled_providers


def test_parse_error_no_extra_api_retry():
    """Invalid JSON response produces exactly 1 request (no extra API retries)."""
    client = LLMClient(provider="mistral")
    client.cache = None
    settings.llm_provider_chain = ["mistral"]
    settings.mistral_api_key = "test_key"
    settings.offline = False
    settings.llm_enabled = True

    non_json_body = '{"choices": [{"message": {"content": "I cannot classify this as JSON."}}]}'
    mock_post = MagicMock(return_value=MockResponse(200, non_json_body))

    with patch("httpx.Client.post", mock_post):
        res = client.generate_structured("prompt_parse_error_test")

    assert mock_post.call_count == 1
    assert res.ok is False
    assert res.status == "parse_error"


def test_provider_chain_fallthrough():
    """When provider 1 fails with quota, provider 2 in chain is attempted."""
    client = LLMClient()
    client.cache = None
    settings.llm_provider_chain = ["mistral", "groq"]
    settings.mistral_api_key = "mistral_key"
    settings.groq_api_key = "groq_key"
    settings.offline = False
    settings.llm_enabled = True

    quota_body = '{"error": {"message": "Resource exhausted", "retryDelay": "77553s"}}'
    success_body = '{"choices": [{"message": {"content": "{\\"result\\": \\"success_groq\\"}"}}]}'

    def mock_side_effect(url, headers=None, json=None, **kwargs):
        if "mistral" in url:
            return MockResponse(429, quota_body)
        elif "groq" in url:
            return MockResponse(200, success_body)
        return MockResponse(404, "Not Found")

    mock_post = MagicMock(side_effect=mock_side_effect)

    # Use unique prompt to prevent cache hits
    unique_prompt = f"prompt_fallthrough_{time.time()}"
    with patch("httpx.Client.post", mock_post):
        res = client.generate_structured(unique_prompt)

    assert res.ok is True
    assert res.provider == "groq"
    assert res.data == {"result": "success_groq"}
    assert "mistral" in client.disabled_providers


def test_budget_exceeded():
    """When max calls per run is reached, returns status='budget' immediately."""
    client = LLMClient(provider="mistral")
    settings.llm_provider_chain = ["mistral"]
    settings.mistral_api_key = "test_key"
    settings.llm_max_calls_per_run = 2
    settings.offline = False
    settings.llm_enabled = True

    client.call_count = 2
    mock_post = MagicMock()

    with patch("httpx.Client.post", mock_post):
        res = client.generate_structured("prompt_budget_test")

    assert mock_post.call_count == 0
    assert res.ok is False
    assert res.status == "budget"

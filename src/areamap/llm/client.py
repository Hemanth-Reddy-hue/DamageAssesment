"""Unified LLM/VLM client supporting local execution, cloud APIs, and cached replays.

Fixes WP1:
- Structured return type: LLMResult (never dict with fabricated data)
- Single OpenAI-compatible client implementation driving Mistral, Groq, Ollama, OpenRouter, Gemini
- Provider chain with automatic failover
- Strict error classification (long 429 quota disables provider immediately without sleep)
- Token-bucket rate limiter per provider (replaces unconditional sleep)
- Circuit breaker and call budget enforcement
- Image downscaling and collage helper
- Local JSON repair (strip markdown, extract {...})
- No fabricated fallback: deletes _offline_fallback
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import httpx
from PIL import Image

from areamap.config import settings
from areamap.llm.cache import LLMCache

logger = logging.getLogger(__name__)


@dataclass
class LLMResult:
    ok: bool
    data: Optional[Dict[str, Any]] = None
    status: str = "unavailable"  # "ok" | "unavailable" | "quota" | "parse_error" | "disabled" | "budget"
    provider: Optional[str] = None
    model: Optional[str] = None
    cached: bool = False
    detail: Optional[str] = None  # short reason, no secrets


class TokenBucket:
    """Token-bucket rate limiter for API requests."""

    def __init__(self, rpm: int):
        self.rpm = max(1, rpm)
        self.capacity = float(self.rpm)
        self.tokens = float(self.rpm)
        self.fill_rate = self.capacity / 60.0  # tokens per second
        self.last_update = time.time()

    def acquire(self):
        now = time.time()
        elapsed = now - self.last_update
        self.last_update = now
        self.tokens = min(self.capacity, self.tokens + elapsed * self.fill_rate)
        if self.tokens < 1.0:
            wait_time = (1.0 - self.tokens) / self.fill_rate
            if wait_time > 0.001:
                time.sleep(min(wait_time, 5.0))
            self.tokens = 0.0
        else:
            self.tokens -= 1.0


def downscale_image_bytes(image_input: Union[Path, str, bytes, Image.Image], max_side: int = 768) -> bytes:
    """Downscale an image so its long side is <= max_side and encode as JPEG bytes."""
    if isinstance(image_input, (str, Path)):
        img = Image.open(str(image_input))
    elif isinstance(image_input, bytes):
        img = Image.open(io.BytesIO(image_input))
    elif isinstance(image_input, Image.Image):
        img = image_input
    else:
        raise ValueError(f"Unsupported image input type: {type(image_input)}")

    if img.mode != "RGB":
        img = img.convert("RGB")

    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / float(max(w, h))
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
        img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)

    out = io.BytesIO()
    img.save(out, format="JPEG", quality=80)
    return out.getvalue()


def create_image_collage(images: List[Union[Path, str, bytes, Image.Image]], grid_size: Tuple[int, int] = (2, 2), max_side: int = 768) -> bytes:
    """Combine up to 4 images into a single collage grid."""
    if not images:
        raise ValueError("No images provided for collage")
    if len(images) == 1:
        return downscale_image_bytes(images[0], max_side=max_side)

    pil_images = []
    for item in images[:grid_size[0] * grid_size[1]]:
        if isinstance(item, (str, Path)):
            im = Image.open(str(item))
        elif isinstance(item, bytes):
            im = Image.open(io.BytesIO(item))
        elif isinstance(item, Image.Image):
            im = item
        else:
            continue
        if im.mode != "RGB":
            im = im.convert("RGB")
        pil_images.append(im)

    if not pil_images:
        raise ValueError("Could not load any valid images for collage")

    cell_w = max_side // grid_size[1]
    cell_h = max_side // grid_size[0]
    collage = Image.new("RGB", (cell_w * grid_size[1], cell_h * grid_size[0]), (0, 0, 0))

    for idx, im in enumerate(pil_images):
        row = idx // grid_size[1]
        col = idx % grid_size[1]
        im_thumb = im.copy()
        im_thumb.thumbnail((cell_w, cell_h), Image.Resampling.BILINEAR)
        # Center in cell
        paste_x = col * cell_w + (cell_w - im_thumb.width) // 2
        paste_y = row * cell_h + (cell_h - im_thumb.height) // 2
        collage.paste(im_thumb, (paste_x, paste_y))

    out = io.BytesIO()
    collage.save(out, format="JPEG", quality=80)
    return out.getvalue()


class LLMClient:
    """Robust, non-fabricating LLM client with provider chain, rate limiting, and circuit breakers."""

    def __init__(self, provider: Optional[str] = None):
        self.primary_provider = provider or settings.llm_provider
        self.cache = LLMCache()
        self.call_count = 0
        self.start_time = time.time()
        self.disabled_providers: Dict[str, str] = {}  # provider -> reason
        self.limiters: Dict[str, TokenBucket] = {
            "mistral": TokenBucket(settings.llm_rpm_mistral),
            "groq": TokenBucket(settings.llm_rpm_groq),
            "gemini": TokenBucket(15),
            "openai": TokenBucket(60),
            "openrouter": TokenBucket(20),
        }
        self.stats: Dict[str, Any] = {
            "calls": 0,
            "ok": 0,
            "cached": 0,
            "failed": 0,
            "disabled_providers": {},
        }

    def _get_provider_config(self, provider: str) -> Optional[Dict[str, Any]]:
        """Return base_url, api_key, and default model for a given provider."""
        p = provider.lower().strip()
        if p == "mistral":
            return {
                "base_url": "https://api.mistral.ai/v1",
                "api_key": settings.mistral_api_key,
                "model": settings.llm_model_mistral,
            }
        elif p == "groq":
            return {
                "base_url": "https://api.groq.com/openai/v1",
                "api_key": settings.groq_api_key,
                "model": settings.llm_model_groq,
            }
        elif p == "ollama":
            return {
                "base_url": settings.ollama_base_url,
                "api_key": None,
                "model": settings.llm_model_ollama,
            }
        elif p == "openrouter":
            return {
                "base_url": "https://openrouter.ai/api/v1",
                "api_key": settings.openrouter_api_key,
                "model": settings.llm_model_openrouter,
            }
        elif p == "gemini":
            return {
                "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
                "api_key": settings.gemini_api_key,
                "model": settings.llm_model_gemini,
            }
        elif p == "openai":
            return {
                "base_url": "https://api.openai.com/v1",
                "api_key": settings.openai_api_key,
                "model": "gpt-4o",
            }
        elif p == "anthropic":
            return {
                "base_url": "https://api.anthropic.com/v1",
                "api_key": settings.anthropic_api_key,
                "model": "claude-3-5-sonnet-20241022",
            }
        return None

    def _extract_retry_delay(self, response_text: str, headers: Any) -> Optional[float]:
        """Extract retry delay seconds from Google RetryInfo or HTTP Retry-After headers."""
        # 1. Retry-After header
        if headers:
            retry_after = headers.get("retry-after")
            if retry_after:
                try:
                    return float(retry_after)
                except ValueError:
                    pass

        # 2. Google / JSON error response payload: retryDelay: "77553s"
        match = re.search(r'retryDelay["\']?\s*:\s*["\']?(\d+)(?:\.\d+)?s?["\']?', response_text, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass
        return None

    def _repair_json(self, text: str) -> Optional[Dict[str, Any]]:
        """Locally repair JSON by stripping fences and isolating the first balanced {...} block."""
        if not text or not isinstance(text, str):
            return None

        clean = text.strip()
        # Strip markdown fences
        if clean.startswith("```"):
            lines = clean.splitlines()
            if lines and lines[-1].strip() == "```":
                clean = "\n".join(lines[1:-1])
            else:
                clean = "\n".join(lines[1:])
        clean = clean.strip()

        # Direct parse attempt
        try:
            val = json.loads(clean)
            if isinstance(val, dict):
                return val
        except Exception:
            pass

        # Find first { and matching }
        start = clean.find("{")
        if start != -1:
            depth = 0
            for idx in range(start, len(clean)):
                if clean[idx] == "{":
                    depth += 1
                elif clean[idx] == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = clean[start : idx + 1]
                        try:
                            val = json.loads(candidate)
                            if isinstance(val, dict):
                                return val
                        except Exception:
                            break
        return None

    def _ensure_ollama_running(self, model: str) -> None:
        """Ensure Ollama server is running and the requested model is pulled."""
        import subprocess
        import time
        import os
        
        try:
            with httpx.Client(timeout=1.0) as client:
                resp = client.get("http://localhost:11434/api/tags")
                running = (resp.status_code == 200)
        except Exception:
            running = False
            
        if not running:
            logger.info("[LLM] Ollama server not responding. Attempting to start 'ollama serve' in background...")
            try:
                creationflags = 0x08000000 if os.name == 'nt' else 0
                subprocess.Popen(
                    ["ollama", "serve"], 
                    stdout=subprocess.DEVNULL, 
                    stderr=subprocess.DEVNULL, 
                    creationflags=creationflags
                )
                
                # Wait up to 10s for the API to come up
                for _ in range(10):
                    time.sleep(1.0)
                    try:
                        with httpx.Client(timeout=1.0) as client:
                            if client.get("http://localhost:11434/api/tags").status_code == 200:
                                running = True
                                logger.info("[LLM] Ollama server started successfully.")
                                break
                    except Exception:
                        pass
                        
                if not running:
                    logger.warning("[LLM] Timed out waiting for Ollama to start. Is it installed?")
                    return
            except Exception as e:
                logger.warning("[LLM] Could not launch Ollama: %s", e)
                return
                
        # Check if the required model is pulled
        try:
            with httpx.Client(timeout=5.0) as client:
                resp = client.get("http://localhost:11434/api/tags")
                if resp.status_code == 200:
                    tags = resp.json().get("models", [])
                    has_model = any(m.get("name") == model or m.get("name") == f"{model}:latest" for m in tags)
                    if not has_model:
                        logger.info("[LLM] Ollama model '%s' not found locally. Pulling via API (this may take a few minutes)...", model)
                        # Use the REST API to pull to avoid WinError 2 if ollama is not in PATH
                        pull_resp = client.post("http://localhost:11434/api/pull", json={"name": model}, timeout=900.0)
                        if pull_resp.status_code == 200:
                            logger.info("[LLM] Successfully pulled model '%s'.", model)
                        else:
                            logger.warning("[LLM] Failed to pull model via API: %s", pull_resp.text)
                            # Fallback just in case
                            import subprocess
                            subprocess.run(["ollama", "pull", model], check=True, shell=True)
        except Exception as e:
            logger.warning("[LLM] Failed to check/pull Ollama model '%s': %s", model, e)

    def _call_openai_compat(
        self,
        base_url: str,
        api_key: Optional[str],
        model: str,
        prompt: str,
        image_bytes: Optional[bytes] = None,
        schema: Optional[Dict[str, Any]] = None,
        timeout_s: float = 30.0,
    ) -> Tuple[Optional[Dict[str, Any]], str, Optional[str]]:
        """Call OpenAI-compatible Chat Completions endpoint.

        Returns: (parsed_data, status, detail)
        """
        endpoint = base_url.rstrip("/") + "/chat/completions"
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        system_instruction = "You are a professional architectural and property inspection assistant. Respond ONLY in valid JSON."
        user_content: List[Dict[str, Any]] = []

        full_prompt = prompt
        if schema:
            full_prompt += f"\n\nRespond ONLY with valid JSON conforming to this schema:\n{json.dumps(schema)}"
        else:
            full_prompt += "\n\nRespond ONLY with valid JSON."

        user_content.append({"type": "text", "text": full_prompt})

        if image_bytes:
            b64_str = base64.b64encode(image_bytes).decode("utf-8")
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64_str}"}
            })

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_content}
            ],
            "temperature": 0.0,
        }

        # Attempt with retries according to error classification
        retries = 2
        for attempt in range(retries + 1):
            try:
                with httpx.Client(timeout=timeout_s) as client:
                    resp = client.post(endpoint, headers=headers, json=payload)

                status_code = resp.status_code
                resp_text = resp.text

                if status_code == 200:
                    resp_json = resp.json()
                    content = ""
                    choices = resp_json.get("choices", [])
                    if choices:
                        msg = choices[0].get("message", {})
                        content = msg.get("content", "")

                    parsed = self._repair_json(content)
                    if parsed is not None:
                        return parsed, "ok", None
                    else:
                        logger.warning("LLM response failed to parse as JSON: %s", content[:200])
                        return None, "parse_error", "Failed to parse JSON response"

                elif status_code == 429:
                    delay = self._extract_retry_delay(resp_text, resp.headers)
                    if delay is not None:
                        is_long_reset = (delay > 60.0)
                    else:
                        is_long_reset = any(w in resp_text.lower() for w in ["quota", "per day", "free_tier", "daily"])

                    if is_long_reset:
                        detail = f"Daily quota exhausted (retryDelay={delay}s)"
                        logger.warning("LLM non-retryable 429 quota: %s", detail)
                        return None, "quota", detail
                    else:
                        # Short reset <= 10s: retry once if attempts remain
                        if attempt < retries:
                            wait = min(delay or 2.0, 10.0)
                            logger.info("LLM 429 short rate-limit. Waiting %0.1fs before retry...", wait)
                            time.sleep(wait)
                            continue
                        return None, "quota", f"Rate limit 429 (retryDelay={delay}s)"

                elif status_code in (400, 401, 403, 404):
                    err_msg = resp_text[:300].replace("\n", " ")
                    logger.warning("LLM client error HTTP %d: %s", status_code, err_msg)
                    return None, "unavailable", f"HTTP {status_code}: {err_msg}"

                elif status_code >= 500:
                    if attempt < retries:
                        wait = 1.0 if attempt == 0 else 3.0
                        logger.info("LLM server error HTTP %d. Retrying in %0.1fs...", status_code, wait)
                        time.sleep(wait)
                        continue
                    return None, "unavailable", f"HTTP {status_code} server error"

                else:
                    return None, "unavailable", f"HTTP {status_code}: {resp_text[:200]}"

            except (httpx.TimeoutException, httpx.ConnectError, httpx.NetworkError) as exc:
                if attempt < retries:
                    wait = 1.0 if attempt == 0 else 3.0
                    logger.info("LLM network error (%s). Retrying in %0.1fs...", type(exc).__name__, wait)
                    time.sleep(wait)
                    continue
                return None, "unavailable", f"Network error: {str(exc)[:200]}"
            except Exception as exc:
                logger.warning("LLM unexpected error: %s", str(exc)[:200])
                return None, "unavailable", f"Unexpected error: {str(exc)[:200]}"

        return None, "unavailable", "Max retries exceeded"

    def generate_structured(
        self,
        prompt: str,
        image_path: Union[Path, str, bytes, Image.Image, None] = None,
        schema: Optional[Dict[str, Any]] = None,
        model_name: Optional[str] = None,
    ) -> LLMResult:
        """Generate structured JSON conforming to schema, with no fabricated fallbacks."""
        # 1. Check if LLM is globally enabled
        if not settings.llm_enabled:
            return LLMResult(ok=False, data=None, status="disabled", detail="LLM globally disabled (--no-llm)")

        # 2. Check run budget and timeout
        now = time.time()
        if self.call_count >= settings.llm_max_calls_per_run:
            logger.warning("LLM call budget exceeded (%d calls)", self.call_count)
            return LLMResult(ok=False, data=None, status="budget", detail="Run call budget exceeded")
        if (now - self.start_time) >= settings.llm_max_total_seconds:
            logger.warning("LLM total runtime budget exceeded")
            return LLMResult(ok=False, data=None, status="budget", detail="Run total time exceeded")

        # 3. Process image downscaling if provided
        image_bytes: Optional[bytes] = None
        if image_path:
            try:
                image_bytes = downscale_image_bytes(image_path, max_side=settings.llm_image_max_side)
            except Exception as exc:
                logger.warning("Failed to prepare image for LLM: %s", exc)

        # 4. Determine provider chain
        chain = []
        if self.primary_provider and self.primary_provider not in ("auto", "local"):
            chain.append(self.primary_provider)
        for p in settings.llm_provider_chain:
            if p not in chain:
                chain.append(p)

        # 5. Check cache before API calls (for all candidates in chain)
        if self.cache:
            for provider in chain:
                cfg = self._get_provider_config(provider)
                if not cfg:
                    continue
                model = model_name or cfg["model"]
                cached_data = self.cache.get(prompt, image_bytes, model=f"{provider}:{model}")
                if cached_data is not None and cached_data.get("_source") != "offline_fallback":
                    self.stats["cached"] += 1
                    return LLMResult(
                        ok=True,
                        data=cached_data,
                        status="ok",
                        provider=provider,
                        model=model,
                        cached=True,
                        detail="Loaded from cache",
                    )

        # If offline is forced, we never contact external APIs
        if settings.offline:
            return LLMResult(
                ok=False,
                data=None,
                status="unavailable",
                detail="OFFLINE=1: external LLM calls prohibited and result not in cache",
            )

        # 6. Try each provider in chain
        last_status = "unavailable"
        last_detail = "No provider available"

        for provider in chain:
            # Check circuit breaker
            if provider in self.disabled_providers:
                logger.debug("Skipping provider %s (breaker open: %s)", provider, self.disabled_providers[provider])
                continue

            cfg = self._get_provider_config(provider)
            if not cfg:
                continue

            api_key = cfg["api_key"]
            # Ollama does not need an API key; others do
            if provider != "ollama" and not api_key:
                logger.debug("Provider %s missing API key; skipping", provider)
                continue

            model = model_name or cfg["model"]

            # Token-bucket rate limiting
            limiter = self.limiters.get(provider)
            if limiter:
                limiter.acquire()

            self.call_count += 1
            self.stats["calls"] += 1
            call_t0 = time.time()

            if provider == "ollama":
                self._ensure_ollama_running(model)

            parsed_data, status, detail = self._call_openai_compat(
                base_url=cfg["base_url"],
                api_key=api_key,
                model=model,
                prompt=prompt,
                image_bytes=image_bytes,
                schema=schema,
                timeout_s=settings.llm_timeout_seconds,
            )
            elapsed_ms = int((time.time() - call_t0) * 1000)

            logger.info("[LLM] provider=%s model=%s status=%s elapsed_ms=%d", provider, model, status, elapsed_ms)

            if status == "ok" and parsed_data is not None:
                self.stats["ok"] += 1
                # Cache successful response
                if self.cache:
                    self.cache.set(prompt, parsed_data, image_bytes, model=f"{provider}:{model}")
                return LLMResult(
                    ok=True,
                    data=parsed_data,
                    status="ok",
                    provider=provider,
                    model=model,
                    cached=False,
                    detail=None,
                )

            # Error handling & circuit breaker tripping
            last_status = status
            last_detail = detail

            if status in ("quota", "unavailable") and detail and ("quota" in detail.lower() or "401" in detail or "403" in detail or "404" in detail):
                # Trip breaker for this run
                self.disabled_providers[provider] = detail
                self.stats["disabled_providers"][provider] = detail
                logger.warning("Circuit breaker opened for provider %s: %s", provider, detail)

            elif status == "parse_error":
                # Do not retry another API call on parse error; return immediately
                self.stats["failed"] += 1
                return LLMResult(
                    ok=False,
                    data=None,
                    status="parse_error",
                    provider=provider,
                    model=model,
                    cached=False,
                    detail=detail,
                )

        self.stats["failed"] += 1
        return LLMResult(
            ok=False,
            data=None,
            status=last_status,
            provider=None,
            model=None,
            cached=False,
            detail=last_detail,
        )


_CLIENT_SINGLETON: Optional[LLMClient] = None


def get_llm_client(provider: Optional[str] = None) -> LLMClient:
    """Return LLM client singleton or new instance."""
    global _CLIENT_SINGLETON
    if _CLIENT_SINGLETON is None or (provider and provider != _CLIENT_SINGLETON.primary_provider):
        _CLIENT_SINGLETON = LLMClient(provider=provider)
    return _CLIENT_SINGLETON

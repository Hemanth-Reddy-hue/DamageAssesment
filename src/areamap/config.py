"""Configuration management for AreaMap."""

import os
from pathlib import Path
from pydantic import BaseModel, Field

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def _get_provider_chain() -> list[str]:
    raw = os.getenv("LLM_PROVIDER_CHAIN", "mistral,groq,ollama")
    if isinstance(raw, str):
        return [p.strip() for p in raw.split(",") if p.strip()]
    return list(raw)


class AreaMapConfig(BaseModel):
    # --- Existing keys & HuggingFace token ---
    hf_token: str | None = Field(default_factory=lambda: os.getenv("HF_TOKEN"))
    llm_provider: str = Field(default_factory=lambda: os.getenv("LLM_PROVIDER", "local"))
    anthropic_api_key: str | None = Field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY"))
    gemini_api_key: str | None = Field(default_factory=lambda: os.getenv("GEMINI_API_KEY"))
    gemini_model_name: str = Field(default_factory=lambda: os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash"))
    openai_api_key: str | None = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))

    # --- Cloud LLM Provider keys & models (WP1 / WP2) ---
    mistral_api_key: str | None = Field(default_factory=lambda: os.getenv("MISTRAL_API_KEY"))
    groq_api_key: str | None = Field(default_factory=lambda: os.getenv("GROQ_API_KEY"))
    openrouter_api_key: str | None = Field(default_factory=lambda: os.getenv("OPENROUTER_API_KEY"))

    llm_enabled: bool = Field(default_factory=lambda: os.getenv("LLM_ENABLED", "true").lower() in ("1", "true", "yes"))
    llm_provider_chain: list[str] = Field(default_factory=_get_provider_chain)
    llm_max_calls_per_run: int = Field(default_factory=lambda: int(os.getenv("LLM_MAX_CALLS_PER_RUN", "12")))
    llm_max_total_seconds: int = Field(default_factory=lambda: int(os.getenv("LLM_MAX_TOTAL_SECONDS", "120")))
    llm_timeout_seconds: int = Field(default_factory=lambda: int(os.getenv("LLM_TIMEOUT_SECONDS", "30")))
    llm_image_max_side: int = Field(default_factory=lambda: int(os.getenv("LLM_IMAGE_MAX_SIDE", "768")))

    llm_model_mistral: str = Field(default_factory=lambda: os.getenv("LLM_MODEL_MISTRAL", "pixtral-12b-2409"))
    llm_model_groq: str = Field(default_factory=lambda: os.getenv("LLM_MODEL_GROQ", "llama-3.2-11b-vision-preview"))
    llm_model_ollama: str = Field(default_factory=lambda: os.getenv("LLM_MODEL_OLLAMA", "llava"))
    llm_model_openrouter: str = Field(default_factory=lambda: os.getenv("LLM_MODEL_OPENROUTER", "google/gemini-2.0-flash-exp:free"))
    llm_model_gemini: str = Field(default_factory=lambda: os.getenv("LLM_MODEL_GEMINI", "gemini-2.5-flash"))

    llm_rpm_mistral: int = Field(default_factory=lambda: int(os.getenv("LLM_RPM_MISTRAL", "50")))
    llm_rpm_groq: int = Field(default_factory=lambda: int(os.getenv("LLM_RPM_GROQ", "30")))
    ollama_base_url: str = Field(default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"))

    # --- Local Models (PLAN2.md) ---
    models_preload: bool = Field(default_factory=lambda: os.getenv("MODELS_PRELOAD", "true").lower() in ("1", "true", "yes"))
    models_device: str = Field(default_factory=lambda: os.getenv("MODELS_DEVICE", "auto"))
    room_classifier_model: str = Field(default_factory=lambda: os.getenv("ROOM_CLASSIFIER_MODEL", "openai/clip-vit-base-patch32"))
    room_classifier_min_score: float = Field(default_factory=lambda: float(os.getenv("ROOM_CLASSIFIER_MIN_SCORE", "0.30")))
    local_vlm_model: str | None = Field(default_factory=lambda: os.getenv("LOCAL_VLM_MODEL") or None)
    hf_home: str | None = Field(default_factory=lambda: os.getenv("HF_HOME") or None)
    hf_hub_offline: bool = Field(default_factory=lambda: os.getenv("HF_HUB_OFFLINE", "0").lower() in ("1", "true", "yes"))

    # --- Video Tier Settings (WP2 / WP3 / WP4) ---
    video_max_frames: int = Field(default_factory=lambda: int(os.getenv("VIDEO_MAX_FRAMES", "90")))
    video_max_image_side: int = Field(default_factory=lambda: int(os.getenv("VIDEO_MAX_IMAGE_SIDE", "1600")))
    video_matcher: str = Field(default_factory=lambda: os.getenv("VIDEO_MATCHER", "auto"))
    video_min_reg_ratio: float = Field(default_factory=lambda: float(os.getenv("VIDEO_MIN_REG_RATIO", "0.60")))
    video_stage_timeout_s: int = Field(default_factory=lambda: int(os.getenv("VIDEO_STAGE_TIMEOUT_S", "600")))

    # --- Scale & Safety ---
    allow_synthetic: bool = Field(default_factory=lambda: os.getenv("ALLOW_SYNTHETIC", "false").lower() in ("1", "true", "yes"))
    known_reference_m: float | None = Field(default_factory=lambda: float(os.getenv("KNOWN_REFERENCE_M")) if os.getenv("KNOWN_REFERENCE_M") else None)
    known_reference_kind: str = Field(default_factory=lambda: os.getenv("KNOWN_REFERENCE_KIND", "ceiling_height"))

    # --- General execution ---
    offline: bool = Field(default_factory=lambda: os.getenv("OFFLINE", "1").lower() in ("1", "true", "yes"))
    use_mcp: bool = Field(default_factory=lambda: os.getenv("USE_MCP", "0").lower() in ("1", "true", "yes"))
    device: str = Field(default_factory=lambda: os.getenv("DEVICE", "cpu"))

    llm_cache_dir: Path = Field(default_factory=lambda: Path(os.getenv("LLM_CACHE_DIR", "data/cache")))
    models_dir: Path = Field(default_factory=lambda: Path(os.getenv("MODELS_DIR", "models")))
    data_dir: Path = Field(default_factory=lambda: Path(os.getenv("DATA_DIR", "data")))


# Singleton configuration instance
settings = AreaMapConfig()

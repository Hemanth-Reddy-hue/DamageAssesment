"""Configuration management for AreaMap."""

import os
from pathlib import Path
from pydantic import BaseModel, Field
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

class AreaMapConfig(BaseModel):
    hf_token: str | None = Field(default_factory=lambda: os.getenv("HF_TOKEN"))
    llm_provider: str = Field(default_factory=lambda: os.getenv("LLM_PROVIDER", "local"))
    anthropic_api_key: str | None = Field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY"))
    gemini_api_key: str | None = Field(default_factory=lambda: os.getenv("GEMINI_API_KEY"))
    openai_api_key: str | None = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    
    offline: bool = Field(default_factory=lambda: os.getenv("OFFLINE", "1").lower() in ("1", "true", "yes"))
    use_mcp: bool = Field(default_factory=lambda: os.getenv("USE_MCP", "0").lower() in ("1", "true", "yes"))
    device: str = Field(default_factory=lambda: os.getenv("DEVICE", "cpu"))
    
    llm_cache_dir: Path = Field(default_factory=lambda: Path(os.getenv("LLM_CACHE_DIR", "data/cache")))
    models_dir: Path = Field(default_factory=lambda: Path(os.getenv("MODELS_DIR", "models")))
    data_dir: Path = Field(default_factory=lambda: Path(os.getenv("DATA_DIR", "data")))

# Singleton configuration instance
settings = AreaMapConfig()

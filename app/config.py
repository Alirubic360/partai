"""Runtime settings, read from environment variables (never hard-code keys)."""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    mode: str = "demo"  # "demo" uses canned providers; "live" calls OpenAI/SerpApi
    openai_api_key: str = ""
    serpapi_api_key: str = ""
    vision_model: str = "gpt-4o"
    whisper_model: str = "whisper-1"
    db_path: str = "data/partai.db"
    free_monthly_limit: int = 10
    deadline_seconds: float = 15.0          # hard end-to-end budget
    vision_timeout: float = 11.0
    pricing_timeout: float = 8.0
    part_number_min_confidence: float = 0.75
    max_image_bytes: int = 10 * 1024 * 1024
    max_audio_bytes: int = 5 * 1024 * 1024

    @classmethod
    def from_env(cls) -> "Settings":
        e = os.environ
        return cls(
            mode=e.get("PARTAI_MODE", "demo").lower(),
            openai_api_key=e.get("OPENAI_API_KEY", ""),
            serpapi_api_key=e.get("SERPAPI_API_KEY", ""),
            vision_model=e.get("PARTAI_VISION_MODEL", "gpt-4o"),
            db_path=e.get("PARTAI_DB_PATH", "data/partai.db"),
            free_monthly_limit=int(e.get("PARTAI_FREE_MONTHLY_LIMIT", "10")),
            deadline_seconds=float(e.get("PARTAI_DEADLINE_SECONDS", "15")),
        )

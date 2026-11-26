"""Google Gemini vision (free-tier friendly). Same contract/prompt/gate as the OpenAI provider."""
from __future__ import annotations

import base64

import httpx

from ..schemas import VisionResult
from .base import ProviderError
from .openai_client import VISION_SYSTEM_PROMPT, _extract_json

GEMINI = "https://generativelanguage.googleapis.com/v1beta/models"


class GeminiVision:
    def __init__(self, api_key: str, model: str = "gemini-2.0-flash", timeout: float = 11.0,
                 client: httpx.AsyncClient | None = None):
        self.api_key, self.model, self.timeout, self._client = api_key, model, timeout, client

    async def analyze(self, image_jpeg: bytes, hint: str = "") -> VisionResult:
        body = {
            "systemInstruction": {"parts": [{"text": VISION_SYSTEM_PROMPT}]},
            "contents": [{"role": "user", "parts": [
                {"text": "Identify this part." + (f" Technician context: {hint}" if hint else "")},
                {"inline_data": {"mime_type": "image/jpeg", "data": base64.b64encode(image_jpeg).decode()}},
            ]}],
            "generationConfig": {"temperature": 0, "maxOutputTokens": 500,
                                 "responseMimeType": "application/json"},
        }
        client = self._client or httpx.AsyncClient(timeout=self.timeout)
        try:
            r = await client.post(f"{GEMINI}/{self.model}:generateContent", json=body,
                                  headers={"x-goog-api-key": self.api_key})
        except httpx.HTTPError as exc:
            raise ProviderError(f"gemini request failed: {exc.__class__.__name__}") from exc
        finally:
            if self._client is None:
                await client.aclose()
        if r.status_code != 200:
            raise ProviderError(f"gemini HTTP {r.status_code}")
        try:
            text = r.json()["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, ValueError, TypeError) as exc:
            raise ProviderError("unexpected gemini response shape") from exc
        try:
            return VisionResult.model_validate(_extract_json(text))
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(f"gemini JSON failed validation: {exc}") from exc

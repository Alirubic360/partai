"""GPT-4o Vision + Whisper over plain HTTPS (no SDK dependency)."""
from __future__ import annotations

import base64
import json
import re

import httpx

from ..schemas import VisionResult
from .base import ProviderError

OPENAI = "https://api.openai.com/v1"

VISION_SYSTEM_PROMPT = """\
You identify used appliance parts from a photo for a repair-shop inventory tool.
Return ONLY a JSON object with these keys:
  part_name        string  e.g. "Drain Pump", "Control Board"
  brand            string  manufacturer printed on the part/label, else ""
  part_number      string  the OEM / part number EXACTLY as printed, else ""
  part_number_confidence number 0..1  your confidence every character is right
  legibility       "clear" | "partial" | "blurry" | "none"  (is the label readable?)
  model_numbers    array of strings  appliance model numbers printed on the label
  appliance_type   string  e.g. "washing machine", "refrigerator", else ""
  visual_condition string  what you can SEE (rust, cracks, dirt, burnt marks), max 20 words
  weight_lb_estimate number|null  rough shipping weight of this part in pounds

Hard rules:
- NEVER guess or complete a part number. If characters are blurry, cut off, or
  ambiguous (0/O, 1/I, 5/S, 8/B), lower part_number_confidence below 0.5 and
  set legibility to "partial" or "blurry". If nothing is readable use "" and "none".
- Do not copy a number from memory of what the part "should" be; only what is printed.
- Do not claim the part works; you can only see it.
"""


def _extract_json(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            raise ProviderError("vision response was not JSON")
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError as exc:
            raise ProviderError(f"vision response was not valid JSON: {exc}") from exc


class OpenAIVision:
    def __init__(self, api_key: str, model: str = "gpt-4o", timeout: float = 11.0,
                 client: httpx.AsyncClient | None = None):
        self.api_key, self.model, self.timeout = api_key, model, timeout
        self._client = client

    async def analyze(self, image_jpeg: bytes, hint: str = "") -> VisionResult:
        b64 = base64.b64encode(image_jpeg).decode()
        user_text = "Identify this part." + (f" Technician context: {hint}" if hint else "")
        body = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 500,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": VISION_SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{b64}", "detail": "high"}},
                ]},
            ],
        }
        client = self._client or httpx.AsyncClient(timeout=self.timeout)
        try:
            r = await client.post(f"{OPENAI}/chat/completions", json=body,
                                  headers={"Authorization": f"Bearer {self.api_key}"})
        except httpx.HTTPError as exc:
            raise ProviderError(f"vision request failed: {exc.__class__.__name__}") from exc
        finally:
            if self._client is None:
                await client.aclose()
        if r.status_code != 200:
            raise ProviderError(f"vision HTTP {r.status_code}")
        try:
            content = r.json()["choices"][0]["message"]["content"]
        except (KeyError, IndexError, ValueError) as exc:
            raise ProviderError("unexpected vision response shape") from exc
        data = _extract_json(content)
        try:
            return VisionResult.model_validate(data)
        except Exception as exc:  # pydantic ValidationError
            raise ProviderError(f"vision JSON failed validation: {exc}") from exc


class OpenAIWhisper:
    def __init__(self, api_key: str, model: str = "whisper-1", timeout: float = 10.0,
                 client: httpx.AsyncClient | None = None):
        self.api_key, self.model, self.timeout = api_key, model, timeout
        self._client = client

    async def transcribe(self, audio: bytes, filename: str) -> str:
        client = self._client or httpx.AsyncClient(timeout=self.timeout)
        try:
            r = await client.post(
                f"{OPENAI}/audio/transcriptions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                data={"model": self.model, "language": "en"},
                files={"file": (filename or "note.webm", audio)},
            )
        except httpx.HTTPError as exc:
            raise ProviderError(f"whisper request failed: {exc.__class__.__name__}") from exc
        finally:
            if self._client is None:
                await client.aclose()
        if r.status_code != 200:
            raise ProviderError(f"whisper HTTP {r.status_code}")
        return str(r.json().get("text", "")).strip()

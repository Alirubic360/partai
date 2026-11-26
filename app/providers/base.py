from __future__ import annotations

from typing import Protocol

from ..schemas import Comp, VisionResult


class ProviderError(RuntimeError):
    """An upstream (OpenAI / SerpApi) call failed or returned unusable data."""


class VisionProvider(Protocol):
    async def analyze(self, image_jpeg: bytes, hint: str = "") -> VisionResult: ...


class Transcriber(Protocol):
    async def transcribe(self, audio: bytes, filename: str) -> str: ...


class PriceProvider(Protocol):
    async def search(self, query: str) -> list[Comp]: ...

"""Canned providers for demo mode and tests. Output is ALWAYS flagged demo=True upstream."""
from __future__ import annotations

import asyncio

from ..schemas import Comp, VisionResult


class DemoVision:
    def __init__(self, result: VisionResult | None = None, delay: float = 0.0):
        self.result = result or VisionResult(
            part_name="Drain Pump", brand="Bosch", part_number="00145787",
            part_number_confidence=0.96, legibility="clear",
            model_numbers=["WAE24060UC"], appliance_type="washing machine",
            visual_condition="clean, no visible cracks", weight_lb_estimate=2.0)
        self.delay = delay

    async def analyze(self, image_jpeg: bytes, hint: str = "") -> VisionResult:
        await asyncio.sleep(self.delay)
        return self.result


class DemoTranscriber:
    def __init__(self, text: str = "Drain pump tested working, pulled from a Bosch series 8 washer"):
        self.text = text

    async def transcribe(self, audio: bytes, filename: str) -> str:
        return self.text


class DemoPrices:
    def __init__(self, prices: list[float] | None = None, delay: float = 0.0):
        self.prices = prices if prices is not None else [24.0, 29.5, 32.0, 35.0, 38.0, 41.0, 45.0, 89.0]
        self.delay = delay

    async def search(self, query: str) -> list[Comp]:
        await asyncio.sleep(self.delay)
        return [Comp(title=f"{query} sample", price=p, sold=True) for p in self.prices]


class NoPrices:
    """Used in live mode without a price key: honest 'no data' instead of invented prices."""
    async def search(self, query: str) -> list[Comp]:
        return []

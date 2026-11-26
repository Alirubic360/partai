"""Orchestrates: image + voice -> vision/OCR -> pricing -> eBay-ready listing JSON.

Hard deadline (default 15 s). Vision and Whisper run in parallel; pricing runs after
the part number is known. If pricing is slow/fails the listing still returns, flagged.
"""
from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass

from .config import Settings
from .listing import CONDITION_TEXT, build_description, build_title, shipping_weight
from .notes import parse_note
from .pricing import compute_pricing
from .providers.base import PriceProvider, ProviderError, Transcriber, VisionProvider
from .schemas import AnalysisResult, NoteFacts, PricingInfo, VisionResult

_PN_OK = re.compile(r"^[A-Z0-9][A-Z0-9\-./]{3,24}$")
# letters that OCR/vision commonly swap with digits (I/1, O/0, S/5, B/8, Z/2) sitting next to a digit
_AMBIGUOUS = re.compile(r"(?<=\d)[IOSBZ]|[IOSBZ](?=\d)")
AMBIGUOUS_MIN_CONF = 0.9


class AnalysisError(RuntimeError):
    """Pipeline could not produce a listing (vision down / deadline blown with nothing)."""


@dataclass
class Providers:
    vision: VisionProvider
    transcriber: Transcriber | None
    prices: PriceProvider
    demo: bool = False


def normalize_part_number(pn: str) -> str:
    return re.sub(r"\s+", "", pn or "").upper()


def _ms(t0: float) -> int:
    return int((time.perf_counter() - t0) * 1000)


async def analyze_part(image_jpeg: bytes, audio: bytes | None, audio_name: str, typed_note: str,
                       providers: Providers, cfg: Settings) -> AnalysisResult:
    t_start = time.perf_counter()
    timings: dict[str, int] = {}
    flags: list[str] = []

    async def run_vision() -> VisionResult:
        t = time.perf_counter()
        try:
            return await asyncio.wait_for(providers.vision.analyze(image_jpeg, typed_note), cfg.vision_timeout)
        except asyncio.TimeoutError as exc:
            raise AnalysisError("vision timed out") from exc
        except ProviderError as exc:
            raise AnalysisError(f"vision failed: {exc}") from exc
        finally:
            timings["vision"] = _ms(t)

    async def run_stt() -> str:
        if not audio or providers.transcriber is None:
            return ""
        t = time.perf_counter()
        try:
            return await asyncio.wait_for(providers.transcriber.transcribe(audio, audio_name), 8.0)
        except (asyncio.TimeoutError, ProviderError):
            flags.append("voice_note_unavailable")
            return ""
        finally:
            timings["transcribe"] = _ms(t)

    vision_res, transcript = await asyncio.gather(run_vision(), run_stt())

    note_text = " ".join(x for x in (typed_note.strip(), transcript.strip()) if x)
    note = parse_note(note_text)
    models = sorted(set(m.upper() for m in vision_res.model_numbers) | set(note.model_numbers))

    # --- part-number trust gate: never publish a guessed number ---
    pn = normalize_part_number(vision_res.part_number)
    unreadable = (not pn or not _PN_OK.match(pn)
                  or vision_res.legibility in ("blurry", "none")
                  or vision_res.part_number_confidence < cfg.part_number_min_confidence)
    if unreadable:
        flags.append("part_number_unreadable" if not pn or vision_res.legibility in ("blurry", "none")
                     else "part_number_low_confidence")
        pn_for_listing = "" if (not pn or vision_res.legibility in ("blurry", "none")) else pn
    else:
        pn_for_listing = pn
    if (pn_for_listing and _AMBIGUOUS.search(pn_for_listing)
            and vision_res.part_number_confidence < AMBIGUOUS_MIN_CONF):
        flags.append("part_number_ambiguous_chars")   # e.g. W1I81723 - is that I a 1?
    if not vision_res.brand:
        flags.append("brand_not_detected")
    if not vision_res.part_name:
        flags.append("part_name_not_detected")

    # --- pricing, inside the remaining time budget ---
    pricing = PricingInfo()
    query = " ".join(x for x in (vision_res.brand, pn_for_listing or vision_res.part_name,
                                 "" if pn_for_listing else (models[0] if models else "")) if x).strip()
    remaining = cfg.deadline_seconds - (time.perf_counter() - t_start) - 0.3
    if not query:
        flags.append("pricing_skipped_no_identity")
    elif remaining <= 0.5:
        flags.append("pricing_unavailable_deadline")
    else:
        t = time.perf_counter()
        try:
            comps = await asyncio.wait_for(providers.prices.search(query),
                                           min(cfg.pricing_timeout, remaining))
            pricing = compute_pricing(comps, note.functional_status, query=query)
            if pricing.confidence == "none":
                flags.append("no_price_comps_found")
            elif pricing.confidence == "low":
                flags.append("price_low_confidence")
            if not pn_for_listing:
                flags.append("price_from_fuzzy_query")
        except asyncio.TimeoutError:
            flags.append("pricing_unavailable_timeout")
        except ProviderError:
            flags.append("pricing_unavailable_error")
        finally:
            timings["pricing"] = _ms(t)

    status = note.functional_status
    title = build_title(vision_res, status, pn_for_listing)
    result = AnalysisResult(
        listing_title=title,
        brand=vision_res.brand,
        part_number=pn_for_listing,
        compatible_models=models,
        condition=CONDITION_TEXT[status],
        suggested_description=build_description(vision_res, note, pn_for_listing, models, status),
        estimated_price_usd=pricing.estimated_price_usd,
        shipping_weight_estimate=shipping_weight(vision_res),
        needs_review=bool(set(flags) & {
            "part_number_unreadable", "part_number_low_confidence", "part_number_ambiguous_chars",
            "brand_not_detected", "no_price_comps_found", "price_from_fuzzy_query"}),
        flags=sorted(set(flags)),
        price_confidence=pricing.confidence,
        price_sample_size=pricing.sample_size,
        functional_status=status,
        transcript=transcript,
        demo=providers.demo,
    )
    timings["total"] = _ms(t_start)
    result.timings_ms = timings
    return result

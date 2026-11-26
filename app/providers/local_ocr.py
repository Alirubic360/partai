"""Free, offline vision provider: RapidOCR + label heuristics.

Use when no OpenAI key is set, or as a cross-check. It only reads TEXT printed on
labels (part number / brand / model); it cannot judge condition or recognise a part by shape.
"""
from __future__ import annotations

import asyncio
import difflib
import io
import re
from functools import lru_cache

from PIL import Image, ImageFilter

from ..schemas import VisionResult
from .base import ProviderError

BRANDS = ["Whirlpool", "Samsung", "Bosch", "LG", "GE", "Frigidaire", "Maytag", "Kenmore", "Electrolux",
          "Siemens", "Miele", "Haier", "Speed Queen", "Amana", "KitchenAid", "Hotpoint", "Carrier",
          "Trane", "Lennox", "Goodman", "Rheem", "Honeywell"]
_PN_LABEL = re.compile(r"(?:P/?N|PART\s*(?:NO|NUMBER|#)|MPN|OEM)\s*[:.#\-]?\s*([A-Z0-9][A-Z0-9\-./]{3,23})", re.I)
_MODEL_LABEL = re.compile(r"(?:MODEL|MOD|M/?N)\s*(?:NO|NUMBER|#)?\s*[:.#\-]?\s*([A-Z0-9][A-Z0-9\-./]{4,19})", re.I)
_TOKEN = re.compile(r"^[A-Z0-9][A-Z0-9\-./]{4,23}$")
_PART_WORDS = ["Drain Pump", "Control Board", "Circulation Pump", "Motor", "Compressor", "Door Lock",
               "Heating Element", "Thermostat", "Fan Motor", "Water Inlet Valve", "Pump", "Board"]


@lru_cache(maxsize=1)
def _engine():
    from rapidocr_onnxruntime import RapidOCR
    return RapidOCR()


def sharpness(img: Image.Image) -> float:
    """Variance of the Laplacian-like edge response; low = blurry."""
    import numpy as np
    g = np.asarray(img.convert("L").resize((640, int(640 * img.height / img.width) or 1)), dtype="float32")
    lap = g[1:-1, 1:-1] * 4 - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]
    return float(lap.var())


def _match_part_name(lines: list[tuple[str, float]]) -> str:
    """Tolerant match: ignores case/spacing, allows ~1-2 OCR character errors."""
    norm = lambda x: re.sub(r"[^a-z]", "", x.lower())
    best, best_r = "", 0.0
    for t, _ in lines:
        nt = norm(t)
        if len(nt) < 4:
            continue
        for p in _PART_WORDS:
            npn = norm(p)
            r = 1.0 if npn in nt else difflib.SequenceMatcher(None, npn, nt).ratio()
            if r > best_r:
                best, best_r = p, r
    return best if best_r >= 0.82 else ""


def parse_ocr(lines: list[tuple[str, float]], blur: float) -> VisionResult:
    """lines = [(text, ocr_confidence 0..1)] in reading order."""
    text = "\n".join(t for t, _ in lines)
    upper = text.upper()
    brand = next((b for b in BRANDS if re.search(rf"\b{re.escape(b.upper())}\b", upper)), "")
    pn, conf = "", 0.0
    for i, (t, c) in enumerate(lines):
        m = _PN_LABEL.search(t)
        cand = m.group(1) if m else ""
        if not cand and re.fullmatch(r"(?i)(P/?N|PART\s*(NO|NUMBER|#)?)\s*[:.#\-]?", t.strip()) and i + 1 < len(lines):
            cand, c = lines[i + 1][0].strip(), min(c, lines[i + 1][1])
        cand = re.sub(r"\s+", "", cand).upper().rstrip(".,;:")
        if cand and _TOKEN.match(cand):
            pn, conf = cand, c
            break
    models = []
    for t, _ in lines:
        m = _MODEL_LABEL.search(t)
        if m:
            tok = m.group(1).upper().rstrip(".,;:")
            if tok != pn and _TOKEN.match(tok):
                models.append(tok)
    part = _match_part_name(lines)
    if not pn:
        legibility = "none" if not lines else "blurry"
        conf = 0.0
    else:
        # OCR confidence is optimistic; discount it by image sharpness.
        sharp_factor = min(1.0, blur / 150.0)
        conf = round(conf * (0.6 + 0.4 * sharp_factor), 3)
        legibility = "clear" if conf >= 0.85 else "partial" if conf >= 0.6 else "blurry"
    return VisionResult(part_name=part, brand=brand, part_number=pn, part_number_confidence=conf,
                        legibility=legibility, model_numbers=sorted(set(models)))


class LocalOCRVision:
    async def analyze(self, image_jpeg: bytes, hint: str = "") -> VisionResult:
        return await asyncio.to_thread(self._run, image_jpeg)

    def _run(self, jpeg: bytes) -> VisionResult:
        try:
            import numpy as np
            img = Image.open(io.BytesIO(jpeg)).convert("RGB")
            result, _ = _engine()(np.asarray(img))
        except Exception as exc:
            raise ProviderError(f"local OCR failed: {exc.__class__.__name__}") from exc
        lines = [(str(r[1]), float(r[2])) for r in (result or [])]
        return parse_ocr(lines, sharpness(img))

"""Validate + downscale uploads (faster vision calls, lower cost)."""
from __future__ import annotations

import io

from PIL import Image, ImageOps, UnidentifiedImageError

MAX_SIDE = 1600
MIN_SIDE = 200


class BadImage(ValueError):
    pass


def prepare_image(raw: bytes) -> bytes:
    try:
        img = Image.open(io.BytesIO(raw))
        img.load()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise BadImage("file is not a readable image") from exc
    img = ImageOps.exif_transpose(img).convert("RGB")
    if min(img.size) < MIN_SIDE:
        raise BadImage(f"image too small (min {MIN_SIDE}px on the short side)")
    img.thumbnail((MAX_SIDE, MAX_SIDE))
    out = io.BytesIO()
    img.save(out, "JPEG", quality=88)
    return out.getvalue()

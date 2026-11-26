"""Generate a labeled SYNTHETIC set of appliance-part label photos for pipeline benchmarking.

Synthetic != real. Real photos have glare, curved surfaces, grime, embossed text. Treat results
as a floor-check of the pipeline and the blur gate, not as field accuracy.
Usage: python -m eval.make_synthetic OUT_DIR [N] [SEED] [--hard]
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
BRANDS = ["Bosch", "Samsung", "Whirlpool", "LG", "Frigidaire", "Maytag", "Electrolux", "Siemens"]
PARTS = ["Drain Pump", "Control Board", "Circulation Pump", "Heating Element", "Door Lock"]
PREFIX = {"Bosch": "00", "Samsung": "DC92-", "Whirlpool": "W10", "LG": "EBR", "Frigidaire": "5304",
          "Maytag": "W11", "Electrolux": "8087", "Siemens": "00"}
LEVELS = [("clean", 0.0), ("slight", 1.2), ("heavy", 3.2), ("unreadable", 7.0)]


def make_pn(rng: random.Random, brand: str) -> str:
    n = rng.randint(5, 8) if PREFIX[brand] != "EBR" else 8
    return PREFIX[brand] + "".join(rng.choice("0123456789") for _ in range(n))


def render(rng: random.Random, brand, part, pn, model, blur, rot, noise) -> Image.Image:
    img = Image.new("RGB", (900, 600), (rng.randint(150, 215),) * 3)
    d = ImageDraw.Draw(img)
    for _ in range(40):  # grime
        x, y, r = rng.randint(0, 900), rng.randint(0, 600), rng.randint(3, 25)
        g = rng.randint(90, 170)
        d.ellipse([x, y, x + r, y + r], fill=(g, g, g))
    lab = Image.new("RGB", (760, 360), (245, 245, 240))
    ld = ImageDraw.Draw(lab)
    ld.text((30, 25), brand.upper(), font=ImageFont.truetype(FONT, 52), fill=(10, 10, 10))
    ld.text((30, 105), part, font=ImageFont.truetype(FONT, 30), fill=(40, 40, 40))
    f = ImageFont.truetype(FONT_MONO, 40)
    ld.text((30, 175), f"P/N: {pn}", font=f, fill=(0, 0, 0))
    ld.text((30, 245), f"MODEL: {model}", font=f, fill=(0, 0, 0))
    lab = lab.rotate(rot, expand=True, fillcolor=(190, 190, 190))
    img.paste(lab, (60 + rng.randint(-20, 20), 70 + rng.randint(-10, 10)))
    if blur:
        img = img.filter(ImageFilter.GaussianBlur(blur))
    if noise:
        import numpy as np
        a = np.asarray(img).astype("int16") + np.random.default_rng(rng.randint(0, 9999)).normal(0, noise, (600, 900, 3))
        img = Image.fromarray(a.clip(0, 255).astype("uint8"))
    return img


def harden(rng: random.Random, img: Image.Image) -> Image.Image:
    """Field-like damage: glare hotspot, perspective skew, low resolution, heavy JPEG, dark vignette."""
    import numpy as np
    w, h = img.size
    a = np.asarray(img).astype("float32")
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy, r = rng.randint(200, 700), rng.randint(100, 450), rng.randint(60, 140)
    glare = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r * r)))[..., None] * rng.uniform(120, 230)
    vig = 1 - 0.35 * (((xx - w / 2) ** 2 + (yy - h / 2) ** 2) / ((w / 2) ** 2 + (h / 2) ** 2))[..., None]
    img = Image.fromarray(np.clip(a * vig + glare, 0, 255).astype("uint8"))
    k = rng.uniform(0.0004, 0.0012)   # perspective: tilt the plane like a hand-held shot
    coeffs = (1, rng.uniform(-0.08, 0.08), 0, rng.uniform(-0.08, 0.08), 1, 0, k * rng.choice([-1, 1]), k * rng.choice([-1, 1]))
    img = img.transform((w, h), Image.PERSPECTIVE, coeffs, Image.BICUBIC, fillcolor=(120, 120, 120))
    f = rng.uniform(0.45, 0.8)         # low-res phone / cropped digital zoom
    return img.resize((int(w * f), int(h * f))).resize((w, h))


def main(out: str, n: int = 120, seed: int = 7, hard: bool = False) -> None:
    rng, base = random.Random(seed), Path(out)
    base.mkdir(parents=True, exist_ok=True)
    labels = []
    for i in range(n):
        brand, part = rng.choice(BRANDS), rng.choice(PARTS)
        pn = make_pn(rng, brand)
        model = rng.choice(["WAE", "WM", "RF", "DW"]) + str(rng.randint(10000, 99999)) + rng.choice(["UC", "CW", "SR"])
        level, blur = LEVELS[i % 4]
        img = render(rng, brand, part, pn, model, blur * rng.uniform(0.85, 1.15),
                     rng.uniform(-6, 6), rng.choice([0, 6, 14]))
        if hard:
            img = harden(rng, img)
        name = f"{i:03d}_{level}.jpg"
        img.save(base / name, quality=rng.randint(35, 60) if hard else rng.randint(70, 92))
        labels.append({"image": name, "part_number": pn, "brand": brand, "model": model, "difficulty": level})
    (base / "labels.json").write_text(json.dumps(labels, indent=1))
    print(f"wrote {n} images + labels.json to {base}")


if __name__ == "__main__":
    a = sys.argv[1:]
    nums = [x for x in a[1:] if x.isdigit()]
    main(a[0], int(nums[0]) if nums else 120, int(nums[1]) if len(nums) > 1 else 7, "--hard" in a)

"""Generate a labeled SYNTHETIC set of appliance-part label photos for pipeline benchmarking.

Synthetic != real. Real photos have glare, curved surfaces, grime, embossed text. Treat results
as a floor-check of the pipeline and the blur gate, not as field accuracy.
Usage: python -m eval.make_synthetic OUT_DIR [N] [SEED]
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


def main(out: str, n: int = 120, seed: int = 7) -> None:
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
        name = f"{i:03d}_{level}.jpg"
        img.save(base / name, quality=rng.randint(70, 92))
        labels.append({"image": name, "part_number": pn, "brand": brand, "model": model, "difficulty": level})
    (base / "labels.json").write_text(json.dumps(labels, indent=1))
    print(f"wrote {n} images + labels.json to {base}")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], int(a[1]) if len(a) > 1 else 120, int(a[2]) if len(a) > 2 else 7)

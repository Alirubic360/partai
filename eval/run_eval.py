"""Measure part-identification accuracy on YOUR labeled photos.

Usage:  python -m eval.run_eval DATASET_DIR [--live]
DATASET_DIR/labels.json = [{"image": "a.jpg", "part_number": "00145787", "brand": "Bosch"}, ...]
Use ~100+ real photos from the shop (clean, dirty, angled, blurry) for a number worth trusting.

Metrics (the one that matters for sellers is `wrong_but_published`):
  pn_exact_match        printed PN read correctly (all photos)
  published_precision   of the PNs the pipeline did NOT flag, how many were right
  wrong_but_published   unflagged wrong PNs  -> these become bad eBay listings
  flagged_rate          sent to human review instead of guessed
"""
from __future__ import annotations

import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

from app.config import Settings
from app.imageprep import prepare_image
from app.pipeline import analyze_part, normalize_part_number


async def evaluate(labels: list[dict], base: Path, providers, cfg: Settings) -> dict:
    rows, lat = [], []
    for item in labels:
        jpeg = prepare_image((base / item["image"]).read_bytes())
        t = time.perf_counter()
        res = await analyze_part(jpeg, None, "", "", providers, cfg)
        lat.append(time.perf_counter() - t)
        truth = normalize_part_number(item["part_number"])
        flagged = res.needs_review or not res.part_number
        rows.append({"image": item["image"], "difficulty": item.get("difficulty", "all"), "truth": truth, "got": res.part_number,
                     "match": res.part_number == truth, "flagged": flagged,
                     "brand_ok": res.brand.lower() == item.get("brand", res.brand).lower()})
    n = len(rows)
    published = [r for r in rows if not r["flagged"]]
    wrong_pub = [r for r in published if not r["match"]]
    lat_s = sorted(lat)
    return {
        "n": n,
        "pn_exact_match": round(sum(r["match"] for r in rows) / n, 4) if n else None,
        "brand_accuracy": round(sum(r["brand_ok"] for r in rows) / n, 4) if n else None,
        "flagged_rate": round(sum(r["flagged"] for r in rows) / n, 4) if n else None,
        "published_precision": round(1 - len(wrong_pub) / len(published), 4) if published else None,
        "wrong_but_published": len(wrong_pub),
        "latency_p50_s": round(statistics.median(lat_s), 3) if lat_s else None,
        "latency_p95_s": round(lat_s[min(n - 1, int(0.95 * n))], 3) if lat_s else None,
        "by_difficulty": {d: {"n": len(g), "pn_match": round(sum(x["match"] for x in g) / len(g), 3),
                              "flagged": round(sum(x["flagged"] for x in g) / len(g), 3),
                              "wrong_but_published": sum((not x["flagged"]) and (not x["match"]) for x in g)}
                          for d in sorted({r["difficulty"] for r in rows})
                          for g in [[r for r in rows if r["difficulty"] == d]]},
        "errors": [r for r in rows if not r["match"]][:10],
    }


def main(argv: list[str]) -> None:
    from app.main import build_providers
    base = Path(argv[0])
    cfg = Settings.from_env()
    if "--live" in argv:
        cfg = Settings(**{**cfg.__dict__, "mode": "live"})
    elif cfg.mode == "demo":
        print("DEMO provider: numbers below only test the harness, not real accuracy.", file=sys.stderr)
    labels = json.loads((base / "labels.json").read_text())
    print(json.dumps(asyncio.run(evaluate(labels, base, build_providers(cfg), cfg)), indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])

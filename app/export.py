"""Exports: eBay File-Exchange-style CSV, eBay-ready JSON, and a plain inventory CSV."""
from __future__ import annotations

import csv
import io

from .listing import EBAY_CONDITION_ID

EBAY_COLUMNS = ["*Action(SiteID=US|Country=US|Currency=USD|Version=1193)", "*Category", "*Title",
                "*ConditionID", "*Description", "*Format", "*Duration", "*StartPrice",
                "*Quantity", "CustomLabel", "Brand", "MPN"]
INVENTORY_COLUMNS = ["id", "created_at", "brand", "part_number", "listing_title", "condition",
                     "price_quick", "price_avg", "price_premium", "price_confidence",
                     "compatible_models", "shipping_weight_estimate", "needs_review"]


def _status_cond_id(item: dict) -> int:
    return EBAY_CONDITION_ID.get(item.get("functional_status", "untested"), 3000)


def to_ebay_json(item: dict) -> dict:
    p = item["estimated_price_usd"]
    return {
        "title": item["listing_title"], "conditionId": _status_cond_id(item),
        "description": item["suggested_description"], "format": "FixedPrice",
        "price": {"value": p["avg"], "currency": "USD"},
        "aspects": {"Brand": [item["brand"]], "MPN": [item["part_number"]],
                    "Compatible Model": item["compatible_models"]},
        "needsReview": item["needs_review"],
    }


def to_ebay_csv(items: list[dict]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(EBAY_COLUMNS)
    for it in items:
        w.writerow(["Add", "", it["listing_title"], _status_cond_id(it),
                    it["suggested_description"].replace("\n", " "), "FixedPriceItem", "GTC",
                    it["estimated_price_usd"]["avg"], 1, it.get("part_number", ""),
                    it.get("brand", ""), it.get("part_number", "")])
    return buf.getvalue()


def _safe(v) -> str:
    """Neutralise CSV/Excel formula injection from OCR'd or spoken text."""
    s = str(v)
    return "'" + s if s[:1] in ("=", "+", "-", "@") else s


def to_inventory_csv(items: list[dict]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(INVENTORY_COLUMNS)
    for it in items:
        p = it["estimated_price_usd"]
        w.writerow([it.get("id", ""), it.get("created_at", ""), _safe(it["brand"]), _safe(it["part_number"]),
                    _safe(it["listing_title"]), it["condition"], p["low"], p["avg"], p["high"],
                    it.get("price_confidence", ""), _safe("; ".join(it["compatible_models"])),
                    it["shipping_weight_estimate"], it["needs_review"]])
    return buf.getvalue()

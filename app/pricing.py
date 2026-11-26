"""Turn raw eBay comps into three price points: quick sale / market average / premium."""
from __future__ import annotations

import statistics

from .schemas import Comp, PricePoints, PricingInfo

# Applied because comps mix conditions; documented + tunable, not a market fact.
CONDITION_MULTIPLIER = {"working": 1.0, "untested": 0.85, "for_parts": 0.5}


def _quantile(sorted_vals: list[float], q: float) -> float:
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = q * (len(sorted_vals) - 1)
    lo, hi = int(pos), min(int(pos) + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def _drop_outliers(vals: list[float]) -> list[float]:
    if len(vals) < 5:
        return vals
    s = sorted(vals)
    q1, q3 = _quantile(s, 0.25), _quantile(s, 0.75)
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    kept = [v for v in s if lo <= v <= hi]
    return kept or s


def compute_pricing(comps: list[Comp], functional_status: str, query: str = "",
                    source: str = "ebay") -> PricingInfo:
    prices = [c.price for c in comps if c.price and c.price > 0]
    if not prices:
        return PricingInfo(confidence="none", source=source, query=query)
    sold_n = sum(1 for c in comps if c.sold)
    vals = _drop_outliers(prices)
    s = sorted(vals)
    if len(s) < 3:
        low, avg, high = s[0], statistics.median(s), s[-1]
    else:
        low, avg, high = _quantile(s, 0.25), statistics.median(s), _quantile(s, 0.80)
    m = CONDITION_MULTIPLIER.get(functional_status, 0.85)
    pts = PricePoints(low=round(low * m, 2), avg=round(avg * m, 2), high=round(high * m, 2))
    n = len(vals)
    if n >= 10 and sold_n >= 5:
        conf = "high"
    elif n >= 5:
        conf = "medium"
    else:
        conf = "low"
    return PricingInfo(estimated_price_usd=pts, sample_size=n, confidence=conf,
                       source=source, query=query)

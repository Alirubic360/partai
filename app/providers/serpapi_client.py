"""eBay comps via SerpApi's `ebay` engine (sold + completed listings)."""
from __future__ import annotations

import httpx

from ..schemas import Comp
from .base import ProviderError

SERPAPI = "https://serpapi.com/search.json"


def _price(item: dict) -> float | None:
    p = item.get("price")
    if isinstance(p, dict):
        for k in ("extracted", "from", "raw"):
            v = p.get(k)
            if isinstance(v, (int, float)):
                return float(v)
        if isinstance(p.get("raw"), str):
            return _parse_money(p["raw"])
    elif isinstance(p, (int, float)):
        return float(p)
    elif isinstance(p, str):
        return _parse_money(p)
    return None


def _parse_money(s: str) -> float | None:
    digits = "".join(ch for ch in s.split(" to ")[0] if ch.isdigit() or ch == ".")
    try:
        return float(digits)
    except ValueError:
        return None


def parse_comps(payload: dict, sold: bool) -> list[Comp]:
    comps: list[Comp] = []
    for item in payload.get("organic_results", []) or []:
        price = _price(item)
        if price is None or price <= 0:
            continue
        ship = 0.0
        sh = item.get("shipping")
        if isinstance(sh, dict) and isinstance(sh.get("extracted"), (int, float)):
            ship = float(sh["extracted"])
        comps.append(Comp(title=str(item.get("title", ""))[:120], price=price, shipping=ship, sold=sold))
    return comps


class SerpApiEbay:
    def __init__(self, api_key: str, timeout: float = 8.0, client: httpx.AsyncClient | None = None):
        self.api_key, self.timeout, self._client = api_key, timeout, client

    async def _call(self, client: httpx.AsyncClient, query: str, sold: bool) -> list[Comp]:
        params = {"engine": "ebay", "_nkw": query, "ebay_domain": "ebay.com",
                  "api_key": self.api_key, "_ipg": "60"}
        if sold:
            params.update({"LH_Sold": "1", "LH_Complete": "1"})
        r = await client.get(SERPAPI, params=params)
        if r.status_code != 200:
            raise ProviderError(f"serpapi HTTP {r.status_code}")
        return parse_comps(r.json(), sold)

    async def search(self, query: str) -> list[Comp]:
        client = self._client or httpx.AsyncClient(timeout=self.timeout)
        try:
            comps = await self._call(client, query, sold=True)   # real sale prices first
            if len(comps) < 3:                                    # thin: add active listings
                comps += await self._call(client, query, sold=False)
            return comps
        except httpx.HTTPError as exc:
            raise ProviderError(f"serpapi request failed: {exc.__class__.__name__}") from exc
        finally:
            if self._client is None:
                await client.aclose()

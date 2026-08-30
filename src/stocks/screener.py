"""Bot 4 — daily stock screener. Pure code, no LLM.

Runs once per session at the open. Pulls the day's movers from Alpaca's market
data API and keeps only the names worth spending LLM calls on.
"""

from __future__ import annotations

import logging
from typing import Any

from ..models import Stock

log = logging.getLogger(__name__)


def parse_stock(payload: dict[str, Any]) -> Stock:
    """Map a market-data row onto our Stock model, tolerating key aliases."""

    def pick(*keys: str, default: Any = None) -> Any:
        for key in keys:
            if key in payload and payload[key] is not None:
                return payload[key]
        return default

    return Stock(
        symbol=str(pick("symbol", "ticker", default="")).upper(),
        name=str(pick("name", "company", default="")),
        sector=str(pick("sector", default="unknown")),
        price=float(pick("price", "last", "close", default=0) or 0),
        prev_close=float(pick("prev_close", "previous_close", "prevDailyClose", default=0) or 0),
        avg_volume=float(pick("avg_volume", "average_volume", default=0) or 0),
        volume=float(pick("volume", "day_volume", default=0) or 0),
        market_cap=float(pick("market_cap", "marketCap", default=0) or 0),
        raw=payload,
    )


def filter_reason(stock: Stock, filt: dict[str, Any]) -> str | None:
    """Return the reason the stock is rejected, or None if it passes."""
    if not stock.symbol:
        return "no_symbol"

    min_price = filt.get("min_price")
    if min_price is not None and stock.price < min_price:
        return "price_too_low"

    max_price = filt.get("max_price")
    if max_price is not None and stock.price > max_price:
        return "price_too_high"

    min_avg_vol = filt.get("min_avg_volume")
    if min_avg_vol is not None and stock.avg_volume < min_avg_vol:
        return "illiquid"

    min_cap = filt.get("min_market_cap")
    if min_cap is not None and stock.market_cap < min_cap:
        return "market_cap_too_small"

    max_cap = filt.get("max_market_cap")
    if max_cap is not None and stock.market_cap > max_cap:
        return "market_cap_too_large"

    min_rel_vol = filt.get("min_rel_volume")
    if min_rel_vol is not None and stock.rel_volume < min_rel_vol:
        return "no_relative_volume"

    gap = abs(stock.gap_pct)
    min_gap = filt.get("min_gap_pct")
    if min_gap is not None and gap < min_gap:
        return "gap_too_small"

    max_gap = filt.get("max_gap_pct")
    if max_gap is not None and gap > max_gap:
        return "gap_too_large"

    excluded = {s.lower() for s in (filt.get("excluded_sectors") or [])}
    if stock.sector.lower() in excluded:
        return "excluded_sector"

    return None


def passes_filter(stock: Stock, filt: dict[str, Any]) -> bool:
    return filter_reason(stock, filt) is None


class Screener:
    """Bot 4. One scan per trading day."""

    name = "screener"

    def __init__(self, config: dict[str, Any], fetch=None):
        self.config = config or {}
        self.filter = self.config.get("stock_filter", {}) or {}
        #: async callable returning a list of raw market-data rows. Injected in
        #: tests; defaults to the Alpaca snapshot fetch.
        self._fetch = fetch

    def screen(self, rows: list[dict[str, Any] | Stock], limit: int = 20) -> list[Stock]:
        """Filter and rank raw rows. Highest relative volume first."""
        kept: list[Stock] = []
        for row in rows:
            stock = row if isinstance(row, Stock) else parse_stock(row)
            reason = filter_reason(stock, self.filter)
            if reason is None:
                kept.append(stock)
        kept.sort(key=lambda s: (s.rel_volume, abs(s.gap_pct)), reverse=True)
        return kept[:limit]

    async def run(self, limit: int = 20) -> list[Stock]:
        """Fetch today's universe and return the survivors."""
        if self._fetch is None:
            rows = await self._fetch_alpaca()
        else:
            rows = await self._fetch()
        return self.screen(rows, limit=limit)

    async def _fetch_alpaca(self) -> list[dict[str, Any]]:
        """Pull the day's most active names from Alpaca's screener endpoint."""
        import httpx

        alpaca = self.config.get("alpaca", {}) or {}
        headers = {
            "APCA-API-KEY-ID": alpaca.get("api_key", ""),
            "APCA-API-SECRET-KEY": alpaca.get("api_secret", ""),
        }
        async with httpx.AsyncClient(timeout=30) as client:
            movers = await client.get(
                "https://data.alpaca.markets/v1beta1/screener/stocks/movers",
                headers=headers,
                params={"top": 50},
            )
            movers.raise_for_status()
            body = movers.json()

        rows: list[dict[str, Any]] = []
        for bucket in ("gainers", "losers"):
            for item in body.get(bucket, []):
                price = float(item.get("price", 0) or 0)
                change_pct = float(item.get("percent_change", 0) or 0) / 100.0
                prev_close = price / (1 + change_pct) if change_pct != -1 else 0.0
                rows.append(
                    {
                        "symbol": item.get("symbol"),
                        "price": price,
                        "prev_close": prev_close,
                        **{k: v for k, v in item.items() if k not in {"symbol", "price"}},
                    }
                )
        return rows

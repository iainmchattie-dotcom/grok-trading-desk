"""Bot 4 — daily stock screener. Pure code, no LLM.

Runs once per session at the open. Pulls the day's movers from Alpaca's market
data API and keeps only the names worth spending LLM calls on.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
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
    """Return the reason the stock is rejected, or None if it passes.

    Thresholds whose datum is missing are skipped rather than failed. Alpaca
    exposes no sector and no market cap (see RESEARCH.md), so treating an absent
    value as a rejection would reject the entire universe.
    """
    if not stock.symbol:
        return "no_symbol"

    min_price = filt.get("min_price")
    if min_price is not None and stock.price < min_price:
        return "price_too_low"

    max_price = filt.get("max_price")
    if max_price is not None and stock.price > max_price:
        return "price_too_high"

    if stock.avg_volume > 0:
        min_avg_vol = filt.get("min_avg_volume")
        if min_avg_vol is not None and stock.avg_volume < min_avg_vol:
            return "illiquid"

    # market cap has no Alpaca source; 0 means "unknown", not "tiny"
    if stock.market_cap > 0:
        min_cap = filt.get("min_market_cap")
        if min_cap is not None and stock.market_cap < min_cap:
            return "market_cap_too_small"

        max_cap = filt.get("max_market_cap")
        if max_cap is not None and stock.market_cap > max_cap:
            return "market_cap_too_large"

    if stock.avg_volume > 0 and stock.volume > 0:
        min_rel_vol = filt.get("min_rel_volume")
        if min_rel_vol is not None and stock.rel_volume < min_rel_vol:
            return "no_relative_volume"

    if stock.prev_close > 0:
        gap = abs(stock.gap_pct)
        min_gap = filt.get("min_gap_pct")
        if min_gap is not None and gap < min_gap:
            return "gap_too_small"

        max_gap = filt.get("max_gap_pct")
        if max_gap is not None and gap > max_gap:
            return "gap_too_large"

    # sector likewise has no Alpaca source; "unknown" is not an exclusion
    if stock.sector and stock.sector != "unknown":
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
        """Compose today's universe from three endpoints.

        No single Alpaca endpoint carries what the filter needs. Movers gives
        {symbol, percent_change, change, price}; most-actives gives
        {symbol, volume, trade_count}; neither carries a previous close or an
        average volume. So: take the union of both for the candidate set, read
        real closes and volumes off the snapshot endpoint, and compute a true
        20-day average volume from daily bars.
        """
        symbols = await self._candidate_symbols()
        if not symbols:
            return []
        snapshots = await self._snapshots(symbols)
        averages = await self._average_volumes(symbols)

        rows: list[dict[str, Any]] = []
        for symbol in symbols:
            snapshot = snapshots.get(symbol)
            if snapshot is None:
                continue
            daily = getattr(snapshot, "daily_bar", None)
            previous = getattr(snapshot, "previous_daily_bar", None)
            latest = getattr(snapshot, "latest_trade", None)

            price = float(getattr(latest, "price", 0) or getattr(daily, "close", 0) or 0)
            rows.append(
                {
                    "symbol": symbol,
                    "price": price,
                    "prev_close": float(getattr(previous, "close", 0) or 0),
                    "volume": float(getattr(daily, "volume", 0) or 0),
                    "avg_volume": averages.get(symbol, 0.0),
                    # market_cap and sector are deliberately absent: Alpaca has
                    # no source for them, and the filter skips what it lacks.
                }
            )
        return rows

    async def _candidate_symbols(self) -> list[str]:
        """Union of the movers and most-actives lists."""
        from alpaca.data.historical.screener import ScreenerClient
        from alpaca.data.requests import MarketMoversRequest, MostActivesRequest

        alpaca = self.config.get("alpaca", {}) or {}
        client = ScreenerClient(
            api_key=alpaca.get("api_key", ""), secret_key=alpaca.get("api_secret", "")
        )
        top = int((self.config.get("stock_filter", {}) or {}).get("universe_size", 50))

        movers = await asyncio.to_thread(
            client.get_market_movers, MarketMoversRequest(top=top)
        )
        actives = await asyncio.to_thread(
            client.get_most_actives, MostActivesRequest(top=top)
        )

        symbols: list[str] = []
        for mover in list(getattr(movers, "gainers", [])) + list(getattr(movers, "losers", [])):
            symbols.append(str(mover.symbol).upper())
        for active in getattr(actives, "most_actives", []):
            symbols.append(str(active.symbol).upper())

        seen: set[str] = set()
        return [s for s in symbols if not (s in seen or seen.add(s))]

    async def _snapshots(self, symbols: list[str]) -> dict[str, Any]:
        """Latest trade, today's bar and yesterday's close, per symbol."""
        from alpaca.data.historical.stock import StockHistoricalDataClient
        from alpaca.data.requests import StockSnapshotRequest

        alpaca = self.config.get("alpaca", {}) or {}
        client = StockHistoricalDataClient(
            api_key=alpaca.get("api_key", ""), secret_key=alpaca.get("api_secret", "")
        )
        return await asyncio.to_thread(
            client.get_stock_snapshot, StockSnapshotRequest(symbol_or_symbols=symbols)
        )

    async def _average_volumes(self, symbols: list[str], days: int = 20) -> dict[str, float]:
        """True average daily volume, which no screener endpoint reports."""
        from alpaca.data.historical.stock import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame

        alpaca = self.config.get("alpaca", {}) or {}
        client = StockHistoricalDataClient(
            api_key=alpaca.get("api_key", ""), secret_key=alpaca.get("api_secret", "")
        )
        request = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=TimeFrame.Day,
            start=datetime.now(timezone.utc) - timedelta(days=days * 2),
        )
        bars = await asyncio.to_thread(client.get_stock_bars, request)

        averages: dict[str, float] = {}
        data = getattr(bars, "data", {}) or {}
        for symbol, series in data.items():
            volumes = [float(bar.volume) for bar in series[-days:] if bar.volume]
            if volumes:
                averages[str(symbol).upper()] = sum(volumes) / len(volumes)
        return averages

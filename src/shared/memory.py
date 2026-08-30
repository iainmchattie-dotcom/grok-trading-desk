"""Outcome memory — what happened last time the desk saw something like this.

Every paper in this space converges on the same mechanism: feed realized
outcomes of past decisions back into the next one. FinMem (2311.13743) and
TradingGPT (2309.03736) call it layered memory; FinCon (2407.06567) calls it
conceptual verbal reinforcement; TradingAgents (2412.20138) calls it reflection.

This desk already wrote every ingredient to `logs/desk.jsonl` — the entry, its
full per-agent breakdown, and the eventual PnL — and then never read any of it
back. Every decision was made from a cold start. This module closes that loop.

Deliberately not a vector store. Matching is on the axes that actually recur in
this log: the market, the symbol, the sector, and the theme the narrative bot
assigned. That is cheap, explainable, and needs no extra dependency.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any

from ..models import Market

log = logging.getLogger(__name__)


def _parse_ts(value: Any) -> datetime | None:
    try:
        when = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    return when if when.tzinfo else when.replace(tzinfo=timezone.utc)


class OutcomeMemory:
    """Joins `buy` records to their `close` records and serves the lessons."""

    def __init__(self, config: dict[str, Any] | None = None, event_log: Any = None):
        memory = ((config or {}).get("memory", {}) or {})
        self.enabled = bool(memory.get("enabled", True))
        self.max_examples = int(memory.get("max_examples", 5))
        self.lookback_days = int(memory.get("lookback_days", 30))
        self.log = event_log
        self._trades: list[dict[str, Any]] = []
        self._loaded_from = 0

    # -- building ------------------------------------------------------------------

    def load(self, records: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
        """Pair each closed position with the buy that opened it."""
        if records is None:
            records = self.log.read() if self.log is not None else []

        cutoff = datetime.now(timezone.utc) - timedelta(days=self.lookback_days)
        opens: dict[tuple[str, str], dict[str, Any]] = {}
        trades: list[dict[str, Any]] = []

        for record in records:
            kind = record.get("type")
            key = (str(record.get("market", "")), str(record.get("symbol", "")))

            if kind == "buy":
                opens[key] = record
            elif kind == "close":
                opened = opens.pop(key, None)
                when = _parse_ts(record.get("ts"))
                if when is not None and when < cutoff:
                    continue
                pnl = float(record.get("pnl", 0) or 0)
                amount = float((opened or {}).get("amount", 0) or 0)
                trades.append(
                    {
                        "market": key[0],
                        "symbol": key[1],
                        "pnl": round(pnl, 2),
                        "won": pnl >= 0,
                        "return_pct": round(pnl / amount, 4) if amount else 0.0,
                        "hold_hours": round(float(record.get("hold_time", 0) or 0), 1),
                        "entry_score": float((opened or {}).get("score", 0) or 0),
                        "sector": self._sector_of(opened),
                        "theme": self._theme_of(opened),
                        "closed_at": record.get("ts"),
                    }
                )

        self._trades = trades
        self._loaded_from = len(records)
        return trades

    @staticmethod
    def _theme_of(buy: dict[str, Any] | None) -> str:
        scores = (buy or {}).get("all_agent_scores") or {}
        return str((scores.get("narrative") or {}).get("theme", "") or "")

    @staticmethod
    def _sector_of(buy: dict[str, Any] | None) -> str:
        scores = (buy or {}).get("all_agent_scores") or {}
        stock = (scores.get("matrix") or {}).get("sector")
        return str(stock or (buy or {}).get("sector", "") or "")

    # -- retrieval ------------------------------------------------------------------

    def recall(
        self,
        market: Market | str,
        symbol: str = "",
        sector: str = "",
        theme: str = "",
    ) -> list[dict[str, Any]]:
        """Closed trades most comparable to the one being considered.

        Ranked by how specifically they match: the same symbol beats the same
        sector or theme, which beats merely the same market.
        """
        if not self.enabled or not self._trades:
            return []

        wanted = market.value if isinstance(market, Market) else str(market)
        symbol, sector, theme = symbol.upper(), sector.lower(), theme.lower()

        scored: list[tuple[int, dict[str, Any]]] = []
        for trade in self._trades:
            if trade["market"] != wanted:
                continue
            weight = 1
            if symbol and trade["symbol"].upper() == symbol:
                weight += 8
            if sector and trade["sector"].lower() == sector:
                weight += 3
            if theme and trade["theme"].lower() == theme:
                weight += 3
            scored.append((weight, trade))

        scored.sort(key=lambda pair: (pair[0], str(pair[1].get("closed_at", ""))), reverse=True)
        return [trade for _, trade in scored[: self.max_examples]]

    def summary(self, market: Market | str) -> dict[str, Any]:
        """Aggregate record for one market: the base rate the agent should know."""
        wanted = market.value if isinstance(market, Market) else str(market)
        trades = [t for t in self._trades if t["market"] == wanted]
        if not trades:
            return {}

        wins = [t for t in trades if t["won"]]
        losses = [t for t in trades if not t["won"]]
        by_theme: dict[str, list[float]] = defaultdict(list)
        for trade in trades:
            label = trade["theme"] or trade["sector"]
            if label:
                by_theme[label].append(trade["pnl"])

        worst = sorted(by_theme.items(), key=lambda kv: sum(kv[1]))[:3]
        return {
            "closed_trades": len(trades),
            "win_rate": round(len(wins) / len(trades), 3),
            "total_pnl": round(sum(t["pnl"] for t in trades), 2),
            "avg_win": round(sum(t["pnl"] for t in wins) / len(wins), 2) if wins else 0.0,
            "avg_loss": round(sum(t["pnl"] for t in losses) / len(losses), 2) if losses else 0.0,
            "avg_hold_hours": round(sum(t["hold_hours"] for t in trades) / len(trades), 1),
            "worst_themes": [{"theme": k, "pnl": round(sum(v), 2)} for k, v in worst],
        }

    def context(
        self,
        market: Market | str,
        symbol: str = "",
        sector: str = "",
        theme: str = "",
    ) -> dict[str, Any]:
        """The block injected into an agent's prompt. Empty when there is nothing
        to say, so a cold desk does not ship a misleading 0% win rate."""
        if not self.enabled:
            return {}
        examples = self.recall(market, symbol, sector, theme)
        overall = self.summary(market)
        if not examples and not overall:
            return {}
        return {
            "past_outcomes": {
                "note": (
                    "Realized results from this desk's own closed trades. "
                    "Treat a repeated losing pattern as evidence against a "
                    "similar entry now."
                ),
                "this_market": overall,
                "comparable_trades": examples,
            }
        }

"""Bot 10 — capital allocator (grok-4-fast).

Runs once a day. Splits the budget between crypto and stocks given both regime
reads and the trailing week's realised PnL per market. A failure here must not
express an opinion: it returns 50/50.
"""

from __future__ import annotations

from typing import Any

from ..base_agent import TEXT, UNIT, GrokAgent, schema
from ..models import Allocation

PROMPT = """You allocate one trading desk's daily budget between two markets:
Solana memecoins and US equities.

You are given each market's current regime read and its realised PnL over the
trailing week. Shift capital toward the market whose conditions are actually
favourable, not the one that happened to win last week — but treat a market that
has been losing under conditions that still hold as a reason to shrink it.

Never zero out a market on one week of data; keep at least 10% on each side
unless a regime read is genuinely shut (go_signal below 0.3), in which case that
market may go to zero.

The two percentages must sum to 1.0.

Reply ONLY JSON, no explanation.
Schema: {"crypto_pct": float 0..1, "stocks_pct": float 0..1, "reason": string}"""


class Allocator(GrokAgent):
    name = "allocator"
    model_tier = "fast"
    PROMPT = PROMPT
    SCHEMA = schema({"crypto_pct": UNIT, "stocks_pct": UNIT, "reason": TEXT})
    # Both pulses are already in the payload; no extra retrieval needed.
    SEARCH = None

    def facts(self, payload: dict[str, Any]) -> dict[str, Any]:
        return payload

    def postprocess(self, data: dict[str, Any]) -> dict[str, Any]:
        try:
            crypto = float(data["crypto_pct"])
            stocks = float(data["stocks_pct"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"allocator returned no usable split: {data}") from exc

        crypto = max(0.0, crypto)
        stocks = max(0.0, stocks)
        total = crypto + stocks
        if total <= 0:
            raise ValueError("allocator returned a zero split")

        return {
            "crypto_pct": crypto / total,
            "stocks_pct": stocks / total,
            "reason": str(data.get("reason", "")),
        }

    def fallback(self) -> dict[str, Any]:
        # No read means no tilt.
        return {"crypto_pct": 0.5, "stocks_pct": 0.5, "reason": "allocator_unavailable"}

    async def allocate(
        self,
        crypto_pulse: dict[str, Any],
        market_pulse: dict[str, Any],
        weekly_pnl: dict[str, float],
        risk: dict[str, Any] | None = None,
    ) -> Allocation:
        """Run the bot and return an Allocation already clamped to the ceilings."""
        result = await self.run(
            {
                "crypto_pulse": crypto_pulse,
                "market_pulse": market_pulse,
                "weekly_pnl_usd": weekly_pnl,
            }
        )
        risk = risk or (self.config.get("risk", {}) or {})
        allocation = Allocation(
            crypto_pct=result["crypto_pct"],
            stocks_pct=result["stocks_pct"],
            reason=result.get("reason", ""),
        )
        return allocation.normalized(
            crypto_max_pct=float(risk.get("crypto_max_pct", 1.0)),
            stock_max_pct=float(risk.get("stock_max_pct", 1.0)),
        )

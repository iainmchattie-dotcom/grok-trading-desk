"""Bot 13 — exit manager (grok-4-fast).

Runs every 4 hours over every open position on both markets. Four verbs only:
HOLD, TIGHTEN, TRIM, CLOSE. On any failure it returns HOLD — an unreadable model
must never be the reason a position gets touched.
"""

from __future__ import annotations

from typing import Any

from ..base_agent import NUM, TEXT, UNIT, GrokAgent, clamp01, enum, schema
from ..models import ExitAction, Position

PROMPT = """You manage one open trading position. Decide what to do with it right now.

Choose exactly one action:
- HOLD: the thesis is intact and the position needs nothing.
- TIGHTEN: the thesis is intact but the trade has moved your way — raise the stop
  to protect the gain. Give the new stop as new_stop_pct, a fraction below the
  CURRENT price.
- TRIM: take part of the position off and let the rest run. Give trim_fraction,
  the fraction of the position to sell (0 < f < 1).
- CLOSE: the thesis is broken, the target is reached, or the position has gone
  dead — exit fully.

Weigh: unrealised PnL, how long it has been held versus how long the thesis needed,
distance to the original target and stop, and whether the reason for entry still
holds. Do not close a working position out of impatience, and do not hold a broken
one out of hope.

Reply ONLY JSON, no explanation.
Schema: {"action": "HOLD"|"TIGHTEN"|"TRIM"|"CLOSE", "reason": string,
"confidence": float 0..1, "new_stop_pct": float, "trim_fraction": float}"""


class ExitManager(GrokAgent):
    name = "exit_manager"
    model_tier = "fast"
    PROMPT = PROMPT
    SCHEMA = schema(
        {
            "action": enum("HOLD", "TIGHTEN", "TRIM", "CLOSE"),
            "reason": TEXT,
            "confidence": UNIT,
            "new_stop_pct": NUM,
            "trim_fraction": NUM,
        }
    )
    # Whether the entry thesis still holds is a question about today's news.
    SEARCH = {
        "mode": "auto",
        "sources": [{"type": "news"}, {"type": "x", "post_view_count": 1000}],
        "max_search_results": 10,
    }

    def facts(self, payload: Position | dict[str, Any]) -> dict[str, Any]:
        position = payload if isinstance(payload, Position) else Position(**payload)
        return {
            "market": position.market.value,
            "symbol": position.symbol,
            "entry_price": position.entry_price,
            "current_price": position.current_price,
            "quantity": position.quantity,
            "amount_usd": position.amount_usd,
            "pnl_usd": round(position.pnl_usd, 2),
            "pnl_pct": round(position.pnl_pct, 4),
            "hold_time_hours": round(position.hold_time_hours, 2),
            "stop_price": position.stop_price,
            "take_profit_price": position.take_profit_price,
            "entry_score": position.score,
            "entry_meta": position.meta,
        }

    def memory_context(self, payload: Position | dict[str, Any]) -> dict[str, Any]:
        position = payload if isinstance(payload, Position) else Position(**payload)
        return self.memory.context(
            position.market, symbol=position.symbol, sector=position.sector
        )

    def postprocess(self, data: dict[str, Any]) -> dict[str, Any]:
        raw_action = str(data.get("action", "")).strip().upper()
        try:
            action = ExitAction(raw_action)
        except ValueError as exc:
            raise ValueError(f"unknown exit action {raw_action!r}") from exc

        def fraction(key: str, default: float) -> float:
            try:
                value = float(data.get(key))
            except (TypeError, ValueError):
                return default
            return value if 0.0 < value < 1.0 else default

        return {
            "action": action.value,
            "reason": str(data.get("reason", "")),
            "confidence": clamp01(data.get("confidence"), default=0.0),
            "new_stop_pct": fraction("new_stop_pct", 0.05),
            "trim_fraction": fraction("trim_fraction", 0.5),
        }

    def fallback(self) -> dict[str, Any]:
        # Never let a broken model be the reason a position is touched.
        return {
            "action": ExitAction.HOLD.value,
            "reason": "exit_manager_unavailable",
            "confidence": 0.0,
            "new_stop_pct": 0.05,
            "trim_fraction": 0.5,
        }

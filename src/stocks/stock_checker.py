"""Bot 12 — adversarial stock checker (grok-4, the stronger model)."""

from __future__ import annotations

from typing import Any

from ..base_agent import BOOL, NUM, TEXT, UNIT, GrokAgent, clamp01, schema, string_list

PROMPT = """You are the final adversarial reviewer before real money buys this stock.

The analyst, radar and insider bots want to buy. Argue the other side: the move is
already exhausted and you are buying the top tick, the catalyst is priced in, the
gap will fill by lunch, the fundamentals do not support the multiple, a known event
lands inside the holding window, or the score rests on one optimistic sub-bot.

Check the entry too: is the proposed stop wide enough to survive normal noise, and
does the target sit under an obvious resistance level?

Approve only when you cannot construct a plausible losing path over the intended
holding period.

Reply ONLY JSON, no explanation.
Schema: {"approve": bool, "confidence": float 0..1, "kill_reasons": [string],
"worst_case": string, "adjusted_score": float 0..1,
"suggested_stop_pct": float, "suggested_target_pct": float}"""


class StockChecker(GrokAgent):
    name = "stock_checker"
    model_tier = "deep"
    PROMPT = PROMPT
    SCHEMA = schema(
        {
            "approve": BOOL,
            "confidence": UNIT,
            "kill_reasons": string_list(),
            "worst_case": TEXT,
            "adjusted_score": UNIT,
            "suggested_stop_pct": NUM,
            "suggested_target_pct": NUM,
        }
    )
    SEARCH = {
        "mode": "auto",
        "sources": [{"type": "news"}, {"type": "web"}],
        "max_search_results": 15,
    }

    def facts(self, payload: dict[str, Any]) -> dict[str, Any]:
        return payload

    def postprocess(self, data: dict[str, Any]) -> dict[str, Any]:
        def pct(key: str, default: float) -> float:
            try:
                value = float(data.get(key))
            except (TypeError, ValueError):
                return default
            return value if 0.0 < value < 1.0 else default

        return {
            "approve": bool(data.get("approve", False)),
            "confidence": clamp01(data.get("confidence"), default=0.0),
            "kill_reasons": list(data.get("kill_reasons") or []),
            "worst_case": str(data.get("worst_case", "")),
            "adjusted_score": clamp01(data.get("adjusted_score"), default=0.0),
            "suggested_stop_pct": pct("suggested_stop_pct", 0.08),
            "suggested_target_pct": pct("suggested_target_pct", 0.20),
        }

    def fallback(self) -> dict[str, Any]:
        return {
            "approve": False,
            "confidence": 0.0,
            "kill_reasons": ["checker_unavailable"],
            "worst_case": "no adversarial review was possible",
            "adjusted_score": 0.0,
            "suggested_stop_pct": 0.08,
            "suggested_target_pct": 0.20,
        }

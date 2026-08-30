"""Bot 11 — adversarial crypto checker (grok-4, the stronger model).

Last gate before money moves. It is told to argue against the trade: the cost of
a false "approve" is the whole position, the cost of a false "reject" is a missed
launch, and there is another launch in a minute.
"""

from __future__ import annotations

from typing import Any

from ..base_agent import BOOL, TEXT, UNIT, GrokAgent, clamp01, schema, string_list

PROMPT = """You are the final adversarial reviewer before real money buys this token.

The scout, auditor and narrative bots want to buy. Your job is to argue the other
side and find the reason this is a losing trade: a rug setup the audit missed,
concentration that will dump on the buy, a narrative that is already exhausted,
liquidity too thin to exit, or a score inflated by one optimistic sub-bot.

Approve only if you cannot construct a plausible way this loses most of its value
within the hour. A missed launch costs nothing; there is another in a minute.

Reply ONLY JSON, no explanation.
Schema: {"approve": bool, "confidence": float 0..1, "kill_reasons": [string],
"worst_case": string, "adjusted_score": float 0..1}"""


class CryptoChecker(GrokAgent):
    name = "crypto_checker"
    model_tier = "deep"
    PROMPT = PROMPT
    SCHEMA = schema(
        {
            "approve": BOOL,
            "confidence": UNIT,
            "kill_reasons": string_list(),
            "worst_case": TEXT,
            "adjusted_score": UNIT,
        }
    )
    # The checker gets its own look at the evidence rather than trusting the
    # generators' summary of it.
    SEARCH = {
        "mode": "auto",
        "sources": [{"type": "x", "post_view_count": 500}, {"type": "web"}],
        "max_search_results": 15,
    }

    def facts(self, payload: dict[str, Any]) -> dict[str, Any]:
        return payload

    def memory_context(self, payload: dict[str, Any]) -> dict[str, Any]:
        from ..models import Market

        token = payload.get("token") or {}
        theme = (payload.get("narrative") or {}).get("theme", "")
        return self.memory.context(Market.CRYPTO, symbol=token.get("symbol", ""), theme=theme)

    def postprocess(self, data: dict[str, Any]) -> dict[str, Any]:
        return {
            "approve": bool(data.get("approve", False)),
            "confidence": clamp01(data.get("confidence"), default=0.0),
            "kill_reasons": list(data.get("kill_reasons") or []),
            "worst_case": str(data.get("worst_case", "")),
            "adjusted_score": clamp01(data.get("adjusted_score"), default=0.0),
        }

    def fallback(self) -> dict[str, Any]:
        # An unreachable checker is a rejection.
        return {
            "approve": False,
            "confidence": 0.0,
            "kill_reasons": ["checker_unavailable"],
            "worst_case": "no adversarial review was possible",
            "adjusted_score": 0.0,
        }

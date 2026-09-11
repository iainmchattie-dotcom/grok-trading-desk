"""Bot 11 — adversarial crypto checker (grok-4, the stronger model).

Last gate before money moves. It is told to argue against the trade: the cost of
a false "approve" is the whole position, the cost of a false "reject" is a missed
launch, and there is another launch in a minute.

`hard_reject` is the evidence bar (honeypot / rug / wash / hostile mint). Paper
trading only skips on that bar; `checker_unavailable` and soft rejects become a
score penalty so the desk can actually fill. Live still fail-closes on any
non-approve. See `crypto_scoring.apply_crypto_checker`.
"""

from __future__ import annotations

from typing import Any

from ..base_agent import BOOL, TEXT, UNIT, GrokAgent, clamp01, schema, string_list

PROMPT = """You are the final adversarial reviewer before this token is bought.

The scout, auditor and narrative bots want to buy. Argue the other side: a rug
the audit missed, concentration that dumps on the buy, a narrative that is
already exhausted, liquidity too thin to exit, or a score inflated by one
optimistic sub-bot.

Split your verdict:
- hard_reject: True ONLY with concrete evidence of an unsurvivable trap —
  honeypot / cannot sell, mint or freeze authority retained by a hostile party,
  documented wash trading, or a known scam mint. Hypothetical dumps, thin
  liquidity, a crowded tape, an exhausted meme, or an inflated score are NOT
  hard_reject.
- approve: True when you cannot make a hard_reject case and the trade is not an
  obvious loser. False when you have concerns. False plus hard_reject false is
  an advisory / soft reject, not a proven scam.
- kill_reasons: short evidence labels. Use "honeypot", "rug pull",
  "wash trading", "mint authority" only when those are the actual finding.

A missed launch is cheap on paper; a false hard_reject of a clean tape teaches
the desk nothing. Reserve hard_reject for evidence, not vibes.

Reply ONLY JSON, no explanation.
Schema: {"approve": bool, "hard_reject": bool, "confidence": float 0..1,
"kill_reasons": [string], "worst_case": string, "adjusted_score": float 0..1}"""


class CryptoChecker(GrokAgent):
    name = "crypto_checker"
    model_tier = "deep"
    PROMPT = PROMPT
    SCHEMA = schema(
        {
            "approve": BOOL,
            "hard_reject": BOOL,
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
            "hard_reject": bool(data.get("hard_reject", False)),
            "confidence": clamp01(data.get("confidence"), default=0.0),
            "kill_reasons": list(data.get("kill_reasons") or []),
            "worst_case": str(data.get("worst_case", "")),
            "adjusted_score": clamp01(data.get("adjusted_score"), default=0.0),
        }

    def fallback(self) -> dict[str, Any]:
        # Unreachable is not evidence of a scam. Paper treats this as advisory;
        # live still fail-closes because approve is false.
        return {
            "approve": False,
            "hard_reject": False,
            "confidence": 0.0,
            "kill_reasons": ["checker_unavailable"],
            "worst_case": "no adversarial review was possible",
            "adjusted_score": 0.0,
        }

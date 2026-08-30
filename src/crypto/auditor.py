"""Bot 2 — wallet auditor (grok-4-fast).

Looks at who is holding and who is trading. Its job is to catch the two things
that make a launch unsurvivable: a coordinated buy ring and wash trading. Both
are hard vetoes upstream, so a parse failure here must read as "yes, both".
"""

from __future__ import annotations

from typing import Any

from ..base_agent import GrokAgent, clamp01
from ..models import Token

PROMPT = """You audit Solana pump.fun token launches for wallet-level manipulation.

Given the token's holder distribution and recent trade flow, judge:
- coordinated_buys: are many wallets funded from a common source buying in lockstep?
- wash_trading: is the same capital cycling between related wallets to fake volume?
- bundled_launch: was supply sniped by the deployer's own wallets in the first blocks?
- sniper_pct: fraction of supply held by first-block snipers (0..1)
- insider_pct: fraction held by wallets linked to the deployer (0..1)
- safety_score: 0..1, where 1 is a clean organic launch and 0 is an outright trap
- red_flags: short list of strings naming what you found

Be strict. When the evidence is ambiguous, assume manipulation.

Reply ONLY JSON, no explanation.
Schema: {"coordinated_buys": bool, "wash_trading": bool, "bundled_launch": bool,
"sniper_pct": float, "insider_pct": float, "safety_score": float, "red_flags": [string]}"""


class Auditor(GrokAgent):
    name = "auditor"
    model_tier = "fast"
    PROMPT = PROMPT

    def build_prompt(self, payload: Token | dict[str, Any]) -> str:
        token = payload if isinstance(payload, Token) else Token(**payload)
        facts = {
            "mint": token.mint,
            "symbol": token.symbol,
            "creator": token.creator,
            "age_seconds": token.age_seconds,
            "holders": token.holders,
            "top10_holder_pct": token.top10_holder_pct,
            "dev_holding_pct": token.dev_holding_pct,
            "liquidity_usd": token.liquidity_usd,
            "market_cap_usd": token.market_cap_usd,
            "buys": token.buys,
            "sells": token.sells,
            "buy_sell_ratio": round(token.buy_sell_ratio, 3),
            "mint_revoked": token.mint_revoked,
            "lp_burned": token.lp_burned,
        }
        return f"{self.PROMPT}\n\nTOKEN:\n{facts}"

    def postprocess(self, data: dict[str, Any]) -> dict[str, Any]:
        return {
            "coordinated_buys": bool(data.get("coordinated_buys", True)),
            "wash_trading": bool(data.get("wash_trading", True)),
            "bundled_launch": bool(data.get("bundled_launch", True)),
            "sniper_pct": clamp01(data.get("sniper_pct"), default=1.0),
            "insider_pct": clamp01(data.get("insider_pct"), default=1.0),
            "safety_score": clamp01(data.get("safety_score"), default=0.0),
            "red_flags": list(data.get("red_flags") or []),
        }

    def fallback(self) -> dict[str, Any]:
        # An unreadable audit is a failed audit: veto everything.
        return {
            "coordinated_buys": True,
            "wash_trading": True,
            "bundled_launch": True,
            "sniper_pct": 1.0,
            "insider_pct": 1.0,
            "safety_score": 0.0,
            "red_flags": ["audit_unavailable"],
        }

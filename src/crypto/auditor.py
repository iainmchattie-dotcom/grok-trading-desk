"""Bot 2 — wallet auditor (grok-4-fast).

Looks at who is holding and who is trading. Its job is to catch the two things
that make a launch unsurvivable: a coordinated buy ring and wash trading.

Wash trading is a hard veto upstream. Coordinated buys is only a hard veto when
the watch-window tape corroborates a ring (see crypto_scoring.manipulation_evidence);
an isolated flag on thin data becomes a score penalty. A parse/API failure here
must still read as "yes, both" — we did not audit, so we do not buy.
"""

from __future__ import annotations

from typing import Any

from ..base_agent import BOOL, TEXT, UNIT, GrokAgent, clamp01, schema, string_list
from ..models import Token

PROMPT = """You audit Solana pump.fun token launches for wallet-level manipulation.

You are given watch-window trade stats, NOT a funding graph and NOT an on-chain
holder census. Pump.fun launches in the first 1–5 minutes normally look clustered:
many independent wallets buy the same new ticker at once. That is the product,
not a ring.

Judge from the facts you were given:
- coordinated_buys: True ONLY with evidence that the SAME entity is buying
  through multiple wallets (common funding, lockstep size from a tiny trader
  set, a known bundler). Temporal clustering of independent buys is NOT
  coordinated_buys. If holder concentration is unknown (holder_concentration_known
  is false / top10_holder_pct is null) AND unique_traders_per_buy is not well
  below 0.4, answer false.
- wash_trading: True ONLY if the same capital is cycling — the same small set
  of wallets both buying AND selling to fake volume. A buy-heavy tape with
  many unique traders is not wash trading.
- bundled_launch: True if the deployer's own wallets sniped supply in the
  first blocks.
- sniper_pct: fraction of supply held by first-block snipers (0..1). Use 0
  when you cannot estimate.
- insider_pct: fraction held by wallets linked to the deployer (0..1). Use 0
  when you cannot estimate.
- safety_score: 0..1, where 1 is a clean organic launch and 0 is an outright
  trap. Do not zero this just because the launch is young or buys arrived
  together.
- red_flags: short list of strings naming what you actually found. Omit guesses.

When evidence is missing or thin, leave coordinated_buys and wash_trading false
and lower safety_score modestly rather than inventing a ring. Ambiguity is not proof of manipulation.

Reply ONLY JSON, no explanation.
Schema: {"coordinated_buys": bool, "wash_trading": bool, "bundled_launch": bool,
"sniper_pct": float, "insider_pct": float, "safety_score": float, "red_flags": [string]}"""


class Auditor(GrokAgent):
    name = "auditor"
    model_tier = "fast"
    PROMPT = PROMPT
    SCHEMA = schema(
        {
            "coordinated_buys": BOOL,
            "wash_trading": BOOL,
            "bundled_launch": BOOL,
            "sniper_pct": UNIT,
            "insider_pct": UNIT,
            "safety_score": UNIT,
            "red_flags": string_list(),
        }
    )
    # Deployer wallets and rug post-mortems surface on X long before anywhere
    # else. Engagement floor keeps out the bot replies that quote every mint.
    SEARCH = {
        "mode": "auto",
        "sources": [{"type": "x", "post_view_count": 500}, {"type": "web"}],
        "max_search_results": 10,
    }

    def facts(self, payload: Token | dict[str, Any]) -> dict[str, Any]:
        token = payload if isinstance(payload, Token) else Token(**payload)
        buys = token.buys
        traders = token.unique_traders or token.holders
        concentration_known = token.top10_holder_pct > 0
        dev_known = token.dev_holding_pct > 0
        return {
            "mint": token.mint,
            "symbol": token.symbol,
            "creator": token.creator,
            "age_seconds": token.age_seconds,
            "holders": token.holders,
            "unique_traders": token.unique_traders,
            "unique_traders_per_buy": round(traders / buys, 3) if buys and traders else None,
            "top10_holder_pct": token.top10_holder_pct if concentration_known else None,
            "dev_holding_pct": token.dev_holding_pct if dev_known else None,
            "holder_concentration_known": concentration_known,
            "dev_holding_known": dev_known,
            "liquidity_usd": token.liquidity_usd,
            "market_cap_usd": token.market_cap_usd,
            "buys": token.buys,
            "sells": token.sells,
            "buy_sell_ratio": round(token.buy_sell_ratio, 3),
            "mint_revoked": token.mint_revoked,
            "lp_burned": token.lp_burned,
            "observed_seconds": token.observed_seconds,
            "data_limitations": [
                "holders is unique_traders from the watch window, not an on-chain holder census",
                "top10_holder_pct and dev_holding_pct are unknown when null — not zero concentration",
                "no funding-graph or related-wallet data is available from this feed",
                "clustered buys in a short observation window are the normal pump.fun tape",
            ],
        }

    def postprocess(self, data: dict[str, Any]) -> dict[str, Any]:
        # Missing booleans stay pessimistic: an incomplete object is closer to
        # a failed audit than a clean bill of health. Scoring still requires
        # watch-window corroboration before coordinated_buys becomes a hard veto.
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
        # An unreadable audit is a failed audit: veto everything. Scoring keys
        # off audit_unavailable / red_flags, not the boolean flags alone, so
        # the skip reason is distinguishable from a real coordinated ring.
        return {
            "coordinated_buys": True,
            "wash_trading": True,
            "bundled_launch": True,
            "sniper_pct": 1.0,
            "insider_pct": 1.0,
            "safety_score": 0.0,
            "red_flags": ["audit_unavailable"],
            "audit_unavailable": True,
        }

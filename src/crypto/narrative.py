"""Bot 3 — narrative / meme potential (grok-4-fast).

A clean contract with no story goes nowhere. This bot rates whether the ticker
has anything a crowd can repeat.
"""

from __future__ import annotations

from typing import Any

from ..base_agent import BOOL, TEXT, UNIT, GrokAgent, clamp01, schema
from ..models import Token

PROMPT = """You judge the meme potential of a newly launched Solana token.

Rate how likely this name, ticker and social footprint is to be picked up and
repeated by crypto Twitter in the next few hours. Consider: is the reference
current, is the ticker memorable and typeable, does the art/branding read as
effort or as a template, is there an existing community behind the reference,
and is the name already a crowded copy of a running meme.

Score conservatively: most launches are noise.

Reply ONLY JSON, no explanation.
Schema: {"meme_score": float 0..1, "originality": float 0..1, "virality": float 0..1,
"community_signal": float 0..1, "is_derivative": bool, "theme": string, "reasoning": string}"""


class Narrative(GrokAgent):
    name = "narrative"
    model_tier = "fast"
    PROMPT = PROMPT
    SCHEMA = schema(
        {
            "meme_score": UNIT,
            "originality": UNIT,
            "virality": UNIT,
            "community_signal": UNIT,
            "is_derivative": BOOL,
            "theme": TEXT,
            "reasoning": TEXT,
        }
    )
    # Whether a meme is running *right now* is only answerable from X, and only
    # from the last day or two.
    SEARCH = {
        "mode": "on",
        "sources": [{"type": "x", "post_view_count": 1000}],
        "max_search_results": 15,
    }

    def facts(self, payload: Token | dict[str, Any]) -> dict[str, Any]:
        token = payload if isinstance(payload, Token) else Token(**payload)
        return {
            "symbol": token.symbol,
            "name": token.name,
            "socials": token.socials,
            "holders": token.holders,
            "age_seconds": token.age_seconds,
            "market_cap_usd": token.market_cap_usd,
        }

    def postprocess(self, data: dict[str, Any]) -> dict[str, Any]:
        return {
            "meme_score": clamp01(data.get("meme_score"), default=0.0),
            "originality": clamp01(data.get("originality"), default=0.0),
            "virality": clamp01(data.get("virality"), default=0.0),
            "community_signal": clamp01(data.get("community_signal"), default=0.0),
            "is_derivative": bool(data.get("is_derivative", True)),
            "theme": str(data.get("theme", "unknown")),
            "reasoning": str(data.get("reasoning", "")),
        }

    def fallback(self) -> dict[str, Any]:
        # No story we can read is no story worth buying.
        return {
            "meme_score": 0.0,
            "originality": 0.0,
            "virality": 0.0,
            "community_signal": 0.0,
            "is_derivative": True,
            "theme": "unknown",
            "reasoning": "narrative_unavailable",
        }

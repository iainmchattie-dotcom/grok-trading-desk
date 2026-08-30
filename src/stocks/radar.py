"""Bot 6 — news and sentiment radar (grok-4-fast).

`controversy` is a hard veto upstream, so it must default high on failure.
"""

from __future__ import annotations

from typing import Any

from ..base_agent import GrokAgent, clamp01
from ..models import Stock

PROMPT = """You scan recent news and social sentiment for one US equity.

Cover the last two weeks: earnings and guidance, analyst actions, product and
contract news, regulatory or legal exposure, short reports, executive departures,
and the tone of retail and professional chatter.

controversy means active reputational or legal danger — fraud allegations, an
SEC action, a credible short report, an accounting question. It is a veto, so
only score it low when the name is genuinely clean.

Reply ONLY JSON, no explanation.
Schema: {"sentiment_score": float 0..1, "news_momentum": float 0..1,
"controversy": float 0..1, "catalysts": [string], "headline_risk": string,
"summary": string}"""


class Radar(GrokAgent):
    name = "radar"
    model_tier = "fast"
    PROMPT = PROMPT

    def build_prompt(self, payload: Stock | dict[str, Any]) -> str:
        stock = payload if isinstance(payload, Stock) else Stock(**payload)
        facts = {
            "symbol": stock.symbol,
            "name": stock.name,
            "sector": stock.sector,
            "price": stock.price,
            "gap_pct": round(stock.gap_pct, 4),
        }
        return f"{self.PROMPT}\n\nSTOCK:\n{facts}"

    def postprocess(self, data: dict[str, Any]) -> dict[str, Any]:
        return {
            "sentiment_score": clamp01(data.get("sentiment_score"), default=0.0),
            "news_momentum": clamp01(data.get("news_momentum"), default=0.0),
            "controversy": clamp01(data.get("controversy"), default=1.0),
            "catalysts": list(data.get("catalysts") or []),
            "headline_risk": str(data.get("headline_risk", "")),
            "summary": str(data.get("summary", "")),
        }

    def fallback(self) -> dict[str, Any]:
        # controversy=1.0 trips the hard veto, which is the point.
        return {
            "sentiment_score": 0.0,
            "news_momentum": 0.0,
            "controversy": 1.0,
            "catalysts": [],
            "headline_risk": "radar_unavailable",
            "summary": "radar_unavailable",
        }

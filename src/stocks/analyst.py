"""Bot 5 — fundamentals + technicals (grok-4-fast)."""

from __future__ import annotations

from typing import Any

from ..base_agent import NUM, TEXT, UNIT, GrokAgent, clamp01, enum, schema, string_list
from ..models import Stock

PROMPT = """You analyse one US equity on both fundamentals and price action.

Fundamentals: revenue growth and its direction, margins, balance-sheet strength,
cash burn versus runway, valuation against its own history and its peers, and
whether the last report changed the story.

Technicals: trend across daily and weekly, position relative to the 20/50/200
moving averages, volume behind the current move, distance to obvious support and
resistance, and whether today's move is extended.

Reply ONLY JSON, no explanation.
Schema: {"fundamentals_score": float 0..1, "technicals_score": float 0..1,
"trend": "up"|"sideways"|"down", "support": float, "resistance": float,
"valuation": "cheap"|"fair"|"expensive", "thesis": string, "risks": [string]}"""


class Analyst(GrokAgent):
    name = "analyst"
    model_tier = "fast"
    PROMPT = PROMPT
    SCHEMA = schema(
        {
            "fundamentals_score": UNIT,
            "technicals_score": UNIT,
            "trend": enum("up", "sideways", "down"),
            "support": NUM,
            "resistance": NUM,
            "valuation": enum("cheap", "fair", "expensive"),
            "thesis": TEXT,
            "risks": string_list(),
        }
    )
    # Fundamentals live in filings and financial press, not on X.
    SEARCH = {
        "mode": "on",
        "sources": [
            {"type": "web", "allowed_websites": ["sec.gov", "reuters.com", "bloomberg.com",
                                                 "finance.yahoo.com", "marketwatch.com"]},
            {"type": "news"},
        ],
        "max_search_results": 20,
    }

    def facts(self, payload: Stock | dict[str, Any]) -> dict[str, Any]:
        stock = payload if isinstance(payload, Stock) else Stock(**payload)
        return {
            "symbol": stock.symbol,
            "name": stock.name,
            "sector": stock.sector,
            "price": stock.price,
            "prev_close": stock.prev_close,
            "gap_pct": round(stock.gap_pct, 4),
            "volume": stock.volume,
            "avg_volume": stock.avg_volume,
            "rel_volume": round(stock.rel_volume, 3),
            "market_cap": stock.market_cap,
        }

    def postprocess(self, data: dict[str, Any]) -> dict[str, Any]:
        def num(key: str) -> float:
            try:
                return float(data.get(key) or 0.0)
            except (TypeError, ValueError):
                return 0.0

        return {
            "fundamentals_score": clamp01(data.get("fundamentals_score"), default=0.0),
            "technicals_score": clamp01(data.get("technicals_score"), default=0.0),
            "trend": str(data.get("trend", "sideways")),
            "support": num("support"),
            "resistance": num("resistance"),
            "valuation": str(data.get("valuation", "expensive")),
            "thesis": str(data.get("thesis", "")),
            "risks": list(data.get("risks") or []),
        }

    def fallback(self) -> dict[str, Any]:
        return {
            "fundamentals_score": 0.0,
            "technicals_score": 0.0,
            "trend": "sideways",
            "support": 0.0,
            "resistance": 0.0,
            "valuation": "expensive",
            "thesis": "",
            "risks": ["analyst_unavailable"],
        }

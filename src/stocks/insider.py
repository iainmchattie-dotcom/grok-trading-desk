"""Bot 7 — insider and institutional flow (grok-4-fast).

Reads SEC Form 4 filings and 13F changes. Heavy selling with no buying is a hard
veto upstream, so the fallback has to look exactly like that.
"""

from __future__ import annotations

from typing import Any

from ..base_agent import GrokAgent, clamp01
from ..models import Stock

PROMPT = """You read insider and institutional flow for one US equity.

Form 4: recent open-market purchases and sales by officers and directors. Weight
open-market buys heavily and discount scheduled 10b5-1 sales, option exercises
and tax withholding.

13F: how institutional holders changed position last quarter, and whether the
holder base is concentrating or dispersing.

Reply ONLY JSON, no explanation.
Schema: {"insider_buying": float 0..1, "insider_selling": float 0..1,
"institutional_flow": float 0..1, "cluster_buying": bool, "notable": [string],
"summary": string}"""


class Insider(GrokAgent):
    name = "insider"
    model_tier = "fast"
    PROMPT = PROMPT

    def build_prompt(self, payload: Stock | dict[str, Any]) -> str:
        stock = payload if isinstance(payload, Stock) else Stock(**payload)
        facts = {
            "symbol": stock.symbol,
            "name": stock.name,
            "sector": stock.sector,
            "market_cap": stock.market_cap,
        }
        return f"{self.PROMPT}\n\nSTOCK:\n{facts}"

    def postprocess(self, data: dict[str, Any]) -> dict[str, Any]:
        return {
            "insider_buying": clamp01(data.get("insider_buying"), default=0.0),
            "insider_selling": clamp01(data.get("insider_selling"), default=1.0),
            "institutional_flow": clamp01(data.get("institutional_flow"), default=0.0),
            "cluster_buying": bool(data.get("cluster_buying", False)),
            "notable": list(data.get("notable") or []),
            "summary": str(data.get("summary", "")),
        }

    def fallback(self) -> dict[str, Any]:
        # selling=1.0 with buying=0.0 trips the hard veto.
        return {
            "insider_buying": 0.0,
            "insider_selling": 1.0,
            "institutional_flow": 0.0,
            "cluster_buying": False,
            "notable": [],
            "summary": "insider_unavailable",
        }

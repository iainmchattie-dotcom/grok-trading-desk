"""Bot 9 — equity market regime (grok-4-fast), cached 30 minutes.

The tape's character changes slowly; the screener produces candidates in bursts.
Caching keeps the regime read to a couple of calls a session.
"""

from __future__ import annotations

import time
from typing import Any

from ..base_agent import GrokAgent, clamp01

PROMPT = """You assess the current regime of the US equity market.

Judge: index trend and breadth, VIX level and direction, rate and dollar backdrop,
sector rotation and which side of the market is being paid, and the macro calendar
in the next 48 hours.

go_signal is the single number the desk acts on: 0 means open no new equity
positions today, 1 means conditions are as good as they get.

Reply ONLY JSON, no explanation.
Schema: {"regime": "risk_on"|"neutral"|"risk_off", "go_signal": float 0..1,
"risk_appetite": float 0..1, "volatility": "low"|"normal"|"high",
"leading_sectors": [string], "notes": string}"""


class MarketPulse(GrokAgent):
    name = "market_pulse"
    model_tier = "fast"
    PROMPT = PROMPT

    def __init__(self, config: dict[str, Any], client=None):
        super().__init__(config, client)
        minutes = ((config or {}).get("pulse", {}) or {}).get("market_cache_minutes", 30)
        self.cache_seconds = float(minutes) * 60.0
        self._cache: dict[str, Any] | None = None
        self._cache_time: float = 0.0

    def build_prompt(self, payload: Any = None) -> str:
        return f"{self.PROMPT}\n\nCONTEXT:\n{payload or {}}"

    def postprocess(self, data: dict[str, Any]) -> dict[str, Any]:
        regime = str(data.get("regime", "risk_off")).lower()
        if regime not in {"risk_on", "neutral", "risk_off"}:
            regime = "risk_off"
        return {
            "regime": regime,
            "go_signal": clamp01(data.get("go_signal"), default=0.0),
            "risk_appetite": clamp01(data.get("risk_appetite"), default=0.0),
            "volatility": str(data.get("volatility", "high")),
            "leading_sectors": list(data.get("leading_sectors") or []),
            "notes": str(data.get("notes", "")),
        }

    def fallback(self) -> dict[str, Any]:
        return {
            "regime": "risk_off",
            "go_signal": 0.0,
            "risk_appetite": 0.0,
            "volatility": "high",
            "leading_sectors": [],
            "notes": "market_pulse_unavailable",
        }

    def cache_is_fresh(self, now: float | None = None) -> bool:
        if self._cache is None:
            return False
        now = time.time() if now is None else now
        return (now - self._cache_time) < self.cache_seconds

    async def run(self, payload: Any = None) -> dict[str, Any]:
        if self.cache_is_fresh():
            return dict(self._cache)  # type: ignore[arg-type]
        result = await super().run(payload)
        self._cache = result
        self._cache_time = time.time()
        return result

    def invalidate(self) -> None:
        self._cache = None
        self._cache_time = 0.0

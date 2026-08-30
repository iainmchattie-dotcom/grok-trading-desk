"""Bot 8 — crypto regime (grok-4-fast), cached 15 minutes.

The regime moves far slower than the launch feed, so re-asking on every token
would burn dozens of calls a day for the same answer.
"""

from __future__ import annotations

import time
from datetime import date, timedelta
from typing import Any

from ..base_agent import TEXT, UNIT, GrokAgent, clamp01, enum, schema

PROMPT = """You assess the current risk regime of the Solana memecoin market.

Judge overall conditions right now: SOL price trend, launch volume and survival
rate on pump.fun, how much fresh capital is rotating into new launches versus
sitting out, and whether the tape is rewarding risk or punishing it.

go_signal is the single number the desk acts on: 0 means do not open any new
memecoin position at all, 1 means conditions are as good as they get.

Reply ONLY JSON, no explanation.
Schema: {"regime": "risk_on"|"neutral"|"risk_off", "go_signal": float 0..1,
"risk_appetite": float 0..1, "sol_trend": "up"|"flat"|"down", "notes": string}"""


class CryptoPulse(GrokAgent):
    name = "crypto_pulse"
    model_tier = "fast"
    PROMPT = PROMPT
    SCHEMA = schema(
        {
            "regime": enum("risk_on", "neutral", "risk_off"),
            "go_signal": UNIT,
            "risk_appetite": UNIT,
            "sol_trend": enum("up", "flat", "down"),
            "notes": TEXT,
        }
    )
    # A regime read without today's data is just a guess about February.
    SEARCH = {
        "mode": "on",
        "sources": [{"type": "x", "post_view_count": 2000}, {"type": "news"}, {"type": "web"}],
        "max_search_results": 20,
    }

    def __init__(self, config: dict[str, Any], client=None):
        super().__init__(config, client)
        minutes = ((config or {}).get("pulse", {}) or {}).get("crypto_cache_minutes", 15)
        self.cache_seconds = float(minutes) * 60.0
        self._cache: dict[str, Any] | None = None
        self._cache_time: float = 0.0

    def facts(self, payload: Any = None) -> Any:
        return payload or {"question": "current Solana memecoin regime"}

    def search_parameters(self) -> dict[str, Any] | None:
        params = super().search_parameters()
        if params is not None:
            # Yesterday onward: a regime read must not average in last week.
            params["from_date"] = (date.today() - timedelta(days=1)).isoformat()
        return params

    def postprocess(self, data: dict[str, Any]) -> dict[str, Any]:
        regime = str(data.get("regime", "risk_off")).lower()
        if regime not in {"risk_on", "neutral", "risk_off"}:
            regime = "risk_off"
        return {
            "regime": regime,
            "go_signal": clamp01(data.get("go_signal"), default=0.0),
            "risk_appetite": clamp01(data.get("risk_appetite"), default=0.0),
            "sol_trend": str(data.get("sol_trend", "flat")),
            "notes": str(data.get("notes", "")),
        }

    def fallback(self) -> dict[str, Any]:
        # No read on the regime means the market gate stays shut.
        return {
            "regime": "risk_off",
            "go_signal": 0.0,
            "risk_appetite": 0.0,
            "sol_trend": "flat",
            "notes": "crypto_pulse_unavailable",
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

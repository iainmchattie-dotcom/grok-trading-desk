"""Bot 1 — pump.fun scout. Pure code, no LLM.

Streams new launches over WebSocket and yields only the tokens that clear the
metric filter. Everything downstream costs API calls, so this gate is deliberately
strict.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from typing import Any

from ..models import Token

log = logging.getLogger(__name__)


def parse_token(payload: dict[str, Any]) -> Token:
    """Map a pump.fun event onto our Token model, tolerating key aliases."""

    def pick(*keys: str, default: Any = None) -> Any:
        for key in keys:
            if key in payload and payload[key] is not None:
                return payload[key]
        return default

    return Token(
        mint=str(pick("mint", "mintAddress", "ca", default="")),
        symbol=str(pick("symbol", "ticker", default="")),
        name=str(pick("name", default="")),
        creator=str(pick("creator", "traderPublicKey", "dev", default="")),
        liquidity_usd=float(pick("liquidity_usd", "vSolInBondingCurve_usd", "liquidity", default=0) or 0),
        market_cap_usd=float(pick("market_cap_usd", "marketCapSol_usd", "usd_market_cap", default=0) or 0),
        holders=int(pick("holders", "holder_count", default=0) or 0),
        top10_holder_pct=float(pick("top10_holder_pct", "top10", default=0) or 0),
        dev_holding_pct=float(pick("dev_holding_pct", "dev_pct", default=0) or 0),
        age_seconds=float(pick("age_seconds", "age", default=0) or 0),
        buys=int(pick("buys", "buy_count", default=0) or 0),
        sells=int(pick("sells", "sell_count", default=0) or 0),
        mint_revoked=bool(pick("mint_revoked", "mintAuthorityRevoked", default=False)),
        lp_burned=bool(pick("lp_burned", "lpBurned", default=False)),
        socials={
            k: str(v)
            for k, v in (payload.get("socials") or {}).items()
            if v
        }
        or {k: str(payload[k]) for k in ("twitter", "telegram", "website") if payload.get(k)},
        raw=payload,
    )


def filter_reason(token: Token, filt: dict[str, Any]) -> str | None:
    """Return the reason the token is rejected, or None if it passes."""
    if not token.mint:
        return "no_mint"

    min_liq = filt.get("min_liquidity_usd")
    if min_liq is not None and token.liquidity_usd < min_liq:
        return "liquidity_too_low"

    max_liq = filt.get("max_liquidity_usd")
    if max_liq is not None and token.liquidity_usd > max_liq:
        return "liquidity_too_high"

    min_holders = filt.get("min_holders")
    if min_holders is not None and token.holders < min_holders:
        return "too_few_holders"

    max_top10 = filt.get("max_top10_holder_pct")
    if max_top10 is not None and token.top10_holder_pct > max_top10:
        return "top10_concentration"

    max_dev = filt.get("max_dev_holding_pct")
    if max_dev is not None and token.dev_holding_pct > max_dev:
        return "dev_holding_too_high"

    min_age = filt.get("min_age_seconds")
    if min_age is not None and token.age_seconds < min_age:
        return "too_young"

    max_age = filt.get("max_age_seconds")
    if max_age is not None and token.age_seconds > max_age:
        return "too_old"

    min_buys = filt.get("min_buys")
    if min_buys is not None and token.buys < min_buys:
        return "too_few_buys"

    min_ratio = filt.get("min_buy_sell_ratio")
    if min_ratio is not None and token.buy_sell_ratio < min_ratio:
        return "weak_buy_sell_ratio"

    if filt.get("require_mint_revoked") and not token.mint_revoked:
        return "mint_not_revoked"

    if filt.get("require_lp_burned") and not token.lp_burned:
        return "lp_not_burned"

    return None


def passes_filter(token: Token, filt: dict[str, Any]) -> bool:
    return filter_reason(token, filt) is None


class Scout:
    """Bot 1. Owns the WebSocket subscription and the pre-filter."""

    name = "scout"

    def __init__(self, config: dict[str, Any], connect=None):
        self.config = config or {}
        self.filter = self.config.get("crypto_filter", {}) or {}
        self.ws_url = (self.config.get("pump_fun", {}) or {}).get(
            "ws_url", "wss://pumpportal.fun/api/data"
        )
        self._connect = connect
        self.seen: set[str] = set()

    def _connector(self):
        if self._connect is not None:
            return self._connect
        import websockets  # imported lazily so tests never need the dependency

        return websockets.connect

    async def stream(self, reconnect_delay: float = 5.0) -> AsyncIterator[Token]:
        """Yield filtered, de-duplicated tokens forever, reconnecting on drop."""
        connect = self._connector()
        while True:
            try:
                async with connect(self.ws_url) as ws:
                    await ws.send(json.dumps({"method": "subscribeNewToken"}))
                    async for message in ws:
                        token = self.handle_message(message)
                        if token is not None:
                            yield token
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - a dropped socket is routine
                log.warning("scout socket error, reconnecting in %ss: %s", reconnect_delay, exc)
                await asyncio.sleep(reconnect_delay)

    def handle_message(self, message: str | bytes | dict[str, Any]) -> Token | None:
        """Decode one WS frame; return the token only if it clears the filter."""
        try:
            payload = message if isinstance(message, dict) else json.loads(message)
        except (json.JSONDecodeError, TypeError):
            return None
        if not isinstance(payload, dict):
            return None

        token = parse_token(payload)
        if not token.mint or token.mint in self.seen:
            return None

        reason = filter_reason(token, self.filter)
        if reason is not None:
            return None

        self.seen.add(token.mint)
        return token

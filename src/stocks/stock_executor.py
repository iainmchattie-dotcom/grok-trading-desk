"""Alpaca execution — the working one.

Paper by default. Live trading needs BOTH `mode: "live"` in the config AND the
--i-understand-the-risk flag on the command line; either alone keeps you on paper.
alpaca-py is imported lazily so the rest of the desk (and the test suite) runs
without the SDK installed.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from ..models import Market, Position

log = logging.getLogger(__name__)


class OrderRejected(Exception):
    """A broker refusal that is expected and diagnosable, not a bug.

    `reason` is a stable slug so the event log can be grouped by it.
    """

    def __init__(self, reason: str, detail: str = ""):
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


def classify_rejection(exc: Exception) -> str:
    """Map a broker error onto a stable skip reason.

    A 403 on submit is almost always the PDT guard: Alpaca blocks the order
    rather than let an account under $25k equity get flagged. Reporting that as
    a generic failure hides a limit the desk should respect.
    """
    status = getattr(exc, "status_code", None) or getattr(
        getattr(exc, "response", None), "status_code", None
    )
    text = str(exc).lower()

    if status == 403 or "pattern day" in text or "day trad" in text:
        return "pdt_blocked"
    if "wash trade" in text:
        return "wash_trade_blocked"
    if "insufficient" in text or "buying power" in text:
        return "insufficient_buying_power"
    if "not tradable" in text or "asset is not active" in text:
        return "asset_not_tradable"
    if "market is closed" in text or "outside" in text:
        return "market_closed"
    if status == 429:
        return "broker_rate_limited"
    return "order_rejected"


class StockExecutor:
    """Bracket-order execution against Alpaca."""

    market = Market.STOCKS

    def __init__(self, config: dict[str, Any], client: Any = None, live_ack: bool = False):
        self.config = config or {}
        alpaca = self.config.get("alpaca", {}) or {}
        self.api_key = alpaca.get("api_key", "")
        self.api_secret = alpaca.get("api_secret", "")

        wants_live = str(self.config.get("mode", "paper")).lower() == "live"
        #: paper unless the config says live AND the operator passed the flag
        self.paper = not (wants_live and live_ack)
        if wants_live and not live_ack:
            log.warning(
                "config asks for live trading but --i-understand-the-risk was not "
                "passed; staying on paper"
            )

        self._client = client

    # -- SDK plumbing ---------------------------------------------------------------

    @property
    def client(self) -> Any:
        if self._client is None:
            from alpaca.trading.client import TradingClient

            self._client = TradingClient(
                api_key=self.api_key, secret_key=self.api_secret, paper=self.paper
            )
        return self._client

    @staticmethod
    async def _call(func, *args, **kwargs):
        """Run a blocking SDK call off the event loop."""
        return await asyncio.to_thread(func, *args, **kwargs)

    # -- orders ------------------------------------------------------------------------

    async def buy_bracket(
        self,
        symbol: str,
        amount_usd: float,
        price: float,
        stop_pct: float = 0.08,
        target_pct: float = 0.20,
    ) -> dict[str, Any]:
        """Market buy with an attached stop-loss and take-profit."""
        from alpaca.trading.enums import OrderClass, OrderSide, TimeInForce
        from alpaca.trading.requests import (
            MarketOrderRequest,
            StopLossRequest,
            TakeProfitRequest,
        )

        if price <= 0:
            raise ValueError(f"{symbol}: cannot size an order at price {price}")

        # Bracket orders cannot be fractional, so this floor is a hard rule and
        # not a rounding convenience.
        qty = int(amount_usd // price)
        if qty < 1:
            return {"filled": False, "reason": "amount_below_one_share", "symbol": symbol}

        take_profit = round(price * (1 + target_pct), 2)
        stop_loss = round(price * (1 - stop_pct), 2)

        request = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
            order_class=OrderClass.BRACKET,
            take_profit=TakeProfitRequest(limit_price=take_profit),
            stop_loss=StopLossRequest(stop_price=stop_loss),
        )
        try:
            order = await self._call(self.client.submit_order, request)
        except Exception as exc:  # noqa: BLE001 - classified, then re-raised
            reason = classify_rejection(exc)
            log.warning("bracket buy %s rejected (%s): %s", symbol, reason, exc)
            raise OrderRejected(reason, str(exc)) from exc

        log.info("submitted bracket buy %s x%d @ ~%.2f", symbol, qty, price)
        return {
            "filled": True,
            "symbol": symbol,
            "qty": qty,
            "order_id": str(getattr(order, "id", "")),
            "entry_price": price,
            "stop_price": stop_loss,
            "take_profit_price": take_profit,
            "amount_usd": round(qty * price, 2),
            "paper": self.paper,
        }

    async def tighten_stop(self, order_id: str, new_stop_price: float) -> dict[str, Any]:
        """Move the stop leg of an existing bracket up."""
        from alpaca.trading.requests import ReplaceOrderRequest

        order = await self._call(
            self.client.replace_order_by_id,
            order_id,
            ReplaceOrderRequest(stop_price=round(new_stop_price, 2)),
        )
        return {"order_id": str(getattr(order, "id", order_id)), "stop_price": round(new_stop_price, 2)}

    async def sell_partial(self, symbol: str, qty: float) -> dict[str, Any]:
        """Sell part of a position at market, leaving the rest open."""
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest

        shares = int(qty)
        if shares < 1:
            return {"filled": False, "reason": "qty_below_one_share", "symbol": symbol}

        try:
            order = await self._call(
                self.client.submit_order,
                MarketOrderRequest(
                    symbol=symbol,
                    qty=shares,
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.DAY,
                ),
            )
        except Exception as exc:  # noqa: BLE001
            raise OrderRejected(classify_rejection(exc), str(exc)) from exc

        return {"filled": True, "symbol": symbol, "qty": shares, "order_id": str(getattr(order, "id", ""))}

    async def close_position(self, symbol: str) -> dict[str, Any]:
        """Flatten the whole position."""
        result = await self._call(self.client.close_position, symbol)
        return {"closed": True, "symbol": symbol, "order_id": str(getattr(result, "id", ""))}

    async def get_positions(self) -> list[Position]:
        """All open equity positions, as our Position model."""
        raw = await self._call(self.client.get_all_positions)
        positions: list[Position] = []
        for item in raw or []:
            qty = float(getattr(item, "qty", 0) or 0)
            entry = float(getattr(item, "avg_entry_price", 0) or 0)
            current = float(getattr(item, "current_price", 0) or 0)
            positions.append(
                Position(
                    market=Market.STOCKS,
                    symbol=str(getattr(item, "symbol", "")),
                    quantity=qty,
                    entry_price=entry,
                    current_price=current,
                    amount_usd=round(qty * entry, 2),
                    meta={"asset_id": str(getattr(item, "asset_id", ""))},
                )
            )
        return positions

"""Grok Trading Desk — the orchestrator.

Four concurrent asyncio loops, one shared risk manager, one event log:

  crypto_loop     continuous, 24/7, driven by the pump.fun WebSocket
  stock_loop      wakes every minute, works only inside the RTH window
  exit_loop       every 4 hours over every open position on both books
  allocator_loop  once a day, resets the crypto/stocks budget split

Run:  python -m src.desk --config config.yaml [--dry-run] [--i-understand-the-risk]
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from datetime import date, datetime, time as dtime, timedelta, timezone
from pathlib import Path
from typing import Any

import yaml

from .crypto.auditor import Auditor
from .crypto.crypto_checker import CryptoChecker
from .crypto.crypto_executor import CryptoExecutor
from .crypto.crypto_pulse import CryptoPulse
from .crypto.crypto_scoring import score_token
from .crypto.narrative import Narrative
from .crypto.scout import Scout
from .models import Allocation, Market, Position
from .shared.allocator import Allocator
from .shared.exit_manager import ExitManager
from .shared.log import EventLog
from .shared.risk import RiskManager
from .stocks.analyst import Analyst
from .stocks.insider import Insider
from .stocks.market_pulse import MarketPulse
from .stocks.radar import Radar
from .stocks.screener import Screener
from .stocks.stock_checker import StockChecker
from .stocks.stock_executor import StockExecutor
from .stocks.stock_scoring import score_stock

log = logging.getLogger("desk")


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _parse_hhmm(value: str, default: dtime) -> dtime:
    try:
        hour, minute = (int(part) for part in str(value).split(":", 1))
        return dtime(hour, minute)
    except (TypeError, ValueError):
        return default


class TradingDesk:
    """Owns every bot, the shared risk state and the four loops."""

    def __init__(self, config: dict[str, Any], dry_run: bool = False, live_ack: bool = False):
        self.config = config
        self.dry_run = dry_run

        self.log = EventLog(config)
        self.risk = RiskManager(config)

        # crypto side
        self.scout = Scout(config)
        self.auditor = Auditor(config)
        self.narrative = Narrative(config)
        self.crypto_pulse = CryptoPulse(config)
        self.crypto_checker = CryptoChecker(config)
        self.crypto_executor = CryptoExecutor(config)

        # stock side
        self.screener = Screener(config)
        self.analyst = Analyst(config)
        self.radar = Radar(config)
        self.insider = Insider(config)
        self.market_pulse = MarketPulse(config)
        self.stock_checker = StockChecker(config)
        self.stock_executor = StockExecutor(config, live_ack=live_ack)

        # shared
        self.allocator = Allocator(config)
        self.exit_manager = ExitManager(config)

        self.positions: list[Position] = []
        self.min_go_signal = float((config.get("pulse", {}) or {}).get("min_go_signal", 0.3))
        self.weights = config.get("scoring_weights", {}) or {}
        self.exits_cfg = config.get("exits", {}) or {}
        self.last_stock_session: date | None = None
        self._lock = asyncio.Lock()

    # -- helpers -------------------------------------------------------------------

    def _now_et(self) -> datetime:
        """Current New York time. Falls back to a fixed UTC-4 if tzdata is absent."""
        tz_name = (self.config.get("market_hours", {}) or {}).get("timezone", "America/New_York")
        try:
            from zoneinfo import ZoneInfo

            return datetime.now(ZoneInfo(tz_name))
        except Exception:  # noqa: BLE001 - missing tzdata must not stop the desk
            return datetime.now(timezone.utc) - timedelta(hours=4)

    def market_is_open(self, now: datetime | None = None) -> bool:
        hours = self.config.get("market_hours", {}) or {}
        now = now or self._now_et()
        if now.weekday() >= 5:
            return False
        open_at = _parse_hhmm(hours.get("open", "09:35"), dtime(9, 35))
        close_at = _parse_hhmm(hours.get("close", "15:55"), dtime(15, 55))
        return open_at <= now.time() <= close_at

    # -- crypto loop ----------------------------------------------------------------

    async def evaluate_token(self, token) -> dict[str, Any]:
        """Audit + narrative + pulse -> score -> adversarial check -> buy or skip."""
        pulse, audit, narrative = await asyncio.gather(
            self.crypto_pulse.run(),
            self.auditor.run(token),
            self.narrative.run(token),
        )

        verdict = score_token(
            token,
            audit,
            narrative,
            pulse,
            weights=self.weights.get("crypto"),
            min_go_signal=self.min_go_signal,
        )
        agent_scores = {"audit": audit, "narrative": narrative, "pulse": pulse, "matrix": verdict}

        if not verdict["buy"]:
            self.log.skip(Market.CRYPTO.value, token.symbol or token.mint, verdict["reason"],
                          {"score": verdict["score"]})
            return {"bought": False, "reason": verdict["reason"]}

        check = await self.crypto_checker.run(
            {"token": token.model_dump(mode="json"), "audit": audit,
             "narrative": narrative, "pulse": pulse, "score": verdict}
        )
        agent_scores["checker"] = check
        if not check["approve"]:
            self.log.skip(Market.CRYPTO.value, token.symbol or token.mint, "checker_rejected",
                          {"kill_reasons": check["kill_reasons"]})
            return {"bought": False, "reason": "checker_rejected"}

        return await self._open_crypto(token, verdict, check, agent_scores)

    async def _open_crypto(self, token, verdict, check, agent_scores) -> dict[str, Any]:
        async with self._lock:
            allowed, reason = self.risk.can_open(Market.CRYPTO, self.positions)
            if not allowed:
                self.log.skip(Market.CRYPTO.value, token.symbol or token.mint, reason)
                return {"bought": False, "reason": reason}

            amount = self.risk.position_size(Market.CRYPTO, score=check["adjusted_score"] or verdict["score"])
            if amount <= 0:
                self.log.skip(Market.CRYPTO.value, token.symbol or token.mint, "size_zero")
                return {"bought": False, "reason": "size_zero"}

            if self.dry_run:
                self.log.buy(Market.CRYPTO.value, token.symbol or token.mint, verdict["score"],
                             agent_scores, amount, tx_id="DRY_RUN")
                return {"bought": True, "dry_run": True, "amount": amount}

            try:
                fill = await self.crypto_executor.buy(token.mint, amount)
            except NotImplementedError as exc:
                # the stub is expected until the owner wires signing
                self.log.skip(Market.CRYPTO.value, token.symbol or token.mint,
                              "executor_not_implemented", str(exc))
                return {"bought": False, "reason": "executor_not_implemented"}

            self.risk.record_fill(Market.CRYPTO, amount)
            self.positions.append(
                Position(
                    market=Market.CRYPTO,
                    symbol=token.symbol or token.mint,
                    quantity=float(fill.get("quantity", 0)),
                    entry_price=float(fill.get("price", 0)),
                    amount_usd=amount,
                    score=verdict["score"],
                    meta={"mint": token.mint, "tx_id": fill.get("tx_id", "")},
                )
            )
            self.log.buy(Market.CRYPTO.value, token.symbol or token.mint, verdict["score"],
                         agent_scores, amount, tx_id=str(fill.get("tx_id", "")))
            return {"bought": True, "amount": amount, "tx_id": fill.get("tx_id", "")}

    async def crypto_loop(self) -> None:
        log.info("crypto loop: streaming pump.fun")
        while True:
            try:
                async for token in self.scout.stream():
                    self.risk.maybe_reset_day()
                    await self.evaluate_token(token)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - a loop must not die on one bad token
                log.exception("crypto loop error, restarting in 10s")
                await asyncio.sleep(10)

    # -- stock loop -----------------------------------------------------------------

    async def evaluate_stock(self, stock, pulse: dict[str, Any]) -> dict[str, Any]:
        analyst, radar, insider = await asyncio.gather(
            self.analyst.run(stock),
            self.radar.run(stock),
            self.insider.run(stock),
        )

        verdict = score_stock(
            stock, analyst, radar, insider, pulse,
            weights=self.weights.get("stocks"),
            min_go_signal=self.min_go_signal,
        )
        agent_scores = {"analyst": analyst, "radar": radar, "insider": insider,
                        "pulse": pulse, "matrix": verdict}

        if not verdict["buy"]:
            self.log.skip(Market.STOCKS.value, stock.symbol, verdict["reason"],
                          {"score": verdict["score"]})
            return {"bought": False, "reason": verdict["reason"]}

        check = await self.stock_checker.run(
            {"stock": stock.model_dump(mode="json"), "analyst": analyst, "radar": radar,
             "insider": insider, "pulse": pulse, "score": verdict}
        )
        agent_scores["checker"] = check
        if not check["approve"]:
            self.log.skip(Market.STOCKS.value, stock.symbol, "checker_rejected",
                          {"kill_reasons": check["kill_reasons"]})
            return {"bought": False, "reason": "checker_rejected"}

        return await self._open_stock(stock, verdict, check, agent_scores)

    async def _open_stock(self, stock, verdict, check, agent_scores) -> dict[str, Any]:
        async with self._lock:
            allowed, reason = self.risk.can_open(Market.STOCKS, self.positions, sector=stock.sector)
            if not allowed:
                self.log.skip(Market.STOCKS.value, stock.symbol, reason)
                return {"bought": False, "reason": reason}

            amount = self.risk.position_size(Market.STOCKS, score=check["adjusted_score"] or verdict["score"])
            if amount <= 0:
                self.log.skip(Market.STOCKS.value, stock.symbol, "size_zero")
                return {"bought": False, "reason": "size_zero"}

            if self.dry_run:
                self.log.buy(Market.STOCKS.value, stock.symbol, verdict["score"],
                             agent_scores, amount, tx_id="DRY_RUN")
                return {"bought": True, "dry_run": True, "amount": amount}

            fill = await self.stock_executor.buy_bracket(
                stock.symbol,
                amount,
                stock.price,
                stop_pct=check["suggested_stop_pct"],
                target_pct=check["suggested_target_pct"],
            )
            if not fill.get("filled"):
                self.log.skip(Market.STOCKS.value, stock.symbol, fill.get("reason", "not_filled"))
                return {"bought": False, "reason": fill.get("reason", "not_filled")}

            self.risk.record_fill(Market.STOCKS, fill["amount_usd"])
            self.positions.append(
                Position(
                    market=Market.STOCKS,
                    symbol=stock.symbol,
                    quantity=fill["qty"],
                    entry_price=fill["entry_price"],
                    current_price=fill["entry_price"],
                    amount_usd=fill["amount_usd"],
                    stop_price=fill["stop_price"],
                    take_profit_price=fill["take_profit_price"],
                    sector=stock.sector,
                    score=verdict["score"],
                    meta={"order_id": fill["order_id"]},
                )
            )
            self.log.buy(Market.STOCKS.value, stock.symbol, verdict["score"],
                         agent_scores, fill["amount_usd"], tx_id=fill["order_id"])
            return {"bought": True, "amount": fill["amount_usd"], "order_id": fill["order_id"]}

    async def run_stock_session(self) -> list[dict[str, Any]]:
        """One pass of the equity workflow: pulse -> screen -> evaluate."""
        pulse = await self.market_pulse.run()
        if pulse["go_signal"] < self.min_go_signal:
            self.log.skip(Market.STOCKS.value, "*", "veto_market_paused",
                          {"go_signal": pulse["go_signal"]})
            return []

        candidates = await self.screener.run()
        log.info("stock session: %d candidates cleared the screener", len(candidates))
        return [await self.evaluate_stock(stock, pulse) for stock in candidates]

    async def stock_loop(self, poll_seconds: float = 60.0) -> None:
        log.info("stock loop: polling for the RTH window")
        while True:
            try:
                self.risk.maybe_reset_day()
                today = self._now_et().date()
                if self.market_is_open() and self.last_stock_session != today:
                    self.last_stock_session = today
                    await self.run_stock_session()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                log.exception("stock loop error")
            await asyncio.sleep(poll_seconds)

    # -- exit loop --------------------------------------------------------------------

    async def refresh_positions(self) -> list[Position]:
        """Merge broker truth into our view. Crypto stays desk-side while the
        executor is a stub."""
        try:
            live_stocks = await self.stock_executor.get_positions()
        except Exception as exc:  # noqa: BLE001 - a broker hiccup must not clear the book
            log.warning("could not refresh stock positions: %s", exc)
            return self.positions

        by_symbol = {p.symbol: p for p in self.positions if p.market == Market.STOCKS}
        merged: list[Position] = [p for p in self.positions if p.market == Market.CRYPTO]
        for live in live_stocks:
            known = by_symbol.get(live.symbol)
            if known is not None:
                known.quantity = live.quantity
                known.current_price = live.current_price
                merged.append(known)
            else:
                merged.append(live)
        self.positions = merged
        return self.positions

    async def manage_position(self, position: Position) -> dict[str, Any]:
        decision = await self.exit_manager.run(position)
        action = decision["action"]
        self.log.action(position.symbol, action, decision["reason"],
                        market=position.market.value, pnl_pct=round(position.pnl_pct, 4))

        if action == "HOLD" or self.dry_run:
            return decision

        executor = (
            self.crypto_executor if position.market == Market.CRYPTO else self.stock_executor
        )
        try:
            if action == "TIGHTEN":
                new_stop = position.current_price * (1 - decision["new_stop_pct"])
                if position.market == Market.STOCKS:
                    await executor.tighten_stop(position.meta.get("order_id", ""), new_stop)
                else:
                    await executor.tighten_stop(position.meta.get("mint", ""), new_stop)
                position.stop_price = new_stop

            elif action == "TRIM":
                fraction = decision["trim_fraction"]
                if position.market == Market.STOCKS:
                    await executor.sell_partial(position.symbol, position.quantity * fraction)
                else:
                    await executor.sell(position.meta.get("mint", ""), fraction)
                position.quantity *= 1 - fraction
                position.amount_usd *= 1 - fraction

            elif action == "CLOSE":
                target = (
                    position.symbol
                    if position.market == Market.STOCKS
                    else position.meta.get("mint", "")
                )
                await executor.close_position(target)
                self.risk.record_close(position.market, position.pnl_usd, position.amount_usd)
                self.log.close(position.market.value, position.symbol,
                               round(position.pnl_usd, 2), round(position.hold_time_hours, 2))
                self.positions = [p for p in self.positions if p is not position]

        except NotImplementedError as exc:
            self.log.skip(position.market.value, position.symbol,
                          "executor_not_implemented", str(exc))
        except Exception as exc:  # noqa: BLE001
            log.exception("failed to %s %s", action, position.symbol)
            self.log.skip(position.market.value, position.symbol, "action_failed", str(exc))

        return decision

    async def run_exit_pass(self) -> list[dict[str, Any]]:
        positions = await self.refresh_positions()
        log.info("exit pass over %d positions", len(positions))
        return [await self.manage_position(p) for p in list(positions)]

    async def exit_loop(self) -> None:
        interval = float(self.exits_cfg.get("interval_hours", 4)) * 3600
        log.info("exit loop: every %.1f h", interval / 3600)
        while True:
            await asyncio.sleep(interval)
            try:
                await self.run_exit_pass()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                log.exception("exit loop error")

    # -- allocator loop -----------------------------------------------------------------

    def weekly_pnl(self) -> dict[str, float]:
        """Realised PnL per market over the trailing seven days, from the log."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=7)
        totals = {"crypto": 0.0, "stocks": 0.0}
        for record in self.log.read():
            if record.get("type") != "close":
                continue
            try:
                when = datetime.fromisoformat(record["ts"])
            except (KeyError, ValueError):
                continue
            if when.tzinfo is None:
                when = when.replace(tzinfo=timezone.utc)
            if when < cutoff:
                continue
            market = record.get("market", "")
            if market in totals:
                totals[market] += float(record.get("pnl", 0) or 0)
        return totals

    async def run_allocation(self) -> Allocation:
        crypto_pulse, market_pulse = await asyncio.gather(
            self.crypto_pulse.run(), self.market_pulse.run()
        )
        allocation = await self.allocator.allocate(
            crypto_pulse, market_pulse, self.weekly_pnl(), risk=self.config.get("risk")
        )
        applied = self.risk.set_allocation(allocation)
        self.log.allocation(round(applied.crypto_pct, 4), round(applied.stocks_pct, 4),
                            allocation.reason)
        return applied

    async def allocator_loop(self, interval_seconds: float = 86400.0) -> None:
        log.info("allocator loop: daily")
        while True:
            try:
                await self.run_allocation()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                log.exception("allocator loop error")
            await asyncio.sleep(interval_seconds)

    # -- entry point ---------------------------------------------------------------------

    async def run(self) -> None:
        log.info(
            "desk starting — dry_run=%s, stock execution=%s",
            self.dry_run,
            "paper" if self.stock_executor.paper else "LIVE",
        )
        await asyncio.gather(
            self.crypto_loop(),
            self.stock_loop(),
            self.exit_loop(),
            self.allocator_loop(),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Grok Trading Desk")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--dry-run", action="store_true", help="decide and log, never execute")
    parser.add_argument(
        "--i-understand-the-risk",
        action="store_true",
        dest="live_ack",
        help='required, together with mode: "live" in the config, to leave paper trading',
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    desk = TradingDesk(load_config(args.config), dry_run=args.dry_run, live_ack=args.live_ack)
    try:
        asyncio.run(desk.run())
    except KeyboardInterrupt:
        log.info("desk stopped")


if __name__ == "__main__":
    main()

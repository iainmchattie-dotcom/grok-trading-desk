from datetime import date, timedelta

import pytest

from src.models import Allocation, Market, Position
from src.shared.risk import RiskManager

CONFIG = {
    "risk": {
        "total_budget_usd": 2000.0,
        "daily_loss_limit_usd": 300.0,
        "max_open_total": 5,
        "max_open_crypto": 3,
        "max_open_stocks": 3,
        "max_per_sector": 2,
        "crypto_max_pct": 0.7,
        "stock_max_pct": 0.7,
        "max_position_pct_of_market": 0.15,
        "max_position_pct_of_remaining_loss": 0.25,
    }
}


def rm() -> RiskManager:
    return RiskManager(CONFIG)


def pos(market=Market.CRYPTO, symbol="X", sector="unknown") -> Position:
    return Position(market=market, symbol=symbol, quantity=1, entry_price=1.0, sector=sector)


# --- allocation ---------------------------------------------------------------

def test_default_allocation_is_even():
    r = rm()
    assert r.market_budget(Market.CRYPTO) == 1000.0
    assert r.market_budget(Market.STOCKS) == 1000.0


def test_allocation_is_clamped_to_the_ceiling():
    r = rm()
    r.set_allocation(Allocation(crypto_pct=1.0, stocks_pct=0.0))
    # crypto_max_pct=0.7, stocks clamp to 0 -> renormalized back to 1.0 crypto
    assert r.allocation.crypto_pct == pytest.approx(1.0)

    r.set_allocation(Allocation(crypto_pct=0.9, stocks_pct=0.1))
    assert r.allocation.crypto_pct == pytest.approx(0.7 / 0.8)
    assert r.market_budget(Market.CRYPTO) == pytest.approx(2000 * 0.7 / 0.8)


def test_allocation_shifts_the_budgets():
    r = rm()
    r.set_allocation(Allocation(crypto_pct=0.3, stocks_pct=0.7))
    assert r.market_budget(Market.CRYPTO) == pytest.approx(600.0)
    assert r.market_budget(Market.STOCKS) == pytest.approx(1400.0)


# --- open limits ---------------------------------------------------------------

def test_open_is_allowed_on_an_empty_book():
    assert rm().can_open(Market.CRYPTO, []) == (True, "ok")


def test_max_open_total_counts_both_markets():
    r = rm()
    positions = [pos(Market.CRYPTO)] * 2 + [pos(Market.STOCKS)] * 3
    assert len(positions) == 5
    assert r.can_open(Market.STOCKS, positions) == (False, "max_open_total")


def test_per_market_caps_are_separate():
    r = rm()
    crypto_full = [pos(Market.CRYPTO) for _ in range(3)]
    assert r.can_open(Market.CRYPTO, crypto_full) == (False, "max_open_crypto")
    # stocks still have room even though crypto is full
    assert r.can_open(Market.STOCKS, crypto_full) == (True, "ok")

    stocks_full = [pos(Market.STOCKS) for _ in range(3)]
    assert r.can_open(Market.STOCKS, stocks_full) == (False, "max_open_stocks")


def test_sector_cap_applies_to_stocks_only():
    r = rm()
    two_tech = [pos(Market.STOCKS, "A", "tech"), pos(Market.STOCKS, "B", "tech")]
    assert r.can_open(Market.STOCKS, two_tech, sector="tech") == (False, "max_per_sector")
    assert r.can_open(Market.STOCKS, two_tech, sector="energy") == (True, "ok")
    # unknown sector is never capped
    assert r.can_open(Market.STOCKS, two_tech, sector="unknown") == (True, "ok")


def test_crypto_ignores_the_sector_cap():
    r = rm()
    two = [pos(Market.CRYPTO, "A", "tech"), pos(Market.CRYPTO, "B", "tech")]
    assert r.can_open(Market.CRYPTO, two, sector="tech") == (True, "ok")


# --- daily loss ----------------------------------------------------------------

def test_daily_loss_limit_shuts_both_markets():
    r = rm()
    r.record_close(Market.CRYPTO, -300.0)
    assert r.daily_loss_breached() is True
    assert r.can_open(Market.CRYPTO, []) == (False, "daily_loss_limit_reached")
    assert r.can_open(Market.STOCKS, []) == (False, "daily_loss_limit_reached")


def test_crypto_losses_count_against_the_shared_limit():
    r = rm()
    r.record_close(Market.CRYPTO, -250.0)
    assert r.remaining_loss_room() == pytest.approx(50.0)
    assert r.can_open(Market.STOCKS, [])[0] is True

    r.record_close(Market.STOCKS, -60.0)
    assert r.daily_loss_breached() is True


def test_profits_do_not_inflate_the_loss_room():
    r = rm()
    r.record_close(Market.STOCKS, 500.0)
    assert r.remaining_loss_room() == pytest.approx(300.0)


def test_daily_reset_clears_pnl_and_deployment():
    r = rm()
    r.record_close(Market.CRYPTO, -290.0)
    r.record_fill(Market.CRYPTO, 400.0)
    assert r.maybe_reset_day(r.session_date) is False   # same day, no reset

    assert r.maybe_reset_day(date.today() + timedelta(days=1)) is True
    assert r.realized_pnl_today == 0.0
    assert r.deployed_usd[Market.CRYPTO] == 0.0
    assert r.can_open(Market.CRYPTO, [])[0] is True


# --- budget consumption ----------------------------------------------------------

def test_deployed_capital_exhausts_the_market_budget():
    r = rm()
    r.record_fill(Market.CRYPTO, 1000.0)
    assert r.remaining_market_budget(Market.CRYPTO) == 0.0
    assert r.can_open(Market.CRYPTO, []) == (False, "market_budget_exhausted")
    assert r.can_open(Market.STOCKS, []) == (True, "ok")


def test_an_oversized_order_is_rejected():
    r = rm()
    r.record_fill(Market.STOCKS, 900.0)
    assert r.can_open(Market.STOCKS, [], amount_usd=50.0) == (True, "ok")
    assert r.can_open(Market.STOCKS, [], amount_usd=150.0) == (False, "exceeds_market_budget")


def test_close_returns_budget_and_records_pnl():
    r = rm()
    r.record_fill(Market.CRYPTO, 500.0)
    r.record_close(Market.CRYPTO, 40.0, amount_usd=500.0)
    assert r.deployed_usd[Market.CRYPTO] == 0.0
    assert r.realized_pnl_today == pytest.approx(40.0)


# --- position sizing --------------------------------------------------------------

def test_position_size_respects_the_market_cap():
    r = rm()
    # 15% of a 1000 crypto budget = 150; loss room bound is 0.25*300 = 75 -> 75 binds
    assert r.position_size(Market.CRYPTO, score=1.0) == 75.0


def test_position_size_is_bound_by_remaining_loss_room():
    r = rm()
    r.record_close(Market.STOCKS, -200.0)   # 100 left -> 25 cap
    assert r.position_size(Market.STOCKS, score=1.0) == 25.0


def test_position_size_is_bound_by_free_budget():
    r = rm()
    r.record_fill(Market.CRYPTO, 990.0)     # only 10 free
    assert r.position_size(Market.CRYPTO, score=1.0) == 10.0


def test_position_size_scales_with_score():
    r = rm()
    full = r.position_size(Market.CRYPTO, score=1.0)
    half = r.position_size(Market.CRYPTO, score=0.0)
    assert half == pytest.approx(full * 0.5)
    assert r.position_size(Market.CRYPTO, score=0.5) == pytest.approx(full * 0.75)


def test_position_size_clamps_absurd_scores():
    r = rm()
    assert r.position_size(Market.CRYPTO, score=99) == r.position_size(Market.CRYPTO, score=1.0)
    assert r.position_size(Market.CRYPTO, score=-5) == r.position_size(Market.CRYPTO, score=0.0)


def test_position_size_is_zero_when_the_day_is_blown():
    r = rm()
    r.record_close(Market.CRYPTO, -400.0)
    assert r.position_size(Market.CRYPTO, score=1.0) == 0.0


def test_position_size_follows_the_allocation():
    r = rm()
    r.set_allocation(Allocation(crypto_pct=0.2, stocks_pct=0.8))
    # stocks clamp at the 0.7 ceiling, so the split renormalizes to 0.2/0.9 crypto
    assert r.allocation.crypto_pct == pytest.approx(0.2 / 0.9)
    crypto_budget = 2000 * 0.2 / 0.9
    # 15% of that budget (~66.7) is now tighter than the 75 loss-room bound
    assert r.position_size(Market.CRYPTO, score=1.0) == pytest.approx(
        round(crypto_budget * 0.15, 2)
    )


def test_snapshot_reports_both_books():
    r = rm()
    r.record_fill(Market.CRYPTO, 100.0)
    snap = r.snapshot([pos(Market.CRYPTO), pos(Market.STOCKS)])
    assert snap["open_positions"] == {"total": 2, "crypto": 1, "stocks": 1}
    assert snap["deployed"]["crypto"] == 100.0
    assert snap["daily_loss_breached"] is False

import pytest

from tests.conftest import CONFIG
from src.models import Allocation
from src.shared.allocator import Allocator

CRYPTO_PULSE = {"regime": "risk_on", "go_signal": 0.8}
MARKET_PULSE = {"regime": "neutral", "go_signal": 0.5}
PNL = {"crypto": 120.0, "stocks": -40.0}


async def test_allocator_parses_a_normal_split(client_factory):
    client = client_factory({"crypto_pct": 0.6, "stocks_pct": 0.4, "reason": "crypto regime is hot"})
    result = await Allocator(CONFIG, client=client).run({})
    assert result["crypto_pct"] == pytest.approx(0.6)
    assert result["stocks_pct"] == pytest.approx(0.4)
    assert result["reason"] == "crypto regime is hot"


async def test_allocator_renormalizes_a_split_that_does_not_sum_to_one(client_factory):
    client = client_factory({"crypto_pct": 0.8, "stocks_pct": 0.8})
    result = await Allocator(CONFIG, client=client).run({})
    assert result["crypto_pct"] == pytest.approx(0.5)
    assert result["stocks_pct"] == pytest.approx(0.5)


async def test_allocator_falls_back_to_fifty_fifty_on_broken_json(client_factory, no_sleep):
    result = await Allocator(CONFIG, client=client_factory("I think crypto, mostly")).run({})
    assert result == {"crypto_pct": 0.5, "stocks_pct": 0.5, "reason": "allocator_unavailable"}


async def test_allocator_falls_back_when_keys_are_missing(client_factory, no_sleep):
    result = await Allocator(CONFIG, client=client_factory({"reason": "dunno"})).run({})
    assert result["crypto_pct"] == 0.5
    assert result["reason"] == "allocator_unavailable"


async def test_allocator_falls_back_on_a_zero_split(client_factory, no_sleep):
    result = await Allocator(CONFIG, client=client_factory({"crypto_pct": 0, "stocks_pct": 0})).run({})
    assert result["crypto_pct"] == 0.5


async def test_allocate_clamps_to_the_configured_ceiling(client_factory):
    # model wants 95/5, but crypto_max_pct is 0.7
    client = client_factory({"crypto_pct": 0.95, "stocks_pct": 0.05, "reason": "all in"})
    allocation = await Allocator(CONFIG, client=client).allocate(CRYPTO_PULSE, MARKET_PULSE, PNL)
    assert isinstance(allocation, Allocation)
    assert allocation.crypto_pct == pytest.approx(0.7 / 0.75)
    assert allocation.stocks_pct == pytest.approx(0.05 / 0.75)
    assert allocation.crypto_pct + allocation.stocks_pct == pytest.approx(1.0)


async def test_allocate_leaves_an_in_range_split_alone(client_factory):
    client = client_factory({"crypto_pct": 0.55, "stocks_pct": 0.45})
    allocation = await Allocator(CONFIG, client=client).allocate(CRYPTO_PULSE, MARKET_PULSE, PNL)
    assert allocation.crypto_pct == pytest.approx(0.55)


async def test_allocate_allows_zeroing_one_market(client_factory):
    client = client_factory({"crypto_pct": 0.0, "stocks_pct": 1.0, "reason": "crypto shut"})
    allocation = await Allocator(CONFIG, client=client).allocate(CRYPTO_PULSE, MARKET_PULSE, PNL)
    assert allocation.crypto_pct == 0.0
    assert allocation.stocks_pct == pytest.approx(1.0)


async def test_allocate_sends_both_pulses_and_pnl_to_the_model(client_factory):
    client = client_factory({"crypto_pct": 0.5, "stocks_pct": 0.5})
    await Allocator(CONFIG, client=client).allocate(CRYPTO_PULSE, MARKET_PULSE, PNL)
    body = client.calls[0]["json"]
    # static instructions first (cacheable prefix), variable facts second
    assert "Reply ONLY JSON" in body["messages"][0]["content"]
    sent = body["messages"][1]["content"]
    assert "crypto_pulse" in sent and "market_pulse" in sent and "weekly_pnl_usd" in sent


def test_allocation_normalized_handles_a_degenerate_split():
    assert Allocation(crypto_pct=0.0, stocks_pct=0.0).normalized().crypto_pct == 0.5

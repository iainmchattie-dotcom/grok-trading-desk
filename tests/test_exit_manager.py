from datetime import datetime, timedelta, timezone

import pytest

from tests.conftest import CONFIG
from src.models import ExitAction, Market, Position
from src.shared.exit_manager import ExitManager

POSITION = Position(
    market=Market.STOCKS, symbol="ACME", quantity=10, entry_price=50.0,
    current_price=57.0, amount_usd=500.0, stop_price=46.0, take_profit_price=60.0,
    opened_at=datetime.now(timezone.utc) - timedelta(hours=6),
)


async def test_hold(client_factory):
    client = client_factory({"action": "HOLD", "reason": "thesis intact", "confidence": 0.7})
    result = await ExitManager(CONFIG, client=client).run(POSITION)
    assert result["action"] == ExitAction.HOLD.value
    assert result["reason"] == "thesis intact"


async def test_tighten_returns_a_new_stop(client_factory):
    client = client_factory({"action": "TIGHTEN", "new_stop_pct": 0.03, "confidence": 0.8})
    result = await ExitManager(CONFIG, client=client).run(POSITION)
    assert result["action"] == "TIGHTEN"
    assert result["new_stop_pct"] == 0.03


async def test_trim_returns_a_fraction(client_factory):
    client = client_factory({"action": "TRIM", "trim_fraction": 0.33, "reason": "de-risk"})
    result = await ExitManager(CONFIG, client=client).run(POSITION)
    assert result["action"] == "TRIM"
    assert result["trim_fraction"] == 0.33


async def test_close(client_factory):
    client = client_factory({"action": "CLOSE", "reason": "thesis broken", "confidence": 0.9})
    result = await ExitManager(CONFIG, client=client).run(POSITION)
    assert result["action"] == "CLOSE"


async def test_action_is_case_insensitive(client_factory):
    result = await ExitManager(CONFIG, client=client_factory({"action": " close "})).run(POSITION)
    assert result["action"] == "CLOSE"


async def test_unknown_action_falls_back_to_hold(client_factory, no_sleep):
    result = await ExitManager(CONFIG, client=client_factory({"action": "YOLO"})).run(POSITION)
    assert result["action"] == "HOLD"
    assert result["reason"] == "exit_manager_unavailable"


async def test_broken_json_falls_back_to_hold(client_factory, no_sleep):
    client = client_factory("close it, obviously")
    result = await ExitManager(CONFIG, client=client).run(POSITION)
    assert result["action"] == "HOLD"
    assert result["confidence"] == 0.0
    assert len(client.calls) == 3


async def test_out_of_range_fractions_are_replaced_with_defaults(client_factory):
    client = client_factory({"action": "TRIM", "trim_fraction": 5, "new_stop_pct": -0.2})
    result = await ExitManager(CONFIG, client=client).run(POSITION)
    assert result["trim_fraction"] == 0.5
    assert result["new_stop_pct"] == 0.05


async def test_prompt_carries_the_position_state(client_factory):
    client = client_factory({"action": "HOLD"})
    await ExitManager(CONFIG, client=client).run(POSITION)
    sent = client.calls[0]["json"]["messages"][1]["content"]
    assert "ACME" in sent and "pnl_pct" in sent and "hold_time_hours" in sent


async def test_works_on_a_crypto_position_too(client_factory):
    crypto = Position(market=Market.CRYPTO, symbol="WIF2", quantity=1e6, entry_price=0.0001, current_price=0.00025)
    result = await ExitManager(CONFIG, client=client_factory({"action": "TRIM", "trim_fraction": 0.5})).run(crypto)
    assert result["action"] == "TRIM"


def test_position_pnl_math():
    assert POSITION.pnl_usd == pytest.approx(70.0)
    assert POSITION.pnl_pct == pytest.approx(0.14)
    assert POSITION.hold_time_hours == pytest.approx(6.0, abs=0.01)
    flat = Position(market=Market.CRYPTO, symbol="X", quantity=1, entry_price=0.0)
    assert flat.pnl_usd == 0.0 and flat.pnl_pct == 0.0

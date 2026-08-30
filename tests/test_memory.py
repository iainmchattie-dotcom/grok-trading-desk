"""Outcome memory: pairing buys to closes, recall ranking, prompt injection."""

from datetime import datetime, timedelta, timezone

import pytest

from tests.conftest import CONFIG
from src.crypto.crypto_checker import CryptoChecker
from src.models import Market, Position
from src.shared.allocator import Allocator
from src.shared.exit_manager import ExitManager
from src.shared.memory import OutcomeMemory


def ts(days_ago: float = 0) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()


def buy(symbol, market="crypto", score=0.7, amount=100.0, theme="", sector="", days=1):
    return {
        "ts": ts(days), "type": "buy", "market": market, "symbol": symbol,
        "score": score, "amount": amount,
        "all_agent_scores": {"narrative": {"theme": theme}, "matrix": {"sector": sector}},
    }


def close(symbol, pnl, market="crypto", hold=5.0, days=0):
    return {"ts": ts(days), "type": "close", "market": market,
            "symbol": symbol, "pnl": pnl, "hold_time": hold}


LOG = [
    buy("DOGWIF", theme="dog", days=6), close("DOGWIF", 45.0, days=5),
    buy("CATWIF", theme="cat", days=5), close("CATWIF", -80.0, days=4),
    buy("PUPPY", theme="dog", days=4), close("PUPPY", -60.0, days=3),
    buy("ACME", market="stocks", sector="tech", amount=500.0, days=3),
    close("ACME", 120.0, market="stocks", hold=30.0, days=2),
    buy("BETA", market="stocks", sector="energy", amount=400.0, days=2),
    close("BETA", -40.0, market="stocks", hold=20.0, days=1),
]


def mem(**over) -> OutcomeMemory:
    config = {"memory": {"enabled": True, "max_examples": 5, "lookback_days": 30, **over}}
    m = OutcomeMemory(config)
    m.load(LOG)
    return m


# --- pairing ---------------------------------------------------------------------

def test_buys_are_joined_to_their_closes():
    trades = mem()._trades
    assert len(trades) == 5
    wif = next(t for t in trades if t["symbol"] == "DOGWIF")
    assert wif["pnl"] == 45.0
    assert wif["won"] is True
    assert wif["return_pct"] == pytest.approx(0.45)   # 45 on a 100 entry
    assert wif["theme"] == "dog"


def test_an_open_position_is_not_a_trade_yet():
    m = OutcomeMemory({})
    m.load([buy("OPEN")])
    assert m._trades == []


def test_a_close_without_its_buy_still_records_pnl():
    m = OutcomeMemory({})
    m.load([close("ORPHAN", -25.0)])
    assert m._trades[0]["pnl"] == -25.0
    assert m._trades[0]["return_pct"] == 0.0   # no entry amount to divide by


def test_lookback_window_is_enforced():
    m = OutcomeMemory({"memory": {"lookback_days": 2}})
    m.load(LOG)
    assert all(t["symbol"] not in {"DOGWIF", "CATWIF"} for t in m._trades)


def test_malformed_records_are_skipped_not_fatal():
    m = OutcomeMemory({})
    m.load([{"type": "close", "ts": "not-a-date", "symbol": "X", "market": "crypto", "pnl": 1}])
    assert len(m._trades) == 1     # unparseable timestamp is kept, not crashed on


# --- recall -----------------------------------------------------------------------

def test_recall_is_scoped_to_one_market():
    assert all(t["market"] == "crypto" for t in mem().recall(Market.CRYPTO))
    assert all(t["market"] == "stocks" for t in mem().recall(Market.STOCKS))


def test_the_same_symbol_ranks_above_the_same_theme():
    recalled = mem().recall(Market.CRYPTO, symbol="CATWIF", theme="dog")
    assert recalled[0]["symbol"] == "CATWIF"


def test_the_same_theme_outranks_an_unrelated_trade():
    recalled = mem().recall(Market.CRYPTO, theme="dog")
    assert {t["symbol"] for t in recalled[:2]} == {"DOGWIF", "PUPPY"}


def test_sector_matching_works_for_stocks():
    recalled = mem().recall(Market.STOCKS, sector="energy")
    assert recalled[0]["symbol"] == "BETA"


def test_recall_respects_max_examples():
    assert len(mem(max_examples=2).recall(Market.CRYPTO)) == 2


def test_recall_is_empty_when_disabled():
    assert mem(enabled=False).recall(Market.CRYPTO) == []


# --- summary -----------------------------------------------------------------------

def test_summary_reports_the_base_rate():
    summary = mem().summary(Market.CRYPTO)
    assert summary["closed_trades"] == 3
    assert summary["win_rate"] == pytest.approx(1 / 3, abs=0.01)
    assert summary["total_pnl"] == pytest.approx(-95.0)
    assert summary["avg_win"] == pytest.approx(45.0)
    assert summary["avg_loss"] == pytest.approx(-70.0)


def test_summary_surfaces_the_worst_themes():
    worst = mem().summary(Market.CRYPTO)["worst_themes"]
    assert worst[0]["theme"] == "cat"      # -80 is the single worst bucket
    assert {w["theme"] for w in worst} == {"cat", "dog"}


def test_summary_of_an_empty_market_is_empty():
    assert OutcomeMemory({}).summary(Market.CRYPTO) == {}


# --- prompt injection -----------------------------------------------------------------

def test_context_is_empty_on_a_cold_desk():
    # no history must not ship a misleading 0% win rate
    assert OutcomeMemory({}).context(Market.CRYPTO) == {}


def test_context_carries_both_the_record_and_the_examples():
    block = mem().context(Market.CRYPTO, theme="dog")["past_outcomes"]
    assert block["this_market"]["closed_trades"] == 3
    assert any(t["symbol"] == "PUPPY" for t in block["comparable_trades"])


async def test_crypto_checker_prompt_includes_past_outcomes(client_factory):
    agent = CryptoChecker(CONFIG, client=client_factory({"approve": False}))
    agent.memory = mem()
    await agent.run({"token": {"symbol": "NEWDOG"}, "narrative": {"theme": "dog"}})

    sent = agent._client.calls[0]["json"]["messages"][1]["content"]
    assert "past_outcomes" in sent
    assert "PUPPY" in sent            # the losing dog trade is in front of it


async def test_exit_manager_prompt_includes_past_outcomes(client_factory):
    agent = ExitManager(CONFIG, client=client_factory({"action": "HOLD"}))
    agent.memory = mem()
    await agent.run(Position(market=Market.STOCKS, symbol="ACME", quantity=1,
                             entry_price=10.0, sector="tech"))

    sent = agent._client.calls[0]["json"]["messages"][1]["content"]
    assert "past_outcomes" in sent and "ACME" in sent


async def test_allocator_prompt_includes_both_track_records(client_factory):
    agent = Allocator(CONFIG, client=client_factory({"crypto_pct": 0.5, "stocks_pct": 0.5}))
    agent.memory = mem()
    await agent.run({})

    sent = agent._client.calls[0]["json"]["messages"][1]["content"]
    assert "track_record" in sent and "crypto" in sent and "stocks" in sent


async def test_no_memory_attached_leaves_the_prompt_alone(client_factory):
    agent = CryptoChecker(CONFIG, client=client_factory({"approve": False}))
    await agent.run({"token": {"symbol": "X"}})
    sent = agent._client.calls[0]["json"]["messages"][1]["content"]
    assert "past_outcomes" not in sent

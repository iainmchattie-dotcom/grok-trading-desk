import httpx
import pytest

from tests.conftest import CONFIG, FakeResponse
from src.models import Stock
from src.stocks.analyst import Analyst
from src.stocks.insider import Insider
from src.stocks.market_pulse import MarketPulse
from src.stocks.radar import Radar
from src.stocks.stock_checker import StockChecker

STOCK = Stock(
    symbol="ACME", name="Acme Corp", sector="technology", price=50.0,
    prev_close=46.0, avg_volume=2_000_000, volume=6_000_000, market_cap=5e9,
)


async def test_analyst_parses_full_answer(client_factory):
    client = client_factory(
        {
            "fundamentals_score": 0.72, "technicals_score": 0.64, "trend": "up",
            "support": 45.5, "resistance": 58.0, "valuation": "fair",
            "thesis": "margin inflection", "risks": ["guidance"],
        }
    )
    result = await Analyst(CONFIG, client=client).run(STOCK)
    assert result["fundamentals_score"] == 0.72
    assert result["support"] == 45.5
    assert result["risks"] == ["guidance"]
    assert client.calls[0]["json"]["model"] == "grok-4-fast"


async def test_analyst_falls_back_to_zeros(client_factory, no_sleep):
    result = await Analyst(CONFIG, client=client_factory("]]broken[[")).run(STOCK)
    assert result["fundamentals_score"] == 0.0
    assert result["technicals_score"] == 0.0
    assert result["valuation"] == "expensive"
    assert result["risks"] == ["analyst_unavailable"]


async def test_analyst_survives_non_numeric_levels(client_factory):
    client = client_factory({"fundamentals_score": 0.5, "technicals_score": 0.5, "support": "n/a"})
    result = await Analyst(CONFIG, client=client).run(STOCK)
    assert result["support"] == 0.0


async def test_radar_parses_clean_read(client_factory):
    client = client_factory(
        {"sentiment_score": 0.7, "news_momentum": 0.6, "controversy": 0.05,
         "catalysts": ["earnings"], "headline_risk": "low", "summary": "quiet"}
    )
    result = await Radar(CONFIG, client=client).run(STOCK)
    assert result["controversy"] == 0.05
    assert result["catalysts"] == ["earnings"]


async def test_radar_fallback_trips_the_controversy_veto(client_factory, no_sleep):
    result = await Radar(CONFIG, client=client_factory(httpx.ConnectError("down"))).run(STOCK)
    assert result["controversy"] == 1.0   # > 0.7 hard veto
    assert result["sentiment_score"] == 0.0


async def test_radar_defaults_missing_controversy_to_one(client_factory):
    result = await Radar(CONFIG, client=client_factory({"sentiment_score": 0.9})).run(STOCK)
    assert result["controversy"] == 1.0


async def test_insider_parses_clean_read(client_factory):
    client = client_factory(
        {"insider_buying": 0.6, "insider_selling": 0.1, "institutional_flow": 0.7,
         "cluster_buying": True, "notable": ["CFO bought"], "summary": "accumulation"}
    )
    result = await Insider(CONFIG, client=client).run(STOCK)
    assert result["cluster_buying"] is True
    assert result["insider_selling"] == 0.1


async def test_insider_fallback_trips_the_selling_veto(client_factory, no_sleep):
    result = await Insider(CONFIG, client=client_factory("nope")).run(STOCK)
    assert result["insider_selling"] == 1.0   # > 0.8
    assert result["insider_buying"] == 0.0    # < 0.2


async def test_stock_checker_uses_the_full_model(client_factory):
    client = client_factory(
        {"approve": True, "confidence": 0.8, "adjusted_score": 0.71,
         "suggested_stop_pct": 0.06, "suggested_target_pct": 0.18}
    )
    result = await StockChecker(CONFIG, client=client).run({"symbol": "ACME"})
    assert result["approve"] is True
    assert result["suggested_stop_pct"] == 0.06
    assert client.calls[0]["json"]["model"] == "grok-4"


async def test_stock_checker_rejects_on_failure(client_factory, no_sleep):
    result = await StockChecker(CONFIG, client=client_factory(FakeResponse("", 429))).run({})
    assert result["approve"] is False
    assert result["adjusted_score"] == 0.0
    assert result["suggested_stop_pct"] == 0.08   # sane defaults survive the fallback


async def test_stock_checker_ignores_absurd_stop_suggestions(client_factory):
    client = client_factory({"approve": True, "suggested_stop_pct": 12, "suggested_target_pct": -3})
    result = await StockChecker(CONFIG, client=client).run({})
    assert result["suggested_stop_pct"] == 0.08
    assert result["suggested_target_pct"] == 0.20


async def test_market_pulse_caches_for_thirty_minutes(client_factory):
    client = client_factory({"regime": "risk_on", "go_signal": 0.75, "volatility": "low"})
    pulse = MarketPulse(CONFIG, client=client)

    await pulse.run()
    pulse._cache_time -= 20 * 60   # still inside the 30-minute window
    await pulse.run()
    assert len(client.calls) == 1

    pulse._cache_time -= 15 * 60   # now 35 minutes old
    await pulse.run()
    assert len(client.calls) == 2


async def test_market_pulse_fallback_closes_the_gate(client_factory, no_sleep):
    result = await MarketPulse(CONFIG, client=client_factory("{")).run()
    assert result["go_signal"] == 0.0
    assert result["regime"] == "risk_off"
    assert result["volatility"] == "high"


async def test_market_pulse_does_not_cache_across_instances(client_factory):
    client = client_factory({"regime": "neutral", "go_signal": 0.5})
    await MarketPulse(CONFIG, client=client).run()
    await MarketPulse(CONFIG, client=client).run()
    assert len(client.calls) == 2

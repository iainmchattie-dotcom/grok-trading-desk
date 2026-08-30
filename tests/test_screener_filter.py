import pytest

from src.models import Stock
from src.stocks.screener import Screener, filter_reason, parse_stock, passes_filter

FILT = {
    "min_price": 3.0,
    "max_price": 400.0,
    "min_avg_volume": 750000,
    "min_market_cap": 300_000_000,
    "max_market_cap": 50_000_000_000,
    "min_rel_volume": 1.5,
    "min_gap_pct": 0.02,
    "max_gap_pct": 0.15,
    "excluded_sectors": ["biotech"],
}


def good_stock(**over) -> Stock:
    base = dict(
        symbol="ACME",
        sector="technology",
        price=50.0,
        prev_close=46.0,
        avg_volume=2_000_000,
        volume=6_000_000,
        market_cap=5_000_000_000,
    )
    base.update(over)
    return Stock(**base)


def test_clean_stock_passes():
    assert filter_reason(good_stock(), FILT) is None
    assert passes_filter(good_stock(), FILT)


@pytest.mark.parametrize(
    "over,reason",
    [
        ({"price": 1.0}, "price_too_low"),
        ({"price": 900.0}, "price_too_high"),
        ({"avg_volume": 1000}, "illiquid"),
        ({"market_cap": 1_000_000}, "market_cap_too_small"),
        ({"market_cap": 900_000_000_000}, "market_cap_too_large"),
        ({"volume": 100_000}, "no_relative_volume"),
        ({"prev_close": 50.0}, "gap_too_small"),
        ({"prev_close": 20.0}, "gap_too_large"),
        ({"sector": "Biotech"}, "excluded_sector"),
        ({"symbol": ""}, "no_symbol"),
    ],
)
def test_each_rejection_reason(over, reason):
    assert filter_reason(good_stock(**over), FILT) == reason


def test_gap_uses_absolute_value():
    # a -8% gap down is as tradeable as +8% up
    gapped_down = good_stock(price=46.0, prev_close=50.0)
    assert gapped_down.gap_pct == pytest.approx(-0.08)
    assert filter_reason(gapped_down, FILT) is None


def test_boundaries_are_inclusive():
    assert filter_reason(good_stock(price=3.0, prev_close=2.9, market_cap=300_000_000), FILT) is None
    assert filter_reason(good_stock(avg_volume=750000, volume=1_125_000), FILT) is None
    assert filter_reason(good_stock(price=51.0, prev_close=50.0), FILT) is None  # exactly 2%


def test_rel_volume_and_gap_with_zero_denominators():
    s = Stock(symbol="Z", avg_volume=0, volume=100, prev_close=0, price=10)
    assert s.rel_volume == 0.0
    assert s.gap_pct == 0.0


def test_parse_stock_accepts_aliases():
    stock = parse_stock(
        {"ticker": "msft", "last": 400.0, "previous_close": 380.0, "marketCap": 3e12}
    )
    assert stock.symbol == "MSFT"
    assert stock.price == 400.0
    assert stock.prev_close == 380.0
    assert stock.market_cap == 3e12


def test_screener_sorts_by_relative_volume_and_caps_limit():
    screener = Screener({"stock_filter": FILT})
    rows = [
        good_stock(symbol="LOW", volume=3_200_000).model_dump(mode="json"),
        good_stock(symbol="HIGH", volume=20_000_000).model_dump(mode="json"),
        good_stock(symbol="MID", volume=8_000_000).model_dump(mode="json"),
        good_stock(symbol="JUNK", price=0.5).model_dump(mode="json"),
    ]
    result = screener.screen(rows)
    assert [s.symbol for s in result] == ["HIGH", "MID", "LOW"]
    assert [s.symbol for s in screener.screen(rows, limit=2)] == ["HIGH", "MID"]


async def test_screener_run_uses_injected_fetch():
    async def fetch():
        return [good_stock(symbol="AAA").model_dump(mode="json"), {"symbol": "BAD"}]

    screener = Screener({"stock_filter": FILT}, fetch=fetch)
    result = await screener.run()
    assert [s.symbol for s in result] == ["AAA"]


# --- missing data must not reject the universe -------------------------------------
# Alpaca exposes no sector and no market cap; the old filter treated both as 0
# and rejected every candidate on min_market_cap.

def test_unknown_market_cap_is_not_a_rejection():
    assert filter_reason(good_stock(market_cap=0), FILT) is None


def test_unknown_sector_is_never_excluded():
    assert filter_reason(good_stock(sector="unknown"), FILT) is None
    # but a known, excluded sector still is
    assert filter_reason(good_stock(sector="biotech"), FILT) == "excluded_sector"


def test_a_known_market_cap_is_still_enforced():
    assert filter_reason(good_stock(market_cap=1_000_000), FILT) == "market_cap_too_small"


def test_missing_average_volume_skips_the_volume_filters():
    thin = good_stock(avg_volume=0, volume=0)
    assert filter_reason(thin, FILT) is None


def test_missing_previous_close_skips_the_gap_filter():
    assert filter_reason(good_stock(prev_close=0), FILT) is None


def test_a_row_with_only_a_price_survives():
    # the minimum a snapshot can yield and still be worth an LLM call
    bare = parse_stock({"symbol": "AAA", "price": 50.0})
    assert filter_reason(bare, FILT) is None

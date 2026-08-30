import json

import pytest

from src.crypto.scout import Scout, filter_reason, parse_token, passes_filter
from src.models import Token

FILT = {
    "min_liquidity_usd": 5000.0,
    "max_liquidity_usd": 400000.0,
    "min_holders": 25,
    "max_top10_holder_pct": 0.45,
    "max_dev_holding_pct": 0.10,
    "min_age_seconds": 60,
    "max_age_seconds": 3600,
    "min_buys": 15,
    "min_buy_sell_ratio": 1.2,
    "require_mint_revoked": True,
    "require_lp_burned": False,
}


def good_token(**over) -> Token:
    base = dict(
        mint="MINT1",
        symbol="GOOD",
        liquidity_usd=20000.0,
        holders=100,
        top10_holder_pct=0.20,
        dev_holding_pct=0.02,
        age_seconds=300,
        buys=60,
        sells=20,
        mint_revoked=True,
    )
    base.update(over)
    return Token(**base)


def test_clean_token_passes():
    assert filter_reason(good_token(), FILT) is None
    assert passes_filter(good_token(), FILT)


@pytest.mark.parametrize(
    "over,reason",
    [
        ({"liquidity_usd": 1000.0}, "liquidity_too_low"),
        ({"liquidity_usd": 900000.0}, "liquidity_too_high"),
        ({"holders": 5}, "too_few_holders"),
        ({"top10_holder_pct": 0.8}, "top10_concentration"),
        ({"dev_holding_pct": 0.5}, "dev_holding_too_high"),
        ({"age_seconds": 10}, "too_young"),
        ({"age_seconds": 99999}, "too_old"),
        ({"buys": 3, "sells": 1}, "too_few_buys"),
        ({"buys": 20, "sells": 40}, "weak_buy_sell_ratio"),
        ({"mint_revoked": False}, "mint_not_revoked"),
        ({"mint": ""}, "no_mint"),
    ],
)
def test_each_rejection_reason(over, reason):
    assert filter_reason(good_token(**over), FILT) == reason


def test_boundaries_are_inclusive():
    assert filter_reason(good_token(liquidity_usd=5000.0), FILT) is None
    assert filter_reason(good_token(holders=25), FILT) is None
    assert filter_reason(good_token(age_seconds=60), FILT) is None
    assert filter_reason(good_token(age_seconds=3600), FILT) is None
    assert filter_reason(good_token(buys=15, sells=12), FILT) is None
    assert filter_reason(good_token(top10_holder_pct=0.45), FILT) is None


def test_lp_burned_only_enforced_when_required():
    assert filter_reason(good_token(lp_burned=False), FILT) is None
    strict = {**FILT, "require_lp_burned": True}
    assert filter_reason(good_token(lp_burned=False), strict) == "lp_not_burned"


def test_buy_sell_ratio_with_zero_sells():
    assert good_token(buys=30, sells=0).buy_sell_ratio == 30.0
    assert Token(mint="M", buys=0, sells=0).buy_sell_ratio == 0.0


def test_parse_token_accepts_aliases():
    token = parse_token(
        {
            "mintAddress": "ABC",
            "ticker": "PEPE",
            "traderPublicKey": "DEV",
            "liquidity": 12345,
            "holder_count": 44,
            "mintAuthorityRevoked": True,
            "twitter": "https://x.com/x",
        }
    )
    assert token.mint == "ABC"
    assert token.symbol == "PEPE"
    assert token.creator == "DEV"
    assert token.liquidity_usd == 12345
    assert token.holders == 44
    assert token.mint_revoked is True
    assert token.socials["twitter"] == "https://x.com/x"


def test_scout_handle_message_filters_and_dedups():
    scout = Scout({"crypto_filter": FILT})
    payload = good_token().model_dump(mode="json")
    payload["mint"] = "MINT1"

    first = scout.handle_message(json.dumps(payload))
    assert first is not None and first.mint == "MINT1"

    # same mint again -> already seen
    assert scout.handle_message(json.dumps(payload)) is None


def test_scout_handle_message_rejects_junk():
    scout = Scout({"crypto_filter": FILT})
    assert scout.handle_message("not json") is None
    assert scout.handle_message(json.dumps([1, 2, 3])) is None
    assert scout.handle_message(json.dumps({"mint": "X", "holders": 1})) is None

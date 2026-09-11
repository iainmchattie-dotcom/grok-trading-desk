import pytest

from src.crypto import crypto_scoring as cs
from src.models import Stock, Token
from src.stocks import stock_scoring as ss

TOKEN = Token(mint="M", symbol="X", holders=300, unique_traders=90, buys=90, sells=30, liquidity_usd=100_000)
# Few wallets, many buys: the structural ring the watch window can actually see.
RING_TOKEN = Token(mint="M", symbol="X", holders=4, unique_traders=4, buys=20, sells=1, liquidity_usd=100_000)
# Short window, no holder census — the production pump.fun shape.
THIN_TOKEN = Token(mint="M", symbol="X", holders=14, unique_traders=14, buys=16, sells=2, liquidity_usd=20_000)
STOCK = Stock(symbol="ACME", price=50, prev_close=46, avg_volume=2e6, volume=6e6, market_cap=5e9)

CLEAN_AUDIT = {
    "coordinated_buys": False, "wash_trading": False, "bundled_launch": False,
    "sniper_pct": 0.0, "insider_pct": 0.0, "safety_score": 1.0,
}
STRONG_NARRATIVE = {
    "meme_score": 1.0, "virality": 1.0, "originality": 1.0,
    "community_signal": 1.0, "is_derivative": False,
}
OPEN_PULSE = {"go_signal": 1.0}


# --- crypto ------------------------------------------------------------------

def test_crypto_perfect_inputs_score_one_and_buy():
    result = cs.score_token(TOKEN, CLEAN_AUDIT, STRONG_NARRATIVE, OPEN_PULSE)
    assert result["score"] == 1.0
    assert result["buy"] is True
    assert result["vetoed"] is False


def test_crypto_coordinated_ring_vetoes_even_without_the_llm_flag():
    result = cs.score_token(RING_TOKEN, CLEAN_AUDIT, STRONG_NARRATIVE, OPEN_PULSE)
    assert result["vetoed"] is True
    assert result["reason"] == "veto_coordinated_buys"
    assert result["score"] == 0.0 and result["buy"] is False
    assert "repeat_buyers" in result["manipulation_evidence"]["signals"]


def test_crypto_llm_coordinated_flag_on_organic_tape_is_not_a_hard_veto():
    """The production failure mode: auditor says ring because early buys cluster,
    but unique_traders ≈ buys and concentration was never measured."""
    audit = {**CLEAN_AUDIT, "coordinated_buys": True}
    result = cs.score_token(TOKEN, audit, STRONG_NARRATIVE, OPEN_PULSE)
    assert result["vetoed"] is False
    assert result["buy"] is True
    assert result["manipulation_evidence"]["strength"] == "weak"
    # suspicion is a real penalty, just not a skip
    clean = cs.score_token(TOKEN, CLEAN_AUDIT, STRONG_NARRATIVE, OPEN_PULSE)
    assert result["components"]["audit_safety"] == pytest.approx(
        clean["components"]["audit_safety"] - cs.COORDINATED_SUSPICION_PENALTY
    )


def test_crypto_llm_flag_plus_measured_concentration_is_a_hard_veto():
    concentrated = TOKEN.model_copy(update={"top10_holder_pct": 0.82})
    audit = {**CLEAN_AUDIT, "coordinated_buys": True}
    result = cs.score_token(concentrated, audit, STRONG_NARRATIVE, OPEN_PULSE)
    assert result["reason"] == "veto_coordinated_buys"
    assert "top10_concentration" in result["manipulation_evidence"]["signals"]


def test_crypto_thin_ambiguous_data_does_not_hard_veto():
    audit = {**CLEAN_AUDIT, "coordinated_buys": True, "safety_score": 0.7}
    result = cs.score_token(THIN_TOKEN, audit, STRONG_NARRATIVE, OPEN_PULSE)
    assert result["vetoed"] is False
    assert result["manipulation_evidence"]["strength"] == "weak"
    assert result["manipulation_evidence"]["concentration_known"] is False


def test_crypto_wash_trading_veto():
    audit = {**CLEAN_AUDIT, "wash_trading": True}
    assert cs.score_token(TOKEN, audit, STRONG_NARRATIVE, OPEN_PULSE)["reason"] == "veto_wash_trading"


def test_crypto_paused_market_veto():
    result = cs.score_token(TOKEN, CLEAN_AUDIT, STRONG_NARRATIVE, {"go_signal": 0.29})
    assert result["reason"] == "veto_market_paused"


def test_crypto_go_signal_exactly_at_the_gate_is_not_vetoed():
    result = cs.score_token(TOKEN, CLEAN_AUDIT, STRONG_NARRATIVE, {"go_signal": 0.30})
    assert result["vetoed"] is False


def test_crypto_pessimistic_fallbacks_are_vetoed():
    from src.crypto.auditor import Auditor
    from src.crypto.crypto_pulse import CryptoPulse
    from src.crypto.narrative import Narrative

    result = cs.score_token(
        TOKEN, Auditor({}).fallback(), Narrative({}).fallback(), CryptoPulse({}).fallback()
    )
    assert result["buy"] is False and result["vetoed"] is True
    # Parse/API failure is its own skip reason, not a fake coordinated ring.
    assert result["reason"] == "veto_audit_unavailable"


def test_crypto_threshold_is_exclusive_below_inclusive_at():
    weights = {**cs.DEFAULT_WEIGHTS, "min_score_to_buy": 1.0}
    at = cs.score_token(TOKEN, CLEAN_AUDIT, STRONG_NARRATIVE, OPEN_PULSE, weights=weights)
    assert at["score"] == 1.0 and at["buy"] is True

    weights_above = {**cs.DEFAULT_WEIGHTS, "min_score_to_buy": 1.01}
    below = cs.score_token(TOKEN, CLEAN_AUDIT, STRONG_NARRATIVE, OPEN_PULSE, weights=weights_above)
    assert below["buy"] is False and below["reason"] == "below_threshold"


def test_crypto_audit_score_penalises_snipers_and_bundling():
    assert cs.audit_score(CLEAN_AUDIT) == 1.0
    assert cs.audit_score({**CLEAN_AUDIT, "sniper_pct": 0.4}) == pytest.approx(0.8)
    assert cs.audit_score({**CLEAN_AUDIT, "bundled_launch": True}) == pytest.approx(0.7)
    assert cs.audit_score({"safety_score": 0.1, "sniper_pct": 1.0}) == 0.0  # clamped, never negative
    assert cs.audit_score(CLEAN_AUDIT, suspicion=True) == pytest.approx(1.0 - cs.COORDINATED_SUSPICION_PENALTY)


def test_crypto_zero_holder_pct_is_unknown_not_a_ring():
    """Token.top10_holder_pct defaults to 0.0 because the feed never sends it."""
    evidence = cs.manipulation_evidence(THIN_TOKEN, {**CLEAN_AUDIT, "coordinated_buys": True})
    assert evidence["concentration_known"] is False
    assert evidence["strength"] == "weak"
    assert cs.holder_concentration_known(THIN_TOKEN) is False


def test_crypto_unmeasured_token_does_not_invent_a_ring():
    # buys=0 / holders=0: not enough tape. Missing data is not coordination.
    empty = Token(mint="M")
    evidence = cs.manipulation_evidence(empty, {**CLEAN_AUDIT, "coordinated_buys": True})
    assert evidence["strength"] == "weak"
    assert evidence["diversity"] is None
    assert cs.hard_veto({**CLEAN_AUDIT, "coordinated_buys": True}, OPEN_PULSE, token=empty) is None


def test_crypto_busy_organic_tape_is_not_a_structural_ring():
    """25 unique traders who each bought a few times is dip-buying, not a 4-wallet ring.

    Tokens that already cleared scout min_holders were still eating
    veto_coordinated_buys because unique_traders/buys <= 0.40.
    """
    busy = TOKEN.model_copy(update={"holders": 25, "unique_traders": 25, "buys": 80, "sells": 10})
    clean = cs.score_token(busy, CLEAN_AUDIT, STRONG_NARRATIVE, OPEN_PULSE)
    assert clean["vetoed"] is False
    assert clean["manipulation_evidence"]["strength"] == "none"
    assert "repeat_buyers" not in clean["manipulation_evidence"]["signals"]

    flagged = cs.score_token(
        busy, {**CLEAN_AUDIT, "coordinated_buys": True}, STRONG_NARRATIVE, OPEN_PULSE
    )
    assert flagged["vetoed"] is False
    assert flagged["buy"] is True
    assert flagged["manipulation_evidence"]["strength"] == "weak"


def test_crypto_small_repeat_buyer_set_is_still_a_hard_veto():
    result = cs.score_token(RING_TOKEN, CLEAN_AUDIT, STRONG_NARRATIVE, OPEN_PULSE)
    assert result["reason"] == "veto_coordinated_buys"
    assert result["manipulation_evidence"]["trader_count"] <= cs.MAX_TRADERS_FOR_STRUCTURAL_RING


def test_paper_checker_unavailable_does_not_block():
    check = {
        "approve": False,
        "hard_reject": False,
        "kill_reasons": ["checker_unavailable"],
        "adjusted_score": 0.0,
    }
    gate = cs.apply_crypto_checker(check, paper=True, matrix_score=0.70)
    assert gate["allow"] is True
    assert gate["unavailable"] is True
    assert gate["advisory"] is True
    assert gate["score"] == pytest.approx(0.70 - cs.CHECKER_UNAVAILABLE_PENALTY)


def test_paper_soft_checker_reject_does_not_block():
    check = {
        "approve": False,
        "hard_reject": False,
        "kill_reasons": ["liquidity too thin"],
        "adjusted_score": 0.0,
    }
    gate = cs.apply_crypto_checker(check, paper=True, matrix_score=0.70)
    assert gate["allow"] is True
    assert gate["advisory"] is True
    assert gate["score"] == pytest.approx(0.70 - cs.CHECKER_SOFT_REJECT_PENALTY)


def test_paper_hard_reject_still_blocks():
    check = {
        "approve": False,
        "hard_reject": True,
        "kill_reasons": ["honeypot"],
        "adjusted_score": 0.0,
    }
    gate = cs.apply_crypto_checker(check, paper=True, matrix_score=0.90)
    assert gate["allow"] is False
    assert gate["reason"] == "checker_rejected"


def test_paper_hard_reject_from_kill_reason_text():
    check = {
        "approve": False,
        "hard_reject": False,
        "kill_reasons": ["mint authority retained by deployer"],
        "adjusted_score": 0.0,
    }
    assert cs.apply_crypto_checker(check, paper=True, matrix_score=0.80)["allow"] is False


def test_live_checker_unavailable_still_blocks():
    check = {
        "approve": False,
        "hard_reject": False,
        "kill_reasons": ["checker_unavailable"],
        "adjusted_score": 0.0,
    }
    gate = cs.apply_crypto_checker(check, paper=False, matrix_score=0.90)
    assert gate["allow"] is False
    assert gate["reason"] == "checker_rejected"


def test_crypto_derivative_narrative_is_discounted():
    original = cs.narrative_score(STRONG_NARRATIVE)
    copy = cs.narrative_score({**STRONG_NARRATIVE, "is_derivative": True})
    assert copy == pytest.approx(original * 0.7)


def test_crypto_momentum_saturates_and_floors():
    assert cs.momentum_score(Token(mint="M", buys=0, sells=0, holders=0, liquidity_usd=0)) == 0.0
    hot = Token(mint="M", buys=1000, sells=1, holders=99999, liquidity_usd=1e9)
    assert cs.momentum_score(hot) == 1.0


def test_crypto_score_is_bounded_by_components():
    result = cs.score_token(
        Token(mint="M"), {"safety_score": 0.5}, {"meme_score": 0.5}, {"go_signal": 0.5}
    )
    assert 0.0 <= result["score"] <= 1.0


# --- stocks ------------------------------------------------------------------

CLEAN_RADAR = {"sentiment_score": 1.0, "news_momentum": 1.0, "controversy": 0.0}
CLEAN_INSIDER = {
    "insider_buying": 1.0, "insider_selling": 0.0,
    "institutional_flow": 1.0, "cluster_buying": True,
}
STRONG_ANALYST = {"fundamentals_score": 1.0, "technicals_score": 1.0}


def test_stock_perfect_inputs_score_one_and_buy():
    result = ss.score_stock(STOCK, STRONG_ANALYST, CLEAN_RADAR, CLEAN_INSIDER, OPEN_PULSE)
    assert result["score"] == 1.0
    assert result["buy"] is True


def test_stock_controversy_veto_at_the_boundary():
    just_under = ss.score_stock(STOCK, STRONG_ANALYST, {**CLEAN_RADAR, "controversy": 0.7},
                                CLEAN_INSIDER, OPEN_PULSE)
    assert just_under["vetoed"] is False   # 0.7 is not > 0.7

    just_over = ss.score_stock(STOCK, STRONG_ANALYST, {**CLEAN_RADAR, "controversy": 0.71},
                               CLEAN_INSIDER, OPEN_PULSE)
    assert just_over["reason"] == "veto_controversy"


def test_stock_insider_veto_needs_both_conditions():
    heavy_selling_but_also_buying = {**CLEAN_INSIDER, "insider_selling": 0.9, "insider_buying": 0.5}
    assert ss.score_stock(STOCK, STRONG_ANALYST, CLEAN_RADAR, heavy_selling_but_also_buying,
                          OPEN_PULSE)["vetoed"] is False

    dumping = {**CLEAN_INSIDER, "insider_selling": 0.81, "insider_buying": 0.19}
    assert ss.score_stock(STOCK, STRONG_ANALYST, CLEAN_RADAR, dumping,
                          OPEN_PULSE)["reason"] == "veto_insider_selling"


def test_stock_insider_veto_boundary_is_exclusive():
    edge = {**CLEAN_INSIDER, "insider_selling": 0.8, "insider_buying": 0.2}
    assert ss.score_stock(STOCK, STRONG_ANALYST, CLEAN_RADAR, edge, OPEN_PULSE)["vetoed"] is False


def test_stock_paused_market_veto():
    result = ss.score_stock(STOCK, STRONG_ANALYST, CLEAN_RADAR, CLEAN_INSIDER, {"go_signal": 0.1})
    assert result["reason"] == "veto_market_paused"


def test_stock_pessimistic_fallbacks_are_vetoed():
    from src.stocks.analyst import Analyst
    from src.stocks.insider import Insider
    from src.stocks.market_pulse import MarketPulse
    from src.stocks.radar import Radar

    result = ss.score_stock(
        STOCK, Analyst({}).fallback(), Radar({}).fallback(),
        Insider({}).fallback(), MarketPulse({}).fallback(),
    )
    assert result["buy"] is False and result["vetoed"] is True


def test_stock_news_score_subtracts_controversy():
    assert ss.news_score(CLEAN_RADAR) == 1.0
    assert ss.news_score({**CLEAN_RADAR, "controversy": 0.5}) == pytest.approx(0.75)
    assert ss.news_score({"sentiment_score": 0.0, "news_momentum": 0.0, "controversy": 0.5}) == 0.0


def test_stock_insider_score_rewards_clusters_and_punishes_selling():
    assert ss.insider_score(CLEAN_INSIDER) == 1.0
    assert ss.insider_score({**CLEAN_INSIDER, "cluster_buying": False}) == pytest.approx(0.85)
    assert ss.insider_score({"insider_buying": 0.0, "insider_selling": 1.0}) == 0.0


def test_stock_missing_keys_default_pessimistically():
    result = ss.score_stock(STOCK, {}, {}, {}, {})
    assert result["vetoed"] is True   # missing controversy defaults to 1.0
    assert result["buy"] is False


def test_stock_custom_weights_change_the_verdict():
    mediocre_fundamentals = {"fundamentals_score": 0.0, "technicals_score": 1.0}
    fundamentals_heavy = {**ss.DEFAULT_WEIGHTS, "fundamentals": 0.9, "technicals": 0.02,
                          "news_sentiment": 0.02, "insider": 0.03, "pulse": 0.03}
    result = ss.score_stock(STOCK, mediocre_fundamentals, CLEAN_RADAR, CLEAN_INSIDER,
                            OPEN_PULSE, weights=fundamentals_heavy)
    assert result["buy"] is False

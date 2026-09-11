import json

import httpx
import pytest

from tests.conftest import CONFIG, FakeResponse
from src.crypto.auditor import Auditor
from src.crypto.crypto_checker import CryptoChecker
from src.crypto.crypto_pulse import CryptoPulse
from src.crypto.narrative import Narrative
from src.models import Token

TOKEN = Token(
    mint="MINT1", symbol="WIF2", name="dogwifhat2", liquidity_usd=20000,
    holders=120, buys=80, sells=20, age_seconds=400, mint_revoked=True,
)


def test_auditor_prompt_does_not_assume_manipulation_on_ambiguity():
    prompt = Auditor.PROMPT.lower()
    assert "assume manipulation" not in prompt
    assert "ambiguity is not proof of manipulation" in prompt
    assert "temporal clustering of independent buys is not" in prompt


def test_auditor_facts_mark_unmeasured_concentration_as_unknown():
    facts = Auditor({}).facts(TOKEN)
    assert facts["holder_concentration_known"] is False
    assert facts["dev_holding_known"] is False
    assert facts["top10_holder_pct"] is None
    assert facts["dev_holding_pct"] is None
    assert facts["unique_traders_per_buy"] is None or facts["unique_traders_per_buy"] > 0.4
    assert any("unknown when null" in note for note in facts["data_limitations"])


def test_auditor_facts_include_diversity_when_buys_exist():
    token = TOKEN.model_copy(update={"buys": 20, "unique_traders": 18, "holders": 18})
    facts = Auditor({}).facts(token)
    assert facts["unique_traders_per_buy"] == 0.9
    assert facts["holder_concentration_known"] is False


def test_auditor_fallback_is_marked_unavailable():
    fallback = Auditor({}).fallback()
    assert fallback["audit_unavailable"] is True
    assert "audit_unavailable" in fallback["red_flags"]
    assert fallback["coordinated_buys"] is True
    assert fallback["wash_trading"] is True


async def test_auditor_parses_clean_audit(client_factory):
    client = client_factory(
        {
            "coordinated_buys": False,
            "wash_trading": False,
            "bundled_launch": False,
            "sniper_pct": 0.05,
            "insider_pct": 0.02,
            "safety_score": 0.88,
            "red_flags": [],
        }
    )
    result = await Auditor(CONFIG, client=client).run(TOKEN)
    assert result["safety_score"] == 0.88
    assert result["coordinated_buys"] is False
    assert result["red_flags"] == []
    # the fast model and temperature=0 are non-negotiable
    body = client.calls[0]["json"]
    assert body["model"] == "grok-4.3"
    assert body["temperature"] == 0
    assert client.calls[0]["headers"]["Authorization"] == "Bearer test-key"


async def test_auditor_strips_markdown_fences(client_factory):
    fenced = '```json\n{"coordinated_buys": false, "wash_trading": false, "safety_score": 0.7}\n```'
    result = await Auditor(CONFIG, client=client_factory(fenced)).run(TOKEN)
    assert result["safety_score"] == 0.7
    # unspecified booleans stay pessimistic
    assert result["bundled_launch"] is True


async def test_auditor_handles_prose_wrapped_json(client_factory):
    messy = 'Here is my audit:\n{"coordinated_buys": false, "wash_trading": false, "safety_score": 0.5}\nHope that helps.'
    result = await Auditor(CONFIG, client=client_factory(messy)).run(TOKEN)
    assert result["safety_score"] == 0.5


async def test_auditor_falls_back_pessimistically_on_broken_json(client_factory, no_sleep):
    client = client_factory("not json at all")
    result = await Auditor(CONFIG, client=client).run(TOKEN)
    assert result["coordinated_buys"] is True
    assert result["wash_trading"] is True
    assert result["safety_score"] == 0.0
    assert result["red_flags"] == ["audit_unavailable"]
    assert result["audit_unavailable"] is True
    assert len(client.calls) == 3  # all retries burnt before giving up


async def test_auditor_retries_then_succeeds(client_factory, no_sleep):
    client = client_factory("garbage", {"coordinated_buys": False, "wash_trading": False, "safety_score": 0.6})
    result = await Auditor(CONFIG, client=client).run(TOKEN)
    assert result["safety_score"] == 0.6
    assert len(client.calls) == 2


async def test_auditor_falls_back_on_http_error(client_factory, no_sleep):
    result = await Auditor(CONFIG, client=client_factory(FakeResponse("", 500))).run(TOKEN)
    assert result["safety_score"] == 0.0


async def test_auditor_falls_back_on_timeout(client_factory, no_sleep):
    result = await Auditor(CONFIG, client=client_factory(httpx.TimeoutException("slow"))).run(TOKEN)
    assert result["safety_score"] == 0.0
    assert result["wash_trading"] is True


async def test_auditor_clamps_out_of_range_numbers(client_factory):
    client = client_factory({"coordinated_buys": False, "wash_trading": False, "safety_score": 4.2, "sniper_pct": -1})
    result = await Auditor(CONFIG, client=client).run(TOKEN)
    assert result["safety_score"] == 1.0
    assert result["sniper_pct"] == 0.0


async def test_narrative_parses_and_falls_back(client_factory, no_sleep):
    good = await Narrative(CONFIG, client=client_factory(
        {"meme_score": 0.8, "originality": 0.6, "virality": 0.7, "community_signal": 0.5,
         "is_derivative": False, "theme": "dog", "reasoning": "runs hot"}
    )).run(TOKEN)
    assert good["meme_score"] == 0.8 and good["is_derivative"] is False

    bad = await Narrative(CONFIG, client=client_factory("¯\\_(ツ)_/¯")).run(TOKEN)
    assert bad["meme_score"] == 0.0
    assert bad["is_derivative"] is True


async def test_crypto_checker_uses_the_deep_model(client_factory):
    client = client_factory({"approve": True, "confidence": 0.7, "adjusted_score": 0.66})
    result = await CryptoChecker(CONFIG, client=client).run({"symbol": "WIF2"})
    assert result["approve"] is True
    # the checker must not run the same model as the bots it is checking
    assert client.calls[0]["json"]["model"] == "grok-4.6"


async def test_crypto_checker_rejects_on_failure(client_factory, no_sleep):
    result = await CryptoChecker(CONFIG, client=client_factory("{oops")).run({"symbol": "WIF2"})
    assert result["approve"] is False
    assert result["hard_reject"] is False
    assert result["adjusted_score"] == 0.0
    assert result["kill_reasons"] == ["checker_unavailable"]


def test_crypto_checker_prompt_reserves_hard_reject_for_evidence():
    prompt = CryptoChecker.PROMPT.lower()
    assert "hard_reject" in prompt
    assert "honeypot" in prompt
    assert "advisory" in prompt or "soft reject" in prompt


async def test_crypto_checker_defaults_missing_approve_to_false(client_factory):
    result = await CryptoChecker(CONFIG, client=client_factory({"confidence": 0.9})).run({})
    assert result["approve"] is False
    assert result["hard_reject"] is False


async def test_crypto_pulse_caches_within_window(client_factory):
    client = client_factory({"regime": "risk_on", "go_signal": 0.8, "risk_appetite": 0.7})
    pulse = CryptoPulse(CONFIG, client=client)

    first = await pulse.run()
    second = await pulse.run()
    assert first == second
    assert len(client.calls) == 1  # second read served from cache


async def test_crypto_pulse_refetches_when_cache_expires(client_factory):
    client = client_factory(
        {"regime": "risk_on", "go_signal": 0.8},
        {"regime": "risk_off", "go_signal": 0.1},
    )
    pulse = CryptoPulse(CONFIG, client=client)
    assert (await pulse.run())["go_signal"] == 0.8

    pulse._cache_time -= 16 * 60  # push the cache past 15 minutes
    assert (await pulse.run())["go_signal"] == 0.1
    assert len(client.calls) == 2


async def test_crypto_pulse_invalidate_forces_refetch(client_factory):
    client = client_factory({"regime": "neutral", "go_signal": 0.5})
    pulse = CryptoPulse(CONFIG, client=client)
    await pulse.run()
    pulse.invalidate()
    await pulse.run()
    assert len(client.calls) == 2


async def test_crypto_pulse_fallback_closes_the_gate(client_factory, no_sleep):
    result = await CryptoPulse(CONFIG, client=client_factory("nope")).run()
    assert result["go_signal"] == 0.0
    assert result["regime"] == "risk_off"


async def test_crypto_pulse_rejects_unknown_regime_label(client_factory):
    result = await CryptoPulse(CONFIG, client=client_factory({"regime": "euphoric", "go_signal": 0.9})).run()
    assert result["regime"] == "risk_off"
    assert result["go_signal"] == 0.9

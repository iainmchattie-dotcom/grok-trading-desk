"""Transport-level behaviour: schemas, search, retries, cost accounting."""

import httpx
import pytest

from tests.conftest import CONFIG, FakeResponse
from datetime import date, timedelta

from src.base_agent import (
    CostTracker,
    GrokAgent,
    derive_responses_url,
    extract_message_content,
    parse_json_response,
    schema,
)
from src.crypto.auditor import Auditor
from src.crypto.crypto_checker import CryptoChecker
from src.crypto.crypto_pulse import CryptoPulse
from src.shared.allocator import Allocator
from src.stocks.insider import Insider
from src.stocks.market_pulse import MarketPulse
from src.stocks.radar import Radar


class Probe(GrokAgent):
    name = "probe"
    PROMPT = "static instructions"
    SCHEMA = schema({"ok": {"type": "boolean"}})

    def fallback(self):
        return {"ok": False, "why": "fallback"}


# --- request assembly ------------------------------------------------------------

def test_strict_json_schema_is_attached():
    body = Probe(CONFIG).build_request({"a": 1})
    fmt = body["response_format"]
    assert fmt["type"] == "json_schema"
    assert fmt["json_schema"]["strict"] is True
    assert fmt["json_schema"]["name"] == "probe"
    assert fmt["json_schema"]["schema"]["additionalProperties"] is False


def test_schema_helper_marks_every_property_required():
    built = schema({"a": {"type": "string"}, "b": {"type": "number"}})
    assert built["required"] == ["a", "b"]
    assert built["additionalProperties"] is False


def test_structured_outputs_can_be_switched_off():
    config = {"grok": {**CONFIG["grok"], "structured_outputs": False}}
    assert Probe(config).build_request(None)["response_format"] == {"type": "json_object"}


def test_agent_without_a_schema_uses_json_object():
    class Bare(Probe):
        SCHEMA = None

    assert Bare(CONFIG).build_request(None)["response_format"] == {"type": "json_object"}


def test_static_prompt_comes_first_so_the_prefix_caches():
    body = Probe(CONFIG).build_request({"symbol": "WIF"})
    assert body["messages"][0] == {"role": "system", "content": "static instructions"}
    assert "WIF" in body["messages"][1]["content"]
    assert body["prompt_cache_key"] == "grok-desk:probe"


def test_reasoning_effort_only_goes_to_grok_43():
    # grok-4.3 is the only model that accepts the parameter
    assert Probe(CONFIG).build_request(None)["reasoning_effort"] == "none"

    deep = {"grok": {**CONFIG["grok"], "models": {"fast": "grok-4.6"}, "reasoning_effort": {"fast": "high"}}}
    assert "reasoning_effort" not in Probe(deep).build_request(None)


def test_legacy_model_keys_still_resolve():
    legacy = {"grok": {"fast_model": "grok-4.5", "full_model": "grok-4.6"}}
    assert Probe(legacy).model == "grok-4.5"


def test_model_defaults_split_the_tiers():
    bare = Probe({})
    assert bare.model == "grok-4.3"
    assert CryptoChecker({}).model == "grok-4.6"


# --- Agent Tools (Live Search was retired; search_parameters is HTTP 410) ----------

def test_retrieval_agents_send_tools_not_search_parameters():
    radar = Radar(CONFIG)
    body = radar.build_request({"symbol": "ACME"})
    assert "search_parameters" not in body
    assert {t["type"] for t in body["tools"]} == {"web_search", "x_search"}
    assert body["input"][0]["role"] == "system"
    assert "messages" not in body
    x = next(t for t in body["tools"] if t["type"] == "x_search")
    assert x["from_date"] == (date.today() - timedelta(days=14)).isoformat()
    assert radar.request_url(body).endswith("/responses")


def test_insider_whitelists_primary_filing_sources():
    body = Insider(CONFIG).build_request({"symbol": "ACME"})
    assert "search_parameters" not in body
    web = next(t for t in body["tools"] if t["type"] == "web_search")
    assert "sec.gov" in web["filters"]["allowed_domains"]
    assert len(web["filters"]["allowed_domains"]) <= 5   # API caps the whitelist at 5


def test_agents_that_need_no_retrieval_stay_on_chat_completions():
    alloc = Allocator(CONFIG)
    body = alloc.build_request({})
    assert "search_parameters" not in body
    assert "tools" not in body
    assert "messages" in body
    assert alloc.request_url(body).endswith("/chat/completions")


def test_live_search_can_be_switched_off_globally():
    config = {"grok": {**CONFIG["grok"], "live_search": False}}
    radar = Radar(config)
    body = radar.build_request({"symbol": "A"})
    assert "search_parameters" not in body
    assert "tools" not in body
    assert radar.request_url(body).endswith("/chat/completions")


def test_x_source_becomes_x_search_tool():
    tools = Auditor(CONFIG).build_request({"mint": "M"})["tools"]
    assert any(t["type"] == "x_search" for t in tools)
    # post_view_count is not a tool parameter; do not send unknown fields
    x = next(t for t in tools if t["type"] == "x_search")
    assert "post_view_count" not in x


def test_pulse_agents_use_responses_and_date_window():
    crypto = CryptoPulse(CONFIG)
    crypto_body = crypto.build_request(None)
    assert "search_parameters" not in crypto_body
    assert {t["type"] for t in crypto_body["tools"]} == {"web_search", "x_search"}
    crypto_x = next(t for t in crypto_body["tools"] if t["type"] == "x_search")
    assert crypto_x["from_date"] == (date.today() - timedelta(days=1)).isoformat()
    assert crypto.request_url(crypto_body) == "https://api.x.ai/v1/responses"

    market = MarketPulse(CONFIG)
    market_body = market.build_request(None)
    assert "search_parameters" not in market_body
    market_x = next(t for t in market_body["tools"] if t["type"] == "x_search")
    assert market_x["from_date"] == (date.today() - timedelta(days=2)).isoformat()
    assert market.request_url(market_body).endswith("/responses")


def test_derive_responses_url_from_chat_completions():
    assert derive_responses_url("https://api.x.ai/v1/chat/completions") == (
        "https://api.x.ai/v1/responses"
    )
    assert derive_responses_url("https://api.x.ai/v1") == "https://api.x.ai/v1/responses"
    assert derive_responses_url("https://api.x.ai/v1/responses") == (
        "https://api.x.ai/v1/responses"
    )


def test_extract_message_content_reads_both_envelopes():
    chat = {"choices": [{"message": {"content": '{"ok": true}'}}]}
    assert extract_message_content(chat) == '{"ok": true}'
    responses = {
        "output": [
            {"type": "web_search_call"},
            {"type": "message", "content": [{"type": "output_text", "text": '{"ok": true}'}]},
        ]
    }
    assert extract_message_content(responses) == '{"ok": true}'
    assert extract_message_content({"output_text": '{"ok": true}'}) == '{"ok": true}'


# --- retry policy ---------------------------------------------------------------------

async def test_a_400_is_not_retried(client_factory, no_sleep):
    client = client_factory(FakeResponse("", 400))
    result = await Probe(CONFIG, client=client).run()
    assert result["why"] == "fallback"
    assert len(client.calls) == 1     # our bug; repeating it three times is waste


async def test_a_410_is_not_retried_and_pulse_stays_shut(client_factory, no_sleep):
    client = client_factory(FakeResponse("", 410))
    result = await CryptoPulse(CONFIG, client=client).run()
    assert result["go_signal"] == 0.0
    assert result["regime"] == "risk_off"
    assert result["notes"] == "crypto_pulse_unavailable"
    assert len(client.calls) == 1
    # the retired Live Search field must not be on the wire
    assert "search_parameters" not in client.calls[0]["json"]
    assert client.calls[0]["url"].endswith("/responses")


async def test_a_429_is_retried(client_factory, no_sleep):
    client = client_factory(FakeResponse("", 429), {"ok": True})
    result = await Probe(CONFIG, client=client).run()
    assert result == {"ok": True}
    assert len(client.calls) == 2


async def test_a_500_is_retried(client_factory, no_sleep):
    client = client_factory(FakeResponse("", 500))
    await Probe(CONFIG, client=client).run()
    assert len(client.calls) == 3


async def test_a_timeout_is_retried(client_factory, no_sleep):
    client = client_factory(httpx.TimeoutException("slow"), {"ok": True})
    assert await Probe(CONFIG, client=client).run() == {"ok": True}


def test_retry_after_header_is_honoured():
    agent = Probe(CONFIG)
    exc = httpx.HTTPStatusError("429", request=None, response=None)  # type: ignore[arg-type]
    exc.response = FakeResponse("", 429, headers={"retry-after": "7"})  # type: ignore[assignment]
    assert agent._retry_delay(0, exc) == 7.0


def test_retry_after_is_capped():
    agent = Probe(CONFIG)
    exc = httpx.HTTPStatusError("429", request=None, response=None)  # type: ignore[arg-type]
    exc.response = FakeResponse("", 429, headers={"retry-after": "9999"})  # type: ignore[assignment]
    assert agent._retry_delay(0, exc) == 60.0


def test_backoff_is_jittered_and_bounded():
    agent = Probe(CONFIG)
    exc = httpx.TimeoutException("slow")
    delays = [agent._retry_delay(6, exc) for _ in range(20)]
    assert len(set(delays)) > 1                 # jitter, so retries do not sync up
    assert all(15.0 <= d <= 30.0 for d in delays)   # 2**6 clamped to the 30s ceiling
    early = [agent._retry_delay(1, exc) for _ in range(20)]
    assert all(1.0 <= d <= 2.0 for d in early)


# --- cost accounting ----------------------------------------------------------------

USAGE = {
    "prompt_tokens": 1000,
    "completion_tokens": 200,
    "num_sources_used": 12,
    "cost_in_usd_ticks": 25_000_000_000,        # 1e10 ticks == $1
    "prompt_tokens_details": {"cached_tokens": 400},
    "completion_tokens_details": {"reasoning_tokens": 50},
}


async def test_cost_is_taken_from_the_billed_amount(client_factory):
    costs = CostTracker()
    client = client_factory(FakeResponse('{"ok": true}', usage=USAGE))
    await Probe(CONFIG, client=client, costs=costs).run()

    snap = costs.snapshot()
    assert snap["cost_usd"] == pytest.approx(2.5)
    assert snap["cached_tokens"] == 400
    assert snap["cache_hit_rate"] == 0.4
    assert snap["reasoning_tokens"] == 50
    assert snap["sources_used"] == 12
    assert snap["by_agent"]["probe"] == pytest.approx(2.5)


async def test_costs_accumulate_across_agents(client_factory):
    costs = CostTracker()
    for _ in range(3):
        client = client_factory(FakeResponse('{"ok": true}', usage=USAGE))
        await Probe(CONFIG, client=client, costs=costs).run()
    assert costs.snapshot()["cost_usd"] == pytest.approx(7.5)
    assert costs.calls == 3


async def test_fallbacks_are_counted(client_factory, no_sleep):
    costs = CostTracker()
    client = client_factory(FakeResponse("", 400))
    await Probe(CONFIG, client=client, costs=costs).run()
    assert costs.snapshot()["fallbacks"] == 1
    assert costs.snapshot()["failed_calls"] == 1


async def test_usage_absent_does_not_break_accounting(client_factory):
    costs = CostTracker()
    await Probe(CONFIG, client=client_factory('{"ok": true}'), costs=costs).run()
    assert costs.snapshot()["cost_usd"] == 0.0
    assert costs.calls == 1


async def test_citations_are_captured(client_factory):
    agent = Probe(CONFIG, client=client_factory(
        FakeResponse('{"ok": true}', citations=["https://sec.gov/x", "https://x.com/y"])
    ))
    await agent.run()
    assert agent.last_citations == ["https://sec.gov/x", "https://x.com/y"]


async def test_run_parses_a_responses_envelope(client_factory):
    envelope = {
        "output": [
            {"type": "web_search_call"},
            {
                "type": "message",
                "content": [
                    {
                        "type": "output_text",
                        "text": '{"regime": "neutral", "go_signal": 0.4, "risk_appetite": 0.3}',
                        "annotations": [{"url": "https://example.com/sol"}],
                    }
                ],
            },
        ],
        "usage": USAGE,
    }
    pulse = CryptoPulse(CONFIG, client=client_factory(FakeResponse("", envelope=envelope)))
    result = await pulse.run()
    assert result["regime"] == "neutral"
    assert result["go_signal"] == 0.4
    assert pulse.last_citations == ["https://example.com/sol"]
    assert pulse.costs.snapshot()["cost_usd"] == pytest.approx(2.5)


# --- parsing still guards the json_object path ----------------------------------------

def test_parser_handles_fences_and_prose():
    assert parse_json_response('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_response('Sure:\n{"a": 1}\nHope that helps') == {"a": 1}
    assert parse_json_response('json: {"a": 1}') == {"a": 1}


def test_parser_rejects_non_objects():
    with pytest.raises(ValueError):
        parse_json_response("[1, 2, 3]")
    with pytest.raises(ValueError):
        parse_json_response("")

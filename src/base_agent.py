"""Base class for every LLM-backed bot.

Wraps one xAI chat completion: strict structured output, live search, retry
policy, cost accounting, and — most importantly — the pessimistic fallback. Every
subclass declares what "we could not get an answer" means for it; the desk never
sees a raised exception from an agent.

Three things here come straight from the OpenAPI spec (see RESEARCH.md):

* `response_format: json_schema` with `strict: true` makes the model return the
  shape we asked for, so parsing is no longer the weak link.
* `search_parameters` is what gives an agent real data. Without it the docs are
  explicit that "no data will be acquired by the model" — the model answers from
  a training cutoff months in the past.
* `reasoning_effort` is only supported by grok-4.3, so it is attached by model
  slug rather than sent blindly.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import re
from dataclasses import dataclass, field
from typing import Any

import httpx

log = logging.getLogger(__name__)

_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)

#: usage.cost_in_usd_ticks -> USD. TICKS_IN_USD_CENT = 100_000_000.
TICKS_PER_USD = 10_000_000_000

#: Only these deserve another attempt. A 400/422 is a bug in our request and
#: will fail identically three times in a row.
RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})


def parse_json_response(text: str) -> dict[str, Any]:
    """Pull a JSON object out of an LLM reply.

    With strict structured outputs this is a formality, but it still guards the
    `json_object` path and any model that decides to narrate.
    """
    if not text:
        raise ValueError("empty response")

    cleaned = _FENCE_RE.sub("", text.strip()).strip()
    if cleaned.lower().startswith("json"):
        cleaned = cleaned[4:].lstrip(": \n")

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start == -1 or end <= start:
            raise ValueError(f"no JSON object in response: {text[:200]!r}")
        try:
            parsed = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ValueError(f"unparseable JSON: {exc}") from exc

    if not isinstance(parsed, dict):
        raise ValueError(f"expected a JSON object, got {type(parsed).__name__}")
    return parsed


@dataclass
class CostTracker:
    """Running spend, straight from what the API billed us.

    `cost_in_usd_ticks` is exact, so there is no reason to estimate from token
    counts and a price table that goes stale.
    """

    calls: int = 0
    failed_calls: int = 0
    fallbacks: int = 0
    prompt_tokens: int = 0
    cached_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    sources_used: int = 0
    cost_usd: float = 0.0
    by_agent: dict[str, float] = field(default_factory=dict)

    def record(self, agent: str, usage: dict[str, Any] | None) -> None:
        self.calls += 1
        if not usage:
            return
        self.prompt_tokens += int(usage.get("prompt_tokens", 0) or 0)
        self.completion_tokens += int(usage.get("completion_tokens", 0) or 0)
        self.sources_used += int(usage.get("num_sources_used", 0) or 0)
        self.cached_tokens += int(
            (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0
        )
        self.reasoning_tokens += int(
            (usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0) or 0
        )
        ticks = usage.get("cost_in_usd_ticks")
        if ticks:
            spend = int(ticks) / TICKS_PER_USD
            self.cost_usd += spend
            self.by_agent[agent] = round(self.by_agent.get(agent, 0.0) + spend, 8)

    def snapshot(self) -> dict[str, Any]:
        cache_rate = self.cached_tokens / self.prompt_tokens if self.prompt_tokens else 0.0
        return {
            "calls": self.calls,
            "failed_calls": self.failed_calls,
            "fallbacks": self.fallbacks,
            "cost_usd": round(self.cost_usd, 6),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "cached_tokens": self.cached_tokens,
            "cache_hit_rate": round(cache_rate, 3),
            "sources_used": self.sources_used,
            "by_agent": dict(sorted(self.by_agent.items(), key=lambda kv: -kv[1])),
        }


class GrokAgent:
    """One prompt, one JSON answer, one pessimistic fallback."""

    name: str = "agent"
    #: Static instruction block. Kept first in the message so it caches.
    PROMPT: str = ""
    #: "fast" for generators, "deep" for the adversarial checkers.
    model_tier: str = "fast"
    #: Strict JSON schema for the reply. None falls back to json_object.
    SCHEMA: dict[str, Any] | None = None
    #: Live-search policy. None means the agent needs no external data.
    SEARCH: dict[str, Any] | None = None

    def __init__(
        self,
        config: dict[str, Any],
        client: httpx.AsyncClient | None = None,
        costs: CostTracker | None = None,
    ):
        self.config = config or {}
        grok = self.config.get("grok", {}) or {}
        self.api_key = grok.get("api_key", "")
        self.base_url = grok.get("base_url", "https://api.x.ai/v1/chat/completions")

        models = grok.get("models", {}) or {}
        # Legacy keys stay readable so an old config does not silently pick a
        # different model than its author intended.
        default = "grok-4.6" if self.model_tier == "deep" else "grok-4.3"
        legacy = grok.get("full_model") if self.model_tier == "deep" else grok.get("fast_model")
        self.model = models.get(self.model_tier) or legacy or default

        efforts = grok.get("reasoning_effort", {}) or {}
        self.reasoning_effort = efforts.get(self.model_tier, "none" if self.model_tier == "fast" else None)

        self.timeout = float(grok.get("timeout_seconds", 30))
        self.max_retries = int(grok.get("max_retries", 3))
        self.max_backoff = float(grok.get("max_backoff_seconds", 30))
        self.structured_outputs = bool(grok.get("structured_outputs", True))
        self.live_search = bool(grok.get("live_search", True))
        self.max_search_results = int(grok.get("max_search_results", 15))

        self._client = client
        self.costs = costs if costs is not None else CostTracker()
        #: Optional OutcomeMemory. When set, `facts()` output is augmented with
        #: what happened on comparable past trades.
        self.memory: Any = None
        self.last_citations: list[str] = []
        self.last_usage: dict[str, Any] = {}

    # -- overridden by subclasses -------------------------------------------------

    def facts(self, payload: Any) -> Any:
        """The variable half of the prompt. Subclasses narrow this to what matters."""
        return payload

    def memory_context(self, payload: Any) -> dict[str, Any]:
        """Past-outcome block for this payload. Subclasses that can be matched
        against history override this; the default recalls nothing."""
        return {}

    def build_messages(self, payload: Any) -> list[dict[str, str]]:
        """Static instructions first, variable facts second.

        Ordering is not cosmetic: the cache matches on prefixes, so a constant
        opening block is what makes `cached_tokens` non-zero.
        """
        facts = self.facts(payload)
        recalled = self.memory_context(payload) if self.memory is not None else {}
        if recalled and isinstance(facts, dict):
            facts = {**facts, **recalled}
        rendered = json.dumps(facts, default=str, indent=None) if facts is not None else "{}"
        return [
            {"role": "system", "content": self.PROMPT},
            {"role": "user", "content": rendered},
        ]

    def search_parameters(self) -> dict[str, Any] | None:
        """Live-search policy for this call, or None to answer from the prior."""
        if not self.live_search or self.SEARCH is None:
            return None
        params = {"max_search_results": self.max_search_results, **self.SEARCH}
        return params

    def fallback(self) -> dict[str, Any]:
        """What this agent returns when the model is unusable.

        Subclasses MUST make this the safe answer: refuse to buy, hold the
        position, do not skew the allocation.
        """
        raise NotImplementedError

    def postprocess(self, data: dict[str, Any]) -> dict[str, Any]:
        """Hook for coercion/validation. Raise ValueError to trigger the fallback."""
        return data

    # -- request assembly ---------------------------------------------------------

    def build_request(self, payload: Any) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "temperature": 0,
            "messages": self.build_messages(payload),
            # Sticky routing for prompt-cache hits; stable per agent by design.
            "prompt_cache_key": f"grok-desk:{self.name}",
        }

        if self.structured_outputs and self.SCHEMA is not None:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": self.name, "schema": self.SCHEMA, "strict": True},
            }
        else:
            body["response_format"] = {"type": "json_object"}

        # reasoning_effort is a grok-4.3-only parameter; sending it elsewhere errors.
        if self.reasoning_effort and self.model.startswith("grok-4.3"):
            body["reasoning_effort"] = self.reasoning_effort

        search = self.search_parameters()
        if search:
            body["search_parameters"] = search

        return body

    # -- transport ----------------------------------------------------------------

    async def _post(self, client: httpx.AsyncClient, body: dict[str, Any]) -> dict[str, Any]:
        response = await client.post(
            self.base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=body,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    @staticmethod
    def _status_of(exc: Exception) -> int | None:
        response = getattr(exc, "response", None)
        return getattr(response, "status_code", None)

    def _retry_delay(self, attempt: int, exc: Exception) -> float:
        """Exponential backoff with jitter, honouring Retry-After when given."""
        response = getattr(exc, "response", None)
        headers = getattr(response, "headers", None) or {}
        retry_after = headers.get("retry-after") or headers.get("Retry-After")
        if retry_after:
            try:
                return min(float(retry_after), 60.0)
            except (TypeError, ValueError):
                pass
        # Jitter downward from the ceiling so the cap really is a cap, and so
        # a fleet of agents that failed together does not retry in lockstep.
        ceiling = min(2.0**attempt, self.max_backoff)
        return random.uniform(ceiling / 2, ceiling)

    def _should_retry(self, exc: Exception) -> bool:
        status = self._status_of(exc)
        if status is None:
            # Timeouts, connection resets, malformed JSON: worth another go.
            return True
        return status in RETRYABLE_STATUS

    async def run(self, payload: Any = None) -> dict[str, Any]:
        """Call the model, returning parsed JSON or the pessimistic fallback."""
        body = self.build_request(payload)
        client = self._client
        owns_client = client is None
        if owns_client:
            client = httpx.AsyncClient(timeout=self.timeout)

        try:
            last_error: Exception | None = None
            for attempt in range(self.max_retries):
                try:
                    envelope = await asyncio.wait_for(
                        self._post(client, body), timeout=self.timeout
                    )
                    self.last_usage = envelope.get("usage") or {}
                    self.last_citations = list(envelope.get("citations") or [])
                    self.costs.record(self.name, self.last_usage)
                    content = envelope["choices"][0]["message"]["content"]
                    return self.postprocess(parse_json_response(content))
                except Exception as exc:  # noqa: BLE001 - any failure means fallback
                    last_error = exc
                    self.costs.failed_calls += 1
                    status = self._status_of(exc)
                    log.warning(
                        "%s attempt %d/%d failed (status=%s): %s",
                        self.name, attempt + 1, self.max_retries, status, exc,
                    )
                    if not self._should_retry(exc):
                        log.error("%s: not retryable, falling back immediately", self.name)
                        break
                    if attempt + 1 < self.max_retries:
                        await asyncio.sleep(self._retry_delay(attempt, exc))

            log.error("%s falling back: %s", self.name, last_error)
            self.costs.fallbacks += 1
            return self.fallback()
        finally:
            if owns_client:
                await client.aclose()


def clamp01(value: Any, default: float = 0.0) -> float:
    """Coerce anything to a 0..1 float; unparseable values become `default`."""
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def schema(properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    """Build a strict object schema.

    `additionalProperties` must be explicitly false for strict mode, and the API
    requires every property to be listed in `required`.
    """
    return {
        "type": "object",
        "properties": properties,
        "required": required if required is not None else list(properties),
        "additionalProperties": False,
    }


# Small schema helpers, so agent modules read as declarations rather than JSON.
UNIT = {"type": "number", "minimum": 0, "maximum": 1}
BOOL = {"type": "boolean"}
TEXT = {"type": "string"}
NUM = {"type": "number"}


def enum(*values: str) -> dict[str, Any]:
    return {"type": "string", "enum": list(values)}


def string_list(max_items: int = 12) -> dict[str, Any]:
    return {"type": "array", "items": {"type": "string"}, "maxItems": max_items}

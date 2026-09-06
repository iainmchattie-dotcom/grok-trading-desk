"""Base class for every LLM-backed bot.

Wraps one xAI call: strict structured output, Agent Tools retrieval, retry
policy, cost accounting, and — most importantly — the pessimistic fallback. Every
subclass declares what "we could not get an answer" means for it; the desk never
sees a raised exception from an agent.

Three things here come straight from the upstream docs (see RESEARCH.md):

* `response_format: json_schema` with `strict: true` makes the model return the
  shape we asked for, so parsing is no longer the weak link.
* Live Search (`search_parameters` on `/v1/chat/completions`) was retired and
  now returns HTTP 410 Gone. Agents that need current data send `web_search` /
  `x_search` tools on `/v1/responses` instead. `grok.live_search: false` skips
  retrieval and answers from the training cutoff.
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
#: will fail identically three times in a row. 410 means the feature is gone.
RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})

_CHAT_COMPLETIONS_SUFFIX = "/chat/completions"


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


def derive_responses_url(chat_url: str) -> str:
    """Map a chat/completions URL to the Responses API sibling.

    Agent Tools (`web_search`, `x_search`) are served on `/v1/responses`.
    Allocators and other no-retrieval agents stay on chat/completions.
    """
    url = (chat_url or "").rstrip("/")
    if url.endswith(_CHAT_COMPLETIONS_SUFFIX):
        return url[: -len(_CHAT_COMPLETIONS_SUFFIX)] + "/responses"
    if url.endswith("/responses"):
        return url
    if url.endswith("/v1"):
        return f"{url}/responses"
    return f"{url}/responses" if url else "https://api.x.ai/v1/responses"


def _text_from_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content") or ""
                if text:
                    parts.append(str(text))
        return "".join(parts)
    return str(content)


def extract_message_content(envelope: dict[str, Any]) -> str:
    """Pull the model text out of a chat/completions or Responses envelope."""
    choices = envelope.get("choices")
    if isinstance(choices, list) and choices:
        message = (choices[0] or {}).get("message") or {}
        text = _text_from_content(message.get("content"))
        if text:
            return text

    output_text = envelope.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text

    texts: list[str] = []
    for item in envelope.get("output") or []:
        if not isinstance(item, dict):
            continue
        if item.get("type") in {None, "message", "output_text"}:
            text = _text_from_content(item.get("content") or item.get("text"))
            if text:
                texts.append(text)
    if texts:
        return "".join(texts)
    raise ValueError("no message content in response")


def extract_citations(envelope: dict[str, Any]) -> list[str]:
    """Citations from either API shape (top-level list or output annotations)."""
    raw = envelope.get("citations")
    if isinstance(raw, list) and raw:
        return [str(item) for item in raw if item]

    found: list[str] = []
    for item in envelope.get("output") or []:
        if not isinstance(item, dict):
            continue
        for part in item.get("content") or []:
            if not isinstance(part, dict):
                continue
            for annotation in part.get("annotations") or []:
                if not isinstance(annotation, dict):
                    continue
                url = annotation.get("url") or annotation.get("source")
                if url:
                    found.append(str(url))
    return found


def search_tools_from_policy(params: dict[str, Any] | None) -> list[dict[str, Any]] | None:
    """Translate a Live Search policy dict into Agent Tools.

    `news` has no dedicated tool; it is folded into `web_search`. Engagement
    floors (`post_view_count`) are not accepted by `x_search` and are dropped.
    """
    if not params:
        return None

    from_date = params.get("from_date")
    to_date = params.get("to_date")
    sources = params.get("sources") or []

    want_web = False
    want_x = False
    allowed_domains: list[str] = []
    excluded_domains: list[str] = []
    allowed_handles: list[str] = []
    excluded_handles: list[str] = []

    if not sources:
        want_web = True
        want_x = True

    for src in sources:
        if not isinstance(src, dict):
            continue
        stype = src.get("type")
        if stype in {"web", "news"}:
            want_web = True
            allowed_domains.extend(src.get("allowed_websites") or src.get("allowed_domains") or [])
            excluded_domains.extend(src.get("excluded_websites") or src.get("excluded_domains") or [])
        elif stype == "x":
            want_x = True
            allowed_handles.extend(src.get("included_x_handles") or src.get("allowed_x_handles") or [])
            excluded_handles.extend(src.get("excluded_x_handles") or [])

    tools: list[dict[str, Any]] = []
    if want_web:
        tool: dict[str, Any] = {"type": "web_search"}
        filters: dict[str, Any] = {}
        if allowed_domains:
            filters["allowed_domains"] = list(dict.fromkeys(allowed_domains))[:5]
        elif excluded_domains:
            filters["excluded_domains"] = list(dict.fromkeys(excluded_domains))[:5]
        if filters:
            tool["filters"] = filters
        tools.append(tool)
    if want_x:
        tool = {"type": "x_search"}
        if from_date:
            tool["from_date"] = from_date
        if to_date:
            tool["to_date"] = to_date
        if allowed_handles:
            tool["allowed_x_handles"] = list(dict.fromkeys(allowed_handles))[:20]
        elif excluded_handles:
            tool["excluded_x_handles"] = list(dict.fromkeys(excluded_handles))[:20]
        tools.append(tool)
    return tools or None


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
        self.responses_url = grok.get("responses_url") or derive_responses_url(self.base_url)

        models = grok.get("models", {}) or {}
        # Legacy keys stay readable so an old config does not silently pick a
        # different model than its author intended.
        default = "grok-4.6" if self.model_tier == "deep" else "grok-4.3"
        legacy = grok.get("full_model") if self.model_tier == "deep" else grok.get("fast_model")
        self.model = models.get(self.model_tier) or legacy or default

        efforts = grok.get("reasoning_effort", {}) or {}
        self.reasoning_effort = efforts.get(self.model_tier, "none" if self.model_tier == "fast" else None)

        self.timeout = float(grok.get("timeout_seconds", 30))
        # Agent Tools run a server-side loop; 30s is often tight for pulse.
        self.tool_timeout = float(grok.get("tool_timeout_seconds", max(self.timeout, 60.0)))
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
        """Retrieval policy for this call, or None to answer from the prior.

        This is the in-process policy object (sources, `from_date`). It is
        never sent on the wire — Live Search's `search_parameters` field
        returns HTTP 410. `search_tools()` is what the request actually carries.
        """
        if not self.live_search or self.SEARCH is None:
            return None
        params = {"max_search_results": self.max_search_results, **self.SEARCH}
        return params

    def search_tools(self) -> list[dict[str, Any]] | None:
        """Agent Tools for this call (`web_search` / `x_search`), or None."""
        return search_tools_from_policy(self.search_parameters())

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
        messages = self.build_messages(payload)
        body: dict[str, Any] = {
            "model": self.model,
            "temperature": 0,
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

        tools = self.search_tools()
        if tools:
            # Responses API: tools run server-side. Never send search_parameters;
            # that field is retired and the chat/completions endpoint returns 410.
            body["input"] = messages
            body["tools"] = tools
            body["store"] = False
        else:
            body["messages"] = messages

        return body

    def request_url(self, body: dict[str, Any]) -> str:
        """Chat/completions when there is no retrieval; Responses when there is."""
        if body.get("tools"):
            return self.responses_url
        return self.base_url

    def request_timeout(self, body: dict[str, Any]) -> float:
        return self.tool_timeout if body.get("tools") else self.timeout

    # -- transport ----------------------------------------------------------------

    async def _post(self, client: httpx.AsyncClient, body: dict[str, Any]) -> dict[str, Any]:
        response = await client.post(
            self.request_url(body),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=body,
            timeout=self.request_timeout(body),
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
            client = httpx.AsyncClient(timeout=max(self.timeout, self.tool_timeout))

        try:
            last_error: Exception | None = None
            timeout = self.request_timeout(body)
            for attempt in range(self.max_retries):
                try:
                    envelope = await asyncio.wait_for(
                        self._post(client, body), timeout=timeout
                    )
                    self.last_usage = envelope.get("usage") or {}
                    self.last_citations = extract_citations(envelope)
                    self.costs.record(self.name, self.last_usage)
                    content = extract_message_content(envelope)
                    return self.postprocess(parse_json_response(content))
                except Exception as exc:  # noqa: BLE001 - any failure means fallback
                    last_error = exc
                    self.costs.failed_calls += 1
                    status = self._status_of(exc)
                    log.warning(
                        "%s attempt %d/%d failed (status=%s): %s",
                        self.name, attempt + 1, self.max_retries, status, exc,
                    )
                    if status == 410:
                        log.error(
                            "%s: HTTP 410 Gone — a requested xAI feature is retired. "
                            "Live Search (search_parameters) is gone; retrieval now "
                            "uses Agent Tools on /v1/responses. Set grok.live_search: "
                            "false to skip retrieval and answer from the cutoff.",
                            self.name,
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

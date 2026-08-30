"""Base class for every LLM-backed bot.

Encapsulates the xAI call, JSON extraction, retries with exponential backoff,
timeout, and — most importantly — the pessimistic fallback. Every subclass
declares what "we could not get an answer" means for it; the desk never sees a
raised exception from an agent.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

import httpx

log = logging.getLogger(__name__)

_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


def parse_json_response(text: str) -> dict[str, Any]:
    """Pull a JSON object out of an LLM reply.

    Strips markdown fences and a leading bare ``json`` token, then falls back to
    the outermost brace pair if the model wrapped the object in prose.
    Raises ValueError when nothing parses.
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


class GrokAgent:
    """One prompt, one JSON answer, one pessimistic fallback."""

    name: str = "agent"
    #: PROMPT constant lives on the subclass.
    PROMPT: str = ""
    #: "fast" for generators, "full" for the adversarial checkers.
    model_tier: str = "fast"

    def __init__(self, config: dict[str, Any], client: httpx.AsyncClient | None = None):
        self.config = config or {}
        grok = self.config.get("grok", {}) or {}
        self.api_key = grok.get("api_key", "")
        self.base_url = grok.get("base_url", "https://api.x.ai/v1/chat/completions")
        self.model = (
            grok.get("full_model", "grok-4")
            if self.model_tier == "full"
            else grok.get("fast_model", "grok-4-fast")
        )
        self.timeout = float(grok.get("timeout_seconds", 30))
        self.max_retries = int(grok.get("max_retries", 3))
        self._client = client

    # -- overridden by subclasses -------------------------------------------------

    def build_prompt(self, payload: Any) -> str:
        """Render the user message. Default: PROMPT plus the payload as JSON."""
        return f"{self.PROMPT}\n\nINPUT:\n{json.dumps(payload, default=str)}"

    def fallback(self) -> dict[str, Any]:
        """What this agent returns when the model is unusable.

        Subclasses MUST make this the safe answer: refuse to buy, hold the
        position, do not skew the allocation.
        """
        raise NotImplementedError

    def postprocess(self, data: dict[str, Any]) -> dict[str, Any]:
        """Hook for coercion/validation. Raise ValueError to trigger the fallback."""
        return data

    # -- transport ----------------------------------------------------------------

    async def _post(self, client: httpx.AsyncClient, prompt: str) -> str:
        response = await client.post(
            self.base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "temperature": 0,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        body = response.json()
        return body["choices"][0]["message"]["content"]

    async def run(self, payload: Any = None) -> dict[str, Any]:
        """Call the model, returning parsed JSON or the pessimistic fallback."""
        prompt = self.build_prompt(payload)
        client = self._client
        owns_client = client is None
        if owns_client:
            client = httpx.AsyncClient(timeout=self.timeout)

        try:
            last_error: Exception | None = None
            for attempt in range(self.max_retries):
                try:
                    raw = await asyncio.wait_for(
                        self._post(client, prompt), timeout=self.timeout
                    )
                    return self.postprocess(parse_json_response(raw))
                except Exception as exc:  # noqa: BLE001 - any failure means fallback
                    last_error = exc
                    log.warning(
                        "%s attempt %d/%d failed: %s",
                        self.name,
                        attempt + 1,
                        self.max_retries,
                        exc,
                    )
                    if attempt + 1 < self.max_retries:
                        await asyncio.sleep(2**attempt)
            log.error("%s falling back after %d attempts: %s", self.name, self.max_retries, last_error)
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

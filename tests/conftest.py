import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import json as _json
from typing import Any

import httpx
import pytest


class FakeResponse:
    """Minimal stand-in for httpx.Response as base_agent uses it."""

    def __init__(
        self,
        content: str,
        status_code: int = 200,
        usage: dict[str, Any] | None = None,
        citations: list[str] | None = None,
        headers: dict[str, str] | None = None,
        envelope: dict[str, Any] | None = None,
    ):
        self._content = content
        self.status_code = status_code
        self.usage = usage
        self.citations = citations
        self.headers = headers or {}
        self._envelope = envelope

    def raise_for_status(self):
        if self.status_code >= 400:
            error = httpx.HTTPStatusError(
                f"status {self.status_code}",
                request=httpx.Request("POST", "https://api.x.ai/v1/chat/completions"),
                response=None,  # type: ignore[arg-type]
            )
            # base_agent reads exc.response.status_code to decide retryability
            error.response = self  # type: ignore[assignment]
            raise error

    def json(self) -> dict[str, Any]:
        if self._envelope is not None:
            return self._envelope
        body: dict[str, Any] = {"choices": [{"message": {"content": self._content}}]}
        if self.usage is not None:
            body["usage"] = self.usage
        if self.citations is not None:
            body["citations"] = self.citations
        return body


class FakeClient:
    """AsyncClient stub. Replays `replies` in order; the last one repeats."""

    def __init__(self, replies: list[Any]):
        self.replies = list(replies)
        self.calls: list[dict[str, Any]] = []

    async def post(self, url, headers=None, json=None, timeout=None):
        self.calls.append({"url": url, "headers": headers, "json": json})
        reply = self.replies[min(len(self.calls) - 1, len(self.replies) - 1)]
        if isinstance(reply, Exception):
            raise reply
        # Duck-typed on purpose: pytest imports this file as `conftest` while
        # the test modules import it as `tests.conftest`, so isinstance against
        # FakeResponse compares two different class objects and always fails.
        if hasattr(reply, "raise_for_status"):
            return reply
        if isinstance(reply, (dict, list)):
            return FakeResponse(_json.dumps(reply))
        return FakeResponse(str(reply))

    async def aclose(self):
        pass


@pytest.fixture
def client_factory():
    def make(*replies):
        return FakeClient(list(replies))

    return make


@pytest.fixture
def no_sleep(monkeypatch):
    """Kill the exponential backoff so retry paths run instantly."""
    import asyncio

    async def instant(_seconds):
        return None

    monkeypatch.setattr(asyncio, "sleep", instant)


CONFIG: dict[str, Any] = {
    "grok": {
        "api_key": "test-key",
        "base_url": "https://api.x.ai/v1/chat/completions",
        "models": {"fast": "grok-4.3", "deep": "grok-4.6"},
        "reasoning_effort": {"fast": "none"},
        "structured_outputs": True,
        "live_search": True,
        "timeout_seconds": 5,
        "max_retries": 3,
    },
    "risk": {"crypto_max_pct": 0.7, "stock_max_pct": 0.7},
    "pulse": {"crypto_cache_minutes": 15, "market_cache_minutes": 30},
}

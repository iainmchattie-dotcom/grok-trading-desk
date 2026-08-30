import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import json as _json
from typing import Any

import httpx
import pytest


class FakeResponse:
    """Minimal stand-in for httpx.Response as base_agent uses it."""

    def __init__(self, content: str, status_code: int = 200):
        self._content = content
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                f"status {self.status_code}",
                request=httpx.Request("POST", "https://api.x.ai/v1/chat/completions"),
                response=None,  # type: ignore[arg-type]
            )

    def json(self) -> dict[str, Any]:
        return {"choices": [{"message": {"content": self._content}}]}


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
        if isinstance(reply, FakeResponse):
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
        "fast_model": "grok-4-fast",
        "full_model": "grok-4",
        "timeout_seconds": 5,
        "max_retries": 3,
    },
    "risk": {"crypto_max_pct": 0.7, "stock_max_pct": 0.7},
    "pulse": {"crypto_cache_minutes": 15, "market_cache_minutes": 30},
}

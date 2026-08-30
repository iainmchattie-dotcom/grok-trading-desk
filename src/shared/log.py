"""JSONL event log. One line per record, append-only.

Everything the desk decides ends up here: it is the only record `scripts/replay.py`
and `scripts/dashboard.py` read. Records are never rewritten.
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


class EventLog:
    """Thread-safe append-only JSONL writer."""

    def __init__(self, config: dict[str, Any] | None = None, path: str | Path | None = None):
        logging_cfg = ((config or {}).get("logging", {}) or {})
        self.path = Path(path or logging_cfg.get("path", "logs/desk.jsonl"))
        self.echo_stdout = bool(logging_cfg.get("echo_stdout", True))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(self, record_type: str, **fields: Any) -> dict[str, Any]:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "type": record_type,
            **fields,
        }
        line = json.dumps(record, default=str, ensure_ascii=False)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        if self.echo_stdout:
            log.info("%s", line)
        return record

    # -- the five record types ----------------------------------------------------

    def buy(
        self,
        market: str,
        symbol: str,
        score: float,
        all_agent_scores: dict[str, Any],
        amount: float,
        tx_id: str = "",
    ) -> dict[str, Any]:
        return self.write(
            "buy",
            market=market,
            symbol=symbol,
            score=score,
            all_agent_scores=all_agent_scores,
            amount=amount,
            tx_id=tx_id,
        )

    def skip(self, market: str, symbol: str, reason: str, detail: Any = None) -> dict[str, Any]:
        return self.write("skip", market=market, symbol=symbol, reason=reason, detail=detail)

    def close(
        self, market: str, symbol: str, pnl: float, hold_time: float, **extra: Any
    ) -> dict[str, Any]:
        return self.write(
            "close", market=market, symbol=symbol, pnl=pnl, hold_time=hold_time, **extra
        )

    def action(self, symbol: str, action: str, reason: str, **extra: Any) -> dict[str, Any]:
        return self.write("action", symbol=symbol, action=action, reason=reason, **extra)

    def allocation(self, crypto_pct: float, stocks_pct: float, reason: str) -> dict[str, Any]:
        return self.write(
            "allocation", crypto_pct=crypto_pct, stocks_pct=stocks_pct, reason=reason
        )

    # -- reading back ----------------------------------------------------------------

    def read(self) -> list[dict[str, Any]]:
        """Load every record. Malformed lines are skipped, not fatal."""
        return read_log(self.path)


def read_log(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                log.warning("skipping malformed log line %d in %s", line_number, path)
    return records

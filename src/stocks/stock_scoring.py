"""Stock scoring matrix. Pure code.

Hard vetoes live here, not in prompts: active controversy, or heavy insider
selling with no offsetting buying, kills the candidate before scoring.
"""

from __future__ import annotations

from typing import Any

from ..models import Stock

DEFAULT_WEIGHTS = {
    "fundamentals": 0.25,
    "technicals": 0.25,
    "news_sentiment": 0.20,
    "insider": 0.15,
    "pulse": 0.15,
    "min_score_to_buy": 0.60,
}

CONTROVERSY_VETO = 0.7
INSIDER_SELLING_VETO = 0.8
INSIDER_BUYING_FLOOR = 0.2


def hard_veto(
    radar: dict[str, Any],
    insider: dict[str, Any],
    pulse: dict[str, Any],
    min_go_signal: float = 0.3,
) -> str | None:
    """Return the veto reason, or None. Checked before scoring."""
    if float(radar.get("controversy", 1.0)) > CONTROVERSY_VETO:
        return "veto_controversy"
    selling = float(insider.get("insider_selling", 1.0))
    buying = float(insider.get("insider_buying", 0.0))
    if selling > INSIDER_SELLING_VETO and buying < INSIDER_BUYING_FLOOR:
        return "veto_insider_selling"
    if float(pulse.get("go_signal", 0.0)) < min_go_signal:
        return "veto_market_paused"
    return None


def news_score(radar: dict[str, Any]) -> float:
    sentiment = float(radar.get("sentiment_score", 0.0))
    momentum = float(radar.get("news_momentum", 0.0))
    controversy = float(radar.get("controversy", 1.0))
    return round(max(0.0, min(1.0, 0.6 * sentiment + 0.4 * momentum - 0.5 * controversy)), 4)


def insider_score(insider: dict[str, Any]) -> float:
    buying = float(insider.get("insider_buying", 0.0))
    selling = float(insider.get("insider_selling", 1.0))
    flow = float(insider.get("institutional_flow", 0.0))
    score = 0.55 * buying + 0.3 * flow - 0.25 * selling
    if insider.get("cluster_buying"):
        score += 0.15  # several officers buying at once is the strongest single signal
    return round(max(0.0, min(1.0, score)), 4)


def score_stock(
    stock: Stock,
    analyst: dict[str, Any],
    radar: dict[str, Any],
    insider: dict[str, Any],
    pulse: dict[str, Any],
    weights: dict[str, Any] | None = None,
    min_go_signal: float = 0.3,
) -> dict[str, Any]:
    """Weighted score plus the buy/skip verdict and its components."""
    w = {**DEFAULT_WEIGHTS, **(weights or {})}

    veto = hard_veto(radar, insider, pulse, min_go_signal)
    components = {
        "fundamentals": max(0.0, min(1.0, float(analyst.get("fundamentals_score", 0.0)))),
        "technicals": max(0.0, min(1.0, float(analyst.get("technicals_score", 0.0)))),
        "news_sentiment": news_score(radar),
        "insider": insider_score(insider),
        "pulse": float(pulse.get("go_signal", 0.0)),
    }

    if veto is not None:
        return {
            "score": 0.0,
            "buy": False,
            "reason": veto,
            "vetoed": True,
            "components": components,
        }

    denominator = sum(w[k] for k in components) or 1.0
    score = round(sum(components[k] * w[k] for k in components) / denominator, 4)
    threshold = float(w["min_score_to_buy"])
    return {
        "score": score,
        "buy": score >= threshold,
        "reason": "above_threshold" if score >= threshold else "below_threshold",
        "vetoed": False,
        "components": components,
        "threshold": threshold,
    }

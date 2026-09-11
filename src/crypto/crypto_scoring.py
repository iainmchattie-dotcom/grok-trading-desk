"""Crypto scoring matrix. Pure code — no model gets to move these weights.

Hard vetoes live here, not in prompts. Wash trading still kills the candidate
before any weighted score is computed. Coordinated-buy rings do too — but only
when the watch-window tape (or measured holder concentration) actually looks
like a ring. An LLM flag on thin or ambiguous data is a score penalty, not a
skip: pump.fun's first minutes always look "clustered", and the feed does not
ship a funding graph.
"""

from __future__ import annotations

from typing import Any

from ..models import Token

DEFAULT_WEIGHTS = {
    "audit_safety": 0.35,
    "narrative": 0.30,
    "momentum": 0.20,
    "pulse": 0.15,
    "min_score_to_buy": 0.62,
}

# Watch-window corroboration. PumpPortal never sends a holder graph, so the
# only structural ring signal on a typical launch is "few wallets, many buys".
MIN_BUYS_FOR_RING = 8
STRONG_DIVERSITY_MAX = 0.40  # unique traders / buys; 12 buys from 5 wallets = 0.42
STRONG_TOP10 = 0.70          # only when concentration is actually measured (> 0)
STRONG_DEV_HOLDING = 0.25
# LLM said ring, but the tape does not corroborate. Keeps safety in the score
# without zeroing the whole candidate.
COORDINATED_SUSPICION_PENALTY = 0.20


def _trader_count(token: Token) -> int:
    """Watch window sets both; tests sometimes populate only `holders`."""
    return int(token.unique_traders or token.holders or 0)


def trader_diversity(token: Token) -> float | None:
    """Unique traders per buy. Near 1.0 is organic; well below 0.4 is a ring.

    Returns None when there is not enough tape to judge — missing counts are
    unknown, not proof of coordination.
    """
    traders = _trader_count(token)
    if token.buys < MIN_BUYS_FOR_RING or traders <= 0:
        return None
    return round(traders / token.buys, 4)


def holder_concentration_known(token: Token) -> bool:
    """`top10_holder_pct` defaults to 0.0 on Token because the feed never
    sends it. Zero therefore means unmeasured, not 'evenly distributed'."""
    return token.top10_holder_pct > 0


def manipulation_evidence(token: Token | None, audit: dict[str, Any]) -> dict[str, Any]:
    """How strongly the facts we actually have back a coordinated-buy ring.

    strength:
      strong — hard-veto. Repeat-buyer concentration, or the LLM flag plus
               measured holder/dev concentration.
      weak   — LLM flagged a ring but the tape is organic or too thin to
               corroborate. Soft penalty only.
      none   — no LLM flag and no structural ring.
    """
    if token is None:
        flagged = bool(audit.get("coordinated_buys"))
        return {
            "strength": "weak" if flagged else "none",
            "signals": [],
            "diversity": None,
            "concentration_known": False,
            "trader_count": 0,
        }

    signals: list[str] = []
    diversity = trader_diversity(token)
    if diversity is not None and diversity <= STRONG_DIVERSITY_MAX:
        signals.append("repeat_buyers")

    concentration_known = holder_concentration_known(token)
    if concentration_known and token.top10_holder_pct >= STRONG_TOP10:
        signals.append("top10_concentration")
    dev_known = token.dev_holding_pct > 0
    if dev_known and token.dev_holding_pct >= STRONG_DEV_HOLDING:
        signals.append("dev_holding")

    flagged = bool(audit.get("coordinated_buys"))
    structural_ring = "repeat_buyers" in signals
    corroborating_concentration = bool(
        {"top10_concentration", "dev_holding"} & set(signals)
    )

    if structural_ring or (flagged and corroborating_concentration):
        strength = "strong"
    elif flagged:
        strength = "weak"
    else:
        strength = "none"

    return {
        "strength": strength,
        "signals": signals,
        "diversity": diversity,
        "concentration_known": concentration_known,
        "trader_count": _trader_count(token),
    }


def audit_unavailable(audit: dict[str, Any]) -> bool:
    """Parse/API failure. Distinct from a model that actually audited."""
    if audit.get("audit_unavailable"):
        return True
    return "audit_unavailable" in (audit.get("red_flags") or [])


def hard_veto(
    audit: dict[str, Any],
    pulse: dict[str, Any],
    min_go_signal: float = 0.3,
    token: Token | None = None,
    evidence: dict[str, Any] | None = None,
) -> str | None:
    """Return the veto reason, or None. Checked before scoring.

    Fail-closed on an unreadable audit: we did not get a verdict, so we do
    not buy. Wash trading stays a hard veto. Coordinated buys veto only
    when evidence is strong — an isolated LLM flag is not enough.
    """
    if audit_unavailable(audit):
        return "veto_audit_unavailable"
    if audit.get("wash_trading"):
        return "veto_wash_trading"
    evidence = evidence if evidence is not None else manipulation_evidence(token, audit)
    if evidence.get("strength") == "strong":
        return "veto_coordinated_buys"
    if float(pulse.get("go_signal", 0.0)) < min_go_signal:
        return "veto_market_paused"
    return None


def momentum_score(token: Token) -> float:
    """Buy pressure and holder count, squashed into 0..1."""
    ratio = token.buy_sell_ratio
    ratio_component = min(ratio / 3.0, 1.0) if ratio > 0 else 0.0
    holder_component = min(token.holders / 300.0, 1.0)
    liquidity_component = min(token.liquidity_usd / 100_000.0, 1.0)
    return round(
        0.5 * ratio_component + 0.3 * holder_component + 0.2 * liquidity_component, 4
    )


def narrative_score(narrative: dict[str, Any]) -> float:
    meme = float(narrative.get("meme_score", 0.0))
    virality = float(narrative.get("virality", 0.0))
    originality = float(narrative.get("originality", 0.0))
    community = float(narrative.get("community_signal", 0.0))
    score = 0.4 * meme + 0.3 * virality + 0.15 * originality + 0.15 * community
    if narrative.get("is_derivative"):
        score *= 0.7  # a copy of a running meme is worth less than the original
    return round(max(0.0, min(1.0, score)), 4)


def audit_score(audit: dict[str, Any], *, suspicion: bool = False) -> float:
    safety = float(audit.get("safety_score", 0.0))
    penalty = 0.5 * float(audit.get("sniper_pct", 0.0)) + 0.5 * float(audit.get("insider_pct", 0.0))
    if audit.get("bundled_launch"):
        penalty += 0.3
    if suspicion:
        penalty += COORDINATED_SUSPICION_PENALTY
    return round(max(0.0, min(1.0, safety - penalty)), 4)


def score_token(
    token: Token,
    audit: dict[str, Any],
    narrative: dict[str, Any],
    pulse: dict[str, Any],
    weights: dict[str, Any] | None = None,
    min_go_signal: float = 0.3,
) -> dict[str, Any]:
    """Weighted score plus the buy/skip verdict and its components."""
    w = {**DEFAULT_WEIGHTS, **(weights or {})}

    evidence = manipulation_evidence(token, audit)
    veto = hard_veto(audit, pulse, min_go_signal, token=token, evidence=evidence)
    suspicion = evidence.get("strength") == "weak"
    components = {
        "audit_safety": audit_score(audit, suspicion=suspicion),
        "narrative": narrative_score(narrative),
        "momentum": momentum_score(token),
        "pulse": float(pulse.get("go_signal", 0.0)),
    }

    if veto is not None:
        return {
            "score": 0.0,
            "buy": False,
            "reason": veto,
            "vetoed": True,
            "components": components,
            "manipulation_evidence": evidence,
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
        "manipulation_evidence": evidence,
    }

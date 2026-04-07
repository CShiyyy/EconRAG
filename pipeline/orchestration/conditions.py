"""Re-query condition evaluator — deterministic Python, no LLM.

Compares Agent A per-ticker sentiment against Agent B per-ticker health scores.
Three trigger conditions, any one fires a re-query.
"""

from __future__ import annotations

from pipeline.config import DRIFT_BREACH_THRESHOLD


def evaluate_requery(
    agent_a_output: dict,
    agent_b_output: dict | None,
    source_health: list[dict],
) -> dict:
    """Evaluate whether a re-query is needed.

    Returns:
        {"should_requery": bool, "flagged_tickers": list[str], "reason": str}
    """
    if agent_b_output is None:
        return {"should_requery": False, "flagged_tickers": [], "reason": ""}

    a_tickers = agent_a_output.get("per_ticker", {})
    b_tickers = agent_b_output.get("per_ticker", {})

    flagged: list[str] = []
    reasons: list[str] = []

    common_tickers = set(a_tickers.keys()) & set(b_tickers.keys())

    for ticker in common_tickers:
        a_data = a_tickers[ticker]
        b_data = b_tickers[ticker]

        sentiment = a_data.get("sentiment", "neutral")
        confidence = a_data.get("confidence", "weak")
        health = b_data.get("health_score", "normal")
        drift = abs(b_data.get("drift", 0.0))

        # Condition 1: Conflicting signals
        if sentiment == "bullish" and health in ("warning", "breach"):
            flagged.append(ticker)
            reasons.append(f"Conflicting signals: {ticker} bullish sentiment vs {health} health")
        elif sentiment == "bearish" and health == "normal" and b_data.get("drift", 0.0) > 0:
            flagged.append(ticker)
            reasons.append(f"Conflicting signals: {ticker} bearish sentiment vs normal health with positive drift")

        # Condition 3: Magnitude breach with no catalyst explanation
        elif health == "breach" and sentiment == "neutral" and confidence == "weak":
            flagged.append(ticker)
            reasons.append(f"Magnitude breach: {ticker} health breach but neutral/weak sentiment")
        elif drift > DRIFT_BREACH_THRESHOLD and sentiment == "neutral" and confidence == "weak":
            flagged.append(ticker)
            reasons.append(f"Magnitude breach: {ticker} drift {drift:.3f} exceeds threshold but neutral/weak sentiment")

    if flagged:
        return {
            "should_requery": True,
            "flagged_tickers": flagged,
            "reason": "; ".join(reasons),
        }

    return {"should_requery": False, "flagged_tickers": [], "reason": ""}

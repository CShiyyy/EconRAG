"""Tests for pipeline.orchestration.conditions."""

from pipeline.orchestration.conditions import evaluate_requery


def _agent_a_output(ticker_sentiments: dict) -> dict:
    """Build Agent A output. ticker_sentiments: {ticker: {sentiment, confidence}}."""
    per_ticker = {}
    for ticker, data in ticker_sentiments.items():
        per_ticker[ticker] = {
            "sentiment": data.get("sentiment", "neutral"),
            "confidence": data.get("confidence", "moderate"),
        }
    return {"per_ticker": per_ticker}


def _agent_b_output(ticker_health: dict) -> dict:
    """Build Agent B output. ticker_health: {ticker: {health_score, drift}}."""
    per_ticker = {}
    for ticker, data in ticker_health.items():
        per_ticker[ticker] = {
            "health_score": data.get("health_score", "normal"),
            "drift": data.get("drift", 0.0),
        }
    return {"per_ticker": per_ticker}


def test_first_run_never_triggers():
    result = evaluate_requery(
        _agent_a_output({"AAPL": {"sentiment": "bullish"}}),
        agent_b_output=None,
        source_health=[],
    )
    assert result["should_requery"] is False
    assert result["flagged_tickers"] == []


def test_no_conflict_no_requery():
    result = evaluate_requery(
        _agent_a_output({"AAPL": {"sentiment": "bullish", "confidence": "strong"}}),
        _agent_b_output({"AAPL": {"health_score": "normal", "drift": 0.01}}),
        source_health=[],
    )
    assert result["should_requery"] is False


def test_conflicting_signals_triggers():
    result = evaluate_requery(
        _agent_a_output({"AAPL": {"sentiment": "bullish", "confidence": "strong"}}),
        _agent_b_output({"AAPL": {"health_score": "warning", "drift": 0.04}}),
        source_health=[],
    )
    assert result["should_requery"] is True
    assert "AAPL" in result["flagged_tickers"]


def test_bearish_vs_normal_positive_drift_triggers():
    result = evaluate_requery(
        _agent_a_output({"AAPL": {"sentiment": "bearish", "confidence": "moderate"}}),
        _agent_b_output({"AAPL": {"health_score": "normal", "drift": 0.02}}),
        source_health=[],
    )
    assert result["should_requery"] is True
    assert "AAPL" in result["flagged_tickers"]


def test_magnitude_breach_neutral_triggers():
    result = evaluate_requery(
        _agent_a_output({"AAPL": {"sentiment": "neutral", "confidence": "weak"}}),
        _agent_b_output({"AAPL": {"health_score": "breach", "drift": 0.06}}),
        source_health=[],
    )
    assert result["should_requery"] is True
    assert "AAPL" in result["flagged_tickers"]


def test_multiple_tickers_partial_conflict():
    result = evaluate_requery(
        _agent_a_output({
            "AAPL": {"sentiment": "bullish", "confidence": "strong"},
            "MSFT": {"sentiment": "neutral", "confidence": "moderate"},
        }),
        _agent_b_output({
            "AAPL": {"health_score": "warning", "drift": 0.04},
            "MSFT": {"health_score": "normal", "drift": 0.01},
        }),
        source_health=[],
    )
    assert result["should_requery"] is True
    assert "AAPL" in result["flagged_tickers"]
    assert "MSFT" not in result["flagged_tickers"]

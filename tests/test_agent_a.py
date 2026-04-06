"""Tests for Agent A — Local LLM Context Retriever."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pipeline.agents.agent_a import run_agent_a


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TICKERS = ["AAPL", "NVDA"]


def _valid_ticker_response(sentiment="bullish", confidence="strong"):
    return {
        "sentiment": sentiment,
        "confidence": confidence,
        "key_catalysts": ["earnings beat", "new product"],
        "narrative": "The stock looks strong due to recent earnings." * 3,
    }


def _valid_macro_response():
    return {"macro_overview": "Markets are driven by AI hype and rate expectations."}


def _valid_standing_response(canonical_ids):
    return {
        cid: {
            "still_relevant": True,
            "current_impact": "Ongoing effect on markets.",
            "affected_tickers_update": ["NVDA"],
        }
        for cid in canonical_ids
    }


def _ollama_response(data: dict):
    """Create a mock ollama AsyncClient.chat return value."""
    return {"message": {"content": json.dumps(data)}}


def _seed_run_log(conn) -> int:
    """Insert a run_log row and return run_id."""
    cur = conn.execute(
        "INSERT INTO run_log (timestamp, run_type) VALUES ('2026-04-06T08:30:00', 'pre_open')"
    )
    conn.commit()
    return cur.lastrowid


def _seed_standing_event(conn, run_id: int, canonical_id: str = "FED_RATE_HIKE",
                         affected_tickers: list[str] | None = None):
    """Seed canonical entity + standing event. Returns standing_id."""
    affected = affected_tickers or ["NVDA", "AAPL"]
    conn.execute(
        "INSERT OR IGNORE INTO canonical_entities (canonical_id, entity_type, display_name) "
        "VALUES (?, 'MACRO_THEME', ?)",
        (canonical_id, canonical_id),
    )
    cur = conn.execute(
        "INSERT INTO standing_events "
        "(canonical_id, status, category, summary, affected_tickers, "
        " promoted_at, promotion_source, created_from_run_id) "
        "VALUES (?, 'active', 'monetary_policy', 'Fed rate hike concerns', ?, "
        " '2026-04-05T10:00:00', 'auto', ?)",
        (canonical_id, json.dumps(affected), run_id),
    )
    conn.commit()
    return cur.lastrowid


def _make_mock_client(side_effect):
    """Create a mock AsyncClient whose chat() returns from side_effect list."""
    mock_client = AsyncMock()
    mock_client.chat = AsyncMock(side_effect=side_effect)
    return mock_client


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_rag():
    return MagicMock()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@patch("pipeline.agents.agent_a.query_graph", new_callable=AsyncMock)
@patch("pipeline.agents.agent_a.ollama.AsyncClient")
async def test_output_structure(MockAsyncClient, mock_qg, db_conn, mock_rag):
    """Full run output has macro_overview, standing_context_assessment, per_ticker."""
    mock_qg.return_value = "some context"
    MockAsyncClient.return_value = _make_mock_client([
        _ollama_response(_valid_macro_response()),
        _ollama_response(_valid_ticker_response()),
        _ollama_response(_valid_ticker_response("bearish", "moderate")),
    ])
    run_id = _seed_run_log(db_conn)
    result = await run_agent_a(db_conn, mock_rag, TICKERS, run_id)

    assert "macro_overview" in result
    assert "standing_context_assessment" in result
    assert "per_ticker" in result
    assert set(result["per_ticker"].keys()) == set(TICKERS)


@pytest.mark.asyncio
@patch("pipeline.agents.agent_a.query_graph", new_callable=AsyncMock)
@patch("pipeline.agents.agent_a.ollama.AsyncClient")
async def test_per_ticker_sentiment_values(MockAsyncClient, mock_qg, db_conn, mock_rag):
    """Sentiment and confidence values are within valid sets."""
    mock_qg.return_value = "context"
    MockAsyncClient.return_value = _make_mock_client([
        _ollama_response(_valid_macro_response()),
        _ollama_response(_valid_ticker_response("bearish", "weak")),
        _ollama_response(_valid_ticker_response("neutral", "moderate")),
    ])
    run_id = _seed_run_log(db_conn)
    result = await run_agent_a(db_conn, mock_rag, TICKERS, run_id)

    for ticker_data in result["per_ticker"].values():
        assert ticker_data["sentiment"] in {"bullish", "bearish", "neutral"}
        assert ticker_data["confidence"] in {"strong", "moderate", "weak"}


@pytest.mark.asyncio
@patch("pipeline.agents.agent_a.query_graph", new_callable=AsyncMock)
@patch("pipeline.agents.agent_a.ollama.AsyncClient")
async def test_standing_context_included(MockAsyncClient, mock_qg, db_conn, mock_rag):
    """When standing events exist, standing_context_assessment is populated."""
    mock_qg.return_value = "context"
    run_id = _seed_run_log(db_conn)
    _seed_standing_event(db_conn, run_id)

    MockAsyncClient.return_value = _make_mock_client([
        _ollama_response(_valid_macro_response()),
        _ollama_response(_valid_standing_response(["FED_RATE_HIKE"])),
        _ollama_response(_valid_ticker_response()),
        _ollama_response(_valid_ticker_response()),
    ])
    result = await run_agent_a(db_conn, mock_rag, TICKERS, run_id)

    assert result["standing_context_assessment"] != {}
    assert "FED_RATE_HIKE" in result["standing_context_assessment"]


@pytest.mark.asyncio
@patch("pipeline.agents.agent_a.query_graph", new_callable=AsyncMock)
@patch("pipeline.agents.agent_a.ollama.AsyncClient")
async def test_standing_context_empty(MockAsyncClient, mock_qg, db_conn, mock_rag):
    """When no standing events, standing_context_assessment is empty dict."""
    mock_qg.return_value = "context"
    MockAsyncClient.return_value = _make_mock_client([
        _ollama_response(_valid_macro_response()),
        _ollama_response(_valid_ticker_response()),
        _ollama_response(_valid_ticker_response()),
    ])
    run_id = _seed_run_log(db_conn)
    result = await run_agent_a(db_conn, mock_rag, TICKERS, run_id)

    assert result["standing_context_assessment"] == {}


@pytest.mark.asyncio
@patch("pipeline.agents.agent_a.query_graph", new_callable=AsyncMock)
@patch("pipeline.agents.agent_a.ollama.AsyncClient")
async def test_requery_updates_only_flagged(MockAsyncClient, mock_qg, db_conn, mock_rag):
    """Re-query mode only updates flagged tickers."""
    mock_qg.return_value = "context"
    MockAsyncClient.return_value = _make_mock_client([
        _ollama_response(_valid_ticker_response("bearish", "strong")),
    ])
    run_id = _seed_run_log(db_conn)
    result = await run_agent_a(
        db_conn, mock_rag, TICKERS, run_id, flagged_tickers=["NVDA"]
    )

    assert "per_ticker" in result
    assert list(result["per_ticker"].keys()) == ["NVDA"]
    assert "macro_overview" not in result
    assert "standing_context_assessment" not in result


@pytest.mark.asyncio
@patch("pipeline.agents.agent_a.query_graph", new_callable=AsyncMock)
@patch("pipeline.agents.agent_a.ollama.AsyncClient")
async def test_corrective_retry(MockAsyncClient, mock_qg, db_conn, mock_rag):
    """First Ollama call returns invalid JSON, second succeeds."""
    mock_qg.return_value = "context"
    valid_ticker = _valid_ticker_response()

    # Macro succeeds, first ticker call fails then succeeds (retry), second ticker succeeds
    MockAsyncClient.return_value = _make_mock_client([
        _ollama_response(_valid_macro_response()),
        # First ticker: invalid then valid (retry)
        {"message": {"content": "not json at all"}},
        _ollama_response(valid_ticker),
        # Second ticker: succeeds
        _ollama_response(valid_ticker),
    ])
    run_id = _seed_run_log(db_conn)
    result = await run_agent_a(db_conn, mock_rag, TICKERS, run_id)

    # Both tickers should have valid output (first via retry)
    assert result["per_ticker"]["AAPL"]["sentiment"] == "bullish"
    assert result["per_ticker"]["NVDA"]["sentiment"] == "bullish"


@pytest.mark.asyncio
@patch("pipeline.agents.agent_a.query_graph", new_callable=AsyncMock)
@patch("pipeline.agents.agent_a.ollama.AsyncClient")
async def test_degraded_output(MockAsyncClient, mock_qg, db_conn, mock_rag):
    """When Ollama fails twice, degraded neutral/weak output is produced."""
    mock_qg.return_value = "context"

    # Macro succeeds, AAPL fails twice, NVDA succeeds
    MockAsyncClient.return_value = _make_mock_client([
        _ollama_response(_valid_macro_response()),
        {"message": {"content": "bad"}},
        {"message": {"content": "still bad"}},
        _ollama_response(_valid_ticker_response()),
    ])
    run_id = _seed_run_log(db_conn)
    result = await run_agent_a(db_conn, mock_rag, TICKERS, run_id)

    degraded = result["per_ticker"]["AAPL"]
    assert degraded["sentiment"] == "neutral"
    assert degraded["confidence"] == "weak"
    assert degraded["key_catalysts"] == []
    assert "degraded" in degraded["narrative"].lower()


@pytest.mark.asyncio
@patch("pipeline.agents.agent_a.query_graph", new_callable=AsyncMock)
@patch("pipeline.agents.agent_a.ollama.AsyncClient")
async def test_output_stored(MockAsyncClient, mock_qg, db_conn, mock_rag):
    """Verify agent output is stored in agent_outputs table."""
    mock_qg.return_value = "context"
    MockAsyncClient.return_value = _make_mock_client([
        _ollama_response(_valid_macro_response()),
        _ollama_response(_valid_ticker_response()),
        _ollama_response(_valid_ticker_response()),
    ])
    run_id = _seed_run_log(db_conn)
    await run_agent_a(db_conn, mock_rag, TICKERS, run_id)

    row = db_conn.execute(
        "SELECT * FROM agent_outputs WHERE run_id = ? AND agent = 'A'",
        (run_id,),
    ).fetchone()
    assert row is not None
    blob = json.loads(row["output_blob"])
    assert "macro_overview" in blob
    assert "per_ticker" in blob


@pytest.mark.asyncio
@patch("pipeline.agents.agent_a.query_graph", new_callable=AsyncMock)
@patch("pipeline.agents.agent_a.ollama.AsyncClient")
async def test_ticker_with_standing_event(MockAsyncClient, mock_qg, db_conn, mock_rag):
    """Ticker in affected_tickers gets standing context appended to its query."""
    run_id = _seed_run_log(db_conn)
    _seed_standing_event(db_conn, run_id, affected_tickers=["NVDA"])

    mock_qg.return_value = "context"
    MockAsyncClient.return_value = _make_mock_client([
        _ollama_response(_valid_macro_response()),
        _ollama_response(_valid_standing_response(["FED_RATE_HIKE"])),
        _ollama_response(_valid_ticker_response()),  # AAPL
        _ollama_response(_valid_ticker_response()),  # NVDA
    ])
    result = await run_agent_a(db_conn, mock_rag, TICKERS, run_id)

    # Verify query_graph was called with standing context for NVDA
    calls = mock_qg.call_args_list
    # Find the per-ticker calls (local mode)
    local_calls = [c for c in calls if c.kwargs.get("mode") == "local" or
                   (len(c.args) >= 3 and c.args[2] == "local")]
    # NVDA query should contain the standing event summary
    nvda_calls = [c for c in local_calls
                  if "NVDA" in (c.args[1] if len(c.args) > 1 else c.kwargs.get("query", ""))]
    assert len(nvda_calls) >= 1
    nvda_query = nvda_calls[0].args[1] if len(nvda_calls[0].args) > 1 else nvda_calls[0].kwargs["query"]
    assert "FED_RATE_HIKE" in nvda_query

    # AAPL should NOT have standing event in query
    aapl_calls = [c for c in local_calls
                  if "AAPL" in (c.args[1] if len(c.args) > 1 else c.kwargs.get("query", ""))]
    assert len(aapl_calls) >= 1
    aapl_query = aapl_calls[0].args[1] if len(aapl_calls[0].args) > 1 else aapl_calls[0].kwargs["query"]
    assert "FED_RATE_HIKE" not in aapl_query

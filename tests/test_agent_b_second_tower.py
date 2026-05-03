"""Tests for Agent B v0.2 — Second Tower strategy wrapper."""

from __future__ import annotations

import datetime
import json
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

from pipeline.agents.agent_b import run_agent_b
from pipeline.ingestion.models import MarketDataPoint


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TICKERS = ["AAPL", "MSFT", "JPM", "JNJ", "XOM"]


def _seed_account(conn, cash: float = 100_000.0):
    conn.execute(
        "INSERT OR REPLACE INTO account "
        "(account_id, cash_balance, initial_cash, universe, initialized_at) "
        "VALUES (1, ?, ?, 'djia30', '2026-01-01T00:00:00')",
        (cash, cash),
    )
    conn.commit()


def _seed_constraints(conn):
    for name, val in [
        ("cash_floor", 0.05),
        ("max_single_position", 0.15),
        ("max_sector_concentration", 0.35),
        ("min_position_size", 0.02),
    ]:
        conn.execute(
            "INSERT OR REPLACE INTO constraints "
            "(constraint_name, value, description) VALUES (?, ?, '')",
            (name, val),
        )
    conn.commit()


def _seed_watchlist(conn, tickers: list[str]):
    for t in tickers:
        conn.execute(
            "INSERT OR IGNORE INTO watchlist "
            "(ticker, company_name, sector, added_at) VALUES (?, ?, ?, ?)",
            (t, t, "Information Technology", "2026-01-01"),
        )
    conn.commit()


def _seed_run(conn, run_id: int):
    conn.execute(
        "INSERT INTO run_log (run_id, timestamp, run_type) "
        "VALUES (?, '2026-04-01T08:30:00', 'pre_open')",
        (run_id,),
    )
    conn.commit()


def _seed_computed_target(conn, run_id: int, tickers: list[str]):
    n = len(tickers)
    targets = {
        t: {"target_weight": 1.0 / n, "conviction_weight": 0.5, "action": "Buy"}
        for t in tickers
    }
    conn.execute(
        "INSERT INTO computed_targets (run_id, timestamp, per_ticker_json) "
        "VALUES (?, '2026-01-01', ?)",
        (run_id, json.dumps(targets)),
    )
    conn.commit()


def _seed_ohlcv_cache(conn, tickers: list[str], n_days: int = 800):
    """Insert synthetic OHLCV rows covering ~3 years."""
    base = datetime.date(2023, 1, 2)
    rng = np.random.default_rng(42)
    for t in tickers:
        price = 100.0 + rng.uniform(0, 50)
        for i in range(n_days):
            d = base + datetime.timedelta(days=i)
            ret = rng.normal(0, 0.012)
            price = max(price * (1 + ret), 1.0)
            conn.execute(
                "INSERT OR IGNORE INTO market_history_cache "
                "(ticker, date, open, high, low, close, volume) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (t, d.isoformat(), price * 0.99, price * 1.015,
                 price * 0.985, price, 1_000_000),
            )
    conn.commit()


def _make_market_data(tickers: list[str], price: float = 100.0) -> dict:
    return {
        t: MarketDataPoint(
            ticker=t, current_price=price, open_price=price, previous_close=price,
            pe_ratio=None, market_cap=None, price_history_30d=[price] * 30,
            fetched_at="2026-04-01T08:30:00",
        )
        for t in tickers
    }


def _mock_rag():
    return AsyncMock(spec=["chunk_entity_relation_graph", "ainsert_custom_kg", "aquery"])


def _fake_weights(n: int) -> np.ndarray:
    w = np.ones(n) / n
    return w


def _fake_cov(n: int) -> np.ndarray:
    # Diagonal covariance → zero off-diagonal correlations
    return np.eye(n) * 0.0004


# ---------------------------------------------------------------------------
# Test 1: first run returns None (no computed_targets)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_first_run_returns_none(db_conn):
    """Empty computed_targets → is_first_run=True → returns None without calling Second Tower."""
    _seed_account(db_conn)
    _seed_constraints(db_conn)
    _seed_watchlist(db_conn, TICKERS)
    _seed_run(db_conn, 1)

    result = await run_agent_b(db_conn, _mock_rag(), _make_market_data(TICKERS), 1)
    assert result is None


# ---------------------------------------------------------------------------
# Test 2: output schema conformance on a non-first run
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_output_schema_conformance(db_conn):
    """Second run: output must contain all required top-level and per-ticker keys."""
    _seed_account(db_conn)
    _seed_constraints(db_conn)
    _seed_watchlist(db_conn, TICKERS)
    _seed_run(db_conn, 1)
    _seed_run(db_conn, 2)
    _seed_computed_target(db_conn, 1, TICKERS)
    _seed_ohlcv_cache(db_conn, TICKERS + ["SPY"])

    n = len(TICKERS)
    with (
        patch("pipeline.agents.agent_b.refresh_ohlcv_cache"),
        patch("pipeline.agents.agent_b.refresh_edgar_cache"),
        patch("pipeline.agents.agent_b._compute_rebal_weights",
              return_value=(None, None, _fake_weights(n), None)),
        patch("pipeline.agents.agent_b._shrinkage_cov",
              return_value=_fake_cov(n)),
        patch("pipeline.agents.agent_b.get_sector_map",
              return_value={t: "Information Technology" for t in TICKERS}),
        patch("pipeline.agents.agent_b.inject_correlation_edges", new_callable=AsyncMock),
    ):
        result = await run_agent_b(db_conn, _mock_rag(), _make_market_data(TICKERS), 2)

    assert result is not None

    # Top-level structure
    for key in ("portfolio_level", "per_ticker", "sector_concentrations", "constraint_violations"):
        assert key in result, f"Missing top-level key: {key}"

    pl = result["portfolio_level"]
    for key in ("total_value", "cash_pct", "portfolio_volatility_30d",
                "max_drawdown_30d", "overall_status"):
        assert key in pl, f"Missing portfolio_level.{key}"
    assert pl["overall_status"] in ("normal", "warning", "breach")

    # Per-ticker structure for every seeded ticker
    for ticker in TICKERS:
        assert ticker in result["per_ticker"], f"{ticker} missing from per_ticker"
        entry = result["per_ticker"][ticker]
        for key in ("current_weight", "previous_target_weight", "target_weight",
                    "drift", "volatility_30d", "sector", "health_score", "flags"):
            assert key in entry, f"Missing per_ticker[{ticker}].{key}"
        assert entry["health_score"] in ("normal", "warning", "breach")
        assert isinstance(entry["flags"], list)
        assert isinstance(entry["drift"], float)
        assert isinstance(entry["volatility_30d"], float)


# ---------------------------------------------------------------------------
# Test 3: side effects — inject_correlation_edges called, output stored in DB
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_side_effects_called(db_conn):
    """inject_correlation_edges must be awaited with correct run_id; output row must appear in DB."""
    _seed_account(db_conn)
    _seed_constraints(db_conn)
    _seed_watchlist(db_conn, TICKERS)
    _seed_run(db_conn, 1)
    _seed_run(db_conn, 2)
    _seed_computed_target(db_conn, 1, TICKERS)
    _seed_ohlcv_cache(db_conn, TICKERS + ["SPY"])

    n = len(TICKERS)
    inject_calls: list = []

    async def _fake_inject(rag_arg, correlations, run_id):
        inject_calls.append((correlations, run_id))

    with (
        patch("pipeline.agents.agent_b.refresh_ohlcv_cache"),
        patch("pipeline.agents.agent_b.refresh_edgar_cache"),
        patch("pipeline.agents.agent_b._compute_rebal_weights",
              return_value=(None, None, _fake_weights(n), None)),
        patch("pipeline.agents.agent_b._shrinkage_cov",
              return_value=_fake_cov(n)),
        patch("pipeline.agents.agent_b.get_sector_map",
              return_value={t: "Information Technology" for t in TICKERS}),
        patch("pipeline.agents.agent_b.inject_correlation_edges",
              side_effect=_fake_inject),
    ):
        result = await run_agent_b(db_conn, _mock_rag(), _make_market_data(TICKERS), 2)

    assert result is not None

    # inject_correlation_edges called exactly once with run_id=2
    assert len(inject_calls) == 1
    correlations, rid = inject_calls[0]
    assert rid == 2
    assert isinstance(correlations, list)
    # n*(n-1)/2 pairs from diagonal cov (all non-zero vols)
    expected_pairs = n * (n - 1) // 2
    assert len(correlations) == expected_pairs
    for c in correlations:
        assert "ticker_a" in c and "ticker_b" in c and "correlation" in c
        assert -1.0 <= c["correlation"] <= 1.0

    # output row stored in agent_outputs
    row = db_conn.execute(
        "SELECT output_blob FROM agent_outputs WHERE agent = 'B' AND run_id = 2"
    ).fetchone()
    assert row is not None
    blob = json.loads(row["output_blob"])
    assert "portfolio_level" in blob
    assert "per_ticker" in blob


# ---------------------------------------------------------------------------
# Test 4: SPY unavailable → returns None gracefully
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_spy_unavailable_returns_none(db_conn):
    """Missing SPY OHLCV cache → Agent B logs an error and returns None without crashing."""
    _seed_account(db_conn)
    _seed_constraints(db_conn)
    _seed_watchlist(db_conn, TICKERS)
    _seed_run(db_conn, 1)
    _seed_run(db_conn, 2)
    _seed_computed_target(db_conn, 1, TICKERS)
    # Intentionally omit SPY from the cache
    _seed_ohlcv_cache(db_conn, TICKERS)

    with (
        patch("pipeline.agents.agent_b.refresh_ohlcv_cache"),
        patch("pipeline.agents.agent_b.refresh_edgar_cache"),
    ):
        result = await run_agent_b(db_conn, _mock_rag(), _make_market_data(TICKERS), 2)

    assert result is None

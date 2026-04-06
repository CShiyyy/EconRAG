"""Tests for Agent B — Deterministic Quant Script."""

from __future__ import annotations

import json
import math
from unittest.mock import AsyncMock

import numpy as np
import pytest

from pipeline.agents.agent_b import (
    _compute_correlations,
    _compute_drift,
    _compute_max_drawdown,
    _compute_sector_concentrations,
    _compute_volatility,
    _compute_portfolio_volatility,
    _compute_health_scores,
    _detect_constraint_violations,
    _determine_overall_status,
    run_agent_b,
)
from pipeline.ingestion.models import MarketDataPoint


# ---------------------------------------------------------------------------
# Helpers for seeding test data
# ---------------------------------------------------------------------------

def _seed_account(conn, cash: float = 10000.0):
    conn.execute(
        "INSERT OR REPLACE INTO account (account_id, cash_balance, initial_cash, universe, initialized_at) "
        "VALUES (1, ?, ?, 'djia30', '2026-01-01T00:00:00')",
        (cash, cash),
    )
    conn.commit()


def _seed_constraints(conn):
    for name, (val, desc) in [
        ("cash_floor", (0.05, "min cash")),
        ("max_single_position", (0.15, "max pos")),
        ("max_sector_concentration", (0.35, "max sector")),
        ("min_position_size", (0.02, "min pos")),
    ]:
        conn.execute(
            "INSERT OR REPLACE INTO constraints (constraint_name, value, description) VALUES (?, ?, ?)",
            (name, val, desc),
        )
    conn.commit()


def _seed_run(conn, run_id: int = 1):
    conn.execute(
        "INSERT INTO run_log (run_id, timestamp, run_type) VALUES (?, '2026-04-01T08:30:00', 'pre_open')",
        (run_id,),
    )
    conn.commit()


def _seed_holdings(conn, holdings: list[tuple]):
    """holdings: list of (ticker, shares, cost_basis, sector)"""
    for ticker, shares, cost, sector in holdings:
        conn.execute(
            "INSERT INTO holdings (ticker, shares, cost_basis_per_share, sector) VALUES (?, ?, ?, ?)",
            (ticker, shares, cost, sector),
        )
    conn.commit()


def _seed_computed_targets(conn, run_id: int, targets: dict):
    conn.execute(
        "INSERT INTO computed_targets (run_id, timestamp, per_ticker_json) VALUES (?, '2026-04-01', ?)",
        (run_id, json.dumps(targets)),
    )
    conn.commit()


def _seed_snapshots(conn, snapshots: list[tuple]):
    """snapshots: list of (run_id, total_value, cash)"""
    for run_id, total_value, cash in snapshots:
        conn.execute(
            "INSERT INTO snapshots (run_id, timestamp, total_value, cash, per_ticker_json) "
            "VALUES (?, '2026-04-01', ?, ?, '{}')",
            (run_id, total_value, cash),
        )
    conn.commit()


def _make_market_data(prices: dict[str, float], histories: dict[str, list[float]] | None = None) -> dict[str, MarketDataPoint]:
    result = {}
    for ticker, price in prices.items():
        hist = (histories or {}).get(ticker, [price] * 30)
        result[ticker] = MarketDataPoint(
            ticker=ticker,
            current_price=price,
            open_price=price,
            previous_close=price,
            pe_ratio=None,
            market_cap=None,
            price_history_30d=hist,
            fetched_at="2026-04-01T08:30:00",
        )
    return result


def _mock_rag():
    return AsyncMock(spec=["chunk_entity_relation_graph", "ainsert_custom_kg", "aquery"])


# ---------------------------------------------------------------------------
# Test: first run returns None
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_first_run_returns_none(db_conn):
    """Empty computed_targets table -> returns None."""
    _seed_account(db_conn)
    _seed_constraints(db_conn)
    _seed_run(db_conn, 1)
    market_data = _make_market_data({"AAPL": 150.0})
    rag = _mock_rag()

    result = await run_agent_b(db_conn, rag, market_data, 1)
    assert result is None


# ---------------------------------------------------------------------------
# Test: drift calculation
# ---------------------------------------------------------------------------

def test_drift_calculation():
    current = {"AAPL": 0.12, "MSFT": 0.08}
    targets = {
        "AAPL": {"target_weight": 0.10, "conviction_weight": 0.7, "action": "Buy"},
        "MSFT": {"target_weight": 0.10, "conviction_weight": 0.5, "action": "Buy"},
    }
    drift = _compute_drift(current, targets)
    assert abs(drift["AAPL"] - 0.02) < 1e-9
    assert abs(drift["MSFT"] - (-0.02)) < 1e-9


# ---------------------------------------------------------------------------
# Test: volatility calculation
# ---------------------------------------------------------------------------

def test_volatility_calculation():
    # Create a known price series and verify annualized vol
    prices = [100.0, 101.0, 99.5, 100.5, 102.0, 101.0, 100.0, 99.0, 100.0, 101.0]
    log_returns = np.diff(np.log(prices))
    expected_vol = float(np.std(log_returns, ddof=1) * math.sqrt(252))

    vols = _compute_volatility({"TEST": prices})
    assert abs(vols["TEST"] - expected_vol) < 1e-9


# ---------------------------------------------------------------------------
# Test: portfolio volatility
# ---------------------------------------------------------------------------

def test_portfolio_volatility():
    per_ticker_vol = {"AAPL": 0.20, "MSFT": 0.30}
    weights = {"AAPL": 0.60, "MSFT": 0.40}
    # Weighted average: 0.60*0.20 + 0.40*0.30 = 0.12 + 0.12 = 0.24
    pv = _compute_portfolio_volatility(per_ticker_vol, weights)
    assert abs(pv - 0.24) < 1e-9


# ---------------------------------------------------------------------------
# Test: sector concentration
# ---------------------------------------------------------------------------

def test_sector_concentration():
    holdings = [
        {"ticker": "AAPL", "sector": "Tech"},
        {"ticker": "MSFT", "sector": "Tech"},
        {"ticker": "NVDA", "sector": "Tech"},
        {"ticker": "GOOG", "sector": "Tech"},
    ]
    weights = {"AAPL": 0.10, "MSFT": 0.10, "NVDA": 0.10, "GOOG": 0.10}
    # Total Tech = 0.40, limit = 0.35 -> breach
    result = _compute_sector_concentrations(holdings, weights, 0.35)
    assert result["Tech"]["status"] == "breach"
    assert abs(result["Tech"]["weight"] - 0.40) < 1e-6

    # Within warning buffer (limit=0.44, weight=0.40, buffer=0.05 -> 0.40 > 0.44-0.05=0.39)
    result2 = _compute_sector_concentrations(holdings, weights, 0.44)
    assert result2["Tech"]["status"] == "warning"

    # Normal (limit=0.50, weight=0.40, 0.40 < 0.50-0.05=0.45)
    result3 = _compute_sector_concentrations(holdings, weights, 0.50)
    assert result3["Tech"]["status"] == "normal"


# ---------------------------------------------------------------------------
# Test: max drawdown
# ---------------------------------------------------------------------------

def test_max_drawdown(db_conn):
    _seed_account(db_conn)
    _seed_run(db_conn, 1)
    _seed_run(db_conn, 2)
    _seed_run(db_conn, 3)
    _seed_run(db_conn, 4)
    # Peak at 110k, trough at 95k -> dd = (95-110)/110 = -0.13636...
    _seed_snapshots(db_conn, [
        (1, 100000.0, 10000.0),
        (2, 110000.0, 10000.0),
        (3, 95000.0, 10000.0),
        (4, 105000.0, 10000.0),
    ])
    dd = _compute_max_drawdown(db_conn)
    expected = (95000 - 110000) / 110000
    assert abs(dd - expected) < 1e-6


# ---------------------------------------------------------------------------
# Test: health scores
# ---------------------------------------------------------------------------

def test_health_scores():
    # Ticker A: drift 0.06 -> breach
    # Ticker B: drift 0.04 -> warning
    # Ticker C: drift 0.01 -> normal
    drift = {"A": 0.06, "B": 0.04, "C": 0.01}
    # All same vol so vol doesn't trigger anything
    vols = {"A": 0.20, "B": 0.20, "C": 0.20}
    weights = {"A": 0.30, "B": 0.30, "C": 0.30}

    scores = _compute_health_scores(drift, vols, weights)
    assert scores["A"] == "breach"
    assert scores["B"] == "warning"
    assert scores["C"] == "normal"


# ---------------------------------------------------------------------------
# Test: constraint violations
# ---------------------------------------------------------------------------

def test_constraint_violations():
    weights = {"AAPL": 0.20, "MSFT": 0.05}  # AAPL > 0.15 limit
    cash_pct = 0.10
    sector_conc = {"Tech": {"weight": 0.25, "limit": 0.35, "status": "normal"}}
    constraints = {
        "cash_floor": 0.05,
        "max_single_position": 0.15,
        "max_sector_concentration": 0.35,
        "min_position_size": 0.02,
    }

    violations = _detect_constraint_violations(weights, cash_pct, sector_conc, constraints)
    assert len(violations) == 1
    assert violations[0]["constraint"] == "max_single_position"
    assert violations[0]["ticker"] == "AAPL"


# ---------------------------------------------------------------------------
# Test: correlation injection
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_correlation_injection(db_conn):
    """Verify inject_correlation_edges called with correct pairs."""
    _seed_account(db_conn, cash=10000.0)
    _seed_constraints(db_conn)
    _seed_run(db_conn, 1)
    _seed_run(db_conn, 2)
    _seed_holdings(db_conn, [
        ("AAPL", 100, 150.0, "Tech"),
        ("MSFT", 50, 300.0, "Tech"),
    ])
    _seed_computed_targets(db_conn, 1, {
        "AAPL": {"target_weight": 0.50, "conviction_weight": 0.7, "action": "Buy"},
        "MSFT": {"target_weight": 0.40, "conviction_weight": 0.5, "action": "Buy"},
    })

    # Generate correlated price histories
    np.random.seed(42)
    base = np.cumsum(np.random.randn(30)) + 150
    prices_aapl = base.tolist()
    prices_msft = (base * 2 + np.random.randn(30) * 0.5).tolist()

    market_data = _make_market_data(
        {"AAPL": prices_aapl[-1], "MSFT": prices_msft[-1]},
        {"AAPL": prices_aapl, "MSFT": prices_msft},
    )

    rag = _mock_rag()
    # Mock inject_correlation_edges at the module level
    import pipeline.agents.agent_b as agent_b_mod
    original = agent_b_mod.inject_correlation_edges
    call_args = []

    async def mock_inject(rag_arg, correlations, run_id):
        call_args.append((correlations, run_id))

    agent_b_mod.inject_correlation_edges = mock_inject
    try:
        result = await run_agent_b(db_conn, rag, market_data, 2)
        assert result is not None
        assert len(call_args) == 1
        correlations, rid = call_args[0]
        assert rid == 2
        # Should have exactly one pair: AAPL-MSFT
        assert len(correlations) == 1
        assert correlations[0]["ticker_a"] == "AAPL"
        assert correlations[0]["ticker_b"] == "MSFT"
        assert isinstance(correlations[0]["correlation"], float)
    finally:
        agent_b_mod.inject_correlation_edges = original


# ---------------------------------------------------------------------------
# Test: output stored in agent_outputs
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_output_stored(db_conn):
    _seed_account(db_conn, cash=10000.0)
    _seed_constraints(db_conn)
    _seed_run(db_conn, 1)
    _seed_run(db_conn, 2)
    _seed_holdings(db_conn, [
        ("AAPL", 100, 150.0, "Tech"),
    ])
    _seed_computed_targets(db_conn, 1, {
        "AAPL": {"target_weight": 0.90, "conviction_weight": 0.7, "action": "Buy"},
    })

    market_data = _make_market_data({"AAPL": 150.0})

    rag = _mock_rag()
    import pipeline.agents.agent_b as agent_b_mod
    original = agent_b_mod.inject_correlation_edges
    agent_b_mod.inject_correlation_edges = AsyncMock()
    try:
        await run_agent_b(db_conn, rag, market_data, 2)
    finally:
        agent_b_mod.inject_correlation_edges = original

    row = db_conn.execute(
        "SELECT * FROM agent_outputs WHERE agent = 'B' AND run_id = 2"
    ).fetchone()
    assert row is not None
    blob = json.loads(row["output_blob"])
    assert "portfolio_level" in blob
    assert "per_ticker" in blob


# ---------------------------------------------------------------------------
# Test: overall status
# ---------------------------------------------------------------------------

def test_overall_status():
    # Breach if any violation
    assert _determine_overall_status(
        {"A": "normal"}, [{"constraint": "cash_floor"}]
    ) == "breach"

    # Warning if any health warning, no violations
    assert _determine_overall_status(
        {"A": "warning", "B": "normal"}, []
    ) == "warning"

    # Breach if any health breach, no violations
    assert _determine_overall_status(
        {"A": "breach", "B": "normal"}, []
    ) == "breach"

    # Normal otherwise
    assert _determine_overall_status(
        {"A": "normal", "B": "normal"}, []
    ) == "normal"

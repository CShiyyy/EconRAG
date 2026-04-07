"""End-to-end orchestration tests with mocked agents and ingestion."""

from __future__ import annotations

import asyncio
import sqlite3
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pipeline.db.connection import get_connection
from pipeline.db.schema import create_tables
from pipeline.orchestration.graph import run_pipeline


# ---------------------------------------------------------------------------
# Test data builders
# ---------------------------------------------------------------------------

TICKERS = ["AAPL", "JPM", "MSFT"]


def _make_ingestion_result() -> dict:
    """Return a serialised IngestionResult dict (as dataclasses.asdict would)."""
    market = {}
    prices = {"AAPL": 175.0, "MSFT": 410.0, "JPM": 195.0}
    for t in TICKERS:
        market[t] = {
            "ticker": t,
            "current_price": prices[t],
            "open_price": prices[t] - 1.0,
            "previous_close": prices[t] - 2.0,
            "pe_ratio": 25.0,
            "market_cap": 2.5e12,
            "price_history_30d": [prices[t]] * 30,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }
    return {
        "market_data": market,
        "news_hits": [],
        "social_hits": [],
        "parsed_content": [],
        "health": [
            {"source": "yfinance", "status": "success", "items_fetched": 3,
             "duration_ms": 200, "error_detail": None},
        ],
    }


def _agent_a_output() -> dict:
    per_ticker = {}
    for t in TICKERS:
        per_ticker[t] = {
            "sentiment": "bullish",
            "confidence": "strong",
            "summary": f"Positive outlook for {t}",
        }
    return {"per_ticker": per_ticker}


def _agent_b_output() -> dict:
    per_ticker = {}
    for t in TICKERS:
        per_ticker[t] = {
            "health_score": "normal",
            "drift": 0.01,
            "metrics": {"pe_ratio": 25.0},
        }
    return {"per_ticker": per_ticker}


def _agent_c_assessment() -> dict:
    per_ticker = {}
    for t in TICKERS:
        per_ticker[t] = {
            "action": "assessment",
            "conviction": {
                "narrative_alignment": "high",
                "quant_support": "medium",
                "signal_agreement": "high",
            },
            "rationale": f"Assessment for {t}",
            "key_risk_factors": [],
        }
    return {"per_ticker": per_ticker, "standing_event_actions": None}


def _agent_c_first_run() -> dict:
    per_ticker = {}
    for t in TICKERS:
        per_ticker[t] = {
            "action": "Buy",
            "conviction": {
                "narrative_alignment": "high",
                "quant_support": "n/a",
                "signal_agreement": "n/a",
            },
            "rationale": f"Initial buy for {t}",
            "key_risk_factors": [],
        }
    return {"per_ticker": per_ticker, "standing_event_actions": None}


# ---------------------------------------------------------------------------
# Fixture: initialized database
# ---------------------------------------------------------------------------

@pytest.fixture
def initialized_db(tmp_path):
    """Create a fully-initialised DB and return its path as a string."""
    db_path = str(tmp_path / "test.db")
    conn = get_connection(db_path)
    create_tables(conn)

    ts = datetime.now(timezone.utc).isoformat()

    conn.execute(
        "INSERT INTO account (account_id, cash_balance, initial_cash, universe, initialized_at) "
        "VALUES (1, 100000, 100000, 'djia30', ?)",
        (ts,),
    )
    for ticker, name, sector in [
        ("AAPL", "Apple Inc.", "Technology"),
        ("MSFT", "Microsoft Corporation", "Technology"),
        ("JPM", "JPMorgan Chase", "Financials"),
    ]:
        conn.execute(
            "INSERT INTO watchlist (ticker, company_name, sector, added_at) VALUES (?, ?, ?, ?)",
            (ticker, name, sector, ts),
        )
    for cname, val, desc in [
        ("cash_floor", 0.05, "Min cash %"),
        ("max_single_position", 0.15, "Max single position %"),
        ("max_sector_concentration", 0.35, "Max sector %"),
        ("min_position_size", 0.02, "Min position size %"),
    ]:
        conn.execute(
            "INSERT INTO constraints (constraint_name, value, description) VALUES (?, ?, ?)",
            (cname, val, desc),
        )
    conn.commit()
    conn.close()
    return db_path


# ---------------------------------------------------------------------------
# Helper: patch _create_run_log so it uses the correct schema columns
# ---------------------------------------------------------------------------

def _fixed_create_run_log(conn, run_type, first_run):
    ts = datetime.now(timezone.utc).isoformat()
    cursor = conn.execute(
        "INSERT INTO run_log (timestamp, run_type, is_first_run) VALUES (?, ?, ?)",
        (ts, run_type, 1 if first_run else 0),
    )
    conn.commit()
    return cursor.lastrowid


# ---------------------------------------------------------------------------
# Common patches context manager
# ---------------------------------------------------------------------------

def _ensure_mock_modules():
    """Inject stub modules into sys.modules so patch() can resolve dotted paths
    without triggering real imports (which need yfinance, etc.).

    Only creates stub modules for leaf modules that cannot be imported.
    Never overwrites already-imported parent packages.
    """
    import importlib
    import sys
    import types

    stubs = {
        "pipeline.ingestion.orchestrator": "run_ingestion",
        "pipeline.knowledge.extraction": "process_content_batch",
        "pipeline.knowledge.lightrag_config": "get_rag_instance",
        "pipeline.knowledge.pruner": "prune_expired_edges",
        "pipeline.agents.agent_a": "run_agent_a",
        "pipeline.agents.agent_b": "run_agent_b",
        "pipeline.agents.agent_c": "run_agent_c",
        "pipeline.sizing.engine": "run_sizing_engine",
    }

    for mod_name, func_name in stubs.items():
        # Try real import first; only stub if it fails
        try:
            importlib.import_module(mod_name)
        except (ImportError, ModuleNotFoundError):
            # Ensure all parent packages exist
            parts = mod_name.split(".")
            for i in range(1, len(parts)):
                parent = ".".join(parts[:i])
                if parent not in sys.modules:
                    sys.modules[parent] = types.ModuleType(parent)

            if mod_name not in sys.modules:
                sys.modules[mod_name] = types.ModuleType(mod_name)

        # Ensure the target function attribute exists
        setattr(sys.modules[mod_name], func_name, None)

        # Wire child module as attribute on parent
        parent_name, child_name = mod_name.rsplit(".", 1)
        if parent_name in sys.modules:
            setattr(sys.modules[parent_name], child_name, sys.modules[mod_name])


def _common_patches(agent_a_ret, agent_b_ret, agent_c_ret, agent_a_mock=None):
    """Return a list of patch context managers for all external calls."""
    _ensure_mock_modules()

    if agent_a_mock is None:
        agent_a_mock = AsyncMock(return_value=agent_a_ret)

    patches = [
        patch("pipeline.orchestration.graph._create_run_log", side_effect=_fixed_create_run_log),
        patch("pipeline.ingestion.orchestrator.run_ingestion",
              new_callable=AsyncMock, return_value=_make_ingestion_result_obj()),
        patch("pipeline.knowledge.extraction.process_content_batch",
              new_callable=AsyncMock, return_value=MagicMock(total_processed=0, total_valid=0, total_failed=0)),
        patch("pipeline.knowledge.lightrag_config.get_rag_instance",
              new_callable=AsyncMock,
              return_value=MagicMock(
                  finalize_storages=AsyncMock(),
                  initialize_storages=AsyncMock(),
              )),
        patch("pipeline.knowledge.pruner.prune_expired_edges",
              new_callable=AsyncMock,
              return_value=MagicMock(edges_scanned=0, edges_expired=0, edges_removed=0)),
        patch("pipeline.agents.agent_a.run_agent_a", agent_a_mock),
        patch("pipeline.agents.agent_b.run_agent_b",
              new_callable=AsyncMock, return_value=agent_b_ret),
        patch("pipeline.agents.agent_c.run_agent_c",
              new_callable=AsyncMock, return_value=agent_c_ret),
        patch("pipeline.sizing.engine.run_sizing_engine",
              side_effect=_mock_sizing_engine),
    ]
    return patches


def _make_ingestion_result_obj():
    """Build a real IngestionResult dataclass for run_ingestion mock."""
    from pipeline.ingestion.models import IngestionResult, MarketDataPoint, SourceHealth

    market = {}
    prices = {"AAPL": 175.0, "MSFT": 410.0, "JPM": 195.0}
    for t in TICKERS:
        market[t] = MarketDataPoint(
            ticker=t,
            current_price=prices[t],
            open_price=prices[t] - 1.0,
            previous_close=prices[t] - 2.0,
            pe_ratio=25.0,
            market_cap=2.5e12,
            price_history_30d=[prices[t]] * 30,
            fetched_at=datetime.now(timezone.utc).isoformat(),
        )
    health = [SourceHealth(source="yfinance", status="success", items_fetched=3,
                           duration_ms=200, error_detail=None)]
    return IngestionResult(
        market_data=market, news_hits=[], social_hits=[], parsed_content=[], health=health,
    )


def _mock_sizing_engine(conn, run_id, agent_c_output, fill_prices, is_first_run=False):
    """Mock sizing engine that creates holdings and adjusts cash on first run."""
    import json

    if is_first_run and agent_c_output:
        # Simulate buying equal-weight positions
        cash = conn.execute("SELECT cash_balance FROM account WHERE account_id=1").fetchone()[0]
        num = len(agent_c_output)
        per_ticker_cash = (cash * 0.90) / num  # spend 90% of cash
        trade_list = []

        # look up sectors from watchlist
        sectors = {}
        for row in conn.execute("SELECT ticker, sector FROM watchlist").fetchall():
            sectors[row["ticker"]] = row["sector"]

        for ticker, entry in agent_c_output.items():
            if entry.get("action") == "Buy" and ticker in fill_prices:
                price = fill_prices[ticker]
                shares = round(per_ticker_cash / price, 4)
                cost = shares * price
                conn.execute(
                    "INSERT OR REPLACE INTO holdings (ticker, shares, cost_basis_per_share, sector) "
                    "VALUES (?, ?, ?, ?)",
                    (ticker, shares, price, sectors.get(ticker, "Unknown")),
                )
                cash -= cost
                trade_list.append({
                    "ticker": ticker, "action": "Buy", "shares": shares,
                    "fill_price": price, "notional": cost,
                })
        conn.execute("UPDATE account SET cash_balance=? WHERE account_id=1", (cash,))
        conn.commit()

        # Store computed targets
        ts = datetime.now(timezone.utc).isoformat()
        targets = {t: {"weight": 0.30} for t in agent_c_output}
        conn.execute(
            "INSERT INTO computed_targets (run_id, timestamp, per_ticker_json) VALUES (?, ?, ?)",
            (run_id, ts, json.dumps(targets)),
        )
        conn.commit()

        return {"target_weights": targets, "trade_list": trade_list}

    return {"target_weights": {}, "trade_list": []}


# ---------------------------------------------------------------------------
# Test 1: Post-close full run
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_post_close_full_run(initialized_db, tmp_path):
    rag_dir = str(tmp_path / "rag")
    patches = _common_patches(_agent_a_output(), _agent_b_output(), _agent_c_assessment())

    for p in patches:
        p.start()
    try:
        result = await run_pipeline(initialized_db, "post_close", rag_storage_dir=rag_dir)
    finally:
        for p in patches:
            p.stop()

    # Verify snapshot recorded
    conn = get_connection(initialized_db)
    try:
        snap = conn.execute("SELECT * FROM snapshots").fetchone()
        assert snap is not None, "Snapshot should be recorded"

        # Verify run_log
        row = conn.execute("SELECT * FROM run_log WHERE run_id=?", (result["run_id"],)).fetchone()
        assert row is not None
        assert row["run_type"] == "post_close"
        assert row["wall_clock_seconds"] is not None

        # No trades on post_close
        trades = conn.execute("SELECT * FROM trades").fetchall()
        assert len(trades) == 0, "Post-close should produce no trades"
    finally:
        conn.close()

    # Agent C output captured
    assert result.get("risk_assessment") is not None


# ---------------------------------------------------------------------------
# Test 2: First run pipeline
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_first_run_pipeline(initialized_db, tmp_path):
    rag_dir = str(tmp_path / "rag")
    patches = _common_patches(_agent_a_output(), None, _agent_c_first_run())

    for p in patches:
        p.start()
    try:
        result = await run_pipeline(initialized_db, "pre_open", rag_storage_dir=rag_dir)
    finally:
        for p in patches:
            p.stop()

    # First run detection
    assert result.get("is_first_run") is True

    # Agent B skipped on first run
    assert result.get("quant_assessment") is None

    # Holdings created
    conn = get_connection(initialized_db)
    try:
        holdings = conn.execute("SELECT * FROM holdings").fetchall()
        assert len(holdings) > 0, "Holdings should be created on first run"

        # Cash reduced
        cash = conn.execute("SELECT cash_balance FROM account WHERE account_id=1").fetchone()[0]
        assert cash < 100000, f"Cash should be reduced from 100000, got {cash}"
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Test 3: Re-query triggers on conflict
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_requery_triggers_on_conflict(initialized_db, tmp_path):
    """Agent B reports breach for AAPL while Agent A says bullish -> re-query."""
    import json

    rag_dir = str(tmp_path / "rag")

    # Setup: insert holdings and computed_targets so NOT first_run
    conn = get_connection(initialized_db)
    ts = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT INTO holdings (ticker, shares, cost_basis_per_share, sector) VALUES (?, ?, ?, ?)",
        ("AAPL", 100, 170.0, "Technology"),
    )
    conn.execute(
        "INSERT INTO holdings (ticker, shares, cost_basis_per_share, sector) VALUES (?, ?, ?, ?)",
        ("MSFT", 50, 400.0, "Technology"),
    )
    # Insert a computed_targets row so is_first_run returns False
    conn.execute(
        "INSERT INTO run_log (timestamp, run_type, is_first_run) VALUES (?, 'pre_open', 0)",
        (ts,),
    )
    prev_run_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    targets = {"AAPL": {"weight": 0.4}, "MSFT": {"weight": 0.4}, "JPM": {"weight": 0.2}}
    conn.execute(
        "INSERT INTO computed_targets (run_id, timestamp, per_ticker_json) VALUES (?, ?, ?)",
        (prev_run_id, ts, json.dumps(targets)),
    )
    conn.commit()
    conn.close()

    # Agent B with AAPL breach
    agent_b_out = {
        "per_ticker": {
            "AAPL": {"health_score": "breach", "drift": 0.06, "metrics": {"pe_ratio": 35.0}},
            "MSFT": {"health_score": "normal", "drift": 0.01, "metrics": {"pe_ratio": 25.0}},
            "JPM": {"health_score": "normal", "drift": 0.01, "metrics": {"pe_ratio": 12.0}},
        }
    }

    # Track Agent A call count
    call_count = 0

    async def counting_agent_a(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return _agent_a_output()

    agent_a_mock = AsyncMock(side_effect=counting_agent_a)

    patches = _common_patches(_agent_a_output(), agent_b_out, _agent_c_assessment(),
                              agent_a_mock=agent_a_mock)

    for p in patches:
        p.start()
    try:
        result = await run_pipeline(initialized_db, "pre_open", rag_storage_dir=rag_dir)
    finally:
        for p in patches:
            p.stop()

    # Re-query should have been triggered
    assert result.get("requery_count", 0) == 1, f"Expected requery_count=1, got {result.get('requery_count')}"
    assert result.get("requery_reason") is not None, "requery_reason should be set"

    # Agent A called at least twice (initial + re-query)
    assert call_count >= 2, f"Agent A should be called >=2 times, got {call_count}"

"""Tests for pipeline.orchestration.executor."""

import json
from datetime import datetime, timezone

import pytest

from pipeline.db.connection import get_connection
from pipeline.db.schema import create_tables
from pipeline.orchestration.executor import log_trades


@pytest.fixture
def exec_conn(tmp_path):
    conn = get_connection(tmp_path / "test.db")
    create_tables(conn)
    ts = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT INTO account (account_id, cash_balance, initial_cash, universe, initialized_at) "
        "VALUES (1, 100000, 100000, 'djia30', ?)", (ts,),
    )
    conn.execute(
        "INSERT INTO run_log (run_id, timestamp, run_type, is_first_run) VALUES (1, ?, 'pre_open', 0)", (ts,),
    )
    # Insert recommendation for AAPL
    conn.execute(
        "INSERT INTO recommendations (recommendation_id, run_id, timestamp, ticker, action, conviction_scores, rationale) "
        "VALUES (1, 1, ?, 'AAPL', 'Buy', '{}', 'test')", (ts,),
    )
    # Insert recommendation for MSFT
    conn.execute(
        "INSERT INTO recommendations (recommendation_id, run_id, timestamp, ticker, action, conviction_scores, rationale) "
        "VALUES (2, 1, ?, 'MSFT', 'Exit', '{}', 'test')", (ts,),
    )
    conn.commit()
    yield conn
    conn.close()


def test_log_buy_trade(exec_conn):
    trade_list = [{"ticker": "AAPL", "action": "Buy", "shares": 50, "fill_price": 175.0}]
    market_data = {"AAPL": {"open_price": 175.0, "previous_close": 173.0, "current_price": 176.0}}
    log_trades(exec_conn, trade_list, market_data, {"AAPL": 1}, run_id=1)

    row = exec_conn.execute("SELECT * FROM trades WHERE ticker = 'AAPL'").fetchone()
    assert row is not None
    assert row["action"] == "Buy"
    assert row["shares"] == 50
    assert row["simulated_fill_price"] == 175.0
    assert row["slippage_applied"] == 0.0
    expected_gap = (175.0 - 173.0) / 173.0
    assert row["gap_pct"] == pytest.approx(expected_gap)


def test_log_exit_trade_with_pnl(exec_conn):
    trade_list = [{"ticker": "MSFT", "action": "Exit", "shares": 30, "fill_price": 320.0, "realized_pnl": 1500.0}]
    market_data = {"MSFT": {"open_price": 320.0, "previous_close": 315.0, "current_price": 321.0}}
    log_trades(exec_conn, trade_list, market_data, {"MSFT": 2}, run_id=1)

    row = exec_conn.execute("SELECT * FROM trades WHERE ticker = 'MSFT'").fetchone()
    assert row is not None
    assert row["realized_pnl"] == 1500.0
    assert row["action"] == "Exit"


def test_log_trades_empty_list(exec_conn):
    log_trades(exec_conn, [], {}, {}, run_id=1)
    count = exec_conn.execute("SELECT COUNT(*) as c FROM trades").fetchone()["c"]
    assert count == 0

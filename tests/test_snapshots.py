"""Tests for pipeline.orchestration.snapshots."""

import json
from datetime import datetime, timezone

import pytest

from pipeline.db.connection import get_connection
from pipeline.db.schema import create_tables
from pipeline.orchestration.snapshots import record_snapshot


@pytest.fixture
def snap_conn(tmp_path):
    conn = get_connection(tmp_path / "test.db")
    create_tables(conn)
    conn.execute(
        "INSERT INTO account (account_id, cash_balance, initial_cash, universe, initialized_at) "
        "VALUES (1, 100000, 100000, 'djia30', ?)",
        (datetime.now(timezone.utc).isoformat(),),
    )
    conn.execute(
        "INSERT INTO run_log (run_id, timestamp, run_type, is_first_run) VALUES (1, ?, 'pre_open', 0)",
        (datetime.now(timezone.utc).isoformat(),),
    )
    conn.commit()
    yield conn
    conn.close()


def test_snapshot_cash_only(snap_conn):
    record_snapshot(snap_conn, run_id=1, market_data=None, run_type="pre_open")
    row = snap_conn.execute("SELECT * FROM snapshots WHERE run_id = 1").fetchone()
    assert row is not None
    assert row["total_value"] == 100000
    assert row["cash"] == 100000
    per_ticker = json.loads(row["per_ticker_json"])
    assert per_ticker == {}


def test_snapshot_with_holdings(snap_conn):
    snap_conn.execute(
        "INSERT INTO holdings (ticker, shares, cost_basis_per_share, sector) "
        "VALUES ('AAPL', 100, 150.0, 'Information Technology')"
    )
    snap_conn.commit()

    market_data = {"AAPL": {"current_price": 175.0, "open_price": 174.0, "previous_close": 173.0}}
    record_snapshot(snap_conn, run_id=1, market_data=market_data, run_type="post_close")

    row = snap_conn.execute("SELECT * FROM snapshots WHERE run_id = 1").fetchone()
    assert row["total_value"] == pytest.approx(117500.0)
    assert row["cash"] == 100000

    per_ticker = json.loads(row["per_ticker_json"])
    assert "AAPL" in per_ticker
    assert per_ticker["AAPL"]["shares"] == 100
    assert per_ticker["AAPL"]["price"] == 175.0
    assert per_ticker["AAPL"]["value"] == pytest.approx(17500.0)


def test_snapshot_pre_open_uses_current_price(snap_conn):
    snap_conn.execute(
        "INSERT INTO holdings (ticker, shares, cost_basis_per_share, sector) "
        "VALUES ('MSFT', 50, 300.0, 'Information Technology')"
    )
    snap_conn.commit()

    market_data = {"MSFT": {"current_price": 320.0, "open_price": 318.0, "previous_close": 315.0}}
    record_snapshot(snap_conn, run_id=1, market_data=market_data, run_type="pre_open")

    row = snap_conn.execute("SELECT * FROM snapshots WHERE run_id = 1").fetchone()
    per_ticker = json.loads(row["per_ticker_json"])
    assert per_ticker["MSFT"]["price"] == 320.0
    assert per_ticker["MSFT"]["value"] == pytest.approx(16000.0)

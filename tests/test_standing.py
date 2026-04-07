"""Tests for pipeline.orchestration.standing."""

import json
from datetime import datetime, timezone

import pytest

from pipeline.db.connection import get_connection
from pipeline.db.schema import create_tables
from pipeline.orchestration.standing import run_maintenance, process_actions


@pytest.fixture
def stand_conn(tmp_path):
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
    conn.commit()
    yield conn
    conn.close()


def _insert_canonical(conn, canonical_id, entity_type="MACRO_THEME"):
    conn.execute(
        "INSERT INTO canonical_entities (canonical_id, entity_type, display_name, aliases) "
        "VALUES (?, ?, ?, '[]')",
        (canonical_id, entity_type, canonical_id),
    )
    conn.commit()


def _insert_standing(conn, canonical_id, status="active", category="geopolitical", standing_id=None):
    _insert_canonical(conn, canonical_id)
    ts = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT INTO standing_events "
        "(canonical_id, status, category, summary, affected_tickers, promoted_at, promotion_source, last_reinforced, reinforcement_count, created_from_run_id) "
        "VALUES (?, ?, ?, 'test summary', '[]', ?, 'agent_c', ?, 0, 1)",
        (canonical_id, status, category, ts, ts),
    )
    conn.commit()


def test_run_maintenance_returns_active_events(stand_conn):
    _insert_standing(stand_conn, "theme:tariff_war")
    result = run_maintenance(stand_conn, run_id=1)
    assert len(result) == 1
    assert result[0]["canonical_id"] == "theme:tariff_war"
    assert result[0]["status"] == "active"


def test_process_promote_to_standing(stand_conn):
    _insert_canonical(stand_conn, "theme:fed_rate_hike")
    agent_c_output = {
        "standing_event_actions": {
            "promote_to_standing": [{
                "canonical_id": "theme:fed_rate_hike",
                "category": "monetary_policy",
                "summary": "Fed rate hike cycle",
                "affected_tickers": ["JPM", "GS"],
            }],
            "recommend_resolution": [],
        }
    }
    process_actions(stand_conn, agent_c_output, run_id=1)
    row = stand_conn.execute(
        "SELECT * FROM standing_events WHERE canonical_id = 'theme:fed_rate_hike'"
    ).fetchone()
    assert row is not None
    assert row["status"] == "active"
    assert row["category"] == "monetary_policy"
    assert json.loads(row["affected_tickers"]) == ["JPM", "GS"]


def test_process_recommend_resolution(stand_conn):
    _insert_standing(stand_conn, "theme:chip_shortage")
    se = stand_conn.execute("SELECT standing_id FROM standing_events").fetchone()
    agent_c_output = {
        "standing_event_actions": {
            "promote_to_standing": [],
            "recommend_resolution": [{"standing_id": se["standing_id"]}],
        }
    }
    process_actions(stand_conn, agent_c_output, run_id=1)
    row = stand_conn.execute(
        "SELECT * FROM standing_events WHERE standing_id = ?", (se["standing_id"],)
    ).fetchone()
    assert row["status"] == "resolved"
    assert row["resolved_at"] is not None


def test_process_actions_no_actions(stand_conn):
    # No standing_event_actions key — should not crash
    process_actions(stand_conn, {}, run_id=1)
    process_actions(stand_conn, {"other_key": "value"}, run_id=1)


def test_promote_creates_canonical_entity_if_missing(stand_conn):
    # No pre-inserted canonical entity
    agent_c_output = {
        "standing_event_actions": {
            "promote_to_standing": [{
                "canonical_id": "theme:new_trade_war",
                "category": "trade_policy",
                "summary": "New trade war escalation",
                "affected_tickers": ["AAPL"],
            }],
            "recommend_resolution": [],
        }
    }
    process_actions(stand_conn, agent_c_output, run_id=1)

    ce = stand_conn.execute(
        "SELECT * FROM canonical_entities WHERE canonical_id = 'theme:new_trade_war'"
    ).fetchone()
    assert ce is not None
    assert ce["entity_type"] == "MACRO_THEME"
    assert ce["display_name"] == "New Trade War"

    se = stand_conn.execute(
        "SELECT * FROM standing_events WHERE canonical_id = 'theme:new_trade_war'"
    ).fetchone()
    assert se is not None
    assert se["status"] == "active"

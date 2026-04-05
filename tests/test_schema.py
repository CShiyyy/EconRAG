import sqlite3

import pytest

from pipeline.db.schema import EXPECTED_TABLES, create_tables


def test_create_tables_creates_all_12(db_conn):
    rows = db_conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    ).fetchall()
    table_names = {row["name"] for row in rows}
    assert table_names == EXPECTED_TABLES


def test_create_tables_idempotent(db_conn):
    # Second call should not raise
    create_tables(db_conn)
    rows = db_conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    ).fetchall()
    assert len(rows) == 12


def test_wal_mode_active(db_conn):
    mode = db_conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode == "wal"


def test_foreign_keys_enforced(db_conn):
    with pytest.raises(sqlite3.IntegrityError):
        db_conn.execute(
            "INSERT INTO computed_targets (run_id, timestamp, per_ticker_json) "
            "VALUES (9999, '2026-01-01T00:00:00', '{}')"
        )


def test_account_check_constraint(db_conn):
    with pytest.raises(sqlite3.IntegrityError):
        db_conn.execute(
            "INSERT INTO account (account_id, cash_balance, initial_cash, universe, initialized_at) "
            "VALUES (2, 100000, 100000, 'djia30', '2026-01-01T00:00:00')"
        )


def test_canonical_entities_entity_type_check(db_conn):
    with pytest.raises(sqlite3.IntegrityError):
        db_conn.execute(
            "INSERT INTO canonical_entities (canonical_id, entity_type, display_name) "
            "VALUES ('test', 'INVALID', 'Test')"
        )


def test_recommendations_action_check(db_conn):
    # First insert a valid run_log row for the FK
    db_conn.execute(
        "INSERT INTO run_log (timestamp, run_type) VALUES ('2026-01-01T00:00:00', 'pre_open')"
    )
    with pytest.raises(sqlite3.IntegrityError):
        db_conn.execute(
            "INSERT INTO recommendations (run_id, timestamp, ticker, action, conviction_scores, rationale) "
            "VALUES (1, '2026-01-01T00:00:00', 'AAPL', 'invalid', '{}', 'test')"
        )


def test_standing_events_fk_to_canonical(db_conn):
    # Insert a run_log row first
    db_conn.execute(
        "INSERT INTO run_log (timestamp, run_type) VALUES ('2026-01-01T00:00:00', 'pre_open')"
    )
    with pytest.raises(sqlite3.IntegrityError):
        db_conn.execute(
            "INSERT INTO standing_events "
            "(canonical_id, category, summary, promoted_at, promotion_source, created_from_run_id) "
            "VALUES ('nonexistent', 'other', 'test', '2026-01-01T00:00:00', 'manual', 1)"
        )

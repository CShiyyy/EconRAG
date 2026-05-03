"""FastAPI dependencies — database connection and run-status registry."""

import itertools
import sqlite3
from datetime import datetime, timezone
from typing import Generator

from pipeline.config import DB_PATH
from pipeline.db.connection import get_connection


def get_db() -> Generator[sqlite3.Connection, None, None]:
    """Yield a SQLite connection per request, closing on teardown."""
    conn = get_connection(DB_PATH)
    try:
        yield conn
    finally:
        conn.close()


# In-memory registry for triggered pipeline runs.
# Keyed by trigger_id -> {status, started_at, completed_at, run_id, error}
RUN_STATUS: dict[int, dict] = {}

_trigger_counter = itertools.count(1)


def next_trigger_id() -> int:
    return next(_trigger_counter)


def register_trigger(trigger_id: int, session_date: str = "", run_type: str = "") -> None:
    RUN_STATUS[trigger_id] = {
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": None,
        "run_id": None,
        "error": None,
        "session_date": session_date,
        "run_type": run_type,
    }


def complete_trigger(trigger_id: int, run_id: int) -> None:
    RUN_STATUS[trigger_id].update(
        status="completed",
        completed_at=datetime.now(timezone.utc).isoformat(),
        run_id=run_id,
    )


def fail_trigger(trigger_id: int, error: str) -> None:
    RUN_STATUS[trigger_id].update(
        status="failed",
        completed_at=datetime.now(timezone.utc).isoformat(),
        error=error,
    )

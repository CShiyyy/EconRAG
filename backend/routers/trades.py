"""Trade history and latest-run trade endpoints."""

import sqlite3
from fastapi import APIRouter, Depends, Query
from backend.deps import get_db
from backend.schemas import PaginatedResponse

router = APIRouter(tags=["trades"])


def _enrich_trade(row: dict) -> dict:
    """Join run metadata onto a trade row."""
    return row


@router.get("/trades")
def list_trades(
    status: str | None = None,
    skip: int = 0,
    limit: int = Query(default=50, le=200),
    conn: sqlite3.Connection = Depends(get_db),
):
    """Return paginated trade history, optionally filtered by status."""
    where = "WHERE t.status = ?" if status else ""
    params_filter = [status] if status else []

    rows = conn.execute(
        f"""SELECT t.*,
               rl_q.session_date  AS queued_session_date,
               rl_q.run_type      AS queued_run_type,
               rl_e.session_date  AS execution_session_date,
               rl_e.run_type      AS execution_run_type
            FROM trades t
            LEFT JOIN run_log rl_q ON rl_q.run_id = t.queued_run_id
            LEFT JOIN run_log rl_e ON rl_e.run_id = t.execution_run_id
            {where}
            ORDER BY t.trade_id DESC
            LIMIT ? OFFSET ?""",
        [*params_filter, limit, skip],
    ).fetchall()

    count_row = conn.execute(
        f"SELECT COUNT(*) FROM trades t {where}", params_filter
    ).fetchone()
    total = count_row[0]

    return PaginatedResponse(
        items=[dict(r) for r in rows],
        total=total,
        skip=skip,
        limit=limit,
    )


@router.get("/trades/latest-run")
def get_latest_run_trades(conn: sqlite3.Connection = Depends(get_db)):
    """Return trades associated with the most recent run_log entry."""
    run_row = conn.execute(
        "SELECT * FROM run_log ORDER BY run_id DESC LIMIT 1"
    ).fetchone()
    if run_row is None:
        return {"run": None, "trades": [], "context": {}}

    run = dict(run_row)
    run_id = run["run_id"]

    # Trades queued by this run
    queued_rows = conn.execute(
        """SELECT t.*,
               rl_e.session_date AS execution_session_date,
               rl_e.run_type     AS execution_run_type
            FROM trades t
            LEFT JOIN run_log rl_e ON rl_e.run_id = t.execution_run_id
            WHERE t.queued_run_id = ?""",
        (run_id,),
    ).fetchall()

    # Trades executed by this run (may belong to a different queued_run_id)
    executed_rows = conn.execute(
        """SELECT t.*,
               rl_q.session_date AS queued_session_date,
               rl_q.run_type     AS queued_run_type
            FROM trades t
            LEFT JOIN run_log rl_q ON rl_q.run_id = t.queued_run_id
            WHERE t.execution_run_id = ? AND t.queued_run_id != ?""",
        (run_id, run_id),
    ).fetchall()

    trades = [dict(r) for r in queued_rows] + [dict(r) for r in executed_rows]

    queued_count = sum(1 for t in trades if t["status"] == "queued")
    executed_count = sum(1 for t in trades if t["status"] == "executed")
    overwritten_count = sum(1 for t in trades if t["status"] == "overwritten")

    return {
        "run": run,
        "trades": trades,
        "context": {
            "queued_count": queued_count,
            "executed_count": executed_count,
            "overwritten_count": overwritten_count,
        },
    }

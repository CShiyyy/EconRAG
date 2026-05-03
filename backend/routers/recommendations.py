"""Recommendations endpoints."""

import json
import sqlite3

from fastapi import APIRouter, Depends, Query

from backend.deps import get_db
from backend.schemas import PaginatedResponse

router = APIRouter(tags=["recommendations"])


def _parse_recommendation(row: sqlite3.Row) -> dict:
    d = dict(row)
    if d.get("conviction_scores"):
        d["conviction_scores"] = json.loads(d["conviction_scores"])
    if d.get("key_quant_metrics"):
        d["key_quant_metrics"] = json.loads(d["key_quant_metrics"])
    if d.get("key_risk_factors"):
        d["key_risk_factors"] = json.loads(d["key_risk_factors"])
    return d


@router.get("/recommendations/dates")
def list_recommendation_dates(conn: sqlite3.Connection = Depends(get_db)):
    """Return distinct session dates that have recommendations, newest first."""
    rows = conn.execute(
        """SELECT DISTINCT rl.session_date
           FROM recommendations r
           JOIN run_log rl ON r.run_id = rl.run_id
           WHERE rl.session_date IS NOT NULL
           ORDER BY rl.session_date DESC"""
    ).fetchall()
    return {"dates": [r["session_date"] for r in rows]}


@router.get("/recommendations/latest-run")
def get_latest_run_recommendations(conn: sqlite3.Connection = Depends(get_db)):
    """Return recommendations for the most recent run."""
    run_row = conn.execute(
        "SELECT * FROM run_log ORDER BY run_id DESC LIMIT 1"
    ).fetchone()
    if run_row is None:
        return {"run": None, "items": []}
    run = dict(run_row)
    rows = conn.execute(
        "SELECT * FROM recommendations WHERE run_id = ? ORDER BY recommendation_id",
        (run["run_id"],),
    ).fetchall()
    return {"run": run, "items": [_parse_recommendation(r) for r in rows]}


@router.get("/recommendations", response_model=PaginatedResponse)
def list_recommendations(
    session_date: str | None = Query(default=None),
    skip: int = 0,
    limit: int = Query(default=20, le=100),
    conn: sqlite3.Connection = Depends(get_db),
):
    """List recommendations, optionally filtered by session_date.

    When session_date is supplied, all rows for that date are returned
    (no pagination — max ~60 rows/day).
    """
    if session_date:
        rows = conn.execute(
            """SELECT r.*
               FROM recommendations r
               JOIN run_log rl ON r.run_id = rl.run_id
               WHERE rl.session_date = ?
               ORDER BY r.run_id DESC, r.recommendation_id ASC""",
            (session_date,),
        ).fetchall()
        items = [_parse_recommendation(r) for r in rows]
        return PaginatedResponse(items=items, total=len(items), skip=0, limit=len(items) or 1)

    rows = conn.execute(
        "SELECT * FROM recommendations ORDER BY recommendation_id DESC LIMIT ? OFFSET ?",
        (limit, skip),
    ).fetchall()
    total = conn.execute("SELECT COUNT(*) FROM recommendations").fetchone()[0]
    return PaginatedResponse(
        items=[_parse_recommendation(r) for r in rows],
        total=total,
        skip=skip,
        limit=limit,
    )


@router.get("/recommendations/{run_id}")
def get_recommendations_by_run(
    run_id: int,
    conn: sqlite3.Connection = Depends(get_db),
):
    rows = conn.execute(
        "SELECT * FROM recommendations WHERE run_id = ? ORDER BY recommendation_id",
        (run_id,),
    ).fetchall()
    return {"items": [_parse_recommendation(r) for r in rows]}

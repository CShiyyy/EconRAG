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
    return d


@router.get("/recommendations", response_model=PaginatedResponse)
def list_recommendations(
    skip: int = 0,
    limit: int = Query(default=20, le=100),
    conn: sqlite3.Connection = Depends(get_db),
):
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

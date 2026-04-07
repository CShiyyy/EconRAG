"""Standing events endpoints."""

import json
import sqlite3
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query

from backend.deps import get_db
from backend.schemas import StandingEventCreate, StandingEventPatch

router = APIRouter(tags=["standing-events"])


def _parse_event(row: sqlite3.Row) -> dict:
    d = dict(row)
    if d.get("affected_tickers"):
        d["affected_tickers"] = json.loads(d["affected_tickers"])
    return d


@router.get("/standing-events")
def list_standing_events(
    status: str | None = Query(default=None),
    conn: sqlite3.Connection = Depends(get_db),
):
    if status:
        rows = conn.execute(
            "SELECT * FROM standing_events WHERE status = ? ORDER BY standing_id DESC",
            (status,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM standing_events ORDER BY standing_id DESC"
        ).fetchall()
    return {"items": [_parse_event(r) for r in rows]}


@router.post("/standing-events", status_code=201)
def create_standing_event(
    body: StandingEventCreate,
    conn: sqlite3.Connection = Depends(get_db),
):
    now = datetime.now(timezone.utc).isoformat()

    # Ensure canonical entity exists, create if not
    existing = conn.execute(
        "SELECT canonical_id FROM canonical_entities WHERE canonical_id = ?",
        (body.canonical_id,),
    ).fetchone()
    if existing is None:
        from pipeline.orchestration.standing import _infer_entity_type

        entity_type = _infer_entity_type(body.canonical_id)
        conn.execute(
            "INSERT INTO canonical_entities (canonical_id, entity_type, display_name, aliases) "
            "VALUES (?, ?, ?, ?)",
            (body.canonical_id, entity_type, body.canonical_id, json.dumps([])),
        )

    cursor = conn.execute(
        """INSERT INTO standing_events
           (canonical_id, status, category, summary, affected_tickers,
            promoted_at, promotion_source, last_reinforced,
            reinforcement_count, stale_run_threshold)
           VALUES (?, 'active', ?, ?, ?, ?, 'manual', ?, 0, ?)""",
        (
            body.canonical_id,
            body.category,
            body.summary,
            json.dumps(body.affected_tickers),
            now,
            now,
            body.stale_run_threshold,
        ),
    )
    conn.commit()
    standing_id = cursor.lastrowid

    row = conn.execute(
        "SELECT * FROM standing_events WHERE standing_id = ?", (standing_id,)
    ).fetchone()
    return _parse_event(row)


@router.patch("/standing-events/{standing_id}")
def update_standing_event(
    standing_id: int,
    body: StandingEventPatch,
    conn: sqlite3.Connection = Depends(get_db),
):
    row = conn.execute(
        "SELECT * FROM standing_events WHERE standing_id = ?", (standing_id,)
    ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Standing event not found")

    updates = []
    params = []

    if body.summary is not None:
        updates.append("summary = ?")
        params.append(body.summary)

    if body.affected_tickers is not None:
        updates.append("affected_tickers = ?")
        params.append(json.dumps(body.affected_tickers))

    if body.status is not None:
        updates.append("status = ?")
        params.append(body.status)
        if body.status == "resolved":
            updates.append("resolved_at = ?")
            params.append(datetime.now(timezone.utc).isoformat())

    if not updates:
        raise HTTPException(status_code=400, detail="No fields to update")

    params.append(standing_id)
    conn.execute(
        f"UPDATE standing_events SET {', '.join(updates)} WHERE standing_id = ?",
        params,
    )
    conn.commit()

    row = conn.execute(
        "SELECT * FROM standing_events WHERE standing_id = ?", (standing_id,)
    ).fetchone()
    return _parse_event(row)

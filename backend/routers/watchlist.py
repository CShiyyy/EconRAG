"""Watchlist endpoints."""

import json
import sqlite3
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException

from backend.deps import get_db
from pipeline.db.helpers import get_account, get_watchlist

router = APIRouter(tags=["watchlist"])


@router.get("/watchlist")
def list_watchlist(conn: sqlite3.Connection = Depends(get_db)):
    try:
        get_account(conn)  # verify initialized
    except (RuntimeError, sqlite3.OperationalError):
        raise HTTPException(status_code=404, detail="System not initialized")
    return {"items": get_watchlist(conn)}


@router.post("/watchlist/refresh")
def refresh_watchlist(conn: sqlite3.Connection = Depends(get_db)):
    try:
        account = get_account(conn)
    except (RuntimeError, sqlite3.OperationalError):
        raise HTTPException(status_code=404, detail="System not initialized")

    from pipeline.scrapers.watchlist import scrape_universe

    universe = account["universe"]
    entries = scrape_universe(universe)

    existing = {r["ticker"] for r in get_watchlist(conn)}
    now = datetime.now(timezone.utc).isoformat()
    added = []

    for ticker, company_name, sector in entries:
        if ticker in existing:
            continue
        conn.execute(
            "INSERT INTO watchlist (ticker, company_name, sector, added_at) VALUES (?, ?, ?, ?)",
            (ticker, company_name, sector, now),
        )
        # Seed canonical entity for new ticker
        ce_exists = conn.execute(
            "SELECT 1 FROM canonical_entities WHERE canonical_id = ?", (ticker,)
        ).fetchone()
        if ce_exists is None:
            conn.execute(
                "INSERT INTO canonical_entities (canonical_id, entity_type, display_name, aliases) "
                "VALUES (?, 'COMPANY', ?, ?)",
                (ticker, company_name, json.dumps([company_name])),
            )
        added.append(ticker)

    conn.commit()
    return {"added": added, "total": len(get_watchlist(conn))}

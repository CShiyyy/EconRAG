"""Profiles endpoints — exposes seeded LightRAG profile text."""

import sqlite3

from fastapi import APIRouter, Depends, HTTPException

from backend.deps import get_db
from backend.schemas import ProfileDetail, ProfileItem

router = APIRouter(tags=["profiles"])


@router.get("/profiles", response_model=list[ProfileItem])
def list_profiles(conn: sqlite3.Connection = Depends(get_db)):
    rows = conn.execute(
        "SELECT seed_type, seed_key, seeded_at, source_id FROM kg_seed_log ORDER BY seed_type, seed_key"
    ).fetchall()
    return [
        ProfileItem(seed_type=r[0], seed_key=r[1], seeded_at=r[2], source_id=r[3])
        for r in rows
    ]


@router.get("/profiles/{key}", response_model=ProfileDetail)
def get_profile(key: str, conn: sqlite3.Connection = Depends(get_db)):
    row = conn.execute(
        "SELECT seed_type, seed_key, seeded_at, source_id, seed_text FROM kg_seed_log WHERE seed_key = ?",
        (key,),
    ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail=f"No seeded profile found for '{key}'")
    return ProfileDetail(
        seed_type=row[0],
        seed_key=row[1],
        seeded_at=row[2],
        source_id=row[3],
        seed_text=row[4],
    )

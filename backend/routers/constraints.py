"""Constraints endpoints."""

import sqlite3

from fastapi import APIRouter, Depends, HTTPException

from backend.deps import get_db
from backend.schemas import ConstraintPatch

router = APIRouter(tags=["constraints"])


@router.get("/constraints")
def get_constraints(conn: sqlite3.Connection = Depends(get_db)):
    rows = conn.execute(
        "SELECT constraint_name, value, description FROM constraints"
    ).fetchall()
    return {"constraints": [dict(r) for r in rows]}


@router.patch("/constraints")
def update_constraints(
    body: ConstraintPatch,
    conn: sqlite3.Connection = Depends(get_db),
):
    existing = conn.execute(
        "SELECT constraint_name FROM constraints"
    ).fetchall()
    valid_keys = {r["constraint_name"] for r in existing}

    invalid = set(body.constraints) - valid_keys
    if invalid:
        raise HTTPException(
            status_code=400, detail=f"Unknown constraint keys: {invalid}"
        )

    for name, value in body.constraints.items():
        conn.execute(
            "UPDATE constraints SET value = ? WHERE constraint_name = ?",
            (value, name),
        )
    conn.commit()

    rows = conn.execute(
        "SELECT constraint_name, value, description FROM constraints"
    ).fetchall()
    return {"constraints": [dict(r) for r in rows]}

"""Initialization endpoints."""

import sqlite3

from fastapi import APIRouter, Depends, HTTPException

from backend.deps import get_db
from backend.schemas import InitRequest, InitStatusResponse
from pipeline.db.helpers import get_account
from pipeline.db.init import initialize

router = APIRouter(tags=["init"])


@router.post("/init")
def init_system(body: InitRequest, conn: sqlite3.Connection = Depends(get_db)):
    try:
        initialize(conn, body.universe, body.starting_cash, body.constraints)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return get_account(conn)


@router.get("/init/status", response_model=InitStatusResponse)
def init_status(conn: sqlite3.Connection = Depends(get_db)):
    try:
        account = get_account(conn)
        return InitStatusResponse(initialized=True, account=account)
    except (RuntimeError, sqlite3.OperationalError):
        return InitStatusResponse(initialized=False)

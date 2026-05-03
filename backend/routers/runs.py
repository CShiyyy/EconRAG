"""Run log and pipeline trigger endpoints."""

import json
import logging
import sqlite3

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query

from backend.deps import (
    RUN_STATUS,
    complete_trigger,
    fail_trigger,
    get_db,
    next_trigger_id,
    register_trigger,
)
from backend.schemas import (
    PaginatedResponse,
    SlotStatusResponse,
    TriggerRunRequest,
    TriggerRunResponse,
    TriggerStatusResponse,
)
from pipeline.config import DB_PATH, LIGHTRAG_STORAGE_DIR
from pipeline.orchestration.clock import market_phase

logger = logging.getLogger(__name__)

router = APIRouter(tags=["runs"])


@router.get("/runs", response_model=PaginatedResponse)
def list_runs(
    skip: int = 0,
    limit: int = Query(default=20, le=100),
    conn: sqlite3.Connection = Depends(get_db),
):
    rows = conn.execute(
        """SELECT rl.*,
               (SELECT COUNT(*) FROM trades WHERE queued_run_id = rl.run_id) AS queued_count,
               (SELECT COUNT(*) FROM trades WHERE execution_run_id = rl.run_id) AS executed_count
            FROM run_log rl
            ORDER BY rl.run_id DESC LIMIT ? OFFSET ?""",
        (limit, skip),
    ).fetchall()
    total = conn.execute("SELECT COUNT(*) FROM run_log").fetchone()[0]
    items = []
    for row in rows:
        d = dict(row)
        if d.get("source_status"):
            d["source_status"] = json.loads(d["source_status"])
        d.setdefault("started_at", d.get("timestamp"))
        items.append(d)
    return PaginatedResponse(items=items, total=total, skip=skip, limit=limit)


# Must be defined before /runs/{run_id} so FastAPI matches static segment first
@router.get("/runs/slot-status", response_model=SlotStatusResponse)
def get_slot_status(conn: sqlite3.Connection = Depends(get_db)):
    """Return current market phase and whether there are pending queued trades."""
    from datetime import datetime, timezone
    import pytz
    et = pytz.timezone("America/New_York")
    session_date = datetime.now(timezone.utc).astimezone(et).strftime("%Y-%m-%d")
    phase = market_phase()
    has_pending = conn.execute(
        "SELECT 1 FROM trades WHERE status='queued' LIMIT 1"
    ).fetchone() is not None
    return SlotStatusResponse(
        session_date=session_date,
        run_type="scheduled",
        existing=False,
        trading_day=phase != "closed_day",
        market_phase=phase,
        has_pending_queue=has_pending,
    )


@router.get("/runs/{run_id}")
def get_run_detail(run_id: int, conn: sqlite3.Connection = Depends(get_db)):
    run = conn.execute(
        "SELECT * FROM run_log WHERE run_id = ?", (run_id,)
    ).fetchone()
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")

    run_dict = dict(run)
    if run_dict.get("source_status"):
        run_dict["source_status"] = json.loads(run_dict["source_status"])
    run_dict.setdefault("started_at", run_dict.get("timestamp"))

    # Agent outputs
    agent_rows = conn.execute(
        "SELECT * FROM agent_outputs WHERE run_id = ?", (run_id,)
    ).fetchall()
    agent_outputs = {}
    for r in agent_rows:
        agent_outputs[r["agent"]] = json.loads(r["output_blob"])

    # Recommendations
    rec_rows = conn.execute(
        "SELECT * FROM recommendations WHERE run_id = ?", (run_id,)
    ).fetchall()
    recommendations = []
    for r in rec_rows:
        d = dict(r)
        if d.get("conviction_scores"):
            d["conviction_scores"] = json.loads(d["conviction_scores"])
        if d.get("key_quant_metrics"):
            d["key_quant_metrics"] = json.loads(d["key_quant_metrics"])
        recommendations.append(d)

    # Trades
    trade_rows = conn.execute(
        "SELECT * FROM trades WHERE recommendation_id IN "
        "(SELECT recommendation_id FROM recommendations WHERE run_id = ?)",
        (run_id,),
    ).fetchall()
    trades = [dict(t) for t in trade_rows]

    return {
        "run": run_dict,
        "agent_outputs": agent_outputs,
        "recommendations": recommendations,
        "trades": trades,
    }


@router.post("/runs/trigger", response_model=TriggerRunResponse)
def trigger_run(
    body: TriggerRunRequest,
    background_tasks: BackgroundTasks,
):
    from datetime import datetime, timezone
    import pytz
    et = pytz.timezone("America/New_York")
    session_date = datetime.now(timezone.utc).astimezone(et).strftime("%Y-%m-%d")

    # Guard against concurrent in-flight trigger
    for entry in RUN_STATUS.values():
        if entry.get("status") == "running":
            raise HTTPException(
                status_code=409,
                detail={"code": "run_in_progress"},
            )

    trigger_id = next_trigger_id()
    register_trigger(trigger_id, session_date=session_date, run_type="scheduled")
    background_tasks.add_task(_execute_pipeline, trigger_id, session_date)
    return TriggerRunResponse(
        trigger_id=trigger_id, status="running",
        session_date=session_date, run_type="scheduled",
    )


@router.get("/runs/trigger/{trigger_id}/status", response_model=TriggerStatusResponse)
def get_trigger_status(trigger_id: int):
    entry = RUN_STATUS.get(trigger_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="Trigger ID not found")
    return TriggerStatusResponse(
        trigger_id=trigger_id,
        status=entry["status"],
        run_id=entry.get("run_id"),
        error=entry.get("error"),
    )


async def _execute_pipeline(trigger_id: int, session_date: str) -> None:
    """Background task that runs the full pipeline."""
    try:
        from pipeline.orchestration.graph import run_pipeline

        result = await run_pipeline(
            db_path=str(DB_PATH),
            session_date=session_date,
            rag_storage_dir=str(LIGHTRAG_STORAGE_DIR),
        )
        run_id = result.get("run_id")
        complete_trigger(trigger_id, run_id)
    except Exception as e:
        logger.exception("Pipeline run failed for trigger %s", trigger_id)
        fail_trigger(trigger_id, str(e))

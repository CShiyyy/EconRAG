"""Parent orchestration graph — composes data, reasoning, and execution subgraphs."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

from langgraph.graph import StateGraph, START, END

from pipeline.orchestration.data_graph import build_data_graph
from pipeline.orchestration.execution_graph import build_execution_graph
from pipeline.orchestration.reasoning_graph import build_reasoning_graph
from pipeline.orchestration.state import PipelineState

logger = logging.getLogger(__name__)


def _create_run_log(conn, run_type: str, first_run: bool) -> int:
    """Insert a new run_log entry and return the run_id."""
    ts = datetime.now(timezone.utc).isoformat()
    cursor = conn.execute(
        """INSERT INTO run_log
           (timestamp, run_type, is_first_run)
           VALUES (?, ?, ?)""",
        (ts, run_type, 1 if first_run else 0),
    )
    conn.commit()
    return cursor.lastrowid


def build_pipeline_graph() -> StateGraph:
    """Build the parent pipeline graph composing all three subgraphs.

    Returns an uncompiled StateGraph. Call .compile() before invoking.
    """
    data_graph = build_data_graph().compile()
    reasoning_graph = build_reasoning_graph().compile()
    execution_graph = build_execution_graph().compile()

    parent = StateGraph(PipelineState)
    parent.add_node("data_pipeline", data_graph)
    parent.add_node("reasoning", reasoning_graph)
    parent.add_node("execution", execution_graph)

    parent.add_edge(START, "data_pipeline")
    parent.add_edge("data_pipeline", "reasoning")
    parent.add_edge("reasoning", "execution")
    parent.add_edge("execution", END)

    return parent


async def run_pipeline(
    db_path: str,
    run_type: str,
    rag_storage_dir: str | None = None,
) -> dict:
    """Async entry point for the full pipeline.

    Args:
        db_path: Path to the SQLite database.
        run_type: "pre_open" or "post_close".
        rag_storage_dir: Override LightRAG storage directory.

    Returns:
        Final PipelineState dict after execution.
    """
    from pipeline.config import LIGHTRAG_STORAGE_DIR
    from pipeline.db.connection import get_connection
    from pipeline.db.helpers import get_previous_assessment, get_watchlist, is_first_run
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    if rag_storage_dir is None:
        rag_storage_dir = str(LIGHTRAG_STORAGE_DIR)

    # Initialize run
    conn = get_connection(db_path)
    try:
        first_run = is_first_run(conn)
        run_id = _create_run_log(conn, run_type, first_run)
        tickers = [row["ticker"] for row in get_watchlist(conn)]
        previous_assessment = get_previous_assessment(conn)
    finally:
        conn.close()

    if not tickers:
        logger.warning("No tickers in watchlist, aborting pipeline")
        return {}

    initial_state: PipelineState = {
        "run_id": run_id,
        "run_type": run_type,
        "is_first_run": first_run,
        "start_time": time.time(),
        "db_path": db_path,
        "rag_storage_dir": rag_storage_dir,
        "tickers": tickers,
        "previous_assessment": previous_assessment,
        "standing_context": [],
        "requery_count": 0,
        "requery_reason": None,
        "flagged_tickers": [],
        "ingestion_result": None,
        "source_health": [],
        "market_narrative": None,
        "quant_assessment": None,
        "risk_assessment": None,
        "sizing_result": None,
        "trade_list": [],
    }

    graph = build_pipeline_graph()

    checkpoint_path = db_path + ".checkpoints"
    async with AsyncSqliteSaver.from_conn_string(checkpoint_path) as checkpointer:
        app = graph.compile(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": f"run-{run_id}"}}
        final_state = await app.ainvoke(initial_state, config=config)

    logger.info("Pipeline run %s completed", run_id)
    return final_state

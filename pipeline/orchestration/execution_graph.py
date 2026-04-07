"""Execution Subgraph — standing_actions -> position_sizing -> execute_trades -> log."""

from __future__ import annotations

import logging
import time

from langgraph.graph import StateGraph, START, END

from pipeline.orchestration.state import PipelineState

logger = logging.getLogger(__name__)


async def standing_actions_node(state: PipelineState) -> dict:
    """Process standing event actions from Agent C output."""
    from pipeline.db.connection import get_connection
    from pipeline.orchestration.standing import process_actions

    risk_assessment = state.get("risk_assessment")
    if not risk_assessment:
        logger.info("No risk assessment, skipping standing actions")
        return {}

    conn = get_connection(state["db_path"])
    try:
        process_actions(conn, risk_assessment, state["run_id"])
        logger.info("Standing actions processed")
    finally:
        conn.close()
    return {}


async def position_sizing_node(state: PipelineState) -> dict:
    """Run position sizing engine to compute trades."""
    from pipeline.db.connection import get_connection
    from pipeline.sizing.engine import run_sizing_engine

    risk_assessment = state.get("risk_assessment", {})
    agent_c_per_ticker = risk_assessment.get("per_ticker", {})

    # Build fill prices from ingestion market data (use open_price)
    ingestion = state.get("ingestion_result", {})
    market_data = ingestion.get("market_data", {})
    fill_prices = {
        ticker: md["open_price"] for ticker, md in market_data.items()
        if "open_price" in md
    }

    conn = get_connection(state["db_path"])
    try:
        result = run_sizing_engine(
            conn,
            state["run_id"],
            agent_c_per_ticker,
            fill_prices,
            is_first_run=state.get("is_first_run", False),
        )
        logger.info(
            "Position sizing complete: %d trades",
            len(result.get("trade_list", [])),
        )
    finally:
        conn.close()
    return {
        "sizing_result": result,
        "trade_list": result.get("trade_list", []),
    }


async def execute_trades_node(state: PipelineState) -> dict:
    """Log executed trades to the trades table."""
    from pipeline.db.connection import get_connection
    from pipeline.orchestration.executor import log_trades

    trade_list = state.get("trade_list", [])
    if not trade_list:
        logger.info("No trades to execute")
        return {}

    ingestion = state.get("ingestion_result", {})
    market_data = ingestion.get("market_data", {})

    # Look up recommendation IDs from DB
    conn = get_connection(state["db_path"])
    try:
        rows = conn.execute(
            "SELECT ticker, recommendation_id FROM recommendations WHERE run_id = ?",
            (state["run_id"],),
        ).fetchall()
        recommendation_ids = {row["ticker"]: row["recommendation_id"] for row in rows}

        log_trades(conn, trade_list, market_data, recommendation_ids, state["run_id"])
        logger.info("Executed %d trades", len(trade_list))
    finally:
        conn.close()
    return {}


async def log_node(state: PipelineState) -> dict:
    """Update run log with final timing and status."""
    from pipeline.db.connection import get_connection

    wall_clock = time.time() - state.get("start_time", time.time())

    source_health = state.get("source_health", [])
    source_statuses = [h.get("status", "unknown") for h in source_health]
    all_ok = all(s == "success" for s in source_statuses)
    source_status = "all_ok" if all_ok else "degraded"

    requery_triggered = state.get("requery_count", 0) > 0

    conn = get_connection(state["db_path"])
    try:
        conn.execute(
            """UPDATE run_log
               SET wall_clock_seconds = ?,
                   source_status = ?,
                   requery_triggered = ?,
                   status = 'completed'
               WHERE run_id = ?""",
            (round(wall_clock, 2), source_status, 1 if requery_triggered else 0, state["run_id"]),
        )
        conn.commit()
        logger.info(
            "Run %s completed in %.1fs (source: %s, requery: %s)",
            state["run_id"],
            wall_clock,
            source_status,
            requery_triggered,
        )
    finally:
        conn.close()
    return {}


def _route_after_standing(state: PipelineState) -> str:
    """Route after standing actions: skip sizing on post-close runs."""
    if state.get("run_type") == "post_close":
        return "log"
    return "position_sizing"


def build_execution_graph() -> StateGraph:
    """Build the execution subgraph (uncompiled)."""
    graph = StateGraph(PipelineState)

    graph.add_node("standing_actions", standing_actions_node)
    graph.add_node("position_sizing", position_sizing_node)
    graph.add_node("execute_trades", execute_trades_node)
    graph.add_node("log", log_node)

    graph.add_edge(START, "standing_actions")
    graph.add_conditional_edges(
        "standing_actions", _route_after_standing, ["position_sizing", "log"]
    )
    graph.add_edge("position_sizing", "execute_trades")
    graph.add_edge("execute_trades", "log")
    graph.add_edge("log", END)

    return graph

"""Execution Subgraph — execute_pending -> standing_actions -> position_sizing -> overwrite_and_queue -> snapshot -> log."""

from __future__ import annotations

import logging
import time

from langgraph.graph import StateGraph, START, END

from pipeline.orchestration.state import PipelineState

logger = logging.getLogger(__name__)


async def execute_pending_node(state: PipelineState) -> dict:
    """Execute any queued trades whose queued_at predates the most recent market close."""
    from datetime import datetime, timezone
    from pipeline.db.connection import get_connection
    from pipeline.orchestration.clock import previous_close
    from pipeline.orchestration.executor import execute_queued_batch

    conn = get_connection(state["db_path"])
    try:
        last_close_dt = previous_close()
        has_queued = conn.execute(
            "SELECT 1 FROM trades WHERE status='queued' AND queued_at < ? LIMIT 1",
            (last_close_dt.isoformat(),),
        ).fetchone()

        if not has_queued:
            logger.info("No queued trades ready for execution")
            return {"last_close_dt": last_close_dt.isoformat()}

        ingestion = state.get("ingestion_result") or {}
        market_data = ingestion.get("market_data") or {}

        # Use ingestion current_price for today's close; yfinance history for older closes
        import pytz
        from datetime import timedelta
        et = pytz.timezone("America/New_York")
        now_et = datetime.now(timezone.utc).astimezone(et)
        last_close_date = last_close_dt.astimezone(et).date()
        is_todays_close = last_close_date == now_et.date()

        def price_source(ticker: str) -> float | None:
            if is_todays_close:
                md = market_data.get(ticker, {})
                price = md.get("current_price")
                return float(price) if price else None
            # Historical close via yfinance
            try:
                import yfinance as yf
                hist = yf.Ticker(ticker).history(
                    start=last_close_date,
                    end=last_close_date + timedelta(days=1),
                )
                if not hist.empty:
                    return float(hist["Close"].iloc[-1])
            except Exception:
                pass
            return None

        result = execute_queued_batch(conn, state["run_id"], last_close_dt, price_source)
        logger.info(
            "Executed %d trades, skipped %d (%s)",
            result["executed"], result["skipped"], result.get("skipped_tickers", []),
        )
        return {"last_close_dt": last_close_dt.isoformat(), "execution_result": result}
    finally:
        conn.close()


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
    """Run position sizing engine — computes the trade plan without mutating portfolio."""
    from pipeline.db.connection import get_connection
    from pipeline.sizing.engine import run_sizing_engine

    risk_assessment = state.get("risk_assessment", {})
    agent_c_per_ticker = risk_assessment.get("per_ticker", {})

    ingestion = state.get("ingestion_result", {})
    market_data = ingestion.get("market_data", {})
    fill_prices = {
        ticker: md["previous_close"]
        for ticker, md in market_data.items()
        if "previous_close" in md
    }

    conn = get_connection(state["db_path"])
    try:
        result = run_sizing_engine(
            conn,
            state["run_id"],
            agent_c_per_ticker,
            fill_prices,
            is_first_run=state.get("is_first_run", False),
            agent_b_output=state.get("quant_assessment"),
            agent_a_output=state.get("market_narrative"),
            mutate_portfolio=False,
        )
        per_ticker_data = result.get("per_ticker_data", {})
        for ticker, data in per_ticker_data.items():
            cw = data.get("conviction_weight")
            tw = data.get("target_weight")
            if cw is not None or tw is not None:
                conn.execute(
                    "UPDATE recommendations SET conviction_weight = ?, target_weight = ? WHERE run_id = ? AND ticker = ?",
                    (cw, tw, state["run_id"], ticker),
                )
        conn.commit()
        logger.info("Position sizing complete: %d trades planned", len(result.get("trade_list", [])))
    finally:
        conn.close()

    return {
        "sizing_result": result,
        "trade_list": result.get("trade_list", []),
    }


async def overwrite_and_queue_node(state: PipelineState) -> dict:
    """Overwrite any pending same-window queue then queue the fresh trade list."""
    from pipeline.db.connection import get_connection
    from pipeline.orchestration.clock import next_close, previous_close
    from pipeline.orchestration.executor import mark_overwritten, queue_trades

    trade_list = state.get("trade_list", [])

    conn = get_connection(state["db_path"])
    try:
        last_close_dt_str = state.get("last_close_dt")
        if last_close_dt_str:
            from datetime import datetime, timezone
            last_close_dt = datetime.fromisoformat(last_close_dt_str)
        else:
            last_close_dt = previous_close()

        overwritten = mark_overwritten(conn, state["run_id"], last_close_dt)
        if overwritten:
            logger.info("Overwrote %d pending queued trades from current window", overwritten)

        if not trade_list:
            logger.info("No trades to queue")
            return {}

        target_close_at = next_close()
        rows = conn.execute(
            "SELECT ticker, recommendation_id FROM recommendations WHERE run_id = ?",
            (state["run_id"],),
        ).fetchall()
        recommendation_ids = {row["ticker"]: row["recommendation_id"] for row in rows}

        trade_ids = queue_trades(conn, trade_list, recommendation_ids, state["run_id"], target_close_at)
        logger.info("Queued %d trades (target close: %s)", len(trade_ids), target_close_at.isoformat())
    finally:
        conn.close()

    return {}


async def snapshot_node(state: PipelineState) -> dict:
    """Record post-execution portfolio snapshot using current prices."""
    from pipeline.db.connection import get_connection
    from pipeline.orchestration.snapshots import record_snapshot

    ingestion = state.get("ingestion_result") or {}
    market_data = ingestion.get("market_data") or None

    conn = get_connection(state["db_path"])
    try:
        record_snapshot(conn, state["run_id"], market_data)
        logger.info("Snapshot recorded for run %s", state["run_id"])
    finally:
        conn.close()
    return {}


async def log_node(state: PipelineState) -> dict:
    """Update run log with final timing and status."""
    from pipeline.db.connection import get_connection

    wall_clock = time.time() - state.get("start_time", time.time())
    source_health = state.get("source_health", [])
    all_ok = all(h.get("status") == "success" for h in source_health)
    requery_triggered = state.get("requery_count", 0) > 0

    conn = get_connection(state["db_path"])
    try:
        conn.execute(
            """UPDATE run_log
               SET wall_clock_seconds = ?,
                   source_status = ?,
                   requery_triggered = ?,
                   requery_reason = ?
               WHERE run_id = ?""",
            (round(wall_clock, 2), "all_ok" if all_ok else "degraded",
             1 if requery_triggered else 0, state.get("requery_reason"), state["run_id"]),
        )
        conn.commit()
        logger.info("Run %s completed in %.1fs", state["run_id"], wall_clock)
    finally:
        conn.close()
    return {}


def build_execution_graph() -> StateGraph:
    """Build the execution subgraph (uncompiled)."""
    graph = StateGraph(PipelineState)

    graph.add_node("execute_pending", execute_pending_node)
    graph.add_node("standing_actions", standing_actions_node)
    graph.add_node("position_sizing", position_sizing_node)
    graph.add_node("overwrite_and_queue", overwrite_and_queue_node)
    graph.add_node("snapshot", snapshot_node)
    graph.add_node("log", log_node)

    graph.add_edge(START, "execute_pending")
    graph.add_edge("execute_pending", "standing_actions")
    graph.add_edge("standing_actions", "position_sizing")
    graph.add_edge("position_sizing", "overwrite_and_queue")
    graph.add_edge("overwrite_and_queue", "snapshot")
    graph.add_edge("snapshot", "log")
    graph.add_edge("log", END)

    return graph

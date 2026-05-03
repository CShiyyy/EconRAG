"""Agent Reasoning Subgraph — agent_a -> agent_b -> requery_check -> agent_c.

Conditional edges handle re-query loop (max 1 re-query per run, disabled on
first run). Agent B runs on every run including first run.
"""

from __future__ import annotations

import logging

from langgraph.graph import StateGraph, START, END

from pipeline.orchestration.state import PipelineState

logger = logging.getLogger(__name__)


async def agent_a_node(state: PipelineState) -> dict:
    """Run Agent A (local LLM context retriever)."""
    from pipeline.db.connection import get_connection
    from pipeline.knowledge.lightrag_config import get_rag_instance
    from pipeline.agents.agent_a import run_agent_a

    rag = await get_rag_instance(state.get("rag_storage_dir"))
    conn = get_connection(state["db_path"])
    try:
        flagged = state.get("flagged_tickers", [])
        result = await run_agent_a(
            conn,
            rag,
            state["tickers"],
            state["run_id"],
            flagged_tickers=flagged if flagged else None,
        )
        logger.info("Agent A complete: %d ticker analyses", len(result.get("per_ticker", {})))
    finally:
        conn.close()
    return {"market_narrative": result}


async def agent_b_node(state: PipelineState) -> dict:
    """Run Agent B (deterministic quant script)."""
    from pipeline.db.connection import get_connection
    from pipeline.knowledge.lightrag_config import get_rag_instance
    from pipeline.agents.agent_b import run_agent_b
    from pipeline.ingestion.models import MarketDataPoint

    ingestion = state.get("ingestion_result", {})
    market_data_dicts = ingestion.get("market_data", {})

    # Reconstruct MarketDataPoint objects from serialized dicts
    market_data = {
        ticker: MarketDataPoint(**md) for ticker, md in market_data_dicts.items()
    }

    rag = await get_rag_instance(state.get("rag_storage_dir"))
    conn = get_connection(state["db_path"])
    try:
        result = await run_agent_b(conn, rag, market_data, state["run_id"])
        logger.info("Agent B complete")
    finally:
        conn.close()
    return {"quant_assessment": result}


async def requery_check_node(state: PipelineState) -> dict:
    """Evaluate whether re-query is needed based on agent outputs."""
    from pipeline.orchestration.conditions import evaluate_requery

    # Re-query disabled on first run (no prior baseline to compare against)
    if state.get("is_first_run", False):
        logger.info("First run — re-query disabled")
        return {"flagged_tickers": [], "requery_reason": None}

    requery_count = state.get("requery_count", 0)

    # If already re-queried, don't do it again
    if requery_count > 0:
        logger.info("Re-query already performed, skipping")
        return {"flagged_tickers": [], "requery_reason": None}

    result = evaluate_requery(
        state.get("market_narrative", {}),
        state.get("quant_assessment"),
        state.get("source_health", []),
    )

    if result["should_requery"]:
        logger.info(
            "Re-query triggered for tickers: %s. Reason: %s",
            result["flagged_tickers"],
            result["reason"],
        )
        return {
            "flagged_tickers": result["flagged_tickers"],
            "requery_count": 1,
            "requery_reason": result["reason"],
        }

    logger.info("No re-query needed")
    return {"flagged_tickers": [], "requery_reason": None}


async def agent_c_node(state: PipelineState) -> dict:
    """Run Agent C (cloud LLM portfolio synthesizer)."""
    from pipeline.db.connection import get_connection
    from pipeline.agents.agent_c import run_agent_c

    conn = get_connection(state["db_path"])
    try:
        result = await run_agent_c(
            conn,
            state.get("market_narrative", {}),
            state.get("quant_assessment"),
            state.get("source_health", []),
            state["run_id"],
            state["run_type"],
            state.get("is_first_run", False),
        )
        logger.info("Agent C complete")
    finally:
        conn.close()
    return {"risk_assessment": result}


def _route_after_agent_a(state: PipelineState) -> str:
    """Route after Agent A: always run agent_b, except on a re-query pass."""
    if state.get("requery_count", 0) > 0:
        return "agent_c"
    return "agent_b"


def _route_after_requery(state: PipelineState) -> str:
    """Route after requery check: loop to agent_a if flagged, else agent_c."""
    flagged = state.get("flagged_tickers", [])
    if flagged:
        return "agent_a"
    return "agent_c"


def build_reasoning_graph() -> StateGraph:
    """Build the agent reasoning subgraph (uncompiled)."""
    graph = StateGraph(PipelineState)

    graph.add_node("agent_a", agent_a_node)
    graph.add_node("agent_b", agent_b_node)
    graph.add_node("requery_check", requery_check_node)
    graph.add_node("agent_c", agent_c_node)

    graph.add_edge(START, "agent_a")
    graph.add_conditional_edges("agent_a", _route_after_agent_a, ["agent_b", "agent_c"])
    graph.add_edge("agent_b", "requery_check")
    graph.add_conditional_edges("requery_check", _route_after_requery, ["agent_a", "agent_c"])
    graph.add_edge("agent_c", END)

    return graph

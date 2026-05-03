"""Data Pipeline Subgraph — seed_profiles -> ingest -> extract_and_resolve -> embed_and_prune -> standing_maintenance."""

from __future__ import annotations

import dataclasses
import logging

from langgraph.graph import StateGraph, START, END

from pipeline.orchestration.state import PipelineState

logger = logging.getLogger(__name__)


async def seed_profiles_node(state: PipelineState) -> dict:
    """Seed LightRAG with ticker and macro profiles before the first ingest.

    Idempotent: profiles already in kg_seed_log are skipped in O(1) SQL.
    Runs on every pipeline call so new tickers added via watchlist refresh
    are picked up automatically on the next run.
    """
    from pipeline.agents.cloud_client import create_seeding_client
    from pipeline.config import PROFILE_SEED_ENABLED
    from pipeline.db.connection import get_connection
    from pipeline.knowledge.lightrag_config import get_rag_instance
    from pipeline.knowledge.profile_seeder import seed_missing_profiles

    if not PROFILE_SEED_ENABLED:
        return {}

    conn = get_connection(state["db_path"])
    try:
        rag = await get_rag_instance(state.get("rag_storage_dir"))
        client = create_seeding_client()
        result = await seed_missing_profiles(conn, rag, client, state["run_id"])
        return {
            "profile_seed_result": {
                "tickers_seeded": result.tickers_seeded,
                "macro_seeded": result.macro_seeded,
                "skipped_existing": result.skipped_existing,
                "tickers_failed": result.tickers_failed,
                "macro_error": result.macro_error,
            }
        }
    finally:
        conn.close()


async def ingest_node(state: PipelineState) -> dict:
    """Run full ingestion pipeline and serialize result for checkpointing."""
    from pipeline.ingestion.orchestrator import run_ingestion

    result = await run_ingestion(state["tickers"])
    serialized = dataclasses.asdict(result)

    source_health = serialized.get("health", [])
    logger.info(
        "Ingestion complete: %d market points, %d news, %d social, %d parsed",
        len(serialized.get("market_data", {})),
        len(serialized.get("news_hits", [])),
        len(serialized.get("social_hits", [])),
        len(serialized.get("parsed_content", [])),
    )
    for h in source_health:
        if h.get("status") != "success":
            logger.warning(
                "Source '%s': status=%s, error=%s",
                h.get("source"), h.get("status"), h.get("error_detail") or "none",
            )
    return {"ingestion_result": serialized, "source_health": source_health}


async def extract_and_resolve_node(state: PipelineState) -> dict:
    """Extract entities/relations from parsed content and insert into knowledge graph."""
    from pipeline.db.connection import get_connection
    from pipeline.ingestion.models import ParsedContent
    from pipeline.knowledge.extraction import process_content_batch
    from pipeline.knowledge.lightrag_config import get_rag_instance

    ingestion = state.get("ingestion_result")
    if not ingestion:
        logger.warning("No ingestion result, skipping extraction")
        return {}

    parsed_dicts = ingestion.get("parsed_content", [])
    if not parsed_dicts:
        logger.info("No parsed content, skipping extraction")
        return {}

    parsed_contents = [
        ParsedContent(**d) for d in parsed_dicts if d.get("success")
    ]
    if not parsed_contents:
        logger.info("No successful parsed content, skipping extraction")
        return {}

    rag = await get_rag_instance(state.get("rag_storage_dir"))
    conn = get_connection(state["db_path"])
    try:
        result = await process_content_batch(parsed_contents, conn, rag, state["run_id"])
        logger.info(
            "Extraction complete: %d processed, %d valid, %d failed",
            result.total_processed,
            result.total_valid,
            result.total_failed,
        )
    finally:
        conn.close()
    return {}


async def embed_and_prune_node(state: PipelineState) -> dict:
    """Prune expired ephemeral edges from knowledge graph."""
    from pipeline.db.connection import get_connection
    from pipeline.knowledge.lightrag_config import get_rag_instance
    from pipeline.knowledge.pruner import prune_expired_edges

    rag = await get_rag_instance(state.get("rag_storage_dir"))
    conn = get_connection(state["db_path"])
    try:
        result = await prune_expired_edges(rag, conn)
        logger.info(
            "Prune complete: %d scanned, %d expired, %d removed",
            result.edges_scanned,
            result.edges_expired,
            result.edges_removed,
        )
    finally:
        conn.close()
    return {}


async def standing_maintenance_node(state: PipelineState) -> dict:
    """Run standing event maintenance and return active events."""
    from pipeline.db.connection import get_connection
    from pipeline.orchestration.standing import run_maintenance

    conn = get_connection(state["db_path"])
    try:
        events = run_maintenance(conn, state["run_id"])
        logger.info("Standing maintenance: %d active events", len(events))
    finally:
        conn.close()
    return {"standing_context": events}


def build_data_graph() -> StateGraph:
    """Build the data pipeline subgraph (uncompiled).

    Edge order: seed_profiles -> ingest -> extract_and_resolve
                -> embed_and_prune -> standing_maintenance
    """
    graph = StateGraph(PipelineState)

    graph.add_node("seed_profiles", seed_profiles_node)
    graph.add_node("ingest", ingest_node)
    graph.add_node("extract_and_resolve", extract_and_resolve_node)
    graph.add_node("embed_and_prune", embed_and_prune_node)
    graph.add_node("standing_maintenance", standing_maintenance_node)

    graph.add_edge(START, "seed_profiles")
    graph.add_edge("seed_profiles", "ingest")
    graph.add_edge("ingest", "extract_and_resolve")
    graph.add_edge("extract_and_resolve", "embed_and_prune")
    graph.add_edge("embed_and_prune", "standing_maintenance")
    graph.add_edge("standing_maintenance", END)

    return graph

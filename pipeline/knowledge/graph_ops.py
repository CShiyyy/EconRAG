"""Graph insertion and query helpers bridging validated data to LightRAG."""

from __future__ import annotations

import json
from typing import Any

from lightrag import LightRAG, QueryParam

from pipeline.knowledge.validator import ValidationResult


# Metadata is appended to relationship descriptions with this delimiter.
# The pruner parses metadata from this suffix.
META_DELIMITER = "|||META:"


def _encode_rel_metadata(rel) -> str:
    """Encode relationship metadata as a JSON suffix on the description."""
    meta = {
        "relationship_type": rel.relationship_type,
        "tier": rel.tier,
        "significance_score": rel.significance_score,
        "attributes": rel.attributes,
    }
    if rel.extracted_at is not None:
        meta["extracted_at"] = rel.extracted_at
    if rel.source_run_id is not None:
        meta["source_run_id"] = rel.source_run_id
    if rel.effective_ttl_hours is not None:
        meta["effective_ttl_hours"] = rel.effective_ttl_hours
    return f"{rel.description}{META_DELIMITER}{json.dumps(meta)}"


def parse_edge_metadata(description: str) -> dict | None:
    """Extract JSON metadata from a relationship description.

    Returns parsed dict or None if no metadata found.
    """
    if META_DELIMITER not in description:
        return None
    try:
        _, json_str = description.rsplit(META_DELIMITER, 1)
        return json.loads(json_str)
    except (ValueError, json.JSONDecodeError):
        return None


async def insert_validated_data(
    rag: LightRAG,
    validation_result: ValidationResult,
    source_text: str,
    source_id: str,
) -> None:
    """Insert validated entities and relationships into LightRAG via ainsert_custom_kg.

    Args:
        rag: Initialized LightRAG instance.
        validation_result: Output from validate_extraction().
        source_text: Original markdown chunk (stored as a LightRAG chunk).
        source_id: Unique identifier for this content piece (e.g., URL or doc hash).
    """
    if not validation_result.entities and not validation_result.relationships:
        return

    custom_kg: dict[str, Any] = {
        "chunks": [
            {
                "content": source_text,
                "source_id": source_id,
            },
        ],
        "entities": [
            {
                "entity_name": e.canonical_id,
                "entity_type": e.entity_type,
                "description": e.description,
                "source_id": source_id,
            }
            for e in validation_result.entities
        ],
        "relationships": [
            {
                "src_id": r.src_canonical_id,
                "tgt_id": r.tgt_canonical_id,
                "description": _encode_rel_metadata(r),
                "keywords": r.relationship_type,
                "weight": r.significance_score if r.tier == 2 else 1.0,
                "source_id": source_id,
            }
            for r in validation_result.relationships
        ],
    }

    await rag.ainsert_custom_kg(custom_kg)


async def query_graph(
    rag: LightRAG,
    query: str,
    mode: str = "hybrid",
    only_context: bool = False,
) -> str:
    """Query the knowledge graph via LightRAG.

    Args:
        rag: Initialized LightRAG instance.
        query: Natural language query.
        mode: "naive", "local", "global", "hybrid", or "mix".
        only_context: If True, return retrieved context without LLM generation.

    Returns:
        LightRAG response string (or context string if only_context=True).
    """
    result = await rag.aquery(
        query,
        param=QueryParam(mode=mode, only_need_context=only_context),
    )
    return result


async def inject_correlation_edges(
    rag: LightRAG,
    correlations: list[dict],
    run_id: int,
) -> None:
    """Inject CORRELATED_WITH edges from Agent B.

    First removes all existing CORRELATED_WITH edges, then inserts new ones.

    Args:
        rag: Initialized LightRAG instance.
        correlations: List of {"ticker_a": str, "ticker_b": str, "correlation": float}.
        run_id: Current run ID.
    """
    # Remove previous Agent B edges by scanning graph
    graph_storage = rag.chunk_entity_relation_graph
    graph = await graph_storage._get_graph()

    edges_to_remove = []
    for src, tgt, data in graph.edges(data=True):
        desc = data.get("description", "")
        meta = parse_edge_metadata(desc)
        if meta and meta.get("relationship_type") == "CORRELATED_WITH":
            edges_to_remove.append((src, tgt))

    if edges_to_remove:
        await graph_storage.remove_edges(edges_to_remove)

    # Insert new correlation edges
    if not correlations:
        return

    for corr in correlations:
        meta = {
            "relationship_type": "CORRELATED_WITH",
            "tier": 3,
            "significance_score": abs(corr["correlation"]),
            "attributes": {"source": "agent_b", "correlation": corr["correlation"]},
            "source_run_id": run_id,
        }
        desc = (
            f"{corr['ticker_a']} and {corr['ticker_b']} have "
            f"correlation {corr['correlation']:.2f}"
            f"{META_DELIMITER}{json.dumps(meta)}"
        )
        await graph_storage.upsert_edge(
            corr["ticker_a"],
            corr["ticker_b"],
            {
                "description": desc,
                "keywords": "CORRELATED_WITH",
                "weight": abs(corr["correlation"]),
                "source_id": f"agent_b_run_{run_id}",
            },
        )

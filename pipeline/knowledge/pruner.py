"""Ephemeral TTL pruner: scans graph for expired Tier 2 edges and removes them."""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from lightrag import LightRAG

from pipeline.knowledge.graph_ops import parse_edge_metadata

logger = logging.getLogger(__name__)


@dataclass
class PruneResult:
    edges_scanned: int = 0
    edges_expired: int = 0
    edges_removed: int = 0
    nodes_orphaned_removed: int = 0
    standing_edges_skipped: int = 0
    structural_edges_skipped: int = 0


def _is_standing_entity(conn: sqlite3.Connection, canonical_id: str) -> bool:
    """Check if a canonical_id has an active standing event."""
    row = conn.execute(
        "SELECT 1 FROM standing_events WHERE canonical_id = ? AND status = 'active' LIMIT 1",
        (canonical_id,),
    ).fetchone()
    return row is not None


async def prune_expired_edges(
    rag: LightRAG,
    conn: sqlite3.Connection,
    now: datetime | None = None,
) -> PruneResult:
    """Scan graph for expired Tier 2 edges and remove them.

    Algorithm:
    1. Iterate all edges in the NetworkX graph.
    2. Parse metadata from description field.
    3. Skip Tier 1 (structural) edges — never pruned.
    4. Skip CORRELATED_WITH edges — managed by Agent B.
    5. Skip edges connected to active standing events.
    6. For Tier 2 edges: check if now > extracted_at + effective_ttl_hours.
    7. Remove expired edges and orphaned nodes.
    """
    if now is None:
        now = datetime.now(timezone.utc)

    result = PruneResult()
    graph_storage = rag.chunk_entity_relation_graph
    graph = await graph_storage._get_graph()

    edges_to_remove: list[tuple[str, str]] = []

    for src, tgt, data in graph.edges(data=True):
        result.edges_scanned += 1
        desc = data.get("description", "")
        meta = parse_edge_metadata(desc)

        if meta is None:
            # No metadata — skip (could be a manually inserted edge)
            continue

        tier = meta.get("tier", 0)
        rel_type = meta.get("relationship_type", "")

        # Skip Tier 1 structural edges
        if tier == 1:
            result.structural_edges_skipped += 1
            continue

        # Skip Agent B correlation edges
        if rel_type == "CORRELATED_WITH":
            continue

        # Skip standing event edges
        if _is_standing_entity(conn, src) or _is_standing_entity(conn, tgt):
            result.standing_edges_skipped += 1
            continue

        # Check TTL for Tier 2 edges
        if tier == 2:
            extracted_at_str = meta.get("extracted_at")
            ttl_hours = meta.get("effective_ttl_hours")
            if extracted_at_str is None or ttl_hours is None:
                continue

            try:
                extracted_at = datetime.fromisoformat(extracted_at_str)
                expiry = extracted_at + timedelta(hours=ttl_hours)
                if now >= expiry:
                    result.edges_expired += 1
                    edges_to_remove.append((src, tgt))
            except (ValueError, TypeError):
                logger.warning("Could not parse edge timestamp: %s", extracted_at_str)
                continue

    # Remove expired edges
    if edges_to_remove:
        await graph_storage.remove_edges(edges_to_remove)
        result.edges_removed = len(edges_to_remove)

    # Find and remove orphaned nodes (nodes with no remaining edges)
    graph = await graph_storage._get_graph()
    orphaned = [n for n in graph.nodes() if graph.degree(n) == 0]

    # Only remove orphaned nodes that are Tier 2 entity types (events, themes)
    tier2_entity_prefixes = ("event:", "theme:")
    orphans_to_remove = [
        n for n in orphaned
        if any(n.startswith(p) for p in tier2_entity_prefixes)
    ]

    if orphans_to_remove:
        await graph_storage.remove_nodes(orphans_to_remove)
        result.nodes_orphaned_removed = len(orphans_to_remove)

    return result

"""Tests for the ephemeral TTL pruner."""

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import networkx as nx
import pytest

from pipeline.knowledge.graph_ops import META_DELIMITER, parse_edge_metadata
from pipeline.knowledge.pruner import PruneResult, prune_expired_edges


class TestEdgeMetadataParsing:
    def test_parse_valid_metadata(self):
        meta = {"tier": 2, "extracted_at": "2026-04-01T12:00:00+00:00",
                "effective_ttl_hours": 72.0, "relationship_type": "AFFECTED_BY_EVENT"}
        desc = f"Some description{META_DELIMITER}{json.dumps(meta)}"
        parsed = parse_edge_metadata(desc)
        assert parsed is not None
        assert parsed["tier"] == 2
        assert parsed["effective_ttl_hours"] == 72.0

    def test_parse_missing_metadata_returns_none(self):
        assert parse_edge_metadata("Just a plain description") is None

    def test_parse_corrupted_metadata_returns_none(self):
        assert parse_edge_metadata(f"desc{META_DELIMITER}not-json") is None


class TestTTLCalculation:
    def test_base_ttl_zero_significance(self):
        assert 48 * (1 + 0.0) == 48.0

    def test_base_ttl_mid_significance(self):
        assert 48 * (1 + 0.5) == 72.0

    def test_base_ttl_max_significance(self):
        assert 48 * (1 + 1.0) == 96.0


def _make_edge_desc(rel_type, tier, significance, extracted_at, ttl_hours=None):
    """Helper to build an edge description with metadata."""
    if ttl_hours is None:
        ttl_hours = 48 * (1 + significance) if tier == 2 else None
    meta = {
        "relationship_type": rel_type,
        "tier": tier,
        "significance_score": significance,
        "attributes": {},
    }
    if extracted_at is not None:
        meta["extracted_at"] = extracted_at
    if ttl_hours is not None:
        meta["effective_ttl_hours"] = ttl_hours
    return f"Description{META_DELIMITER}{json.dumps(meta)}"


@pytest.fixture
def mock_graph_data():
    """Build edge data for testing. Returns list of (src, tgt, data) tuples."""
    now = datetime.now(timezone.utc)
    old = (now - timedelta(hours=72)).isoformat()     # 72h ago
    recent = (now - timedelta(hours=24)).isoformat()   # 24h ago

    return [
        # Expired Tier 2 edge: 72h old, significance 0.0, TTL=48h -> expired
        ("NVDA", "event:20260401:old_event", {
            "description": _make_edge_desc("AFFECTED_BY_EVENT", 2, 0.0, old),
            "keywords": "AFFECTED_BY_EVENT",
        }),
        # Surviving Tier 2 edge: 72h old, significance 0.9, TTL=91.2h -> alive
        ("AAPL", "event:20260401:big_event", {
            "description": _make_edge_desc("AFFECTED_BY_EVENT", 2, 0.9, old),
            "keywords": "AFFECTED_BY_EVENT",
        }),
        # Recent Tier 2 edge: 24h old, significance 0.0, TTL=48h -> alive
        ("MSFT", "event:20260402:recent", {
            "description": _make_edge_desc("AFFECTED_BY_EVENT", 2, 0.0, recent),
            "keywords": "AFFECTED_BY_EVENT",
        }),
        # Tier 1 structural edge: never pruned regardless of age
        ("NVDA", "sector:semiconductors", {
            "description": _make_edge_desc("BELONGS_TO_SECTOR", 1, 1.0, old),
            "keywords": "BELONGS_TO_SECTOR",
        }),
        # Agent B CORRELATED_WITH edge: skipped by pruner
        ("NVDA", "AMD", {
            "description": _make_edge_desc("CORRELATED_WITH", 3, 0.8, old),
            "keywords": "CORRELATED_WITH",
        }),
    ]


async def _run_pruner(edge_data, conn):
    """Helper: create a mock graph, populate edges, run pruner, return result."""
    graph = nx.Graph()
    for src, tgt, data in edge_data:
        graph.add_edge(src, tgt, **data)

    mock_storage = AsyncMock()
    mock_storage._get_graph = AsyncMock(return_value=graph)
    mock_storage.remove_edges = AsyncMock()
    mock_storage.remove_nodes = AsyncMock()

    mock_rag = MagicMock()
    mock_rag.chunk_entity_relation_graph = mock_storage

    result = await prune_expired_edges(mock_rag, conn)
    return result


class TestPruner:
    @pytest.mark.asyncio
    async def test_expired_tier2_edge_removed(self, mock_graph_data, db_conn):
        result = await _run_pruner(mock_graph_data, db_conn)
        assert result.edges_expired == 1
        assert result.edges_removed == 1

    @pytest.mark.asyncio
    async def test_tier1_never_pruned(self, mock_graph_data, db_conn):
        result = await _run_pruner(mock_graph_data, db_conn)
        assert result.structural_edges_skipped >= 1

    @pytest.mark.asyncio
    async def test_prune_result_counts(self, mock_graph_data, db_conn):
        result = await _run_pruner(mock_graph_data, db_conn)
        assert result.edges_scanned == 5
        assert result.edges_expired == 1
        assert result.structural_edges_skipped >= 1

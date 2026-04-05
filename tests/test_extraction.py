"""Tests for Phase 4: extraction, canonical resolution, and validation."""

import json
import pytest

from tests.conftest import MOCK_WATCHLIST
from pipeline.knowledge.validator import (
    ValidatedEntity,
    ValidatedRelationship,
    ValidationResult,
    validate_extraction,
    VALID_ENTITY_TYPES,
    VALID_RELATIONSHIP_TYPES,
)


@pytest.fixture
def seeded_db(db_conn):
    """db_conn with MOCK_WATCHLIST in watchlist and canonical_entities."""
    from pipeline.db.init import _populate_watchlist, _seed_canonical_entities
    from unittest.mock import patch

    with patch("pipeline.db.init.scrape_universe", return_value=MOCK_WATCHLIST):
        entries = _populate_watchlist(db_conn, "djia30")
    _seed_canonical_entities(db_conn, entries)
    db_conn.commit()
    return db_conn


class TestCanonicalResolution:
    def test_resolve_known_company_by_alias(self, seeded_db):
        from pipeline.knowledge.canonical import resolve_entity

        result = resolve_entity(
            seeded_db,
            raw_name="NVIDIA Corporation",
            proposed_canonical_id="NVDA",
            proposed_type="COMPANY",
            description="GPU manufacturer",
        )
        assert result == "NVDA"

    def test_resolve_known_company_by_ticker(self, seeded_db):
        from pipeline.knowledge.canonical import resolve_entity

        result = resolve_entity(
            seeded_db,
            raw_name="NVDA",
            proposed_canonical_id="NVDA",
            proposed_type="COMPANY",
            description="Nvidia",
        )
        assert result == "NVDA"

    def test_resolve_case_insensitive(self, seeded_db):
        from pipeline.knowledge.canonical import resolve_entity

        result = resolve_entity(
            seeded_db,
            raw_name="nvidia corporation",
            proposed_canonical_id="NVDA",
            proposed_type="COMPANY",
            description="GPU maker",
        )
        assert result == "NVDA"

    def test_resolve_unknown_creates_new_entry(self, seeded_db):
        from pipeline.knowledge.canonical import resolve_entity

        result = resolve_entity(
            seeded_db,
            raw_name="Jensen Huang",
            proposed_canonical_id="person:jensen_huang",
            proposed_type="PERSON",
            description="CEO of Nvidia",
        )
        assert result == "person:jensen_huang"

        # Verify it was inserted
        row = seeded_db.execute(
            "SELECT * FROM canonical_entities WHERE canonical_id = ?",
            ("person:jensen_huang",),
        ).fetchone()
        assert row is not None
        assert row["entity_type"] == "PERSON"
        assert "Jensen Huang" in json.loads(row["aliases"])

    def test_resolve_adds_alias_to_existing(self, seeded_db):
        from pipeline.knowledge.canonical import resolve_entity

        resolve_entity(
            seeded_db,
            raw_name="Nvidia Corp",
            proposed_canonical_id="NVDA",
            proposed_type="COMPANY",
            description="GPU maker",
        )
        row = seeded_db.execute(
            "SELECT aliases FROM canonical_entities WHERE canonical_id = 'NVDA'"
        ).fetchone()
        aliases = json.loads(row["aliases"])
        assert "Nvidia Corp" in aliases

    def test_resolve_invalid_id_format_returns_none(self, seeded_db):
        from pipeline.knowledge.canonical import resolve_entity

        result = resolve_entity(
            seeded_db,
            raw_name="Some Person",
            proposed_canonical_id="bad_format",  # PERSON needs person: prefix
            proposed_type="PERSON",
            description="Unknown",
        )
        assert result is None

    def test_validate_all_id_formats(self):
        from pipeline.knowledge.canonical import validate_canonical_id_format

        # Valid formats
        assert validate_canonical_id_format("NVDA", "COMPANY") is True
        assert validate_canonical_id_format("person:jensen_huang", "PERSON") is True
        assert validate_canonical_id_format("sector:semiconductors", "SECTOR") is True
        assert validate_canonical_id_format("index:SPX", "INDEX") is True
        assert validate_canonical_id_format("product:NVDA:H100", "PRODUCT") is True
        assert validate_canonical_id_format("event:20260401:nvda_q1", "EVENT") is True
        assert validate_canonical_id_format("theme:ai_capex_cycle", "MACRO_THEME") is True
        assert validate_canonical_id_format("inst:federal_reserve", "INSTITUTION") is True

        # Invalid formats
        assert validate_canonical_id_format("person:jensen_huang", "COMPANY") is False
        assert validate_canonical_id_format("NVDA", "PERSON") is False
        assert validate_canonical_id_format("no_prefix", "SECTOR") is False
        assert validate_canonical_id_format("event:baddate:slug", "EVENT") is False


@pytest.fixture
def sample_extraction():
    """A realistic raw extraction result dict."""
    return {
        "entities": [
            {
                "entity_name": "NVIDIA",
                "entity_type": "COMPANY",
                "proposed_canonical_id": "NVDA",
                "description": "Leading GPU manufacturer",
            },
            {
                "entity_name": "Jensen Huang",
                "entity_type": "PERSON",
                "proposed_canonical_id": "person:jensen_huang",
                "description": "CEO of Nvidia",
            },
            {
                "entity_name": "Q1 2026 Earnings",
                "entity_type": "EVENT",
                "proposed_canonical_id": "event:20260401:nvda_q1_earnings",
                "description": "Nvidia Q1 FY2026 earnings report",
            },
        ],
        "relationships": [
            {
                "src_entity": "NVIDIA",
                "tgt_entity": "Q1 2026 Earnings",
                "relationship_type": "AFFECTED_BY_EVENT",
                "description": "Nvidia affected by earnings report",
                "significance_score": 0.8,
                "attributes": {},
            },
            {
                "src_entity": "Q1 2026 Earnings",
                "tgt_entity": "Jensen Huang",
                "relationship_type": "ANNOUNCED_BY",
                "description": "Earnings announced by Jensen Huang",
                "significance_score": 0.6,
                "attributes": {},
            },
        ],
    }


@pytest.fixture
def extraction_with_untyped():
    """Extraction with UNTYPED and invalid types."""
    return {
        "entities": [
            {
                "entity_name": "NVIDIA",
                "entity_type": "COMPANY",
                "proposed_canonical_id": "NVDA",
                "description": "GPU maker",
            },
            {
                "entity_name": "Some Widget",
                "entity_type": "GADGET",  # invalid type
                "proposed_canonical_id": "gadget:widget",
                "description": "Unknown thing",
            },
        ],
        "relationships": [
            {
                "src_entity": "NVIDIA",
                "tgt_entity": "Some Widget",
                "relationship_type": "UNTYPED",
                "description": "Some vague connection",
                "significance_score": 0.3,
                "attributes": {},
            },
            {
                "src_entity": "NVIDIA",
                "tgt_entity": "Some Widget",
                "relationship_type": "INVENTED_BY",  # invalid type
                "description": "Not a valid relationship",
                "significance_score": 0.2,
                "attributes": {},
            },
        ],
    }


class TestValidator:
    def test_valid_extraction_produces_entities_and_rels(self, seeded_db, sample_extraction):
        result = validate_extraction(sample_extraction, seeded_db, run_id=1)
        assert len(result.entities) == 3
        assert len(result.relationships) == 2
        assert len(result.untyped_edges) == 0
        assert len(result.rejected_entities) == 0

    def test_invalid_entity_type_rejected(self, seeded_db, extraction_with_untyped):
        result = validate_extraction(extraction_with_untyped, seeded_db, run_id=1)
        assert len(result.rejected_entities) == 1
        assert result.rejected_entities[0]["entity_type"] == "GADGET"

    def test_untyped_edges_logged_separately(self, seeded_db, extraction_with_untyped):
        result = validate_extraction(extraction_with_untyped, seeded_db, run_id=1)
        assert len(result.untyped_edges) == 1
        assert result.untyped_edges[0]["relationship_type"] == "UNTYPED"

    def test_invalid_relationship_type_rejected(self, seeded_db, extraction_with_untyped):
        result = validate_extraction(extraction_with_untyped, seeded_db, run_id=1)
        assert len(result.rejected_relationships) == 1
        assert result.rejected_relationships[0]["relationship_type"] == "INVENTED_BY"

    def test_tier2_gets_temporal_metadata(self, seeded_db, sample_extraction):
        result = validate_extraction(sample_extraction, seeded_db, run_id=42)
        # AFFECTED_BY_EVENT is Tier 2
        tier2_rels = [r for r in result.relationships if r.tier == 2]
        assert len(tier2_rels) > 0
        for rel in tier2_rels:
            assert rel.extracted_at is not None
            assert rel.source_run_id == 42
            assert rel.effective_ttl_hours is not None

    def test_tier1_no_ttl(self, seeded_db):
        """Tier 1 relationships should not have TTL metadata."""
        raw = {
            "entities": [
                {"entity_name": "NVIDIA", "entity_type": "COMPANY",
                 "proposed_canonical_id": "NVDA", "description": "GPU maker"},
                {"entity_name": "Semiconductors", "entity_type": "SECTOR",
                 "proposed_canonical_id": "sector:semiconductors", "description": "Chip sector"},
            ],
            "relationships": [
                {"src_entity": "NVIDIA", "tgt_entity": "Semiconductors",
                 "relationship_type": "BELONGS_TO_SECTOR", "description": "NVDA is in semis",
                 "significance_score": 1.0, "attributes": {}},
            ],
        }
        result = validate_extraction(raw, seeded_db, run_id=1)
        assert len(result.relationships) == 1
        rel = result.relationships[0]
        assert rel.tier == 1
        assert rel.effective_ttl_hours is None

    def test_effective_ttl_calculation(self, seeded_db, sample_extraction):
        result = validate_extraction(sample_extraction, seeded_db, run_id=1, base_ttl_hours=48)
        tier2_rels = [r for r in result.relationships if r.tier == 2]
        for rel in tier2_rels:
            expected = 48 * (1 + rel.significance_score)
            assert rel.effective_ttl_hours == pytest.approx(expected)

    def test_significance_score_clamped(self, seeded_db):
        """Significance scores outside 0-1 should be clamped."""
        raw = {
            "entities": [
                {"entity_name": "NVIDIA", "entity_type": "COMPANY",
                 "proposed_canonical_id": "NVDA", "description": "GPU maker"},
                {"entity_name": "Big Event", "entity_type": "EVENT",
                 "proposed_canonical_id": "event:20260401:big_event", "description": "Big"},
            ],
            "relationships": [
                {"src_entity": "NVIDIA", "tgt_entity": "Big Event",
                 "relationship_type": "AFFECTED_BY_EVENT", "description": "Affected",
                 "significance_score": 1.5, "attributes": {}},
            ],
        }
        result = validate_extraction(raw, seeded_db, run_id=1)
        assert result.relationships[0].significance_score == 1.0

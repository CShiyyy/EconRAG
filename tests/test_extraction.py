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


class TestEndToEnd:
    """End-to-end: mock Ollama -> validate -> verify entities/relationships."""

    @pytest.fixture
    def nvda_article_markdown(self):
        return """\
# NVIDIA Reports Record Q1 2026 Revenue

NVIDIA Corporation (NVDA) reported record first-quarter revenue of $44 billion,
driven by surging demand for its H100 and H200 data center GPUs. CEO Jensen Huang
said the AI infrastructure buildout is "just getting started."

The semiconductor sector continues to benefit from the AI capex cycle, with NVDA
leading the charge. Analysts note that export restrictions to China remain a key
risk factor for the company.

The Federal Reserve held rates steady at its March meeting, providing a stable
macro backdrop. AMD, NVDA's primary competitor, also reported strong results.
"""

    @pytest.fixture
    def mock_ollama_extraction(self):
        """What the LLM would return for the NVDA article."""
        return {
            "entities": [
                {"entity_name": "NVIDIA", "entity_type": "COMPANY",
                 "proposed_canonical_id": "NVDA",
                 "description": "Leading GPU manufacturer"},
                {"entity_name": "Jensen Huang", "entity_type": "PERSON",
                 "proposed_canonical_id": "person:jensen_huang",
                 "description": "CEO of NVIDIA"},
                {"entity_name": "Q1 2026 Earnings", "entity_type": "EVENT",
                 "proposed_canonical_id": "event:20260401:nvda_q1_earnings",
                 "description": "NVIDIA Q1 FY2026 earnings report"},
                {"entity_name": "AI capex cycle", "entity_type": "MACRO_THEME",
                 "proposed_canonical_id": "theme:ai_capex_cycle",
                 "description": "Ongoing AI infrastructure spending wave"},
                {"entity_name": "China export restrictions", "entity_type": "MACRO_THEME",
                 "proposed_canonical_id": "theme:china_export_restrictions",
                 "description": "US restrictions on chip exports to China"},
                {"entity_name": "Federal Reserve", "entity_type": "INSTITUTION",
                 "proposed_canonical_id": "inst:federal_reserve",
                 "description": "US central bank"},
                {"entity_name": "AMD", "entity_type": "COMPANY",
                 "proposed_canonical_id": "AMD",
                 "description": "Semiconductor company, NVDA competitor"},
            ],
            "relationships": [
                {"src_entity": "NVIDIA", "tgt_entity": "Q1 2026 Earnings",
                 "relationship_type": "AFFECTED_BY_EVENT",
                 "description": "NVDA affected by earnings release",
                 "significance_score": 0.9, "attributes": {}},
                {"src_entity": "Q1 2026 Earnings", "tgt_entity": "Jensen Huang",
                 "relationship_type": "ANNOUNCED_BY",
                 "description": "Earnings announced by Jensen Huang",
                 "significance_score": 0.6, "attributes": {}},
                {"src_entity": "NVIDIA", "tgt_entity": "AI capex cycle",
                 "relationship_type": "DRIVEN_BY_THEME",
                 "description": "NVDA revenue driven by AI infrastructure spending",
                 "significance_score": 0.85, "attributes": {}},
                {"src_entity": "NVIDIA", "tgt_entity": "China export restrictions",
                 "relationship_type": "EXPOSED_TO",
                 "description": "NVDA faces risk from export controls",
                 "significance_score": 0.7, "attributes": {}},
                {"src_entity": "NVIDIA", "tgt_entity": "AMD",
                 "relationship_type": "COMPETES_WITH",
                 "description": "Direct competitors in GPU market",
                 "significance_score": 0.8, "attributes": {}},
            ],
        }

    def test_nvda_article_validates_correctly(
        self, seeded_db, mock_ollama_extraction
    ):
        """Full validation pipeline with mock extraction output."""
        result = validate_extraction(mock_ollama_extraction, seeded_db, run_id=1)

        # All 7 entities should resolve
        assert len(result.entities) == 7
        assert len(result.rejected_entities) == 0
        assert len(result.untyped_edges) == 0

        # Check entity types
        entity_types = {e.entity_type for e in result.entities}
        assert "COMPANY" in entity_types
        assert "PERSON" in entity_types
        assert "EVENT" in entity_types
        assert "MACRO_THEME" in entity_types

        # Check relationships
        assert len(result.relationships) == 5
        rel_types = {r.relationship_type for r in result.relationships}
        assert "AFFECTED_BY_EVENT" in rel_types
        assert "DRIVEN_BY_THEME" in rel_types
        assert "COMPETES_WITH" in rel_types

        # Tier 2 edges should have temporal metadata
        tier2 = [r for r in result.relationships if r.tier == 2]
        for r in tier2:
            assert r.extracted_at is not None
            assert r.source_run_id == 1

    def test_canonicalization_deduplicates(self, seeded_db):
        """Two extractions mentioning the same entity should resolve to one canonical ID."""
        extract1 = {
            "entities": [
                {"entity_name": "Nvidia", "entity_type": "COMPANY",
                 "proposed_canonical_id": "NVDA", "description": "GPU maker"},
            ],
            "relationships": [],
        }
        extract2 = {
            "entities": [
                {"entity_name": "NVIDIA Corporation", "entity_type": "COMPANY",
                 "proposed_canonical_id": "NVDA", "description": "GPU company"},
            ],
            "relationships": [],
        }
        r1 = validate_extraction(extract1, seeded_db, run_id=1)
        r2 = validate_extraction(extract2, seeded_db, run_id=2)

        assert r1.entities[0].canonical_id == "NVDA"
        assert r2.entities[0].canonical_id == "NVDA"

        # Only one canonical_entities row for NVDA
        count = seeded_db.execute(
            "SELECT COUNT(*) FROM canonical_entities WHERE canonical_id = 'NVDA'"
        ).fetchone()[0]
        assert count == 1


class TestExtractionPrompt:
    def test_prompt_contains_all_entity_types(self):
        from pipeline.knowledge.extraction_prompt import EXTRACTION_SYSTEM_PROMPT
        for etype in ["COMPANY", "PERSON", "SECTOR", "INDEX",
                      "PRODUCT", "EVENT", "MACRO_THEME", "INSTITUTION"]:
            assert etype in EXTRACTION_SYSTEM_PROMPT, f"Missing entity type: {etype}"

    def test_prompt_contains_all_relationship_types(self):
        from pipeline.knowledge.extraction_prompt import EXTRACTION_SYSTEM_PROMPT
        for rtype in [
            "BELONGS_TO_SECTOR", "CONSTITUENT_OF", "LED_BY", "COMPETES_WITH",
            "SUPPLIES_TO", "PRODUCES", "SUBSIDIARY_OF",
            "AFFECTED_BY_EVENT", "DRIVEN_BY_THEME", "SENTIMENT_TOWARD",
            "ANNOUNCED_BY", "POLICY_AFFECTS", "EXPOSED_TO",
        ]:
            assert rtype in EXTRACTION_SYSTEM_PROMPT, f"Missing relationship type: {rtype}"

    def test_prompt_contains_untyped_fallback(self):
        from pipeline.knowledge.extraction_prompt import EXTRACTION_SYSTEM_PROMPT
        assert "UNTYPED" in EXTRACTION_SYSTEM_PROMPT

    def test_user_template_has_placeholder(self):
        from pipeline.knowledge.extraction_prompt import EXTRACTION_USER_TEMPLATE
        assert "{markdown_chunk}" in EXTRACTION_USER_TEMPLATE

    def test_corrective_prompt_has_placeholder(self):
        from pipeline.knowledge.extraction_prompt import CORRECTIVE_PROMPT
        assert "{markdown_chunk}" in CORRECTIVE_PROMPT

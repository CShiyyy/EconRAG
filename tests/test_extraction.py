"""Tests for Phase 4: extraction, canonical resolution, and validation."""

import json
import pytest

from tests.conftest import MOCK_WATCHLIST


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

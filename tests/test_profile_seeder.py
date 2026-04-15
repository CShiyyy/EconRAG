"""Tests for pipeline/knowledge/profile_seeder.py."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.conftest import MOCK_WATCHLIST


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _seed_run_log(conn, run_type: str = "pre_open") -> int:
    cur = conn.execute(
        "INSERT INTO run_log (timestamp, run_type) VALUES ('2026-04-13T08:30:00', ?)",
        (run_type,),
    )
    conn.commit()
    return cur.lastrowid


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def seeded_db(db_conn):
    """db_conn with watchlist and canonical_entities populated from MOCK_WATCHLIST."""
    from pipeline.db.init import _populate_watchlist, _seed_canonical_entities
    from unittest.mock import patch as _patch

    with _patch("pipeline.db.init.scrape_universe", return_value=MOCK_WATCHLIST):
        entries = _populate_watchlist(db_conn, "djia30")
    _seed_canonical_entities(db_conn, entries)
    db_conn.commit()
    return db_conn


@pytest.fixture
def mock_rag():
    rag = MagicMock()
    rag.ainsert_custom_kg = AsyncMock(return_value=None)
    return rag


@pytest.fixture
def mock_cloud_client():
    client = MagicMock()
    client.generate = AsyncMock(
        return_value=(
            "AAPL belongs to the Information Technology sector. "
            "AAPL is led by CEO Tim Cook. "
            "AAPL competes with MSFT and GOOGL. "
            "AAPL produces the iPhone and MacBook. "
            "**Current state.** AAPL is driven by the services growth theme."
        )
    )
    return client


CANNED_EXTRACTION = {
    "entities": [
        {
            "entity_name": "AAPL",
            "entity_type": "COMPANY",
            "proposed_canonical_id": "AAPL",
            "description": "Apple Inc.",
        },
        {
            "entity_name": "Tim Cook",
            "entity_type": "PERSON",
            "proposed_canonical_id": "person:tim_cook",
            "description": "CEO of Apple",
        },
        {
            "entity_name": "Information Technology",
            "entity_type": "SECTOR",
            "proposed_canonical_id": "sector:information_technology",
            "description": "IT sector",
        },
    ],
    "relationships": [
        {
            "src_entity": "AAPL",
            "tgt_entity": "Information Technology",
            "relationship_type": "BELONGS_TO_SECTOR",
            "description": "AAPL is in IT sector",
            "significance_score": 1.0,
            "attributes": {},
        },
        {
            "src_entity": "AAPL",
            "tgt_entity": "Tim Cook",
            "relationship_type": "LED_BY",
            "description": "Apple led by Tim Cook",
            "significance_score": 1.0,
            "attributes": {},
        },
    ],
}


# ---------------------------------------------------------------------------
# Prompt builder tests
# ---------------------------------------------------------------------------

class TestPromptBuilders:
    def test_ticker_messages_structure(self):
        from pipeline.knowledge.profile_seeder import _build_ticker_messages

        msgs = _build_ticker_messages("AAPL", "Apple Inc.", "Information Technology")
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        assert "AAPL" in msgs[1]["content"]
        assert "Apple Inc." in msgs[1]["content"]
        assert "Information Technology" in msgs[1]["content"]

    def test_ticker_system_contains_phrasing_hints(self):
        from pipeline.knowledge.profile_seeder import _build_ticker_messages

        msgs = _build_ticker_messages("NVDA", "NVIDIA Corporation", "Semiconductors")
        system = msgs[0]["content"]
        assert "belongs to the" in system
        assert "is led by CEO" in system
        assert "competes with" in system
        assert "produces" in system

    def test_macro_messages_structure(self):
        from pipeline.knowledge.profile_seeder import _build_macro_messages

        msgs = _build_macro_messages()
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        assert "macro" in msgs[1]["content"].lower()

    def test_macro_system_contains_institution_hints(self):
        from pipeline.knowledge.profile_seeder import _build_macro_messages

        msgs = _build_macro_messages()
        system = msgs[0]["content"]
        assert "Federal Reserve" in system
        assert "monetary policy" in system.lower()


# ---------------------------------------------------------------------------
# _generate_profile tests
# ---------------------------------------------------------------------------

class TestGenerateProfile:
    @pytest.mark.asyncio
    async def test_returns_llm_output(self):
        from pipeline.knowledge.profile_seeder import _generate_profile

        client = MagicMock()
        client.generate = AsyncMock(return_value="some markdown text")
        result = await _generate_profile(client, [{"role": "user", "content": "test"}])
        assert result == "some markdown text"
        client.generate.assert_called_once_with(
            [{"role": "user", "content": "test"}], json_mode=False
        )

    @pytest.mark.asyncio
    async def test_returns_none_on_exception(self):
        from pipeline.knowledge.profile_seeder import _generate_profile

        client = MagicMock()
        client.generate = AsyncMock(side_effect=RuntimeError("API down"))
        result = await _generate_profile(client, [{"role": "user", "content": "test"}])
        assert result is None


# ---------------------------------------------------------------------------
# _seed_single tests
# ---------------------------------------------------------------------------

class TestSeedSingle:
    @pytest.mark.asyncio
    async def test_success_writes_seed_log_row(self, seeded_db, mock_rag):
        from pipeline.knowledge.profile_seeder import _seed_single, _build_ticker_messages

        run_id = _seed_run_log(seeded_db)
        messages = _build_ticker_messages("AAPL", "Apple Inc.", "Information Technology")

        with patch(
            "pipeline.knowledge.profile_seeder._generate_profile",
            AsyncMock(return_value="AAPL belongs to the Information Technology sector."),
        ), patch(
            "pipeline.knowledge.profile_seeder.extract_from_markdown",
            AsyncMock(return_value=CANNED_EXTRACTION),
        ):
            ok = await _seed_single(
                conn=seeded_db,
                rag=mock_rag,
                client=MagicMock(),
                seed_type="ticker_profile",
                seed_key="AAPL",
                messages=messages,
                source_id="profile:AAPL",
                run_id=run_id,
            )

        assert ok is True
        row = seeded_db.execute(
            "SELECT * FROM kg_seed_log WHERE seed_type='ticker_profile' AND seed_key='AAPL'"
        ).fetchone()
        assert row is not None
        assert row["source_id"] == "profile:AAPL"
        assert row["run_id"] == run_id
        mock_rag.ainsert_custom_kg.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_log_row_when_generation_fails(self, seeded_db, mock_rag):
        from pipeline.knowledge.profile_seeder import _seed_single

        run_id = _seed_run_log(seeded_db)

        with patch(
            "pipeline.knowledge.profile_seeder._generate_profile",
            AsyncMock(return_value=None),
        ):
            ok = await _seed_single(
                conn=seeded_db,
                rag=mock_rag,
                client=MagicMock(),
                seed_type="ticker_profile",
                seed_key="AAPL",
                messages=[],
                source_id="profile:AAPL",
                run_id=run_id,
            )

        assert ok is False
        assert seeded_db.execute(
            "SELECT * FROM kg_seed_log WHERE seed_key='AAPL'"
        ).fetchone() is None
        mock_rag.ainsert_custom_kg.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_log_row_when_extraction_fails(self, seeded_db, mock_rag):
        from pipeline.knowledge.profile_seeder import _seed_single

        run_id = _seed_run_log(seeded_db)

        with patch(
            "pipeline.knowledge.profile_seeder._generate_profile",
            AsyncMock(return_value="some profile text"),
        ), patch(
            "pipeline.knowledge.profile_seeder.extract_from_markdown",
            AsyncMock(return_value=None),
        ):
            ok = await _seed_single(
                conn=seeded_db,
                rag=mock_rag,
                client=MagicMock(),
                seed_type="ticker_profile",
                seed_key="AAPL",
                messages=[],
                source_id="profile:AAPL",
                run_id=run_id,
            )

        assert ok is False
        assert seeded_db.execute(
            "SELECT * FROM kg_seed_log WHERE seed_key='AAPL'"
        ).fetchone() is None

    @pytest.mark.asyncio
    async def test_tier2_edges_use_extended_ttl(self, seeded_db, mock_rag):
        """validate_extraction must be called with PROFILE_SEED_TTL_HOURS, not the default 48h."""
        from pipeline.knowledge.profile_seeder import _seed_single
        from pipeline.config import PROFILE_SEED_TTL_HOURS
        from pipeline.knowledge import validator as validator_module

        run_id = _seed_run_log(seeded_db)

        theme_extraction = {
            "entities": [
                {
                    "entity_name": "AAPL",
                    "entity_type": "COMPANY",
                    "proposed_canonical_id": "AAPL",
                    "description": "Apple",
                },
                {
                    "entity_name": "AI capex cycle",
                    "entity_type": "MACRO_THEME",
                    "proposed_canonical_id": "theme:ai_capex_cycle",
                    "description": "AI capital expenditure theme",
                },
            ],
            "relationships": [
                {
                    "src_entity": "AAPL",
                    "tgt_entity": "AI capex cycle",
                    "relationship_type": "DRIVEN_BY_THEME",
                    "description": "Driven by AI capex",
                    "significance_score": 0.8,
                    "attributes": {},
                },
            ],
        }

        captured_ttl = []

        original_validate = validator_module.validate_extraction

        def capturing_validate(raw, conn, run_id, base_ttl_hours=48):
            captured_ttl.append(base_ttl_hours)
            return original_validate(raw, conn, run_id, base_ttl_hours)

        with patch(
            "pipeline.knowledge.profile_seeder._generate_profile",
            AsyncMock(return_value="AAPL is driven by the AI capital expenditure cycle."),
        ), patch(
            "pipeline.knowledge.profile_seeder.extract_from_markdown",
            AsyncMock(return_value=theme_extraction),
        ), patch(
            "pipeline.knowledge.profile_seeder.validate_extraction",
            side_effect=capturing_validate,
        ):
            await _seed_single(
                conn=seeded_db,
                rag=mock_rag,
                client=MagicMock(),
                seed_type="ticker_profile",
                seed_key="AAPL",
                messages=[],
                source_id="profile:AAPL",
                run_id=run_id,
            )

        assert captured_ttl, "validate_extraction was not called"
        assert captured_ttl[0] == PROFILE_SEED_TTL_HOURS, (
            f"Expected TTL={PROFILE_SEED_TTL_HOURS}, got {captured_ttl[0]}"
        )


# ---------------------------------------------------------------------------
# seed_missing_profiles tests
# ---------------------------------------------------------------------------

class TestSeedMissingProfiles:
    @pytest.mark.asyncio
    async def test_seeds_all_tickers_and_macro_on_first_run(self, seeded_db, mock_rag, mock_cloud_client):
        from pipeline.knowledge.profile_seeder import seed_missing_profiles

        run_id = _seed_run_log(seeded_db)

        with patch(
            "pipeline.knowledge.profile_seeder.extract_from_markdown",
            AsyncMock(return_value=CANNED_EXTRACTION),
        ):
            result = await seed_missing_profiles(seeded_db, mock_rag, mock_cloud_client, run_id)

        assert len(result.tickers_seeded) == len(MOCK_WATCHLIST)
        assert result.macro_seeded is True
        assert result.tickers_failed == []
        assert result.macro_error is None

        count = seeded_db.execute("SELECT COUNT(*) FROM kg_seed_log").fetchone()[0]
        assert count == len(MOCK_WATCHLIST) + 1  # tickers + macro

    @pytest.mark.asyncio
    async def test_noop_on_second_call(self, seeded_db, mock_rag, mock_cloud_client):
        """Second call must skip all already-seeded profiles without calling the cloud LLM."""
        from pipeline.knowledge.profile_seeder import seed_missing_profiles

        run_id = _seed_run_log(seeded_db)

        with patch(
            "pipeline.knowledge.profile_seeder.extract_from_markdown",
            AsyncMock(return_value=CANNED_EXTRACTION),
        ):
            await seed_missing_profiles(seeded_db, mock_rag, mock_cloud_client, run_id)
            calls_after_first = mock_cloud_client.generate.call_count

            result2 = await seed_missing_profiles(seeded_db, mock_rag, mock_cloud_client, run_id)

        assert result2.tickers_seeded == []
        assert result2.macro_seeded is False
        # No new cloud calls on the second pass
        assert mock_cloud_client.generate.call_count == calls_after_first

    @pytest.mark.asyncio
    async def test_seeds_only_new_tickers_after_watchlist_growth(self, seeded_db, mock_rag, mock_cloud_client):
        """Adding a new ticker to watchlist causes only that ticker to be seeded next run."""
        from pipeline.knowledge.profile_seeder import seed_missing_profiles

        run_id = _seed_run_log(seeded_db)

        with patch(
            "pipeline.knowledge.profile_seeder.extract_from_markdown",
            AsyncMock(return_value=CANNED_EXTRACTION),
        ):
            await seed_missing_profiles(seeded_db, mock_rag, mock_cloud_client, run_id)
            calls_after_first = mock_cloud_client.generate.call_count

            # Simulate watchlist refresh adding a new ticker
            seeded_db.execute(
                "INSERT INTO watchlist (ticker, company_name, sector, added_at) "
                "VALUES ('GS', 'Goldman Sachs Group Inc.', 'Financials', '2026-04-13T00:00:00')"
            )
            seeded_db.commit()

            result = await seed_missing_profiles(seeded_db, mock_rag, mock_cloud_client, run_id)

        assert result.tickers_seeded == ["GS"]
        assert result.macro_seeded is False  # macro already seeded
        assert mock_cloud_client.generate.call_count == calls_after_first + 1

    @pytest.mark.asyncio
    async def test_ticker_failure_does_not_abort_batch(self, seeded_db, mock_rag):
        """A cloud call failure on one ticker must not prevent others from seeding."""
        from pipeline.knowledge.profile_seeder import seed_missing_profiles

        run_id = _seed_run_log(seeded_db)
        call_count = [0]

        async def flaky_generate(messages, json_mode=True):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("rate limit hit")
            return "Some company belongs to the Information Technology sector."

        client = MagicMock()
        client.generate = flaky_generate

        with patch(
            "pipeline.knowledge.profile_seeder.extract_from_markdown",
            AsyncMock(return_value=CANNED_EXTRACTION),
        ):
            result = await seed_missing_profiles(seeded_db, mock_rag, client, run_id)

        # At most one ticker failed; the rest seeded successfully
        assert len(result.tickers_failed) <= 1
        assert len(result.tickers_seeded) >= len(MOCK_WATCHLIST) - 1

    @pytest.mark.asyncio
    async def test_disabled_by_config_returns_empty_result(self, seeded_db, mock_rag, mock_cloud_client):
        from pipeline.knowledge.profile_seeder import seed_missing_profiles

        run_id = _seed_run_log(seeded_db)

        with patch("pipeline.knowledge.profile_seeder.PROFILE_SEED_ENABLED", False):
            result = await seed_missing_profiles(seeded_db, mock_rag, mock_cloud_client, run_id)

        assert result.tickers_seeded == []
        assert result.macro_seeded is False
        mock_cloud_client.generate.assert_not_called()

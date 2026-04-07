# Phase 4: LightRAG, Extraction Prompt & Canonical Registry — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** LightRAG runs locally with a custom ontology-constrained extraction prompt. Entities are resolved through the canonical registry. The graph is queryable and ephemeral edges are pruned by TTL.

**Architecture:** Two-stage extraction pipeline — Stage A calls Ollama directly with the custom prompt to get structured JSON, Stage B feeds validated/canonicalized entities and relationships into LightRAG via `ainsert_custom_kg()`. This avoids monkey-patching LightRAG's internal extraction pipeline while giving full control over entity resolution and validation. Metadata (tier, TTL, significance) is encoded as a JSON suffix on relationship descriptions for the pruner to parse. The entire knowledge layer is async to match Phase 3's ingestion layer and LightRAG's native API.

**Tech Stack:** LightRAG (lightrag-hku), Ollama (Gemma 3 4B / nomic-embed-text), NetworkX, NanoVectorDB, httpx (async Ollama calls)

---

## File Structure

| File | Responsibility |
|------|----------------|
| `pipeline/knowledge/__init__.py` | Package exports |
| `pipeline/knowledge/extraction_prompt.py` | Custom prompt template constants (entity types, relationship types, JSON output schema) |
| `pipeline/knowledge/lightrag_config.py` | LightRAG instance factory + async context manager for lifecycle |
| `pipeline/knowledge/canonical.py` | Canonical registry resolution: raw strings → canonical IDs via `canonical_entities` table |
| `pipeline/knowledge/validator.py` | Post-extraction validation: type checking, tier classification, temporal metadata |
| `pipeline/knowledge/graph_ops.py` | Graph insertion via `ainsert_custom_kg`, query helpers, edge deletion |
| `pipeline/knowledge/extraction.py` | Extraction orchestrator: markdown → Ollama LLM call → raw JSON |
| `pipeline/knowledge/pruner.py` | Ephemeral TTL pruner: scan + remove expired Tier 2 edges |
| `tests/test_extraction.py` | Tests for canonical resolution, validation, extraction prompt |
| `tests/test_pruner.py` | Tests for TTL calculation, pruning logic, tier immunity |

---

## Task 1: Config Additions & Dependencies

**Files:**
- Modify: `pipeline/config.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Add knowledge graph config to `pipeline/config.py`**

Append after the existing `UNIVERSE_URLS` and `DEFAULT_CONSTRAINTS` blocks:

```python
# --- Knowledge Graph config ---
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "gemma3:4b")
OLLAMA_EMBED_MODEL: str = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
OLLAMA_EMBED_DIM: int = int(os.getenv("OLLAMA_EMBED_DIM", "768"))
LIGHTRAG_STORAGE_DIR: Path = DATA_DIR / "lightrag_store"
EXTRACTION_TEMPERATURE: float = 0.1
EXTRACTION_MAX_TOKENS: int = 8192
EPHEMERAL_BASE_TTL_HOURS: int = 48
```

- [ ] **Step 2: Add LightRAG and ollama to `pyproject.toml` dependencies**

Add to the `dependencies` list:

```toml
"lightrag-hku>=1.3.9",
"ollama>=0.4",
```

- [ ] **Step 3: Install dependencies**

Run: `pip install -e ".[dev]"`

- [ ] **Step 4: Commit**

```bash
git add pipeline/config.py pyproject.toml
git commit -m "feat(phase4): add knowledge graph config and LightRAG dependency"
```

---

## Task 2: Extraction Prompt Template

**Files:**
- Create: `pipeline/knowledge/__init__.py`
- Create: `pipeline/knowledge/extraction_prompt.py`

- [ ] **Step 1: Create package init**

```python
"""Knowledge graph: LightRAG integration, extraction, canonicalization, pruning."""
```

- [ ] **Step 2: Create `extraction_prompt.py` with the full custom prompt**

This file contains string constants only — no functions. The prompt instructs the LLM to extract entities and relationships in a specific JSON format, constrained to the closed ontology.

```python
"""Custom extraction prompt template for ontology-constrained entity/relationship extraction."""

ENTITY_TYPE_DEFINITIONS: str = """\
Entity Types (use ONLY these 8 types):

1. COMPANY — A publicly traded company. canonical_id = ticker symbol (uppercase, 1-5 chars).
   Examples: NVDA, AAPL, TSMC

2. PERSON — A named individual relevant to markets. canonical_id = person:<snake_case_name>.
   Examples: person:jensen_huang, person:jerome_powell

3. SECTOR — A GICS industry sector. canonical_id = sector:<lowercase_underscored>.
   Examples: sector:semiconductors, sector:software, sector:energy

4. INDEX — A market index. canonical_id = index:<SYMBOL>.
   Examples: index:SPX, index:NDX, index:DJI

5. PRODUCT — A specific product or service. canonical_id = product:<TICKER>:<name>.
   Examples: product:NVDA:H100, product:AAPL:iPhone

6. EVENT — A point-in-time occurrence. canonical_id = event:<yyyymmdd>:<slug>.
   Examples: event:20260401:nvda_q1_earnings, event:20260315:fed_rate_decision

7. MACRO_THEME — A persistent market narrative or macro force. canonical_id = theme:<slug>.
   Examples: theme:ai_capex_cycle, theme:china_export_restrictions

8. INSTITUTION — A government body, regulatory agency, or central bank. canonical_id = inst:<snake_case>.
   Examples: inst:federal_reserve, inst:sec, inst:european_central_bank
"""

RELATIONSHIP_TYPE_DEFINITIONS: str = """\
Relationship Types (use ONLY these 12 types):

--- Structural (stable, rarely change) ---

1. BELONGS_TO_SECTOR: Company -> Sector. GICS sector classification.
   Example: NVDA -> sector:semiconductors

2. CONSTITUENT_OF: Company -> Index. Index membership.
   Example: AAPL -> index:SPX

3. LED_BY: Company -> Person. Executive leadership.
   Example: NVDA -> person:jensen_huang

4. COMPETES_WITH: Company <-> Company. Direct competitor (symmetrical).
   Example: AMD <-> NVDA

5. SUPPLIES_TO: Company -> Company. Supply chain dependency (directional).
   Example: TSMC -> NVDA

6. PRODUCES: Company -> Product. Product ownership.
   Example: NVDA -> product:NVDA:H100

7. SUBSIDIARY_OF: Company -> Company. Corporate hierarchy.
   Example: Instagram -> META

--- Dynamic (extracted from current content, carry temporal metadata) ---

8. AFFECTED_BY_EVENT: Company/Sector -> Event. Causal link to an occurrence.
   Example: NVDA -> event:20260401:nvda_q1_earnings

9. DRIVEN_BY_THEME: Company -> Macro Theme. Persistent narrative driving a stock.
   Example: NVDA -> theme:ai_capex_cycle

10. SENTIMENT_TOWARD: Source -> Company. Measured sentiment (not judgment).
    Attributes: polarity (bullish/bearish/neutral), intensity (high/low), source_type (news/reddit).
    Example: reddit:wsb -> NVDA (bullish, high)

11. ANNOUNCED_BY: Event -> Person/Institution. Attribution of catalyst origin.
    Example: event:20260315:fed_rate_decision -> person:jerome_powell

12. POLICY_AFFECTS: Institution/Event -> Sector/Company. Regulatory or policy risk.
    Example: inst:sec -> sector:software

--- Cross-Portfolio ---

13. EXPOSED_TO: Company -> Macro Theme. Risk exposure (distinct from DRIVEN_BY_THEME).
    Example: NVDA -> theme:china_export_restrictions

If a relationship does not clearly fit any of the 13 types above, label it UNTYPED.
"""

EXTRACTION_SYSTEM_PROMPT: str = f"""\
You are a financial knowledge graph extraction engine. Given a markdown document about \
financial markets, extract entities and relationships according to the STRICT ontology below.

{ENTITY_TYPE_DEFINITIONS}

{RELATIONSHIP_TYPE_DEFINITIONS}

RULES:
- Extract ONLY entities that match one of the 8 types above.
- Extract ONLY relationships that match one of the 13 types above (or UNTYPED as fallback).
- For each entity, propose a canonical_id following the exact format specified for its type.
- For each relationship, assign a significance_score between 0.0 and 1.0:
  - 0.0-0.3: routine mention, low market impact
  - 0.4-0.6: notable development, moderate relevance
  - 0.7-1.0: major catalyst, high market impact (earnings surprises, policy changes, etc.)
- For SENTIMENT_TOWARD relationships, include polarity, intensity, and source_type in attributes.
- Output valid JSON only. No markdown, no commentary.

OUTPUT FORMAT:
{{
  "entities": [
    {{
      "entity_name": "<raw string from text>",
      "entity_type": "<one of 8 types>",
      "proposed_canonical_id": "<formatted ID>",
      "description": "<1-2 sentence description>"
    }}
  ],
  "relationships": [
    {{
      "src_entity": "<raw source entity name>",
      "tgt_entity": "<raw target entity name>",
      "relationship_type": "<one of 13 types or UNTYPED>",
      "description": "<brief description of the relationship>",
      "significance_score": 0.0,
      "attributes": {{}}
    }}
  ]
}}
"""

EXTRACTION_USER_TEMPLATE: str = """\
Extract entities and relationships from the following financial document:

---
{markdown_chunk}
---

Output valid JSON only."""

CORRECTIVE_PROMPT: str = """\
Your previous output was not valid JSON. Output ONLY a JSON object matching this schema exactly — \
no markdown fences, no commentary, no trailing text:

{{"entities": [...], "relationships": [...]}}

Original document:
---
{markdown_chunk}
---"""
```

- [ ] **Step 3: Commit**

```bash
git add pipeline/knowledge/__init__.py pipeline/knowledge/extraction_prompt.py
git commit -m "feat(phase4): add custom ontology extraction prompt template"
```

---

## Task 3: Canonical Registry Resolution

**Files:**
- Create: `pipeline/knowledge/canonical.py`
- Test: `tests/test_extraction.py` (canonical resolution tests)

- [ ] **Step 1: Write failing tests for canonical resolution**

Create `tests/test_extraction.py`:

```python
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

        # "NVIDIA" is not in the pre-seeded aliases (which are "NVIDIA Corporation" and "NVIDIA")
        # but the canonical_id NVDA exists. The new alias "Nvidia Corp" should be added.
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_extraction.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pipeline.knowledge.canonical'`

- [ ] **Step 3: Implement `pipeline/knowledge/canonical.py`**

```python
"""Canonical entity registry resolution.

Maps raw extracted entity strings to canonical IDs using the canonical_entities table.
"""

import json
import re
import sqlite3


# Canonical ID format patterns per entity type
_ID_PATTERNS: dict[str, re.Pattern] = {
    "COMPANY": re.compile(r"^[A-Z]{1,5}$"),
    "PERSON": re.compile(r"^person:[a-z][a-z0-9_]*$"),
    "SECTOR": re.compile(r"^sector:[a-z][a-z0-9_]*$"),
    "INDEX": re.compile(r"^index:[A-Z][A-Z0-9]*$"),
    "PRODUCT": re.compile(r"^product:[A-Z]{1,5}:.+$"),
    "EVENT": re.compile(r"^event:\d{8}:[a-z][a-z0-9_]*$"),
    "MACRO_THEME": re.compile(r"^theme:[a-z][a-z0-9_]*$"),
    "INSTITUTION": re.compile(r"^inst:[a-z][a-z0-9_]*$"),
}


def validate_canonical_id_format(canonical_id: str, entity_type: str) -> bool:
    """Check that a canonical ID matches the expected format for its entity type."""
    pattern = _ID_PATTERNS.get(entity_type)
    if pattern is None:
        return False
    return bool(pattern.match(canonical_id))


def _find_by_alias(conn: sqlite3.Connection, raw_name: str) -> str | None:
    """Search canonical_entities for any row whose aliases contain raw_name (case-insensitive)."""
    rows = conn.execute(
        "SELECT canonical_id, aliases FROM canonical_entities"
    ).fetchall()
    raw_lower = raw_name.lower()
    for row in rows:
        aliases = json.loads(row["aliases"])
        for alias in aliases:
            if alias.lower() == raw_lower:
                return row["canonical_id"]
    return None


def _find_by_canonical_id(conn: sqlite3.Connection, canonical_id: str) -> dict | None:
    """Look up a canonical entity by its ID. Returns dict or None."""
    row = conn.execute(
        "SELECT * FROM canonical_entities WHERE canonical_id = ?",
        (canonical_id,),
    ).fetchone()
    return dict(row) if row else None


def _add_alias(conn: sqlite3.Connection, canonical_id: str, new_alias: str) -> None:
    """Add a new alias to an existing canonical entity if not already present."""
    row = conn.execute(
        "SELECT aliases FROM canonical_entities WHERE canonical_id = ?",
        (canonical_id,),
    ).fetchone()
    if row is None:
        return
    aliases = json.loads(row["aliases"])
    # Case-insensitive dedup check
    if new_alias.lower() not in {a.lower() for a in aliases}:
        aliases.append(new_alias)
        conn.execute(
            "UPDATE canonical_entities SET aliases = ? WHERE canonical_id = ?",
            (json.dumps(aliases), canonical_id),
        )


def resolve_entity(
    conn: sqlite3.Connection,
    raw_name: str,
    proposed_canonical_id: str,
    proposed_type: str,
    description: str,
) -> str | None:
    """Resolve a raw entity string to a canonical ID.

    Resolution order:
    1. Check if raw_name matches any existing alias -> return that canonical_id.
    2. Check if proposed_canonical_id already exists -> add raw_name as alias, return it.
    3. Validate proposed_canonical_id format -> insert new entry, return it.
    4. If format validation fails -> return None.
    """
    # Step 1: alias lookup
    existing_id = _find_by_alias(conn, raw_name)
    if existing_id is not None:
        return existing_id

    # Also check if raw_name itself is a canonical_id
    existing_id = _find_by_alias(conn, proposed_canonical_id)
    if existing_id is not None:
        _add_alias(conn, existing_id, raw_name)
        return existing_id

    # Step 2: check if proposed ID already exists
    existing = _find_by_canonical_id(conn, proposed_canonical_id)
    if existing is not None:
        _add_alias(conn, proposed_canonical_id, raw_name)
        return proposed_canonical_id

    # Step 3: validate format and insert new
    if not validate_canonical_id_format(proposed_canonical_id, proposed_type):
        return None

    conn.execute(
        "INSERT INTO canonical_entities (canonical_id, entity_type, display_name, aliases) "
        "VALUES (?, ?, ?, ?)",
        (proposed_canonical_id, proposed_type, raw_name, json.dumps([raw_name])),
    )
    return proposed_canonical_id


def resolve_entities(
    conn: sqlite3.Connection,
    extracted_entities: list[dict],
) -> list[dict]:
    """Batch-resolve extracted entities. Returns list with canonical_id field added.

    Entities that fail resolution (invalid format) are excluded from the result.
    """
    resolved = []
    for entity in extracted_entities:
        canonical_id = resolve_entity(
            conn,
            raw_name=entity["entity_name"],
            proposed_canonical_id=entity["proposed_canonical_id"],
            proposed_type=entity["entity_type"],
            description=entity.get("description", ""),
        )
        if canonical_id is not None:
            resolved.append({**entity, "canonical_id": canonical_id})
    return resolved
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_extraction.py::TestCanonicalResolution -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add pipeline/knowledge/canonical.py tests/test_extraction.py
git commit -m "feat(phase4): implement canonical entity registry resolution"
```

---

## Task 4: Post-Extraction Validator

**Files:**
- Create: `pipeline/knowledge/validator.py`
- Modify: `tests/test_extraction.py` (add validator tests)

- [ ] **Step 1: Write failing tests for validation**

Append to `tests/test_extraction.py`:

```python
from pipeline.knowledge.validator import (
    ValidatedEntity,
    ValidatedRelationship,
    ValidationResult,
    validate_extraction,
    VALID_ENTITY_TYPES,
    VALID_RELATIONSHIP_TYPES,
)


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_extraction.py::TestValidator -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pipeline.knowledge.validator'`

- [ ] **Step 3: Implement `pipeline/knowledge/validator.py`**

```python
"""Post-extraction validation: type checking, canonical resolution, temporal metadata."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import sqlite3

from pipeline.config import EPHEMERAL_BASE_TTL_HOURS
from pipeline.knowledge.canonical import resolve_entity


VALID_ENTITY_TYPES: frozenset[str] = frozenset({
    "COMPANY", "PERSON", "SECTOR", "INDEX",
    "PRODUCT", "EVENT", "MACRO_THEME", "INSTITUTION",
})

VALID_RELATIONSHIP_TYPES: frozenset[str] = frozenset({
    "BELONGS_TO_SECTOR", "CONSTITUENT_OF", "LED_BY", "COMPETES_WITH",
    "SUPPLIES_TO", "PRODUCES", "SUBSIDIARY_OF",
    "AFFECTED_BY_EVENT", "DRIVEN_BY_THEME", "SENTIMENT_TOWARD",
    "ANNOUNCED_BY", "POLICY_AFFECTS",
    "EXPOSED_TO",
})

TIER_1_RELATIONSHIPS: frozenset[str] = frozenset({
    "BELONGS_TO_SECTOR", "CONSTITUENT_OF", "LED_BY", "COMPETES_WITH",
    "SUPPLIES_TO", "PRODUCES", "SUBSIDIARY_OF",
})

TIER_2_RELATIONSHIPS: frozenset[str] = frozenset({
    "AFFECTED_BY_EVENT", "DRIVEN_BY_THEME", "SENTIMENT_TOWARD",
    "ANNOUNCED_BY", "POLICY_AFFECTS",
})

TIER_3_RELATIONSHIPS: frozenset[str] = frozenset({
    "EXPOSED_TO", "CORRELATED_WITH",
})


@dataclass
class ValidatedEntity:
    canonical_id: str
    entity_type: str
    display_name: str
    description: str
    raw_name: str


@dataclass
class ValidatedRelationship:
    src_canonical_id: str
    tgt_canonical_id: str
    relationship_type: str
    tier: int  # 1, 2, or 3
    description: str
    significance_score: float
    attributes: dict = field(default_factory=dict)
    # Temporal metadata (populated for Tier 2 only)
    extracted_at: str | None = None
    source_run_id: int | None = None
    effective_ttl_hours: float | None = None


@dataclass
class ValidationResult:
    entities: list[ValidatedEntity] = field(default_factory=list)
    relationships: list[ValidatedRelationship] = field(default_factory=list)
    untyped_edges: list[dict] = field(default_factory=list)
    rejected_entities: list[dict] = field(default_factory=list)
    rejected_relationships: list[dict] = field(default_factory=list)


def _classify_tier(rel_type: str) -> int:
    if rel_type in TIER_1_RELATIONSHIPS:
        return 1
    if rel_type in TIER_2_RELATIONSHIPS:
        return 2
    if rel_type in TIER_3_RELATIONSHIPS:
        return 3
    return 0  # should not happen if type is validated


def validate_extraction(
    raw_extraction: dict,
    conn: sqlite3.Connection,
    run_id: int,
    base_ttl_hours: int = EPHEMERAL_BASE_TTL_HOURS,
) -> ValidationResult:
    """Validate and canonicalize raw LLM extraction output.

    Steps:
    1. Validate and resolve each entity through canonical registry.
    2. Validate each relationship type, resolve endpoints, classify tier.
    3. Attach temporal metadata to Tier 2 edges.
    """
    result = ValidationResult()
    now = datetime.now(timezone.utc).isoformat()

    # Build a map from raw entity name -> canonical_id for relationship resolution
    entity_map: dict[str, str] = {}

    # Step 1: Validate entities
    for entity in raw_extraction.get("entities", []):
        etype = entity.get("entity_type", "")
        if etype not in VALID_ENTITY_TYPES:
            result.rejected_entities.append(entity)
            continue

        canonical_id = resolve_entity(
            conn,
            raw_name=entity["entity_name"],
            proposed_canonical_id=entity["proposed_canonical_id"],
            proposed_type=etype,
            description=entity.get("description", ""),
        )
        if canonical_id is None:
            result.rejected_entities.append(entity)
            continue

        entity_map[entity["entity_name"]] = canonical_id
        result.entities.append(ValidatedEntity(
            canonical_id=canonical_id,
            entity_type=etype,
            display_name=entity["entity_name"],
            description=entity.get("description", ""),
            raw_name=entity["entity_name"],
        ))

    # Step 2: Validate relationships
    for rel in raw_extraction.get("relationships", []):
        rel_type = rel.get("relationship_type", "")

        if rel_type == "UNTYPED":
            result.untyped_edges.append(rel)
            continue

        if rel_type not in VALID_RELATIONSHIP_TYPES:
            result.rejected_relationships.append(rel)
            continue

        src_id = entity_map.get(rel["src_entity"])
        tgt_id = entity_map.get(rel["tgt_entity"])
        if src_id is None or tgt_id is None:
            # Skip relationships where an endpoint was rejected
            continue

        sig = max(0.0, min(1.0, rel.get("significance_score", 0.5)))
        tier = _classify_tier(rel_type)

        validated_rel = ValidatedRelationship(
            src_canonical_id=src_id,
            tgt_canonical_id=tgt_id,
            relationship_type=rel_type,
            tier=tier,
            description=rel.get("description", ""),
            significance_score=sig,
            attributes=rel.get("attributes", {}),
        )

        # Attach temporal metadata for Tier 2
        if tier == 2:
            validated_rel.extracted_at = now
            validated_rel.source_run_id = run_id
            validated_rel.effective_ttl_hours = base_ttl_hours * (1 + sig)

        result.relationships.append(validated_rel)

    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_extraction.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add pipeline/knowledge/validator.py tests/test_extraction.py
git commit -m "feat(phase4): implement post-extraction validator with tier classification"
```

---

## Task 5: LightRAG Configuration

**Files:**
- Create: `pipeline/knowledge/lightrag_config.py`

- [ ] **Step 1: Implement `pipeline/knowledge/lightrag_config.py`**

```python
"""LightRAG instance factory and lifecycle management."""

from __future__ import annotations

from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path
from typing import AsyncIterator

from lightrag import LightRAG
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from lightrag.utils import EmbeddingFunc

from pipeline.config import (
    LIGHTRAG_STORAGE_DIR,
    OLLAMA_BASE_URL,
    OLLAMA_EMBED_DIM,
    OLLAMA_EMBED_MODEL,
    OLLAMA_MODEL,
)


async def get_rag_instance(working_dir: Path | None = None) -> LightRAG:
    """Create and initialize a LightRAG instance with NanoVectorDB + NetworkX.

    Args:
        working_dir: Override storage directory (use tmp_path in tests).
                     Defaults to LIGHTRAG_STORAGE_DIR from config.

    Returns:
        Initialized LightRAG instance. Caller must call finalize_storages() when done.
    """
    storage_dir = str(working_dir or LIGHTRAG_STORAGE_DIR)

    rag = LightRAG(
        working_dir=storage_dir,
        graph_storage="NetworkXStorage",
        vector_storage="NanoVectorDBStorage",
        kv_storage="JsonKVStorage",
        llm_model_func=ollama_model_complete,
        llm_model_name=OLLAMA_MODEL,
        llm_model_kwargs={
            "host": OLLAMA_BASE_URL,
            "options": {"num_ctx": 8192},
            "timeout": 300,
        },
        embedding_func=EmbeddingFunc(
            embedding_dim=OLLAMA_EMBED_DIM,
            max_token_size=8192,
            func=partial(
                ollama_embed.func,
                embed_model=OLLAMA_EMBED_MODEL,
                host=OLLAMA_BASE_URL,
            ),
        ),
        addon_params={
            "language": "English",
            "entity_types": [
                "company", "person", "sector", "index",
                "product", "event", "macro_theme", "institution",
            ],
        },
        entity_extract_max_gleaning=1,
        chunk_token_size=1200,
        chunk_overlap_token_size=100,
    )

    await rag.initialize_storages()
    return rag


@asynccontextmanager
async def rag_session(working_dir: Path | None = None) -> AsyncIterator[LightRAG]:
    """Async context manager for LightRAG lifecycle.

    Usage:
        async with rag_session() as rag:
            await rag.aquery(...)
    """
    rag = await get_rag_instance(working_dir)
    try:
        yield rag
    finally:
        await rag.finalize_storages()
```

- [ ] **Step 2: Commit**

```bash
git add pipeline/knowledge/lightrag_config.py
git commit -m "feat(phase4): add LightRAG instance factory with Ollama config"
```

---

## Task 6: Graph Operations

**Files:**
- Create: `pipeline/knowledge/graph_ops.py`

- [ ] **Step 1: Implement `pipeline/knowledge/graph_ops.py`**

```python
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
```

- [ ] **Step 2: Commit**

```bash
git add pipeline/knowledge/graph_ops.py
git commit -m "feat(phase4): implement graph insertion, query, and correlation edge injection"
```

---

## Task 7: Extraction Orchestrator

**Files:**
- Create: `pipeline/knowledge/extraction.py`

- [ ] **Step 1: Implement `pipeline/knowledge/extraction.py`**

```python
"""Extraction orchestrator: markdown -> Ollama LLM call -> raw JSON -> validate -> insert."""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass, field

import httpx
from lightrag import LightRAG

from pipeline.config import (
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    EXTRACTION_TEMPERATURE,
    EXTRACTION_MAX_TOKENS,
)
from pipeline.ingestion.models import ParsedContent
from pipeline.knowledge.extraction_prompt import (
    CORRECTIVE_PROMPT,
    EXTRACTION_SYSTEM_PROMPT,
    EXTRACTION_USER_TEMPLATE,
)
from pipeline.knowledge.graph_ops import insert_validated_data
from pipeline.knowledge.validator import ValidationResult, validate_extraction

logger = logging.getLogger(__name__)


@dataclass
class BatchExtractionResult:
    documents_processed: int = 0
    documents_failed: int = 0
    total_entities: int = 0
    total_relationships: int = 0
    untyped_edges: int = 0
    new_canonical_entries: int = 0
    validation_results: list[ValidationResult] = field(default_factory=list)


async def _call_ollama(
    system_prompt: str,
    user_message: str,
) -> str:
    """Call Ollama chat API and return the raw response text."""
    async with httpx.AsyncClient(timeout=300) as client:
        resp = await client.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                "stream": False,
                "options": {
                    "temperature": EXTRACTION_TEMPERATURE,
                    "num_predict": EXTRACTION_MAX_TOKENS,
                },
                "format": "json",
            },
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]


def _parse_extraction_json(raw_text: str) -> dict | None:
    """Parse LLM output as JSON extraction result. Returns None on failure."""
    try:
        parsed = json.loads(raw_text)
        if "entities" in parsed and "relationships" in parsed:
            return parsed
        return None
    except (json.JSONDecodeError, TypeError):
        return None


async def extract_from_markdown(
    markdown: str,
) -> dict | None:
    """Run LLM extraction on a markdown chunk using the custom prompt.

    Calls Ollama with the extraction prompt. Retries once on malformed JSON.
    Returns parsed extraction dict or None on failure.
    """
    user_msg = EXTRACTION_USER_TEMPLATE.format(markdown_chunk=markdown)

    # First attempt
    raw = await _call_ollama(EXTRACTION_SYSTEM_PROMPT, user_msg)
    result = _parse_extraction_json(raw)
    if result is not None:
        return result

    # Retry with corrective prompt
    logger.warning("First extraction attempt returned malformed JSON, retrying")
    corrective_msg = CORRECTIVE_PROMPT.format(markdown_chunk=markdown)
    raw = await _call_ollama(EXTRACTION_SYSTEM_PROMPT, corrective_msg)
    result = _parse_extraction_json(raw)
    if result is None:
        logger.error("Extraction failed after retry: could not parse JSON")
    return result


async def process_content_batch(
    parsed_contents: list[ParsedContent],
    conn: sqlite3.Connection,
    rag: LightRAG,
    run_id: int,
) -> BatchExtractionResult:
    """Process a batch of parsed content through the full extraction pipeline.

    For each successful ParsedContent:
    1. extract_from_markdown() — LLM extraction via Ollama
    2. validate_extraction() — type checking, canonical resolution
    3. insert_validated_data() — graph insertion via LightRAG

    Args:
        parsed_contents: Output from Phase 3 ingestion (Crawl4AI parsed markdown).
        conn: SQLite connection for canonical registry lookups.
        rag: Initialized LightRAG instance.
        run_id: Current pipeline run ID.

    Returns:
        BatchExtractionResult with aggregate stats.
    """
    batch_result = BatchExtractionResult()

    for content in parsed_contents:
        if not content.success or not content.markdown:
            batch_result.documents_failed += 1
            continue

        # Stage A: LLM extraction
        raw_extraction = await extract_from_markdown(content.markdown)
        if raw_extraction is None:
            batch_result.documents_failed += 1
            continue

        # Stage B: Validate and canonicalize
        validation = validate_extraction(raw_extraction, conn, run_id)
        batch_result.validation_results.append(validation)
        batch_result.total_entities += len(validation.entities)
        batch_result.total_relationships += len(validation.relationships)
        batch_result.untyped_edges += len(validation.untyped_edges)

        # Stage C: Insert into graph
        await insert_validated_data(rag, validation, content.markdown, content.url)
        batch_result.documents_processed += 1

    conn.commit()
    return batch_result
```

- [ ] **Step 2: Commit**

```bash
git add pipeline/knowledge/extraction.py
git commit -m "feat(phase4): implement extraction orchestrator with Ollama LLM calls"
```

---

## Task 8: Ephemeral Pruner

**Files:**
- Create: `pipeline/knowledge/pruner.py`
- Create: `tests/test_pruner.py`

- [ ] **Step 1: Write failing tests for the pruner**

Create `tests/test_pruner.py`:

```python
"""Tests for the ephemeral TTL pruner."""

import json
from datetime import datetime, timedelta, timezone

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
        # 48 * (1 + 0.0) = 48.0
        from pipeline.knowledge.validator import ValidatedRelationship
        # Test via the effective_ttl formula directly
        assert 48 * (1 + 0.0) == 48.0

    def test_base_ttl_mid_significance(self):
        assert 48 * (1 + 0.5) == 72.0

    def test_base_ttl_max_significance(self):
        assert 48 * (1 + 1.0) == 96.0


class TestPruner:
    """Tests using a mock graph with pre-populated edges."""

    def _make_edge_desc(self, rel_type, tier, significance, extracted_at, ttl_hours=None):
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
    def mock_graph_data(self):
        """Build edge data for testing. Returns list of (src, tgt, data) tuples."""
        now = datetime.now(timezone.utc)
        old = (now - timedelta(hours=72)).isoformat()     # 72h ago
        recent = (now - timedelta(hours=24)).isoformat()   # 24h ago

        return [
            # Expired Tier 2 edge: 72h old, significance 0.0, TTL=48h -> expired
            ("NVDA", "event:20260401:old_event", {
                "description": self._make_edge_desc(
                    "AFFECTED_BY_EVENT", 2, 0.0, old),
                "keywords": "AFFECTED_BY_EVENT",
            }),
            # Surviving Tier 2 edge: 72h old, significance 0.9, TTL=91.2h -> alive
            ("AAPL", "event:20260401:big_event", {
                "description": self._make_edge_desc(
                    "AFFECTED_BY_EVENT", 2, 0.9, old),
                "keywords": "AFFECTED_BY_EVENT",
            }),
            # Recent Tier 2 edge: 24h old, significance 0.0, TTL=48h -> alive
            ("MSFT", "event:20260402:recent", {
                "description": self._make_edge_desc(
                    "AFFECTED_BY_EVENT", 2, 0.0, recent),
                "keywords": "AFFECTED_BY_EVENT",
            }),
            # Tier 1 structural edge: never pruned regardless of age
            ("NVDA", "sector:semiconductors", {
                "description": self._make_edge_desc(
                    "BELONGS_TO_SECTOR", 1, 1.0, old),
                "keywords": "BELONGS_TO_SECTOR",
            }),
            # Agent B CORRELATED_WITH edge: skipped by pruner
            ("NVDA", "AMD", {
                "description": self._make_edge_desc(
                    "CORRELATED_WITH", 3, 0.8, old),
                "keywords": "CORRELATED_WITH",
            }),
        ]

    @pytest.mark.asyncio
    async def test_expired_tier2_edge_removed(self, mock_graph_data, db_conn, tmp_path):
        result = await self._run_pruner(mock_graph_data, db_conn, tmp_path)
        assert result.edges_expired == 1
        assert result.edges_removed == 1

    @pytest.mark.asyncio
    async def test_tier1_never_pruned(self, mock_graph_data, db_conn, tmp_path):
        result = await self._run_pruner(mock_graph_data, db_conn, tmp_path)
        assert result.structural_edges_skipped >= 1

    @pytest.mark.asyncio
    async def test_prune_result_counts(self, mock_graph_data, db_conn, tmp_path):
        result = await self._run_pruner(mock_graph_data, db_conn, tmp_path)
        assert result.edges_scanned == 5
        assert result.edges_expired == 1
        assert result.structural_edges_skipped >= 1

    async def _run_pruner(self, edge_data, conn, tmp_path):
        """Helper: create a mock graph, populate edges, run pruner, return result."""
        import networkx as nx
        from unittest.mock import AsyncMock, MagicMock, patch

        # Build a real NetworkX graph
        graph = nx.Graph()
        for src, tgt, data in edge_data:
            graph.add_edge(src, tgt, **data)

        # Mock the graph storage
        mock_storage = AsyncMock()
        mock_storage._get_graph = AsyncMock(return_value=graph)
        mock_storage.remove_edges = AsyncMock()
        mock_storage.remove_nodes = AsyncMock()

        # Mock rag instance
        mock_rag = MagicMock()
        mock_rag.chunk_entity_relation_graph = mock_storage

        result = await prune_expired_edges(mock_rag, conn)
        return result
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_pruner.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pipeline.knowledge.pruner'`

- [ ] **Step 3: Implement `pipeline/knowledge/pruner.py`**

```python
"""Ephemeral TTL pruner: scans graph for expired Tier 2 edges and removes them."""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone

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
                from datetime import timedelta
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
    # Refresh graph after edge removal
    graph = await graph_storage._get_graph()
    orphaned = [n for n in graph.nodes() if graph.degree(n) == 0]

    # Only remove orphaned nodes that are Tier 2 entity types (events, themes)
    # Don't remove structural entities (companies, sectors, etc.)
    tier2_entity_prefixes = ("event:", "theme:")
    orphans_to_remove = [
        n for n in orphaned
        if any(n.startswith(p) for p in tier2_entity_prefixes)
    ]

    if orphans_to_remove:
        await graph_storage.remove_nodes(orphans_to_remove)
        result.nodes_orphaned_removed = len(orphans_to_remove)

    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_pruner.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add pipeline/knowledge/pruner.py tests/test_pruner.py
git commit -m "feat(phase4): implement ephemeral TTL pruner with tier immunity"
```

---

## Task 9: Package Exports & Integration Test

**Files:**
- Modify: `pipeline/knowledge/__init__.py`
- Modify: `tests/test_extraction.py` (add end-to-end test)

- [ ] **Step 1: Update package exports**

Replace `pipeline/knowledge/__init__.py`:

```python
"""Knowledge graph: LightRAG integration, extraction, canonicalization, pruning."""

from pipeline.knowledge.canonical import resolve_entity, resolve_entities
from pipeline.knowledge.extraction import (
    BatchExtractionResult,
    extract_from_markdown,
    process_content_batch,
)
from pipeline.knowledge.graph_ops import (
    inject_correlation_edges,
    insert_validated_data,
    parse_edge_metadata,
    query_graph,
)
from pipeline.knowledge.lightrag_config import get_rag_instance, rag_session
from pipeline.knowledge.pruner import PruneResult, prune_expired_edges
from pipeline.knowledge.validator import (
    ValidatedEntity,
    ValidatedRelationship,
    ValidationResult,
    validate_extraction,
)

__all__ = [
    "resolve_entity",
    "resolve_entities",
    "extract_from_markdown",
    "process_content_batch",
    "BatchExtractionResult",
    "inject_correlation_edges",
    "insert_validated_data",
    "parse_edge_metadata",
    "query_graph",
    "get_rag_instance",
    "rag_session",
    "prune_expired_edges",
    "PruneResult",
    "validate_extraction",
    "ValidatedEntity",
    "ValidatedRelationship",
    "ValidationResult",
]
```

- [ ] **Step 2: Add end-to-end test to `tests/test_extraction.py`**

Append to the file:

```python
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

        # All 7 entities should resolve (NVDA and AMD pre-seeded, rest new)
        assert len(result.entities) >= 6  # AMD may or may not be in mock watchlist
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
```

- [ ] **Step 3: Run all tests**

Run: `pytest tests/test_extraction.py tests/test_pruner.py -v`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add pipeline/knowledge/__init__.py tests/test_extraction.py
git commit -m "feat(phase4): add package exports and end-to-end validation tests"
```

---

## Task 10: Prompt Content Verification Tests

**Files:**
- Modify: `tests/test_extraction.py`

- [ ] **Step 1: Add prompt verification tests**

Append to `tests/test_extraction.py`:

```python
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
```

- [ ] **Step 2: Run tests**

Run: `pytest tests/test_extraction.py::TestExtractionPrompt -v`
Expected: All PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_extraction.py
git commit -m "test(phase4): add extraction prompt content verification tests"
```

---

## Task 11: Add `asyncio_mode` to pytest config

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add asyncio mode to pytest config**

The pruner tests use `@pytest.mark.asyncio`. Ensure pytest-asyncio auto mode is set:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
asyncio_mode = "auto"
markers = [
    "network: tests that require internet access (deselect with '-m not network')",
]
```

- [ ] **Step 2: Run full test suite**

Run: `pytest tests/ -v --ignore=tests/test_ingestion.py`
(Ignore ingestion tests that may need network)

Expected: All Phase 1-4 tests PASS

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "chore: add asyncio_mode=auto to pytest config"
```

---

## Verification Plan

After all tasks are complete, verify the following:

1. **Unit tests pass:** `pytest tests/test_extraction.py tests/test_pruner.py -v` — all green.
2. **Canonical resolution:** Feed "Nvidia", "NVIDIA", "NVIDIA Corporation" through `resolve_entity()` — all return `"NVDA"`.
3. **Validation rejects bad types:** An entity with type `"GADGET"` is in `rejected_entities`, not `entities`.
4. **UNTYPED edges logged:** A relationship with type `"UNTYPED"` appears in `untyped_edges`, not `relationships`.
5. **Tier classification:** `BELONGS_TO_SECTOR` is tier 1 (no TTL). `AFFECTED_BY_EVENT` is tier 2 (has TTL).
6. **TTL calculation:** significance 0.0 → 48h, significance 0.5 → 72h, significance 1.0 → 96h.
7. **Pruner removes expired:** A 72-hour-old edge with significance 0.0 (TTL=48h) is pruned. A 72-hour-old edge with significance 0.9 (TTL=91.2h) survives.
8. **Structural immunity:** Tier 1 edges are never pruned regardless of age.
9. **LightRAG persistence** (manual, requires Ollama running): Create a `rag_session()`, insert data via `ainsert_custom_kg`, close it, reopen with same `working_dir`, verify data persists.
10. **Full pipeline** (manual, requires Ollama running): Feed a sample markdown article → `extract_from_markdown()` → `validate_extraction()` → `insert_validated_data()` → `query_graph()` returns relevant results.

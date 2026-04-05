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

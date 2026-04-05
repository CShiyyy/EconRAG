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

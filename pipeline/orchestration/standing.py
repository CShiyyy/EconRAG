"""Standing event maintenance and Agent C action processing."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone


def run_maintenance(conn: sqlite3.Connection, run_id: int) -> list[dict]:
    """Return active standing events. Deferred: auto-promotion, summary compression."""
    rows = conn.execute(
        "SELECT * FROM standing_events WHERE status = 'active'"
    ).fetchall()
    result = []
    for row in rows:
        d = dict(row)
        d["affected_tickers"] = json.loads(d["affected_tickers"])
        result.append(d)
    return result


def _infer_entity_type(canonical_id: str) -> str:
    prefix = canonical_id.split(":")[0] if ":" in canonical_id else ""
    mapping = {
        "theme": "MACRO_THEME", "event": "EVENT", "person": "PERSON",
        "sector": "SECTOR", "index": "INDEX", "product": "PRODUCT", "inst": "INSTITUTION",
    }
    return mapping.get(prefix, "MACRO_THEME")


def process_actions(conn: sqlite3.Connection, agent_c_output: dict, run_id: int) -> None:
    actions = agent_c_output.get("standing_event_actions")
    if not actions:
        return

    ts = datetime.now(timezone.utc).isoformat()

    for promo in actions.get("promote_to_standing", []):
        canonical_id = promo["canonical_id"]
        existing = conn.execute(
            "SELECT canonical_id FROM canonical_entities WHERE canonical_id = ?",
            (canonical_id,),
        ).fetchone()
        if existing is None:
            entity_type = _infer_entity_type(canonical_id)
            display_name = canonical_id.split(":", 1)[-1].replace("_", " ").title() if ":" in canonical_id else canonical_id
            conn.execute(
                "INSERT INTO canonical_entities (canonical_id, entity_type, display_name, aliases) "
                "VALUES (?, ?, ?, '[]')",
                (canonical_id, entity_type, display_name),
            )
        existing_se = conn.execute(
            "SELECT standing_id FROM standing_events WHERE canonical_id = ? AND status = 'active'",
            (canonical_id,),
        ).fetchone()
        if existing_se is None:
            conn.execute(
                """INSERT INTO standing_events
                   (canonical_id, status, category, summary, affected_tickers,
                    promoted_at, promotion_source, last_reinforced, reinforcement_count,
                    created_from_run_id)
                   VALUES (?, 'active', ?, ?, ?, ?, 'agent_c', ?, 0, ?)""",
                (canonical_id, promo["category"], promo["summary"],
                 json.dumps(promo.get("affected_tickers", [])), ts, ts, run_id),
            )

    for resolution in actions.get("recommend_resolution", []):
        conn.execute(
            "UPDATE standing_events SET status = 'resolved', resolved_at = ? WHERE standing_id = ?",
            (ts, resolution["standing_id"]),
        )

    conn.commit()

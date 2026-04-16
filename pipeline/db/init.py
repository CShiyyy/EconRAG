import json
import re
import sqlite3
from datetime import datetime, timezone

from pipeline.config import DEFAULT_CONSTRAINTS, VALID_UNIVERSES
from pipeline.db.schema import create_tables, migrate_schema
from pipeline.scrapers.watchlist import WatchlistEntry, scrape_universe

# Suffixes to strip when generating company name aliases
_COMPANY_SUFFIXES = re.compile(
    r",?\s*\b(Inc\.?|Corp\.?|Corporation|Incorporated|Ltd\.?|Co\.?|Company|plc|Group|Holdings)\s*$",
    re.IGNORECASE,
)


def _insert_account(
    conn: sqlite3.Connection,
    universe: str,
    starting_cash: float,
) -> None:
    conn.execute(
        "INSERT INTO account (account_id, cash_balance, initial_cash, universe, initialized_at) "
        "VALUES (1, ?, ?, ?, ?)",
        (starting_cash, starting_cash, universe, datetime.now(timezone.utc).isoformat()),
    )


def _populate_watchlist(
    conn: sqlite3.Connection,
    universe: str,
) -> list[WatchlistEntry]:
    entries = scrape_universe(universe)
    now = datetime.now(timezone.utc).isoformat()
    conn.executemany(
        "INSERT INTO watchlist (ticker, company_name, sector, added_at) VALUES (?, ?, ?, ?)",
        [(t, n, s, now) for t, n, s in entries],
    )
    return entries


def _insert_constraints(
    conn: sqlite3.Connection,
    overrides: dict[str, float] | None,
) -> None:
    for name, (default_val, description) in DEFAULT_CONSTRAINTS.items():
        value = overrides.get(name, default_val) if overrides else default_val
        conn.execute(
            "INSERT INTO constraints (constraint_name, value, description) VALUES (?, ?, ?)",
            (name, value, description),
        )


def _build_aliases(company_name: str) -> list[str]:
    """Build alias list from a company name."""
    aliases = [company_name]
    stripped = _COMPANY_SUFFIXES.sub("", company_name).strip()
    if stripped and stripped != company_name:
        aliases.append(stripped)
    return aliases


def _seed_canonical_entities(
    conn: sqlite3.Connection,
    entries: list[WatchlistEntry],
) -> None:
    # Seed COMPANY entities
    for ticker, company_name, _ in entries:
        aliases = _build_aliases(company_name)
        conn.execute(
            "INSERT INTO canonical_entities (canonical_id, entity_type, display_name, aliases) "
            "VALUES (?, 'COMPANY', ?, ?)",
            (ticker, company_name, json.dumps(aliases)),
        )

    # Seed SECTOR entities (deduplicated)
    seen_sectors: set[str] = set()
    for _, _, sector in entries:
        if sector in seen_sectors:
            continue
        seen_sectors.add(sector)
        canonical_id = "sector:" + sector.lower().replace(" ", "_").replace("&", "and")
        conn.execute(
            "INSERT INTO canonical_entities (canonical_id, entity_type, display_name, aliases) "
            "VALUES (?, 'SECTOR', ?, ?)",
            (canonical_id, sector, json.dumps([sector])),
        )


def initialize(
    conn: sqlite3.Connection,
    universe: str,
    starting_cash: float,
    constraint_overrides: dict[str, float] | None = None,
) -> None:
    """Initialize the system: create tables, populate watchlist, seed entities.

    Args:
        conn: SQLite connection.
        universe: One of 'sp500', 'nasdaq100', 'djia30'.
        starting_cash: Initial cash balance (must be > 0).
        constraint_overrides: Optional dict of constraint_name -> value overrides.

    Raises:
        ValueError: Invalid universe, cash, or constraint override keys.
        RuntimeError: System already initialized.
    """
    # Validate inputs
    if universe not in VALID_UNIVERSES:
        raise ValueError(f"Unknown universe '{universe}'. Must be one of {VALID_UNIVERSES}")
    if starting_cash <= 0:
        raise ValueError("starting_cash must be positive")
    if constraint_overrides:
        invalid_keys = set(constraint_overrides) - set(DEFAULT_CONSTRAINTS)
        if invalid_keys:
            raise ValueError(f"Unknown constraint keys: {invalid_keys}")

    # Ensure tables exist and apply incremental migrations
    create_tables(conn)
    migrate_schema(conn)

    # Prevent re-initialization
    try:
        row = conn.execute("SELECT COUNT(*) FROM account").fetchone()
        if row[0] > 0:
            raise RuntimeError("System already initialized. Cannot re-initialize.")
    except sqlite3.OperationalError:
        pass  # Table doesn't exist yet, will be created

    _insert_account(conn, universe, starting_cash)
    entries = _populate_watchlist(conn, universe)
    _insert_constraints(conn, constraint_overrides)
    _seed_canonical_entities(conn, entries)
    conn.commit()

import sqlite3

# Ordered parent-first for FK safety.
TABLES_SQL: list[str] = [
    # 1. run_log — referenced by many tables
    """
    CREATE TABLE IF NOT EXISTS run_log (
        run_id          INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp       TEXT    NOT NULL,
        run_type        TEXT    NOT NULL CHECK(run_type IN ('pre_open', 'post_close', 'non_trading_day')),
        is_first_run    INTEGER NOT NULL DEFAULT 0,
        source_status   TEXT,
        requery_triggered INTEGER NOT NULL DEFAULT 0,
        requery_reason  TEXT,
        wall_clock_seconds REAL
    )
    """,
    # 2. account
    """
    CREATE TABLE IF NOT EXISTS account (
        account_id      INTEGER PRIMARY KEY CHECK(account_id = 1),
        cash_balance    REAL    NOT NULL,
        initial_cash    REAL    NOT NULL,
        universe        TEXT    NOT NULL CHECK(universe IN ('sp500', 'nasdaq100', 'djia30')),
        initialized_at  TEXT    NOT NULL
    )
    """,
    # 3. watchlist
    """
    CREATE TABLE IF NOT EXISTS watchlist (
        ticker          TEXT PRIMARY KEY,
        company_name    TEXT NOT NULL,
        sector          TEXT NOT NULL,
        added_at        TEXT NOT NULL
    )
    """,
    # 4. constraints
    """
    CREATE TABLE IF NOT EXISTS constraints (
        constraint_name TEXT PRIMARY KEY,
        value           REAL NOT NULL,
        description     TEXT NOT NULL
    )
    """,
    # 5. holdings
    """
    CREATE TABLE IF NOT EXISTS holdings (
        ticker              TEXT PRIMARY KEY,
        shares              REAL NOT NULL,
        cost_basis_per_share REAL NOT NULL,
        sector              TEXT NOT NULL
    )
    """,
    # 6. canonical_entities — referenced by standing_events
    """
    CREATE TABLE IF NOT EXISTS canonical_entities (
        canonical_id    TEXT PRIMARY KEY,
        entity_type     TEXT NOT NULL CHECK(entity_type IN (
            'COMPANY', 'PERSON', 'SECTOR', 'INDEX',
            'PRODUCT', 'EVENT', 'MACRO_THEME', 'INSTITUTION'
        )),
        display_name    TEXT NOT NULL,
        aliases         TEXT NOT NULL DEFAULT '[]'
    )
    """,
    # 7. computed_targets — FK to run_log
    """
    CREATE TABLE IF NOT EXISTS computed_targets (
        target_id       INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id          INTEGER NOT NULL REFERENCES run_log(run_id),
        timestamp       TEXT    NOT NULL,
        per_ticker_json TEXT    NOT NULL
    )
    """,
    # 8. snapshots — FK to run_log
    """
    CREATE TABLE IF NOT EXISTS snapshots (
        snapshot_id     INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id          INTEGER NOT NULL REFERENCES run_log(run_id),
        timestamp       TEXT    NOT NULL,
        total_value     REAL    NOT NULL,
        cash            REAL    NOT NULL,
        per_ticker_json TEXT    NOT NULL
    )
    """,
    # 9. recommendations — FK to run_log
    """
    CREATE TABLE IF NOT EXISTS recommendations (
        recommendation_id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id            INTEGER NOT NULL REFERENCES run_log(run_id),
        timestamp         TEXT    NOT NULL,
        ticker            TEXT    NOT NULL,
        action            TEXT    NOT NULL CHECK(action IN ('Hold', 'Buy', 'Trim', 'Exit', 'assessment')),
        conviction_scores TEXT    NOT NULL,
        conviction_weight REAL,
        rationale         TEXT    NOT NULL,
        key_quant_metrics TEXT,
        requery_triggered INTEGER NOT NULL DEFAULT 0,
        requery_reason    TEXT
    )
    """,
    # 10. trades — FK to recommendations
    """
    CREATE TABLE IF NOT EXISTS trades (
        trade_id            INTEGER PRIMARY KEY AUTOINCREMENT,
        recommendation_id   INTEGER NOT NULL REFERENCES recommendations(recommendation_id),
        ticker              TEXT    NOT NULL,
        action              TEXT    NOT NULL,
        shares              REAL    NOT NULL,
        simulated_fill_price REAL   NOT NULL,
        fill_type           TEXT    NOT NULL DEFAULT 'open_price',
        gap_pct             REAL,
        slippage_applied    REAL    NOT NULL DEFAULT 0.0,
        realized_pnl        REAL,
        timestamp           TEXT    NOT NULL
    )
    """,
    # 11. agent_outputs — FK to run_log
    """
    CREATE TABLE IF NOT EXISTS agent_outputs (
        output_id   INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id      INTEGER NOT NULL REFERENCES run_log(run_id),
        agent       TEXT    NOT NULL CHECK(agent IN ('A', 'B', 'C')),
        output_blob TEXT    NOT NULL
    )
    """,
    # 12. standing_events — FK to canonical_entities and run_log
    """
    CREATE TABLE IF NOT EXISTS standing_events (
        standing_id         INTEGER PRIMARY KEY AUTOINCREMENT,
        canonical_id        TEXT    NOT NULL REFERENCES canonical_entities(canonical_id),
        status              TEXT    NOT NULL DEFAULT 'active' CHECK(status IN ('active', 'resolved')),
        category            TEXT    NOT NULL CHECK(category IN (
            'geopolitical', 'monetary_policy', 'regulatory',
            'trade_policy', 'sector_crisis', 'other'
        )),
        summary             TEXT    NOT NULL,
        affected_tickers    TEXT    NOT NULL DEFAULT '[]',
        promoted_at         TEXT    NOT NULL,
        promotion_source    TEXT    NOT NULL CHECK(promotion_source IN ('auto', 'manual', 'agent_c')),
        last_reinforced     TEXT,
        reinforcement_count INTEGER NOT NULL DEFAULT 0,
        stale_run_threshold INTEGER NOT NULL DEFAULT 28,
        resolved_at         TEXT,
        created_from_run_id INTEGER REFERENCES run_log(run_id)
    )
    """,
    # 13. kg_seed_log — tracks which LightRAG profiles have been seeded
    """
    CREATE TABLE IF NOT EXISTS kg_seed_log (
        seed_type  TEXT NOT NULL,
        seed_key   TEXT NOT NULL,
        seeded_at  TEXT NOT NULL,
        run_id     INTEGER REFERENCES run_log(run_id),
        source_id  TEXT NOT NULL,
        PRIMARY KEY (seed_type, seed_key)
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_kg_seed_log_type ON kg_seed_log(seed_type)
    """,
]

EXPECTED_TABLES: set[str] = {
    "run_log", "account", "watchlist", "constraints", "holdings",
    "canonical_entities", "computed_targets", "snapshots",
    "recommendations", "trades", "agent_outputs", "standing_events",
    "kg_seed_log",
}


def create_tables(conn: sqlite3.Connection) -> None:
    """Create all tables. Idempotent (uses IF NOT EXISTS)."""
    for ddl in TABLES_SQL:
        conn.execute(ddl)
    conn.commit()

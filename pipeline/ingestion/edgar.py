"""EDGAR filing cache — fetches SEC 10-K data and stores per-ticker in SQLite."""
from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone, timedelta

import pandas as pd

logger = logging.getLogger(__name__)

_EDGAR_TTL_DAYS = 90


def refresh_edgar_cache(conn: sqlite3.Connection, tickers: list[str]) -> None:
    """Fetch EDGAR data for tickers not recently cached and upsert into edgar_filings_cache."""
    from pipeline.agents.second_tower.sec_edgar_fetcher import fetch_edgar_batch

    stale_tickers = []
    cutoff = datetime.now(timezone.utc) - timedelta(days=_EDGAR_TTL_DAYS)

    for ticker in tickers:
        row = conn.execute(
            "SELECT MAX(cached_at) FROM edgar_filings_cache WHERE ticker = ?", (ticker,)
        ).fetchone()
        if not row or not row[0]:
            stale_tickers.append(ticker)
        else:
            try:
                ts = datetime.fromisoformat(row[0])
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                if ts < cutoff:
                    stale_tickers.append(ticker)
            except Exception:
                stale_tickers.append(ticker)

    if not stale_tickers:
        return

    logger.info(f"Refreshing EDGAR cache for {len(stale_tickers)} tickers")
    results = fetch_edgar_batch(stale_tickers, max_workers=4)
    now_str = datetime.now(timezone.utc).isoformat()

    for ticker, stmts in results.items():
        if not stmts:
            conn.execute(
                """INSERT OR REPLACE INTO edgar_filings_cache
                   (ticker, statement_type, payload_json, cached_at)
                   VALUES (?, 'empty', '{}', ?)""",
                (ticker, now_str),
            )
            continue
        for stmt_name, df in stmts.items():
            if not isinstance(df, pd.DataFrame) or df.empty:
                payload = "{}"
            else:
                payload = df.to_json(date_format="iso")
            conn.execute(
                """INSERT OR REPLACE INTO edgar_filings_cache
                   (ticker, statement_type, payload_json, cached_at)
                   VALUES (?, ?, ?, ?)""",
                (ticker, stmt_name, payload, now_str),
            )
    conn.commit()

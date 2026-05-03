"""Trade executor — queue, execute, and overwrite lifecycle for trades."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import Callable


def queue_trades(
    conn: sqlite3.Connection,
    trade_list: list[dict],
    recommendation_ids: dict[str, int],
    queued_run_id: int,
    target_close_at: datetime,
) -> list[int]:
    """Insert trades with status='queued'. No portfolio mutation.

    Returns list of inserted trade_ids.
    """
    now = datetime.now(timezone.utc).isoformat()
    target_close_iso = (
        target_close_at.isoformat()
        if isinstance(target_close_at, datetime)
        else target_close_at
    )
    trade_ids: list[int] = []

    for trade in trade_list:
        ticker = trade["ticker"]
        rec_id = recommendation_ids.get(ticker)
        if rec_id is None:
            row = conn.execute(
                "SELECT recommendation_id FROM recommendations "
                "WHERE run_id = ? AND ticker = ?",
                (queued_run_id, ticker),
            ).fetchone()
            rec_id = row["recommendation_id"] if row else None
        if rec_id is None:
            continue

        cursor = conn.execute(
            """INSERT INTO trades
               (recommendation_id, ticker, action, shares, simulated_fill_price,
                fill_type, gap_pct, slippage_applied, realized_pnl, timestamp,
                status, queued_at, queued_run_id, target_close_at)
               VALUES (?, ?, ?, ?, ?, 'previous_close', NULL, 0.0, NULL, ?,
                       'queued', ?, ?, ?)""",
            (
                rec_id, ticker, trade["action"], trade["shares"], trade["fill_price"],
                now, now, queued_run_id, target_close_iso,
            ),
        )
        trade_ids.append(cursor.lastrowid)

    conn.commit()
    return trade_ids


def execute_queued_batch(
    conn: sqlite3.Connection,
    execution_run_id: int,
    last_close_dt: datetime,
    price_source: Callable[[str], float | None],
) -> dict:
    """Execute all queued trades whose queued_at predates last_close_dt.

    Mutates portfolio (holdings + cash) per trade and updates trade rows.
    Returns {'executed': int, 'skipped': int, 'skipped_tickers': list[str]}.
    """
    from pipeline.sizing.trade_builder import apply_trade

    last_close_iso = (
        last_close_dt.isoformat()
        if isinstance(last_close_dt, datetime)
        else last_close_dt
    )
    now = datetime.now(timezone.utc).isoformat()

    rows = conn.execute(
        "SELECT * FROM trades WHERE status = 'queued' AND queued_at < ?",
        (last_close_iso,),
    ).fetchall()

    executed = 0
    skipped = 0
    skipped_tickers: list[str] = []

    for row in rows:
        ticker = row["ticker"]
        close_price = price_source(ticker)

        if not close_price or close_price <= 0:
            conn.execute(
                "UPDATE trades SET status='skipped', executed_at=?, execution_run_id=? "
                "WHERE trade_id=?",
                (now, execution_run_id, row["trade_id"]),
            )
            skipped += 1
            skipped_tickers.append(ticker)
            continue

        sector_row = conn.execute(
            "SELECT sector FROM watchlist WHERE ticker = ? "
            "UNION SELECT sector FROM holdings WHERE ticker = ? LIMIT 1",
            (ticker, ticker),
        ).fetchone()
        sector = sector_row["sector"] if sector_row else ""

        realized_pnl = apply_trade(
            conn, ticker, row["action"], row["shares"], close_price, sector
        )

        queued_price = row["simulated_fill_price"]
        gap_pct = (close_price - queued_price) / queued_price if queued_price > 0 else None

        conn.execute(
            """UPDATE trades SET
                status='executed', executed_at=?, execution_run_id=?,
                execution_fill_price=?, fill_type='close_price',
                realized_pnl=?, gap_pct=?
               WHERE trade_id=?""",
            (now, execution_run_id, close_price, realized_pnl, gap_pct, row["trade_id"]),
        )
        executed += 1

    conn.commit()
    return {"executed": executed, "skipped": skipped, "skipped_tickers": skipped_tickers}


def mark_overwritten(
    conn: sqlite3.Connection,
    current_run_id: int,
    last_close_dt: datetime,
) -> int:
    """Mark pending queued trades from the current trading window as overwritten.

    Trades with queued_at >= last_close_dt belong to the current window and are
    replaced by the fresh queue being produced now.
    Returns count of rows updated.
    """
    last_close_iso = (
        last_close_dt.isoformat()
        if isinstance(last_close_dt, datetime)
        else last_close_dt
    )
    cursor = conn.execute(
        "UPDATE trades SET status='overwritten', execution_run_id=? "
        "WHERE status='queued' AND queued_at >= ?",
        (current_run_id, last_close_iso),
    )
    conn.commit()
    return cursor.rowcount

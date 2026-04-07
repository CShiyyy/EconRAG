"""Trade executor — logs executed trades to the trades table.

Note: Portfolio mutations (holdings, cash, cost basis) are handled by
run_sizing_engine() in pipeline.sizing.engine. This module only records
trade entries in the trades table with gap_pct and recommendation FK.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone


def log_trades(
    conn: sqlite3.Connection,
    trade_list: list[dict],
    market_data: dict,
    recommendation_ids: dict[str, int],
    run_id: int,
) -> None:
    timestamp = datetime.now(timezone.utc).isoformat()

    for trade in trade_list:
        ticker = trade["ticker"]
        md = market_data.get(ticker, {})

        open_price = md.get("open_price", trade["fill_price"])
        prev_close = md.get("previous_close")

        gap_pct = None
        if prev_close and prev_close > 0:
            gap_pct = (open_price - prev_close) / prev_close

        rec_id = recommendation_ids.get(ticker)
        if rec_id is None:
            row = conn.execute(
                "SELECT recommendation_id FROM recommendations WHERE run_id = ? AND ticker = ?",
                (run_id, ticker),
            ).fetchone()
            rec_id = row["recommendation_id"] if row else None

        if rec_id is None:
            continue

        conn.execute(
            """INSERT INTO trades
               (recommendation_id, ticker, action, shares, simulated_fill_price,
                fill_type, gap_pct, slippage_applied, realized_pnl, timestamp)
               VALUES (?, ?, ?, ?, ?, 'open_price', ?, 0.0, ?, ?)""",
            (rec_id, ticker, trade["action"], trade["shares"],
             trade["fill_price"], gap_pct, trade.get("realized_pnl"), timestamp),
        )
    conn.commit()

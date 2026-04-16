"""Portfolio snapshot recording — captures full portfolio state per run."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone


def record_snapshot(
    conn: sqlite3.Connection,
    run_id: int,
    market_data: dict | None,
    run_type: str,
) -> None:
    """Record current portfolio state to the snapshots table.

    Args:
        conn: SQLite connection.
        run_id: Current run ID.
        market_data: Dict of {ticker: {current_price, open_price, previous_close}}.
                     None if portfolio is cash-only.
        run_type: "pre_open" or "post_close".
    """
    account = conn.execute(
        "SELECT cash_balance FROM account WHERE account_id = 1"
    ).fetchone()
    cash = account["cash_balance"]

    holdings = conn.execute(
        "SELECT ticker, shares, cost_basis_per_share, sector FROM holdings"
    ).fetchall()

    per_ticker: dict[str, dict] = {}
    holdings_value = 0.0

    # Select valuation price based on run type:
    # Pre-open: previous_close (settled price before today's open).
    # Post-close: current_price (today's closing price).
    val_key = "previous_close" if run_type == "pre_open" else "current_price"

    for row in holdings:
        ticker = row["ticker"]
        shares = row["shares"]

        if market_data and ticker in market_data:
            price = market_data[ticker].get(val_key) or market_data[ticker].get("current_price", 0.0)
        else:
            price = 0.0

        value = shares * price
        holdings_value += value

        per_ticker[ticker] = {
            "shares": shares,
            "price": price,
            "value": value,
            "weight": 0.0,
            "cost_basis": row["cost_basis_per_share"],
            "sector": row["sector"],
        }

    total_value = cash + holdings_value

    if total_value > 0:
        for data in per_ticker.values():
            data["weight"] = data["value"] / total_value

    timestamp = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT INTO snapshots (run_id, timestamp, total_value, cash, per_ticker_json) "
        "VALUES (?, ?, ?, ?, ?)",
        (run_id, timestamp, total_value, cash, json.dumps(per_ticker)),
    )
    conn.commit()

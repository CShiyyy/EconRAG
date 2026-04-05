import json
import sqlite3
from typing import Callable


def get_portfolio_value(
    conn: sqlite3.Connection,
    price_fn: Callable[[str], float] | None = None,
) -> float:
    """Return total portfolio value: cash + sum(shares * current_price).

    Args:
        conn: SQLite connection.
        price_fn: Callable that takes a ticker and returns current price.
                  Required when holdings exist. Can be None if portfolio is cash-only.
    """
    cash = conn.execute("SELECT cash_balance FROM account WHERE account_id = 1").fetchone()
    if cash is None:
        raise RuntimeError("System not initialized")
    cash_balance = cash[0]

    holdings = conn.execute("SELECT ticker, shares FROM holdings").fetchall()
    if not holdings:
        return cash_balance

    if price_fn is None:
        raise ValueError("price_fn required when holdings exist")

    total = cash_balance
    for row in holdings:
        total += row["shares"] * price_fn(row["ticker"])
    return total


def get_derived_weights(
    conn: sqlite3.Connection,
    price_fn: Callable[[str], float] | None = None,
) -> dict[str, float]:
    """Return per-ticker weight as fraction of total portfolio value.

    Returns empty dict if no holdings exist.
    """
    holdings = conn.execute("SELECT ticker, shares FROM holdings").fetchall()
    if not holdings:
        return {}

    if price_fn is None:
        raise ValueError("price_fn required when holdings exist")

    total_value = get_portfolio_value(conn, price_fn)
    if total_value == 0:
        return {}

    return {
        row["ticker"]: (row["shares"] * price_fn(row["ticker"])) / total_value
        for row in holdings
    }


def get_previous_computed_target(conn: sqlite3.Connection) -> dict | None:
    """Return most recent computed_targets row, or None if none exist."""
    row = conn.execute(
        "SELECT * FROM computed_targets ORDER BY target_id DESC LIMIT 1"
    ).fetchone()
    if row is None:
        return None
    return {
        "target_id": row["target_id"],
        "run_id": row["run_id"],
        "timestamp": row["timestamp"],
        "per_ticker_json": json.loads(row["per_ticker_json"]),
    }


def is_first_run(conn: sqlite3.Connection) -> bool:
    """Return True if computed_targets table is empty (no prior runs)."""
    row = conn.execute("SELECT COUNT(*) FROM computed_targets").fetchone()
    return row[0] == 0


def get_account(conn: sqlite3.Connection) -> dict:
    """Return the account row as a dict."""
    row = conn.execute("SELECT * FROM account WHERE account_id = 1").fetchone()
    if row is None:
        raise RuntimeError("System not initialized")
    return dict(row)


def get_constraints(conn: sqlite3.Connection) -> dict[str, float]:
    """Return constraints as {constraint_name: value}."""
    rows = conn.execute("SELECT constraint_name, value FROM constraints").fetchall()
    return {row["constraint_name"]: row["value"] for row in rows}


def get_watchlist(conn: sqlite3.Connection) -> list[dict]:
    """Return all watchlist entries ordered by ticker."""
    rows = conn.execute("SELECT * FROM watchlist ORDER BY ticker").fetchall()
    return [dict(row) for row in rows]

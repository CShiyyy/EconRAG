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


def get_active_standing_events(conn: sqlite3.Connection) -> list[dict]:
    """Return all standing events with status='active', parsing affected_tickers JSON."""
    rows = conn.execute(
        "SELECT * FROM standing_events WHERE status = 'active'"
    ).fetchall()
    result = []
    for row in rows:
        d = dict(row)
        d["affected_tickers"] = json.loads(d["affected_tickers"])
        result.append(d)
    return result


def store_agent_output(conn: sqlite3.Connection, run_id: int, agent: str, output: dict) -> int:
    """Insert agent output JSON into agent_outputs table. Returns output_id."""
    cursor = conn.execute(
        "INSERT INTO agent_outputs (run_id, agent, output_blob) VALUES (?, ?, ?)",
        (run_id, agent, json.dumps(output)),
    )
    conn.commit()
    return cursor.lastrowid


def get_holdings(conn: sqlite3.Connection) -> list[dict]:
    """Return all current holdings rows as dicts."""
    rows = conn.execute("SELECT * FROM holdings").fetchall()
    return [dict(row) for row in rows]


def store_recommendations(
    conn: sqlite3.Connection,
    run_id: int,
    per_ticker: dict[str, dict],
    requery_triggered: bool = False,
    requery_reason: str | None = None,
    quant_assessment: dict | None = None,
) -> list[int]:
    """Insert per-ticker recommendations from Agent C output.

    Each ticker entry becomes a row in the recommendations table.
    conviction_weight is null (computed later by Position Sizing Engine).
    key_quant_metrics stores Agent B metrics; key_risk_factors stores Agent C risk list.
    Returns list of recommendation_ids.
    """
    from datetime import datetime, timezone

    ts = datetime.now(timezone.utc).isoformat()
    ids: list[int] = []
    per_ticker_quant = (quant_assessment or {}).get("per_ticker", {})

    for ticker, entry in per_ticker.items():
        # Extract real Agent B metrics for this ticker
        quant_data = per_ticker_quant.get(ticker, {})
        quant_metrics = {
            k: quant_data.get(k)
            for k in ("drift", "volatility_30d", "health_score", "current_weight", "flags")
            if quant_data.get(k) is not None
        }

        cursor = conn.execute(
            """INSERT INTO recommendations
               (run_id, timestamp, ticker, action, conviction_scores,
                conviction_weight, rationale, key_quant_metrics, key_risk_factors,
                requery_triggered, requery_reason)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                run_id,
                ts,
                ticker,
                entry["action"],
                json.dumps(entry.get("conviction", {})),
                None,
                entry.get("rationale", ""),
                json.dumps(quant_metrics) if quant_metrics else None,
                json.dumps(entry.get("key_risk_factors", [])),
                1 if requery_triggered else 0,
                requery_reason,
            ),
        )
        ids.append(cursor.lastrowid)
    conn.commit()
    return ids


def get_previous_assessment(conn: sqlite3.Connection) -> dict | None:
    """Get the most recent post-close assessment from the recommendations table.

    Queries for the latest run_id where entries have action='assessment'.
    Returns dict keyed by ticker with conviction scores and rationale,
    or None if no assessment run has occurred yet.
    """
    row = conn.execute(
        """SELECT run_id FROM recommendations
           WHERE action = 'assessment'
           ORDER BY recommendation_id DESC LIMIT 1"""
    ).fetchone()
    if row is None:
        return None

    latest_run_id = row["run_id"]
    rows = conn.execute(
        """SELECT ticker, conviction_scores, rationale, key_quant_metrics
           FROM recommendations
           WHERE run_id = ? AND action = 'assessment'""",
        (latest_run_id,),
    ).fetchall()

    result: dict[str, dict] = {}
    for r in rows:
        result[r["ticker"]] = {
            "conviction": json.loads(r["conviction_scores"]),
            "rationale": r["rationale"],
            "key_risk_factors": json.loads(r["key_quant_metrics"])
            if r["key_quant_metrics"]
            else [],
        }
    return result

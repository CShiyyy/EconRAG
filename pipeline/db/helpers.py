import json
import logging
import sqlite3
from io import StringIO
from typing import Callable

logger = logging.getLogger(__name__)


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

    `conviction_scores` (repurposed): JSON of Agent B's quant signals that
        produced the weight — `{factor_drivers: [...], category_scores: {...},
        composite_signal: float}`. The legacy narrative_score / multiplier
        fields are no longer written; weights are 100% deterministic from
        Agent B and LLMs cannot scale them.
    `conviction_weight`: legacy column, set to 1.0 (no longer used by sizing).
    `key_quant_metrics`: drift / volatility / health_score / current_weight /
        flags from Agent B (unchanged).
    `key_risk_factors`: Agent C's risk list (unchanged).

    Returns list of recommendation_ids.
    """
    from datetime import datetime, timezone

    ts = datetime.now(timezone.utc).isoformat()
    ids: list[int] = []
    per_ticker_quant = (quant_assessment or {}).get("per_ticker", {})

    for ticker, entry in per_ticker.items():
        quant_data = per_ticker_quant.get(ticker, {})
        quant_metrics = {
            k: quant_data.get(k)
            for k in ("drift", "volatility_30d", "health_score", "current_weight", "flags")
            if quant_data.get(k) is not None
        }
        conviction_payload = {
            "factor_drivers": quant_data.get("factor_drivers", []),
            "category_scores": quant_data.get("category_scores", {}),
            "composite_signal": quant_data.get("composite_signal", 0.0),
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
                json.dumps(conviction_payload),
                1.0,
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


def load_ohlcv_cache(
    conn: sqlite3.Connection,
    tickers: list[str],
    lookback_days: int,
) -> dict[str, "pd.DataFrame"]:
    """Load OHLCV data from market_history_cache as {ticker: OHLCV DataFrame}."""
    import pandas as pd
    from datetime import datetime, timezone, timedelta

    cutoff = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    result: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        rows = conn.execute(
            """SELECT date, open, high, low, close, volume
               FROM market_history_cache
               WHERE ticker = ? AND date >= ?
               ORDER BY date ASC""",
            (ticker, cutoff),
        ).fetchall()
        if not rows:
            result[ticker] = pd.DataFrame()
            continue
        df = pd.DataFrame(rows, columns=["Date", "Open", "High", "Low", "Close", "Volume"])
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")
        result[ticker] = df
    return result


def load_edgar_cache(conn: sqlite3.Connection, tickers: list[str]) -> dict[str, dict]:
    """Load EDGAR statement data from edgar_filings_cache as {ticker: {stmt: df}}."""
    import pandas as pd

    result: dict[str, dict] = {ticker: {} for ticker in tickers}
    if not tickers:
        return result

    placeholders = ",".join("?" * len(tickers))
    rows = conn.execute(
        f"SELECT ticker, statement_type, payload_json FROM edgar_filings_cache WHERE ticker IN ({placeholders})",
        tickers,
    ).fetchall()

    for ticker, stmt_type, payload in rows:
        if stmt_type == "empty":
            continue
        try:
            df = pd.read_json(StringIO(payload))
            df.index = pd.to_datetime(df.index)
            result[ticker][stmt_type] = df
        except Exception as e:
            logger.warning("Failed to parse EDGAR payload for %s/%s: %s", ticker, stmt_type, e)
            result[ticker][stmt_type] = pd.DataFrame()

    return result


def revert_trades(conn: sqlite3.Connection, run_id: int) -> int:
    """Reverse all portfolio mutations (holdings, cash) from a pre-open run.

    Returns the number of trade rows reversed.
    """
    trades = conn.execute(
        """SELECT t.ticker, t.action, t.shares, t.simulated_fill_price, t.realized_pnl
           FROM trades t
           JOIN recommendations r ON t.recommendation_id = r.recommendation_id
           WHERE r.run_id = ?""",
        (run_id,),
    ).fetchall()

    cash_delta = 0.0  # net cash change that happened during the run (positive = cash in)
    for t in trades:
        action, shares, price = t["action"], t["shares"], t["simulated_fill_price"]
        if action == "Buy":
            cash_delta -= shares * price
        else:  # Trim or Exit
            cash_delta += shares * price

    for t in trades:
        ticker = t["ticker"]
        action, shares, price = t["action"], t["shares"], t["simulated_fill_price"]
        realized_pnl = t["realized_pnl"] or 0.0

        if action == "Buy":
            row = conn.execute(
                "SELECT shares, cost_basis_per_share FROM holdings WHERE ticker = ?", (ticker,)
            ).fetchone()
            if row is None:
                continue  # position already gone, skip
            cur_shares = row["shares"]
            cur_basis = row["cost_basis_per_share"]
            pre_shares = cur_shares - shares
            if pre_shares <= 0.001:
                conn.execute("DELETE FROM holdings WHERE ticker = ?", (ticker,))
            else:
                pre_basis = (cur_basis * cur_shares - shares * price) / pre_shares
                conn.execute(
                    "UPDATE holdings SET shares = ?, cost_basis_per_share = ? WHERE ticker = ?",
                    (pre_shares, pre_basis, ticker),
                )

        elif action in ("Trim", "Exit"):
            row = conn.execute(
                "SELECT shares, cost_basis_per_share FROM holdings WHERE ticker = ?", (ticker,)
            ).fetchone()
            pre_shares = shares  # what we had before the sell
            # cost_basis = fill_price - realized_pnl / shares (from P&L formula)
            pre_basis = price - (realized_pnl / shares if shares > 0 else 0.0)
            sector_row = conn.execute(
                "SELECT sector FROM watchlist WHERE ticker = ?", (ticker,)
            ).fetchone()
            sector = sector_row["sector"] if sector_row else ""
            if row is None:
                # Position was fully liquidated; restore it
                conn.execute(
                    "INSERT INTO holdings (ticker, shares, cost_basis_per_share, sector) VALUES (?, ?, ?, ?)",
                    (ticker, pre_shares, pre_basis, sector),
                )
            else:
                # Partial trim: add shares back (cost basis unchanged)
                conn.execute(
                    "UPDATE holdings SET shares = ? WHERE ticker = ?",
                    (row["shares"] + pre_shares, ticker),
                )

    # Reverse cash change
    conn.execute(
        "UPDATE account SET cash_balance = cash_balance - ? WHERE account_id = 1",
        (cash_delta,),
    )
    conn.commit()
    return len(trades)


class OverwriteResult:
    def __init__(self, prior_run_id: int, reverted_trade_count: int) -> None:
        self.prior_run_id = prior_run_id
        self.reverted_trade_count = reverted_trade_count


def overwrite_slot(conn: sqlite3.Connection, session_date: str, run_type: str) -> OverwriteResult | None:
    """Delete all data for an existing (session_date, run_type) slot so it can be re-run.

    For pre_open runs, reverses simulated trade mutations first.
    For standing_events, NULLs created_from_run_id rather than deleting the event.
    Returns OverwriteResult if a slot existed, None if no prior run found.
    """
    row = conn.execute(
        "SELECT run_id FROM run_log WHERE session_date = ? AND run_type = ?",
        (session_date, run_type),
    ).fetchone()
    if row is None:
        return None

    run_id = row["run_id"]
    trade_count = 0

    if run_type == "pre_open":
        trade_count = revert_trades(conn, run_id)

    # Delete trade records
    conn.execute(
        """DELETE FROM trades WHERE recommendation_id IN (
               SELECT recommendation_id FROM recommendations WHERE run_id = ?
           )""",
        (run_id,),
    )
    # Delete child rows keyed by run_id
    for table in ("recommendations", "snapshots", "agent_outputs", "computed_targets", "kg_seed_log"):
        conn.execute(f"DELETE FROM {table} WHERE run_id = ?", (run_id,))

    # Preserve standing events but sever the run reference
    conn.execute(
        "UPDATE standing_events SET created_from_run_id = NULL WHERE created_from_run_id = ?",
        (run_id,),
    )

    conn.execute("DELETE FROM run_log WHERE run_id = ?", (run_id,))
    conn.commit()
    return OverwriteResult(prior_run_id=run_id, reverted_trade_count=trade_count)


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

"""Position Sizing Engine — orchestrates the full 5-step pipeline.

Steps:
1. Conviction weight mapping (sub-scores -> numeric weights)
2. Normalization (conviction weights -> target portfolio weights)
3. Constraint enforcement (cap/floor/renormalize)
4. Trade list computation (target weights vs current holdings -> share deltas)
5. Store computed targets and execute cost basis updates
"""

import sqlite3
from typing import Any

from pipeline.db.helpers import get_account, get_constraints
from pipeline.sizing.conviction_map import get_conviction_weight
from pipeline.sizing.normalizer import enforce_constraints, normalize_weights
from pipeline.sizing.trade_builder import (
    compute_trade_list,
    store_computed_targets,
    update_cost_basis,
)


def _get_current_holdings(conn: sqlite3.Connection) -> dict[str, dict]:
    """Read current holdings from DB as {ticker: {shares, cost_basis_per_share, sector}}."""
    rows = conn.execute(
        "SELECT ticker, shares, cost_basis_per_share, sector FROM holdings"
    ).fetchall()
    return {
        row["ticker"]: {
            "shares": row["shares"],
            "cost_basis_per_share": row["cost_basis_per_share"],
            "sector": row["sector"],
        }
        for row in rows
    }


def _get_sector_map(conn: sqlite3.Connection, tickers: list[str]) -> dict[str, str]:
    """Build {ticker: sector} from watchlist and holdings."""
    sector_map: dict[str, str] = {}

    # From watchlist
    rows = conn.execute("SELECT ticker, sector FROM watchlist").fetchall()
    for row in rows:
        sector_map[row["ticker"]] = row["sector"]

    # From holdings (sector is cached there too)
    rows = conn.execute("SELECT ticker, sector FROM holdings").fetchall()
    for row in rows:
        sector_map[row["ticker"]] = row["sector"]

    return {t: sector_map[t] for t in tickers if t in sector_map}


def run_sizing_engine(
    conn: sqlite3.Connection,
    run_id: int,
    agent_c_output: dict[str, Any],
    fill_prices: dict[str, float],
    is_first_run: bool = False,
) -> dict:
    """Run the full Position Sizing Engine pipeline.

    Args:
        conn: SQLite connection.
        run_id: Current run ID.
        agent_c_output: Agent C's per_ticker output:
            {ticker: {action, conviction: {narrative_alignment, quant_support, signal_agreement}}}.
        fill_prices: {ticker: expected_fill_price} for trade execution.
        is_first_run: Whether this is the system's first run.

    Returns:
        {
            "target_weights": {ticker: weight},
            "trade_list": [trade dicts],
            "computed_target_id": int,
        }
    """
    constraints = get_constraints(conn)
    account = get_account(conn)
    current_holdings = _get_current_holdings(conn)

    # Step 1: Map conviction sub-scores to numeric weights
    actions: dict[str, str] = {}
    conviction_weights: dict[str, float] = {}

    for ticker, data in agent_c_output.items():
        action = data["action"]
        actions[ticker] = action

        if action in ("Buy", "Hold"):
            weight = get_conviction_weight(data["conviction"], is_first_run)
            conviction_weights[ticker] = weight

    # Step 2: Normalize to target weights
    target_weights = normalize_weights(conviction_weights, constraints["cash_floor"])

    # Step 3: Enforce constraints
    actionable_tickers = list(target_weights.keys())
    sectors = _get_sector_map(conn, actionable_tickers)
    target_weights = enforce_constraints(target_weights, sectors, constraints)

    # Compute total portfolio value for trade sizing
    cash_balance = account["cash_balance"]
    holdings_value = sum(
        h["shares"] * fill_prices.get(t, 0.0)
        for t, h in current_holdings.items()
    )
    total_value = cash_balance + holdings_value

    # Step 4: Compute trade list
    trade_list = compute_trade_list(
        target_weights=target_weights,
        current_holdings=current_holdings,
        total_value=total_value,
        fill_prices=fill_prices,
        actions=actions,
        cash_floor=constraints["cash_floor"],
        cash_balance=cash_balance,
    )

    # Step 5: Store computed targets
    per_ticker_data = {}
    for ticker in set(list(target_weights.keys()) + list(actions.keys())):
        per_ticker_data[ticker] = {
            "target_weight": target_weights.get(ticker, 0.0),
            "conviction_weight": conviction_weights.get(ticker),
            "action": actions.get(ticker),
        }

    target_id = store_computed_targets(conn, run_id, per_ticker_data)

    # Execute trades: update holdings and cost basis
    for trade in trade_list:
        ticker = trade["ticker"]
        sector = sectors.get(ticker, "")
        # For new positions, get sector from watchlist via sector_map
        if not sector:
            row = conn.execute(
                "SELECT sector FROM watchlist WHERE ticker = ?", (ticker,)
            ).fetchone()
            if row:
                sector = row["sector"]

        realized_pnl = update_cost_basis(
            conn, ticker, trade["action"], trade["shares"], trade["fill_price"], sector
        )
        trade["realized_pnl"] = realized_pnl

    # Update account cash balance
    net_cash_change = sum(
        t["dollar_amount"] if t["action"] in ("Trim", "Exit") else -t["dollar_amount"]
        for t in trade_list
    )
    new_cash = cash_balance + net_cash_change
    conn.execute(
        "UPDATE account SET cash_balance = ? WHERE account_id = 1",
        (new_cash,),
    )
    conn.commit()

    return {
        "target_weights": target_weights,
        "trade_list": trade_list,
        "computed_target_id": target_id,
    }

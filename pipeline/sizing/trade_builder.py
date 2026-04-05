"""Trade list computation and cost basis accounting.

Computes the concrete share deltas between current holdings and target weights,
handles cost basis via weighted average method, and stores computed targets.
"""

import json
import sqlite3
from datetime import datetime, timezone


def compute_trade_list(
    target_weights: dict[str, float],
    current_holdings: dict[str, dict],
    total_value: float,
    fill_prices: dict[str, float],
    actions: dict[str, str],
    cash_floor: float,
    cash_balance: float,
) -> list[dict]:
    """Compute trade list from target weights vs current holdings.

    Args:
        target_weights: {ticker: target_weight} after constraint enforcement.
        current_holdings: {ticker: {shares, cost_basis_per_share, sector}}.
        total_value: Current total portfolio value.
        fill_prices: {ticker: expected_fill_price}.
        actions: {ticker: action} from Agent C (Buy/Hold/Trim/Exit).
        cash_floor: Minimum cash fraction.
        cash_balance: Current cash balance.

    Returns:
        List of trade dicts: {ticker, action, shares, dollar_amount, fill_price}.

    Raises:
        ValueError: If trades would violate cash floor.
    """
    trades: list[dict] = []

    # Process Buy/Hold tickers (from target weights)
    for ticker, target_weight in target_weights.items():
        action = actions.get(ticker, "Hold")
        target_dollars = target_weight * total_value
        fill_price = fill_prices[ticker]

        current = current_holdings.get(ticker)
        current_dollars = 0.0
        if current:
            current_dollars = current["shares"] * fill_price

        delta_dollars = target_dollars - current_dollars
        delta_shares = delta_dollars / fill_price if fill_price > 0 else 0.0

        if abs(delta_shares) < 0.001:
            continue

        trades.append({
            "ticker": ticker,
            "action": "Buy" if delta_shares > 0 else "Trim",
            "shares": abs(delta_shares),
            "dollar_amount": abs(delta_dollars),
            "fill_price": fill_price,
        })

    # Process Trim tickers not in target weights (partial reduce)
    for ticker, action in actions.items():
        if action == "Trim" and ticker not in target_weights:
            current = current_holdings.get(ticker)
            if current and current["shares"] > 0:
                fill_price = fill_prices[ticker]
                trades.append({
                    "ticker": ticker,
                    "action": "Trim",
                    "shares": current["shares"],
                    "dollar_amount": current["shares"] * fill_price,
                    "fill_price": fill_price,
                })

    # Process Exit tickers (full liquidation)
    for ticker, action in actions.items():
        if action == "Exit":
            current = current_holdings.get(ticker)
            if current and current["shares"] > 0:
                fill_price = fill_prices[ticker]
                trades.append({
                    "ticker": ticker,
                    "action": "Exit",
                    "shares": current["shares"],
                    "dollar_amount": current["shares"] * fill_price,
                    "fill_price": fill_price,
                })

    # Verify cash floor
    net_cash_change = sum(
        t["dollar_amount"] if t["action"] in ("Trim", "Exit") else -t["dollar_amount"]
        for t in trades
    )
    projected_cash = cash_balance + net_cash_change
    min_cash = cash_floor * total_value

    if projected_cash < min_cash - 0.01:  # small tolerance for float math
        raise ValueError(
            f"Trades would violate cash floor: projected cash ${projected_cash:.2f} "
            f"< minimum ${min_cash:.2f}"
        )

    return trades


def update_cost_basis(
    conn: sqlite3.Connection,
    ticker: str,
    action: str,
    shares: float,
    fill_price: float,
    sector: str = "",
) -> float | None:
    """Update holdings and cost basis after a trade.

    Args:
        conn: SQLite connection.
        ticker: Ticker symbol.
        action: 'Buy', 'Trim', or 'Exit'.
        shares: Number of shares traded (always positive).
        fill_price: Price per share.
        sector: GICS sector (required for new Buy positions).

    Returns:
        Realized P&L for Trim/Exit trades, None for Buy trades.
    """
    row = conn.execute(
        "SELECT shares, cost_basis_per_share, sector FROM holdings WHERE ticker = ?",
        (ticker,),
    ).fetchone()

    if action == "Buy":
        if row is None:
            # New position
            conn.execute(
                "INSERT INTO holdings (ticker, shares, cost_basis_per_share, sector) "
                "VALUES (?, ?, ?, ?)",
                (ticker, shares, fill_price, sector),
            )
        else:
            # Add to existing — weighted average cost basis
            old_shares = row["shares"]
            old_basis = row["cost_basis_per_share"]
            new_total = old_shares + shares
            new_basis = (old_shares * old_basis + shares * fill_price) / new_total
            conn.execute(
                "UPDATE holdings SET shares = ?, cost_basis_per_share = ? WHERE ticker = ?",
                (new_total, new_basis, ticker),
            )
        return None

    elif action == "Trim":
        if row is None:
            raise ValueError(f"Cannot trim {ticker}: no existing position")
        old_shares = row["shares"]
        cost_basis = row["cost_basis_per_share"]
        remaining = old_shares - shares
        if remaining < 0.001:
            # Effectively an exit
            conn.execute("DELETE FROM holdings WHERE ticker = ?", (ticker,))
            return (fill_price - cost_basis) * old_shares
        else:
            # Cost basis unchanged on trim
            conn.execute(
                "UPDATE holdings SET shares = ? WHERE ticker = ?",
                (remaining, ticker),
            )
            return (fill_price - cost_basis) * shares

    elif action == "Exit":
        if row is None:
            raise ValueError(f"Cannot exit {ticker}: no existing position")
        old_shares = row["shares"]
        cost_basis = row["cost_basis_per_share"]
        conn.execute("DELETE FROM holdings WHERE ticker = ?", (ticker,))
        return (fill_price - cost_basis) * old_shares

    else:
        raise ValueError(f"Unknown action: {action}")


def store_computed_targets(
    conn: sqlite3.Connection,
    run_id: int,
    per_ticker_data: dict,
) -> int:
    """Store target allocation to computed_targets table.

    Args:
        conn: SQLite connection.
        run_id: Run ID to associate with.
        per_ticker_data: {ticker: {target_weight, conviction_weight, action}}.

    Returns:
        The target_id of the inserted row.
    """
    timestamp = datetime.now(timezone.utc).isoformat()
    cursor = conn.execute(
        "INSERT INTO computed_targets (run_id, timestamp, per_ticker_json) VALUES (?, ?, ?)",
        (run_id, timestamp, json.dumps(per_ticker_data)),
    )
    return cursor.lastrowid

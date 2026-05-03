"""Trade list computation and cost basis accounting.

Computes the concrete share deltas between current holdings and target weights.
Trade actions are derived from the delta:
  - Ticker in target_weights: Buy (delta > 0) or Trim (delta < 0)
  - Held ticker not in target_weights: full Exit (raw_score was ≤ 0)
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
        actions: {ticker: action} from Agent C (retained for UX labels only).
        cash_floor: Minimum cash fraction.
        cash_balance: Current cash balance.

    Returns:
        List of trade dicts: {ticker, action, shares, dollar_amount, fill_price}.

    Raises:
        ValueError: If trades would violate cash floor.
    """
    trades: list[dict] = []

    # Tickers with a target weight: size by delta vs current holding.
    for ticker, target_weight in target_weights.items():
        fill_price = fill_prices.get(ticker, 0.0)
        if fill_price <= 0:
            continue

        target_dollars = target_weight * total_value
        current = current_holdings.get(ticker)
        current_dollars = current["shares"] * fill_price if current else 0.0
        delta_dollars = target_dollars - current_dollars
        delta_shares = delta_dollars / fill_price

        if abs(delta_shares) < 0.001:
            continue

        trades.append({
            "ticker": ticker,
            "action": "Buy" if delta_shares > 0 else "Trim",
            "shares": abs(delta_shares),
            "dollar_amount": abs(delta_dollars),
            "fill_price": fill_price,
        })

    # Held tickers not in target_weights: full liquidation.
    # This covers any holding whose raw_score was ≤ 0 (Agent B / Agent A signal gone).
    for ticker, holding in current_holdings.items():
        if ticker in target_weights or holding["shares"] <= 0:
            continue
        fill_price = fill_prices.get(ticker, 0.0)
        if fill_price <= 0:
            continue
        trades.append({
            "ticker": ticker,
            "action": "Exit",
            "shares": holding["shares"],
            "dollar_amount": holding["shares"] * fill_price,
            "fill_price": fill_price,
        })

    # Verify cash floor.
    net_cash_change = sum(
        t["dollar_amount"] if t["action"] in ("Trim", "Exit") else -t["dollar_amount"]
        for t in trades
    )
    projected_cash = cash_balance + net_cash_change
    min_cash = cash_floor * total_value

    if projected_cash < min_cash - 0.01:
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
            conn.execute(
                "INSERT INTO holdings (ticker, shares, cost_basis_per_share, sector) "
                "VALUES (?, ?, ?, ?)",
                (ticker, shares, fill_price, sector),
            )
        else:
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
            conn.execute("DELETE FROM holdings WHERE ticker = ?", (ticker,))
            return (fill_price - cost_basis) * old_shares
        else:
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


def apply_trade(
    conn: sqlite3.Connection,
    ticker: str,
    action: str,
    shares: float,
    fill_price: float,
    sector: str = "",
) -> float | None:
    """Apply a single trade: update holdings/cost-basis and adjust cash balance atomically.

    Returns realized P&L for Trim/Exit, None for Buy. Does not commit.
    """
    realized_pnl = update_cost_basis(conn, ticker, action, shares, fill_price, sector)
    dollar_amount = shares * fill_price
    if action in ("Trim", "Exit"):
        conn.execute(
            "UPDATE account SET cash_balance = cash_balance + ? WHERE account_id = 1",
            (dollar_amount,),
        )
    else:
        conn.execute(
            "UPDATE account SET cash_balance = cash_balance - ? WHERE account_id = 1",
            (dollar_amount,),
        )
    return realized_pnl


def store_computed_targets(
    conn: sqlite3.Connection,
    run_id: int,
    per_ticker_data: dict,
) -> int:
    """Store target allocation to computed_targets table."""
    timestamp = datetime.now(timezone.utc).isoformat()
    cursor = conn.execute(
        "INSERT INTO computed_targets (run_id, timestamp, per_ticker_json) VALUES (?, ?, ?)",
        (run_id, timestamp, json.dumps(per_ticker_data)),
    )
    return cursor.lastrowid

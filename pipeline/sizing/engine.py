"""Position Sizing Engine — orchestrates the full sizing pipeline.

Allocation model:
  raw_score = agent_b.target_weight × narrative_multiplier(agent_c.narrative_score)

A zero Agent B weight always produces a zero score regardless of narrative_score.
Agent C action is retained for UX/logging only.
"""

import logging
import sqlite3
from typing import Any

logger = logging.getLogger(__name__)

from pipeline.db.helpers import get_account, get_constraints
from pipeline.sizing.conviction_map import narrative_multiplier
from pipeline.sizing.normalizer import enforce_constraints
from pipeline.sizing.trade_builder import (
    apply_trade,
    compute_trade_list,
    store_computed_targets,
    update_cost_basis,
)


def _get_current_holdings(conn: sqlite3.Connection) -> dict[str, dict]:
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
    sector_map: dict[str, str] = {}
    rows = conn.execute("SELECT ticker, sector FROM watchlist").fetchall()
    for row in rows:
        sector_map[row["ticker"]] = row["sector"]
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
    agent_b_output: dict | None = None,
    agent_a_output: dict | None = None,
    mutate_portfolio: bool = False,
) -> dict:
    """Run the full Position Sizing Engine pipeline.

    Args:
        conn: SQLite connection.
        run_id: Current run ID.
        agent_c_output: Agent C's per_ticker output (action, conviction, rationale).
            Action field is retained for UX labels only; weights come from scores.
        fill_prices: {ticker: expected_fill_price}.
        is_first_run: Whether this is the system's first run (Agent B skipped).
        agent_b_output: Agent B output dict with per_ticker target_weights.
        agent_a_output: Agent A output dict with per_ticker sentiment/confidence.
        mutate_portfolio: If True, execute trades against the DB immediately.

    Returns:
        {target_weights, trade_list, computed_target_id, per_ticker_data}.
    """
    constraints = get_constraints(conn)
    account = get_account(conn)
    current_holdings = _get_current_holdings(conn)

    # Inject implicit Hold for any held ticker Agent C omitted.
    agent_c_output = dict(agent_c_output)
    omitted = [t for t in current_holdings if t not in agent_c_output]
    if omitted:
        logger.warning("Agent C omitted held tickers — applying implicit Hold: %s", omitted)
        for ticker in omitted:
            agent_c_output[ticker] = {
                "action": "Hold",
                "narrative_score": 5.0,
                "rationale": "Implicit Hold: Agent C omitted this held ticker.",
                "key_risk_factors": [],
                "_implicit_hold": True,
            }

    b_per_ticker = (agent_b_output or {}).get("per_ticker", {})

    actions: dict[str, str] = {}
    raw_scores: dict[str, float] = {}

    for ticker, data in agent_c_output.items():
        actions[ticker] = data["action"]
        b_weight = b_per_ticker.get(ticker, {}).get("target_weight", 0.0)
        narrative_score = float(data.get("narrative_score", 5.0))
        raw_scores[ticker] = narrative_multiplier(narrative_score) * b_weight

    # Resolve sector map before constraint enforcement.
    all_tickers = list(raw_scores.keys())
    sectors = _get_sector_map(conn, all_tickers)
    unknown = [t for t in raw_scores if t not in sectors]
    if unknown:
        logger.warning("Dropping tickers with no sector mapping: %s", unknown)
        for t in unknown:
            del raw_scores[t]
            actions.pop(t, None)

    target_weights = enforce_constraints(raw_scores, sectors, constraints)

    # Observability
    n_positive = sum(1 for s in raw_scores.values() if s > 0)
    invested = sum(target_weights.values())
    target_invested = 1.0 - constraints["cash_floor"]
    effective_cash = 1.0 - invested
    logger.info(
        "Sizing: n_scored=%d n_positive=%d invested=%.4f target=%.4f effective_cash=%.4f",
        len(raw_scores), n_positive, invested, target_invested, effective_cash,
    )
    if invested < target_invested - 0.02:
        logger.warning(
            "Under-invested by %.1f%% — conviction breadth too low to fill target allocation",
            (target_invested - invested) * 100,
        )

    cash_balance = account["cash_balance"]
    holdings_value = sum(
        h["shares"] * fill_prices.get(t, 0.0)
        for t, h in current_holdings.items()
    )
    total_value = cash_balance + holdings_value

    trade_list = compute_trade_list(
        target_weights=target_weights,
        current_holdings=current_holdings,
        total_value=total_value,
        fill_prices=fill_prices,
        actions=actions,
        cash_floor=constraints["cash_floor"],
        cash_balance=cash_balance,
    )

    per_ticker_data = {}
    for ticker in set(list(target_weights.keys()) + list(actions.keys())):
        per_ticker_data[ticker] = {
            "target_weight": target_weights.get(ticker, 0.0),
            "raw_score": raw_scores.get(ticker, 0.0),
            "action": actions.get(ticker),
        }

    target_id = store_computed_targets(conn, run_id, per_ticker_data)

    if mutate_portfolio:
        for trade in trade_list:
            ticker = trade["ticker"]
            sector = sectors.get(ticker, "")
            if not sector:
                row = conn.execute(
                    "SELECT sector FROM watchlist WHERE ticker = ?", (ticker,)
                ).fetchone()
                if row:
                    sector = row["sector"]
            trade["realized_pnl"] = apply_trade(
                conn, ticker, trade["action"], trade["shares"], trade["fill_price"], sector
            )
        conn.commit()

    return {
        "target_weights": target_weights,
        "trade_list": trade_list,
        "computed_target_id": target_id,
        "per_ticker_data": per_ticker_data,
    }

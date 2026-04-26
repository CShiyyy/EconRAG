"""Position Sizing Engine — orchestrates the full sizing pipeline.

Allocation model:
  raw_score = agent_b.target_weight × tilt_factor(agent_a.sentiment, agent_a.confidence)
  First-run fallback (no Agent B): FIRST_RUN_TABLE[narrative_alignment] × tilt_factor(...)

The raw scores feed directly into enforce_constraints which handles proportional
allocation, single-position cap (15%), sector cap, and dust floor.

Agent C's categorical action is retained for UX/logging only. Trade generation
is driven by the delta between target_weights and current holdings.
"""

import logging
import sqlite3
from typing import Any

logger = logging.getLogger(__name__)

from pipeline.db.helpers import get_account, get_constraints
from pipeline.sizing.conviction_map import narrative_multiplier

# ---------------------------------------------------------------------------
# Legacy shims — kept here until engine.py is refactored in Task 2.
# These replicate the old conviction_map behaviour so existing engine logic
# continues to work without modification during the transition.
# ---------------------------------------------------------------------------
_SENTIMENT_TILT: dict[tuple[str, str], float] = {
    ("bullish",  "strong"):   1.30,
    ("bullish",  "moderate"): 1.20,
    ("bullish",  "weak"):     1.10,
    ("neutral",  "strong"):   1.00,
    ("neutral",  "moderate"): 1.00,
    ("neutral",  "weak"):     1.00,
    ("bearish",  "weak"):     0.90,
    ("bearish",  "moderate"): 0.80,
    ("bearish",  "strong"):   0.70,
}

FIRST_RUN_TABLE: dict[str, float] = {
    "very_strong": 0.90,
    "strong":      0.70,
    "moderate":    0.45,
    "weak":        0.25,
    "very_weak":   0.10,
}


def tilt_factor(sentiment: str, confidence: str) -> float:
    return _SENTIMENT_TILT.get((sentiment, confidence), 1.0)


def raw_allocation_score(b_weight: float, a_sentiment: str, a_confidence: str) -> float:
    return max(0.0, b_weight * tilt_factor(a_sentiment, a_confidence))
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
                "conviction": {
                    "narrative_alignment": "moderate",
                    "narrative_confidence": 5.0,
                    "quant_support": "n/a",
                    "quant_confidence": 0.0,
                    "signal_agreement": "n/a",
                    "signal_confidence": 0.0,
                },
                "rationale": "Implicit Hold: Agent C omitted this held ticker.",
                "key_risk_factors": [],
                "_implicit_hold": True,
            }

    b_per_ticker = ({} if is_first_run else (agent_b_output or {})).get("per_ticker", {})
    a_per_ticker = (agent_a_output or {}).get("per_ticker", {})

    actions: dict[str, str] = {}
    raw_scores: dict[str, float] = {}

    for ticker, data in agent_c_output.items():
        actions[ticker] = data["action"]

        a_data = a_per_ticker.get(ticker, {})
        sentiment = a_data.get("sentiment", "neutral")
        confidence = a_data.get("confidence", "weak")

        b_data = b_per_ticker.get(ticker, {})
        b_weight = b_data.get("target_weight")

        if b_weight is not None and b_weight > 0:
            score = raw_allocation_score(b_weight, sentiment, confidence)
        else:
            # First run or ticker not covered by Agent B: use narrative table.
            na = data["conviction"].get("narrative_alignment", "weak")
            base = FIRST_RUN_TABLE.get(na, 0.25)
            score = max(0.0, base * tilt_factor(sentiment, confidence))

        raw_scores[ticker] = score

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

"""Agent B — Deterministic Quant Script.

Pure Python/numpy agent that computes portfolio health metrics,
detects constraint violations, and injects correlation edges into
the knowledge graph. No LLM calls.
"""

from __future__ import annotations

import math
import sqlite3
from itertools import combinations

import numpy as np
from lightrag import LightRAG

from pipeline.config import (
    DRIFT_BREACH_THRESHOLD,
    DRIFT_WARNING_THRESHOLD,
    SECTOR_WARNING_BUFFER,
    VOLATILITY_BREACH_MULTIPLIER,
    VOLATILITY_WARNING_MULTIPLIER,
)
from pipeline.db.helpers import (
    get_account,
    get_constraints,
    get_derived_weights,
    get_portfolio_value,
    get_previous_computed_target,
    is_first_run,
    store_agent_output,
)
from pipeline.ingestion.models import MarketDataPoint
from pipeline.knowledge.graph_ops import inject_correlation_edges


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_drift(
    current_weights: dict[str, float],
    previous_targets: dict[str, dict],
) -> dict[str, float]:
    """Per-ticker drift: current_weight - previous_target_weight."""
    drift: dict[str, float] = {}
    all_tickers = set(current_weights) | set(previous_targets)
    for ticker in all_tickers:
        cw = current_weights.get(ticker, 0.0)
        tw = previous_targets.get(ticker, {}).get("target_weight", 0.0)
        drift[ticker] = cw - tw
    return drift


def _compute_volatility(
    price_histories: dict[str, list[float]],
) -> dict[str, float]:
    """Per-ticker annualized volatility from 30-day closing prices.

    vol = std(log_returns) * sqrt(252)
    """
    vols: dict[str, float] = {}
    for ticker, prices in price_histories.items():
        if len(prices) < 2:
            vols[ticker] = 0.0
            continue
        arr = np.array(prices, dtype=float)
        log_returns = np.diff(np.log(arr))
        vols[ticker] = float(np.std(log_returns, ddof=1) * math.sqrt(252))
    return vols


def _compute_portfolio_volatility(
    per_ticker_vol: dict[str, float],
    current_weights: dict[str, float],
) -> float:
    """Weighted average of individual volatilities (simplified)."""
    total = 0.0
    for ticker, vol in per_ticker_vol.items():
        w = current_weights.get(ticker, 0.0)
        total += w * vol
    return total


def _compute_sector_concentrations(
    holdings_rows: list[dict],
    current_weights: dict[str, float],
    sector_limit: float,
) -> dict[str, dict]:
    """Sum weights per sector and classify status."""
    sector_weights: dict[str, float] = {}
    for h in holdings_rows:
        ticker = h["ticker"]
        sector = h["sector"]
        w = current_weights.get(ticker, 0.0)
        sector_weights[sector] = sector_weights.get(sector, 0.0) + w

    result: dict[str, dict] = {}
    for sector, weight in sector_weights.items():
        if weight > sector_limit:
            status = "breach"
        elif weight > sector_limit - SECTOR_WARNING_BUFFER:
            status = "warning"
        else:
            status = "normal"
        result[sector] = {
            "weight": round(weight, 6),
            "limit": sector_limit,
            "status": status,
        }
    return result


def _compute_max_drawdown(conn: sqlite3.Connection) -> float:
    """Peak-to-trough drawdown from snapshot total_value series.

    Returns a negative number (e.g., -0.04 for 4% drawdown), or 0.0 if
    fewer than 2 snapshots.
    """
    rows = conn.execute(
        "SELECT total_value FROM snapshots ORDER BY snapshot_id ASC"
    ).fetchall()
    if len(rows) < 2:
        return 0.0

    values = [r["total_value"] for r in rows]
    peak = values[0]
    max_dd = 0.0
    for v in values[1:]:
        if v > peak:
            peak = v
        dd = (v - peak) / peak
        if dd < max_dd:
            max_dd = dd
    return max_dd


def _compute_health_scores(
    drift: dict[str, float],
    per_ticker_vol: dict[str, float],
    current_weights: dict[str, float],
) -> dict[str, str]:
    """Classify each held ticker as normal/warning/breach.

    Uses drift thresholds and volatility relative to median vol.
    """
    # Only score tickers that are currently held
    held_tickers = {t for t, w in current_weights.items() if w > 0}
    held_vols = [per_ticker_vol.get(t, 0.0) for t in held_tickers]
    median_vol = float(np.median(held_vols)) if held_vols else 0.0

    scores: dict[str, str] = {}
    for ticker in held_tickers:
        d = abs(drift.get(ticker, 0.0))
        v = per_ticker_vol.get(ticker, 0.0)

        drift_status = "normal"
        if d > DRIFT_BREACH_THRESHOLD:
            drift_status = "breach"
        elif d > DRIFT_WARNING_THRESHOLD:
            drift_status = "warning"

        vol_status = "normal"
        if median_vol > 0:
            if v > VOLATILITY_BREACH_MULTIPLIER * median_vol:
                vol_status = "breach"
            elif v > VOLATILITY_WARNING_MULTIPLIER * median_vol:
                vol_status = "warning"

        # Worst of drift and vol
        if drift_status == "breach" or vol_status == "breach":
            scores[ticker] = "breach"
        elif drift_status == "warning" or vol_status == "warning":
            scores[ticker] = "warning"
        else:
            scores[ticker] = "normal"
    return scores


def _build_flags(
    ticker: str,
    drift: dict[str, float],
    per_ticker_vol: dict[str, float],
    median_vol: float,
) -> list[str]:
    """Build human-readable flag list for a ticker."""
    flags: list[str] = []
    d = abs(drift.get(ticker, 0.0))
    v = per_ticker_vol.get(ticker, 0.0)

    if d > DRIFT_BREACH_THRESHOLD:
        flags.append("drift_breach")
    elif d > DRIFT_WARNING_THRESHOLD:
        flags.append("drift_above_threshold")

    if median_vol > 0:
        if v > VOLATILITY_BREACH_MULTIPLIER * median_vol:
            flags.append("volatility_breach")
        elif v > VOLATILITY_WARNING_MULTIPLIER * median_vol:
            flags.append("volatility_above_threshold")

    return flags


def _detect_constraint_violations(
    current_weights: dict[str, float],
    cash_pct: float,
    sector_concentrations: dict[str, dict],
    constraints: dict[str, float],
) -> list[dict]:
    """Check all 4 constraints, return list of violation dicts."""
    violations: list[dict] = []

    # cash_floor
    cash_floor = constraints.get("cash_floor", 0.05)
    if cash_pct < cash_floor:
        violations.append({
            "constraint": "cash_floor",
            "limit": cash_floor,
            "actual": round(cash_pct, 6),
            "detail": f"Cash {cash_pct:.4f} below floor {cash_floor}",
        })

    # max_single_position
    max_pos = constraints.get("max_single_position", 0.15)
    for ticker, w in current_weights.items():
        if w > max_pos:
            violations.append({
                "constraint": "max_single_position",
                "ticker": ticker,
                "limit": max_pos,
                "actual": round(w, 6),
                "detail": f"{ticker} weight {w:.4f} exceeds limit {max_pos}",
            })

    # max_sector_concentration
    max_sector = constraints.get("max_sector_concentration", 0.35)
    for sector, info in sector_concentrations.items():
        if info["weight"] > max_sector:
            violations.append({
                "constraint": "max_sector_concentration",
                "sector": sector,
                "limit": max_sector,
                "actual": info["weight"],
                "detail": f"Sector {sector} weight {info['weight']:.4f} exceeds limit {max_sector}",
            })

    # min_position_size
    min_pos = constraints.get("min_position_size", 0.02)
    for ticker, w in current_weights.items():
        if 0 < w < min_pos:
            violations.append({
                "constraint": "min_position_size",
                "ticker": ticker,
                "limit": min_pos,
                "actual": round(w, 6),
                "detail": f"{ticker} weight {w:.4f} below min {min_pos}",
            })

    return violations


def _compute_correlations(
    price_histories: dict[str, list[float]],
) -> list[dict]:
    """Pairwise Pearson correlation from price histories."""
    tickers = sorted(price_histories.keys())
    correlations: list[dict] = []

    for t_a, t_b in combinations(tickers, 2):
        prices_a = price_histories[t_a]
        prices_b = price_histories[t_b]
        # Need at least 3 prices (2 returns) for meaningful correlation
        min_len = min(len(prices_a), len(prices_b))
        if min_len < 3:
            continue
        # Align to same length
        a = np.array(prices_a[:min_len], dtype=float)
        b = np.array(prices_b[:min_len], dtype=float)
        returns_a = np.diff(np.log(a))
        returns_b = np.diff(np.log(b))
        # Skip if zero variance
        if np.std(returns_a) == 0 or np.std(returns_b) == 0:
            continue
        corr = float(np.corrcoef(returns_a, returns_b)[0, 1])
        if np.isnan(corr):
            continue
        correlations.append({
            "ticker_a": t_a,
            "ticker_b": t_b,
            "correlation": round(corr, 4),
        })

    return correlations


def _determine_overall_status(
    health_scores: dict[str, str],
    constraint_violations: list[dict],
) -> str:
    """Overall portfolio status: breach > warning > normal."""
    if constraint_violations:
        return "breach"
    statuses = set(health_scores.values())
    if "breach" in statuses:
        return "breach"
    if "warning" in statuses:
        return "warning"
    return "normal"


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def run_agent_b(
    conn: sqlite3.Connection,
    rag: LightRAG,
    market_data: dict[str, MarketDataPoint],
    run_id: int,
) -> dict | None:
    """Run Agent B deterministic quant analysis.

    Returns output dict matching the Agent B schema, or None on first run.
    """
    # 1. First-run guard
    if is_first_run(conn):
        return None

    # 2. Data assembly
    def price_fn(ticker: str) -> float:
        return market_data[ticker].current_price

    holdings_rows = [
        dict(r) for r in conn.execute("SELECT * FROM holdings").fetchall()
    ]
    held_tickers = {h["ticker"] for h in holdings_rows}

    current_weights = get_derived_weights(conn, price_fn)
    total_value = get_portfolio_value(conn, price_fn)
    account = get_account(conn)
    cash_pct = account["cash_balance"] / total_value if total_value > 0 else 1.0
    constraints = get_constraints(conn)
    previous_target = get_previous_computed_target(conn)
    previous_targets = previous_target["per_ticker_json"] if previous_target else {}

    # Build price histories for held tickers
    price_histories: dict[str, list[float]] = {}
    for ticker in held_tickers:
        if ticker in market_data:
            price_histories[ticker] = market_data[ticker].price_history_30d

    # 3-6. Compute metrics
    drift = _compute_drift(current_weights, previous_targets)
    per_ticker_vol = _compute_volatility(price_histories)
    portfolio_vol = _compute_portfolio_volatility(per_ticker_vol, current_weights)
    sector_limit = constraints.get("max_sector_concentration", 0.35)
    sector_concentrations = _compute_sector_concentrations(
        holdings_rows, current_weights, sector_limit
    )
    max_drawdown = _compute_max_drawdown(conn)

    # 7. Health scores
    health_scores = _compute_health_scores(drift, per_ticker_vol, current_weights)

    # 8. Constraint violations
    constraint_violations = _detect_constraint_violations(
        current_weights, cash_pct, sector_concentrations, constraints
    )

    # 9. Correlations
    correlations = _compute_correlations(price_histories)

    # 10. Inject correlation edges
    await inject_correlation_edges(rag, correlations, run_id)

    # Build flags and per-ticker detail
    held_vols = [per_ticker_vol.get(t, 0.0) for t in held_tickers]
    median_vol = float(np.median(held_vols)) if held_vols else 0.0

    per_ticker_output: dict[str, dict] = {}
    for h in holdings_rows:
        ticker = h["ticker"]
        per_ticker_output[ticker] = {
            "current_weight": round(current_weights.get(ticker, 0.0), 6),
            "previous_target_weight": round(
                previous_targets.get(ticker, {}).get("target_weight", 0.0), 6
            ),
            "drift": round(drift.get(ticker, 0.0), 6),
            "volatility_30d": round(per_ticker_vol.get(ticker, 0.0), 6),
            "sector": h["sector"],
            "health_score": health_scores.get(ticker, "normal"),
            "flags": _build_flags(ticker, drift, per_ticker_vol, median_vol),
        }

    # 11. Assemble output
    overall_status = _determine_overall_status(health_scores, constraint_violations)

    output = {
        "portfolio_level": {
            "total_value": round(total_value, 2),
            "cash_pct": round(cash_pct, 6),
            "portfolio_volatility_30d": round(portfolio_vol, 6),
            "max_drawdown_30d": round(max_drawdown, 6),
            "overall_status": overall_status,
        },
        "per_ticker": per_ticker_output,
        "sector_concentrations": sector_concentrations,
        "constraint_violations": constraint_violations,
    }

    # 12. Store
    store_agent_output(conn, run_id, "B", output)

    # 13. Return
    return output

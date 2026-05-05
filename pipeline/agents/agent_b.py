"""Agent B — Second Tower Strategy (v0.2).

Replaces the v0.1 deterministic health-checker with the Second Tower:
a signal-adjusted mean-variance portfolio optimizer using 63 cross-sectional
factors (Momentum/Value/Quality/Technical/Risk/Accruals), Ledoit-Wolf
shrinkage covariance, and CVXPY mean-variance optimization.

The strategy code in pipeline.agents.second_tower is ported verbatim from
TradeModem.  This module is the only bespoke layer: it loads data from the
SQLite cache, calls _compute_rebal_weights for the latest date (single-date
live mode), and converts the resulting weight vector into the per-ticker dict
that Agent C and conditions.py expect.
"""
from __future__ import annotations

import math
import sqlite3
from itertools import combinations

import numpy as np
import pandas as pd
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
    get_watchlist,
    load_edgar_cache,
    load_ohlcv_cache,
    store_agent_output,
)
from pipeline.ingestion.edgar import refresh_edgar_cache
from pipeline.ingestion.market_history import refresh_ohlcv_cache
from pipeline.ingestion.models import MarketDataPoint
from pipeline.knowledge.graph_ops import inject_correlation_edges
from pipeline.agents.second_tower.strategy import (
    RISK_CONFIG,
    _compute_rebal_weights,
    _shrinkage_cov,
)
from pipeline.agents.second_tower.universes import get_sector_map

_LOOKBACK_DAYS = int(3 * 365 * 1.3) + 90   # ~Second Tower 3-year warm-up
_TRAIN_DAYS = 3 * 252
_ST_CONFIG = {
    "use_regime_detection": False,
    "use_mc_sl": False,
}


# ---------------------------------------------------------------------------
# v0.1 helpers preserved verbatim — they still handle sectors/violations/drawdown
# ---------------------------------------------------------------------------

def _compute_sector_concentrations(
    holdings_rows: list[dict],
    current_weights: dict[str, float],
    sector_limit: float,
) -> dict[str, dict]:
    sector_weights: dict[str, float] = {}
    for h in holdings_rows:
        w = current_weights.get(h["ticker"], 0.0)
        sector_weights[h["sector"]] = sector_weights.get(h["sector"], 0.0) + w
    result: dict[str, dict] = {}
    for sector, weight in sector_weights.items():
        if weight > sector_limit:
            status = "breach"
        elif weight > sector_limit - SECTOR_WARNING_BUFFER:
            status = "warning"
        else:
            status = "normal"
        result[sector] = {"weight": round(weight, 6), "limit": sector_limit, "status": status}
    return result


def _detect_constraint_violations(
    current_weights: dict[str, float],
    cash_pct: float,
    sector_concentrations: dict[str, dict],
    constraints: dict[str, float],
) -> list[dict]:
    violations: list[dict] = []
    cash_floor = constraints.get("cash_floor", 0.05)
    if cash_pct < cash_floor:
        violations.append({
            "constraint": "cash_floor", "limit": cash_floor,
            "actual": round(cash_pct, 6),
            "detail": f"Cash {cash_pct:.4f} below floor {cash_floor}",
        })
    max_pos = constraints.get("max_single_position", 0.15)
    for ticker, w in current_weights.items():
        if w > max_pos:
            violations.append({
                "constraint": "max_single_position", "ticker": ticker,
                "limit": max_pos, "actual": round(w, 6),
                "detail": f"{ticker} weight {w:.4f} exceeds limit {max_pos}",
            })
    max_sector = constraints.get("max_sector_concentration", 0.35)
    for sector, info in sector_concentrations.items():
        if info["weight"] > max_sector:
            violations.append({
                "constraint": "max_sector_concentration", "sector": sector,
                "limit": max_sector, "actual": info["weight"],
                "detail": f"Sector {sector} weight {info['weight']:.4f} exceeds limit {max_sector}",
            })
    min_pos = constraints.get("min_position_size", 0.02)
    for ticker, w in current_weights.items():
        if 0 < w < min_pos:
            violations.append({
                "constraint": "min_position_size", "ticker": ticker,
                "limit": min_pos, "actual": round(w, 6),
                "detail": f"{ticker} weight {w:.4f} below min {min_pos}",
            })
    return violations


def _compute_max_drawdown(conn: sqlite3.Connection) -> float:
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


def _determine_overall_status(
    health_scores: dict[str, str],
    constraint_violations: list[dict],
) -> str:
    if constraint_violations:
        return "breach"
    statuses = set(health_scores.values())
    if "breach" in statuses:
        return "breach"
    if "warning" in statuses:
        return "warning"
    return "normal"


def _vol_30d(ohlcv_df: pd.DataFrame) -> float:
    """Annualized 30-day log-return volatility from Close prices."""
    if ohlcv_df is None or ohlcv_df.empty or len(ohlcv_df) < 2:
        return 0.0
    close = ohlcv_df["Close"].dropna().tail(31)
    if len(close) < 2:
        return 0.0
    log_ret = np.diff(np.log(close.values.astype(float)))
    return float(np.std(log_ret, ddof=1) * math.sqrt(252))


def _correlations_from_cov(
    cov: np.ndarray,
    tickers: list[str],
) -> list[dict]:
    """Derive pairwise Pearson correlations from a covariance matrix."""
    n = len(tickers)
    vols = np.sqrt(np.diag(cov))
    correlations = []
    for i in range(n):
        for j in range(i + 1, n):
            if vols[i] > 0 and vols[j] > 0:
                corr = cov[i, j] / (vols[i] * vols[j])
                corr = float(np.clip(corr, -1.0, 1.0))
                correlations.append({
                    "ticker_a": tickers[i],
                    "ticker_b": tickers[j],
                    "correlation": round(corr, 4),
                })
    return correlations


# ---------------------------------------------------------------------------
# Second Tower data preparation
# ---------------------------------------------------------------------------

def _prepare_strategy_inputs(
    tickers: list[str],
    ohlcv_data: dict[str, pd.DataFrame],
    market_ohlcv: pd.DataFrame,
) -> tuple[list[str], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DatetimeIndex, int] | None:
    """Filter tickers and build aligned returns matrices.

    Returns (valid_tickers, close_df, returns_df, returns_df_real,
             market_returns, dates, train_days) or None if insufficient data.
    Mirrors the preprocessing in SecondTowerStrategy.run().
    """
    all_first_dates = [
        ohlcv_data[t].index.min()
        for t in tickers if t in ohlcv_data and len(ohlcv_data[t]) >= 60
    ]
    if not all_first_dates:
        return None
    grace_cutoff = min(all_first_dates) + pd.DateOffset(years=1)
    valid_tickers = [
        t for t in tickers
        if t in ohlcv_data
        and len(ohlcv_data[t]) >= 60
        and ohlcv_data[t].index.min() <= grace_cutoff
    ]
    if len(valid_tickers) < 5:
        return None

    close_raw = pd.DataFrame({t: ohlcv_data[t]["Close"] for t in valid_tickers}).dropna(how="all")
    for col in close_raw.columns:
        last_valid = close_raw[col].last_valid_index()
        if last_valid is not None and last_valid < close_raw.index[-1]:
            close_raw.loc[close_raw.index > last_valid, col] = np.nan
    close_df = close_raw.ffill(limit=5).dropna(axis=1)
    valid_tickers = list(close_df.columns)

    returns_df = close_df.pct_change().dropna()
    returns_df_real = close_raw[valid_tickers].pct_change().reindex(returns_df.index)
    market_returns = market_ohlcv["Close"].pct_change().dropna()
    dates = returns_df.index
    train_days = _TRAIN_DAYS

    if len(dates) < train_days + 1:
        return None

    return valid_tickers, close_df, returns_df, returns_df_real, market_returns, dates, train_days


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def run_agent_b(
    conn: sqlite3.Connection,
    rag: LightRAG,
    market_data: dict[str, MarketDataPoint],
    run_id: int,
) -> dict | None:
    """Run Agent B using the Second Tower strategy.

    Returns output dict matching the Agent B schema, or None if Second Tower
    cannot produce weights (e.g. fewer than 5 eligible tickers).
    """
    # 1. Universe = full watchlist
    watchlist_rows = get_watchlist(conn)
    all_tickers = [r["ticker"] for r in watchlist_rows]
    ticker_sector_map = {r["ticker"]: r["sector"] for r in watchlist_rows}
    all_tickers_with_spy = list(set(all_tickers + ["SPY"]))

    # 2. Refresh OHLCV + EDGAR caches (incremental — only fetches missing tail)
    refresh_ohlcv_cache(conn, all_tickers_with_spy, _LOOKBACK_DAYS)
    refresh_edgar_cache(conn, all_tickers)

    ohlcv_data = load_ohlcv_cache(conn, all_tickers, _LOOKBACK_DAYS)
    edgar_data = load_edgar_cache(conn, all_tickers)
    spy_cache = load_ohlcv_cache(conn, ["SPY"], _LOOKBACK_DAYS)
    market_ohlcv = spy_cache.get("SPY", pd.DataFrame())

    if market_ohlcv.empty:
        # Fall back gracefully — signal an error without crashing the pipeline
        import logging
        logging.getLogger(__name__).error("SPY OHLCV not available; Agent B skipped.")
        return None

    # 3. Prepare inputs for _compute_rebal_weights
    prepared = _prepare_strategy_inputs(all_tickers, ohlcv_data, market_ohlcv)
    if prepared is None:
        return None

    valid_tickers, close_df, returns_df, returns_df_real, market_returns, dates, train_days = prepared
    edgar_data_filtered = {t: edgar_data.get(t, {}) for t in valid_tickers}
    sector_map = get_sector_map(valid_tickers)

    constraints = get_constraints(conn)
    risk_cfg = {
        **RISK_CONFIG["medium"],
        "max_weight": constraints["max_single_position"],
        "max_sector_weight": constraints["max_sector_concentration"],
    }
    rebal_idx = len(dates) - 1

    # 4. Compute latest target weights (single-date live mode)
    _, _, weights, _, factor_decomposition = _compute_rebal_weights(
        rebal_idx=rebal_idx,
        dates=dates,
        train_days=train_days,
        valid_tickers=valid_tickers,
        ohlcv_data=ohlcv_data,
        edgar_data=edgar_data_filtered,
        market_returns=market_returns,
        returns_df=returns_df,
        returns_df_real=returns_df_real,
        signal_alpha=risk_cfg["signal_alpha"],
        hist_mean_weight=risk_cfg["hist_mean_weight"],
        max_weight=risk_cfg["max_weight"],
        max_sector_weight=risk_cfg["max_sector_weight"],
        max_turnover=risk_cfg["max_turnover"],
        sector_map=sector_map,
        pre_computed_regime=None,
    )
    target_weights: dict[str, float] = {
        valid_tickers[i]: float(weights[i]) for i in range(len(valid_tickers))
    }

    # 5. Compute covariance for correlation injection
    train_start = max(0, rebal_idx - train_days)
    lw_input = returns_df_real.iloc[train_start:rebal_idx][valid_tickers].fillna(0.0).values
    cov = _shrinkage_cov(lw_input, len(valid_tickers))
    correlations = _correlations_from_cov(cov, valid_tickers)

    # 6. Current portfolio state (uses market_data for live prices)
    def price_fn(ticker: str) -> float:
        if ticker in market_data:
            return market_data[ticker].current_price
        # Fall back to last OHLCV close
        df = ohlcv_data.get(ticker)
        if df is not None and not df.empty:
            return float(df["Close"].iloc[-1])
        return 1.0

    holdings_rows = [dict(r) for r in conn.execute("SELECT * FROM holdings").fetchall()]
    current_weights = get_derived_weights(conn, price_fn)
    total_value = get_portfolio_value(conn, price_fn)
    account = get_account(conn)
    cash_pct = account["cash_balance"] / total_value if total_value > 0 else 1.0
    constraints = get_constraints(conn)
    previous_target = get_previous_computed_target(conn)
    previous_targets: dict = previous_target["per_ticker_json"] if previous_target else {}

    # 7. Build per-ticker output (watchlist-wide, not just holdings)
    all_output_tickers = set(valid_tickers) | set(current_weights.keys())
    per_ticker_output: dict[str, dict] = {}
    health_scores: dict[str, str] = {}

    for ticker in sorted(all_output_tickers):
        cur_w = current_weights.get(ticker, 0.0)
        tgt_w = target_weights.get(ticker, 0.0)
        prev_tgt_w = previous_targets.get(ticker, {}).get("target_weight", 0.0)

        vol_30d = _vol_30d(ohlcv_data.get(ticker))

        # Drift only applies to existing positions. With no current allocation
        # there is nothing to drift from — the gap between 0 and target_weight
        # is a "new position" signal, conveyed via the buy_candidate flag below.
        flags: list[str] = []
        if cur_w <= 0.0:
            drift = 0.0
            health = "normal"
        else:
            drift = cur_w - tgt_w
            if abs(drift) >= DRIFT_BREACH_THRESHOLD:
                health = "breach"
                flags.append("drift_breach")
            elif abs(drift) >= DRIFT_WARNING_THRESHOLD:
                health = "warning"
                flags.append("drift_above_threshold")
            else:
                health = "normal"
        health_scores[ticker] = health

        if tgt_w > 0.01 and cur_w == 0.0:
            flags.append("buy_candidate")
        elif tgt_w > cur_w + DRIFT_WARNING_THRESHOLD:
            flags.append("new_position_suggested")

        sector = ticker_sector_map.get(ticker, sector_map.get(ticker, "Unknown"))
        fdecomp = factor_decomposition.get(ticker, {})
        per_ticker_output[ticker] = {
            "current_weight": round(cur_w, 6),
            "previous_target_weight": round(prev_tgt_w, 6),
            "target_weight": round(tgt_w, 6),
            "drift": round(drift, 6),
            "volatility_30d": round(vol_30d, 6),
            "sector": sector,
            "health_score": health,
            "flags": flags,
            "factor_drivers": fdecomp.get("top_factors", []),
            "category_scores": fdecomp.get("category_scores", {}),
            "composite_signal": fdecomp.get("composite_signal", 0.0),
        }

    # 8. Portfolio-level metrics
    sector_limit = constraints.get("max_sector_concentration", 0.35)
    sector_concentrations = _compute_sector_concentrations(
        holdings_rows, current_weights, sector_limit
    )
    constraint_violations = _detect_constraint_violations(
        current_weights, cash_pct, sector_concentrations, constraints
    )
    max_drawdown = _compute_max_drawdown(conn)
    portfolio_vol = sum(
        per_ticker_output[t]["volatility_30d"] * current_weights.get(t, 0.0)
        for t in current_weights
    )
    overall_status = _determine_overall_status(health_scores, constraint_violations)

    output = {
        "portfolio_level": {
            "total_value": round(total_value, 2),
            "cash_pct": round(cash_pct, 6),
            "portfolio_volatility_30d": round(portfolio_vol, 6),
            "max_drawdown_30d": round(max_drawdown, 6),
            "overall_status": overall_status,
            "regime_label": None,
            "target_vol_scaling": None,
        },
        "per_ticker": per_ticker_output,
        "sector_concentrations": sector_concentrations,
        "constraint_violations": constraint_violations,
    }

    # 9. Side effects
    await inject_correlation_edges(rag, correlations, run_id)
    store_agent_output(conn, run_id, "B", output)

    return output

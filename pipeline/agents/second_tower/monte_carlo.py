"""Bootstrap Monte Carlo calibration of per-ticker stop-loss thresholds.

For each ticker, generates synthetic price paths by resampling historical daily
returns (with replacement), then sweeps candidate soft-SL / wait-days combinations
to find the (sl_soft, wait_days) that maximises the simulated Sharpe ratio.

Bootstrap resampling preserves the empirical fat tails, negative skew, and
volatility clustering of real return distributions — much more realistic than
assuming GBM (Gaussian random walk).
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Sequence
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Default sweep parameters ───────────────────────────────────────────────────

DEFAULT_SL_SOFT_CANDIDATES: list[float] = [-0.02, -0.03, -0.05, -0.07, -0.10, -0.15]
DEFAULT_WAIT_DAYS_CANDIDATES: list[int] = [3, 5, 10]
DEFAULT_N_PATHS: int = 2000
DEFAULT_HARD_MULTIPLIER: float = 2.0   # hard SL = soft SL × this
DEFAULT_RECOVERY_FACTOR: float = 0.30  # cancel warning if price recovers 30% of the way back to soft_sl
DEFAULT_CALIBRATION_WINDOW: int = 504  # 2 trading years of history


@dataclass(frozen=True)
class OptimalSL:
    """Calibrated stop-loss parameters for a single ticker at a single horizon."""
    sl_soft: float        # e.g. -0.05 → exit warning if down 5% from entry
    sl_hard: float        # e.g. -0.10 → immediate exit if down 10% from entry
    wait_days: int        # days to wait during warning before forcing exit
    recovery_factor: float  # fraction of soft_sl recovery needed to cancel warning
    horizon: int          # hold horizon these params were calibrated for (trading days)
    expected_sharpe: float  # MC reward-to-variability ratio (not comparable to realized Sharpe)


def _simulate_paths_sharpe(
    returns: np.ndarray,          # 1-D array of historical daily returns for one ticker
    horizon: int,
    n_paths: int,
    sl_soft: float,               # e.g. -0.05
    sl_hard: float,               # e.g. -0.10
    wait_days: int,
    recovery_factor: float,
    rng: np.random.Generator,
) -> float:
    """Bootstrap-simulate n_paths of length horizon and return the resulting Sharpe.

    Each path:
      - Samples `horizon` returns with replacement from `returns`
      - Applies the soft/hard SL state machine
      - Records terminal cumulative return (accounting for early exit)

    Returns annualised Sharpe across all n_paths terminal returns.
    """
    if len(returns) < 10:
        return 0.0

    # Block bootstrap (block_size=5) preserves volatility clustering
    block_size = 5
    if len(returns) < block_size:
        # Fall back to i.i.d. for very short return series
        sampled = rng.choice(returns, size=(n_paths, horizon), replace=True)
    else:
        n_blocks = math.ceil(horizon / block_size)
        starts = rng.integers(0, max(1, len(returns) - block_size + 1), size=(n_paths, n_blocks))
        sampled = np.empty((n_paths, n_blocks * block_size), dtype=float)
        for b in range(n_blocks):
            for p in range(n_paths):
                s = starts[p, b]
                sampled[p, b * block_size:(b + 1) * block_size] = returns[s:s + block_size]
        sampled = sampled[:, :horizon]

    # Cumulative return path: shape (n_paths, horizon)
    cum_ret = np.cumprod(1.0 + sampled, axis=1) - 1.0  # cum return relative to entry

    terminal = np.zeros(n_paths)
    soft_sl_threshold = sl_soft
    hard_sl_threshold = sl_hard
    # Recovery cancel level — matches portfolio_backtester.py price-level formula:
    # recovery_level = entry_p * (1 + sl_soft) * (1 + abs(sl_soft) * recovery_factor)
    # In cumulative-return space (entry_p = 1.0):
    recovery_cancel_level = (1.0 + soft_sl_threshold) * (1.0 + abs(soft_sl_threshold) * recovery_factor) - 1.0

    for p in range(n_paths):
        path = cum_ret[p]
        warning_active = False
        warning_start = -1
        exit_ret = path[-1]  # default: hold to end

        for t in range(horizon):
            r = path[t]

            # Hard SL: immediate exit
            if r < hard_sl_threshold:
                exit_ret = r
                break

            # Soft SL: start warning
            if r < soft_sl_threshold and not warning_active:
                warning_active = True
                warning_start = t

            # Recovery: cancel warning
            if warning_active and r > recovery_cancel_level:
                warning_active = False
                warning_start = -1

            # Warning expired: forced exit
            if warning_active and (t - warning_start) >= wait_days:
                exit_ret = r
                break
        else:
            exit_ret = path[-1]

        terminal[p] = exit_ret

    if len(terminal) < 2 or np.std(terminal) < 1e-10:
        return float(np.mean(terminal)) * (252.0 / horizon)

    # Cross-path reward-to-variability ratio (not time-series Sharpe).
    # Annualization uses sqrt(252/horizon) as a scaling factor for the
    # MC optimization objective — consistent across candidates but not
    # comparable to realized Sharpe ratios.
    reward_variability = float(np.mean(terminal) / np.std(terminal)) * np.sqrt(252.0 / horizon)
    return reward_variability


def calibrate_sl_thresholds(
    returns_series: pd.Series | np.ndarray,
    horizon: int,
    n_paths: int = DEFAULT_N_PATHS,
    sl_soft_candidates: Sequence[float] = DEFAULT_SL_SOFT_CANDIDATES,
    wait_days_candidates: Sequence[int] = DEFAULT_WAIT_DAYS_CANDIDATES,
    hard_multiplier: float = DEFAULT_HARD_MULTIPLIER,
    recovery_factor: float = DEFAULT_RECOVERY_FACTOR,
    calibration_window: int = DEFAULT_CALIBRATION_WINDOW,
    random_seed: int | None = 42,
) -> OptimalSL:
    """Find the (sl_soft, wait_days) that maximises simulated Sharpe for one ticker.

    Parameters
    ----------
    returns_series : daily return series (pandas or numpy)
    horizon        : planned hold horizon in trading days
    n_paths        : number of bootstrap paths per candidate pair
    sl_soft_candidates : list of soft SL fractions to sweep (negative, e.g. -0.05)
    wait_days_candidates : list of wait-day counts to sweep
    hard_multiplier : sl_hard = sl_soft * hard_multiplier (applied as more negative)
    recovery_factor : fraction of soft_sl recovery needed to cancel warning
    calibration_window : max lookback in trading days
    random_seed    : for reproducibility; None = truly random

    Returns
    -------
    OptimalSL with the best (sl_soft, wait_days) found
    """
    if isinstance(returns_series, pd.Series):
        returns = returns_series.dropna().values
    else:
        returns = np.asarray(returns_series, dtype=float)
        returns = returns[~np.isnan(returns)]

    # Trim to calibration window
    if len(returns) > calibration_window:
        returns = returns[-calibration_window:]

    if len(returns) < 20:
        # Insufficient history — fall back to conservative defaults
        logger.warning("calibrate_sl_thresholds: insufficient history, using defaults")
        return OptimalSL(
            sl_soft=-0.07, sl_hard=-0.14,
            wait_days=5, recovery_factor=recovery_factor,
            horizon=horizon, expected_sharpe=0.0,
        )

    rng = np.random.default_rng(random_seed)

    best_sharpe = -np.inf
    best_sl_soft = sl_soft_candidates[0]
    best_wait_days = wait_days_candidates[0]

    for sl_soft in sl_soft_candidates:
        sl_hard = sl_soft * hard_multiplier  # more negative
        for wd in wait_days_candidates:
            sharpe = _simulate_paths_sharpe(
                returns, horizon, n_paths,
                sl_soft, sl_hard, wd, recovery_factor, rng,
            )
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_sl_soft = sl_soft
                best_wait_days = wd

    return OptimalSL(
        sl_soft=best_sl_soft,
        sl_hard=best_sl_soft * hard_multiplier,
        wait_days=best_wait_days,
        recovery_factor=recovery_factor,
        horizon=horizon,
        expected_sharpe=round(best_sharpe, 4),
    )


def calibrate_portfolio(
    returns_df: pd.DataFrame,
    horizon: int,
    tickers: list[str] | None = None,
    **kwargs,
) -> dict[str, OptimalSL]:
    """Calibrate SL thresholds for each ticker in returns_df.

    Parameters
    ----------
    returns_df : DataFrame with one column per ticker, DatetimeIndex
    horizon    : planned hold horizon in trading days
    tickers    : subset of tickers to calibrate (defaults to all columns)
    **kwargs   : passed through to calibrate_sl_thresholds

    Returns
    -------
    dict mapping ticker → OptimalSL
    """
    cols = tickers if tickers is not None else list(returns_df.columns)
    result: dict[str, OptimalSL] = {}
    for ticker in cols:
        if ticker not in returns_df.columns:
            continue
        try:
            result[ticker] = calibrate_sl_thresholds(
                returns_df[ticker], horizon, **kwargs
            )
        except Exception as e:
            logger.warning(f"calibrate_portfolio: failed for {ticker}: {e}")
            result[ticker] = OptimalSL(
                sl_soft=-0.07, sl_hard=-0.14,
                wait_days=5, recovery_factor=kwargs.get("recovery_factor", DEFAULT_RECOVERY_FACTOR),
                horizon=horizon, expected_sharpe=0.0,
            )
    return result

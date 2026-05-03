"""Second-tower strategy: signal-adjusted mean-variance optimization with walk-forward backtesting.

Key design decisions:
1. Signals recomputed at EACH rebalancing date using only data available at that point.
2. Fundamental factors use SEC EDGAR 10-K data indexed by actual filing date — no
   artificial lag needed; filter is df.index < rebal_date.
3. Ledoit-Wolf covariance shrinkage prevents singular matrices on large universes.
4. Walk-forward: rolling training window; weights applied from the day AFTER each
   rebalancing date (avoids look-ahead bias).
5. Historical-mean contribution is dampened (hist_mean_weight × mu_hist) to reduce
   noise from estimating expected returns from realized returns.
6. Rebalancing computations are parallelised with a ThreadPoolExecutor; folds are
   fully independent (read-only inputs), so this is thread-safe.
7. step_days < rebalance_days produces overlapping hold windows; weights are averaged
   across concurrent windows for each trading day (smoother, less path-dependent).
"""
from __future__ import annotations

import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

import numpy as np
import pandas as pd

from .factors import compute_all_factors
from .optimizer import optimize_portfolio
from .universes import get_sector_map

logger = logging.getLogger(__name__)

# Covariance shrinkage (optional — requires scikit-learn)
try:
    from sklearn.covariance import LedoitWolf as _LedoitWolf, OAS as _OAS
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False


# ── Factor categories ─────────────────────────────────────────────────────────
# Each category gets equal weight (1/N_categories) in the composite signal,
# regardless of how many individual factors it contains.  This prevents the
# implicit momentum tilt that arises from 27/63 factors being momentum/technical.
_FACTOR_CATEGORIES: dict[str, list[str]] = {
    "Momentum":  ["MOM1M", "MOM3M", "MOM6M", "MOM12M", "MOM12M1", "HIGH52W_RATIO",
                  "IND_MOM", "STR", "LTR", "CONSEC_UP", "TSMOM", "PRICE_ACCEL"],
    "Value":     ["BM", "EP", "CFP", "SP", "EV_EBITDA",
                  "TRAILING_PE_INV", "DIV_YIELD", "PAYOUT_YIELD", "FWD_EP"],
    "Quality":   ["ROA", "ROE", "GP_A", "OP_MARGIN", "NET_MARGIN", "ASSET_GROWTH",
                  "CAPEX_GROWTH", "SALES_GROWTH", "PIOTROSKI_F", "ACCRUALS_RATIO",
                  "EARN_QUALITY", "LEV_RATIO"],
    "Technical": ["RSI14", "MACD", "BB_POS", "ATR_NORM", "OBV", "MA50_200",
                  "CROSS_SIGNAL", "VOL_MOM", "STOCH", "WILLIAMS_R", "CCI", "ADX",
                  "LOW52W_DIST", "VOL_RATIO", "DOLLAR_VOL"],
    "Risk":      ["TRAILING_BETA", "IVOL", "TVOL", "MAX_RET", "SKEW", "DOWN_BETA", "KURT", "AMIHUD"],
    "Accruals":  ["BS_ACCRUALS", "WC_ACCRUALS", "DEPR_RATIO", "SUE", "DISC_ACCRUALS"],
}


def _category_balanced_composite(
    factor_z: pd.DataFrame,
    category_weights: dict[str, float] | None = None,
) -> pd.Series:
    """Average factor z-scores within each category first, then weighted-average across categories.

    Each category contributes equally by default (preventing the implicit momentum tilt
    from 27/63 factors being momentum/technical).  When ``category_weights`` is provided,
    categories are weighted accordingly — used for regime-conditional factor tilts
    (e.g., downweight Momentum in Bear, upweight Quality in Crisis).

    Falls back to simple equal-weight mean if no category has any present factors.
    """
    category_scores = []
    weights_list = []
    for cat_name, cat_cols in _FACTOR_CATEGORIES.items():
        present = [c for c in cat_cols if c in factor_z.columns]
        if present:
            category_scores.append(factor_z[present].mean(axis=1))
            w = category_weights.get(cat_name, 1.0) if category_weights else 1.0
            weights_list.append(w)
    if not category_scores:
        return factor_z.mean(axis=1)
    stacked = pd.concat(category_scores, axis=1)
    w_arr = np.array(weights_list)
    w_sum = w_arr.sum()
    if w_sum > 0:
        w_arr = w_arr / w_sum
    else:
        w_arr = np.ones(len(w_arr)) / len(w_arr)
    return (stacked * w_arr).sum(axis=1)


# ── Risk level configuration ──────────────────────────────────────────────────
RISK_CONFIG = {
    "low":    {
        "max_weight": 1.0, "max_sector_weight": 0.40, "max_turnover": 0.15,
        "signal_alpha": 0.01, "hist_mean_weight": 0.20,
    },
    "medium": {
        "max_weight": 1.0, "max_sector_weight": 0.40, "max_turnover": 0.30,
        "signal_alpha": 0.02, "hist_mean_weight": 0.30,
    },
    "high":   {
        "max_weight": 1.0, "max_sector_weight": 0.40, "max_turnover": 0.50,
        "signal_alpha": 0.04, "hist_mean_weight": 0.40,
    },
}


# ── EDGAR filtering ───────────────────────────────────────────────────────────

def _slice_edgar_by_cutoff(edgar_data: dict, cutoff: pd.Timestamp) -> dict[str, dict]:
    """Filter each EDGAR statement DataFrame to rows with filed_date < cutoff."""
    result: dict[str, dict] = {}
    for ticker, stmts in edgar_data.items():
        if not stmts:
            result[ticker] = {}
            continue
        filtered: dict[str, pd.DataFrame] = {}
        for stmt_name, df in stmts.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                fdf = df[df.index < cutoff]
                filtered[stmt_name] = fdf if not fdf.empty else pd.DataFrame()
            else:
                filtered[stmt_name] = df
        result[ticker] = filtered
    return result


# ── Covariance ────────────────────────────────────────────────────────────────

def _shrinkage_cov(
    returns_matrix: np.ndarray,
    n_assets: int,
    method: str = "ledoit_wolf",
    ewm_halflife: int = 0,
) -> np.ndarray:
    """Compute annualised covariance with shrinkage when possible.

    Args:
        method: "ledoit_wolf" (default) or "oas" (Oracle Approximating Shrinkage).
                OAS provides better shrinkage intensity when N/T is moderate (0.1–0.5).
        ewm_halflife: If > 0, apply exponential weighting with this halflife (in
                      trading days) before fitting the shrinkage estimator.  Shorter
                      halflife → heavier weight on recent observations.  Useful in
                      stress regimes where recent correlations are more informative.
    """
    mat = returns_matrix
    if ewm_halflife > 0 and mat.shape[0] > ewm_halflife:
        T = mat.shape[0]
        # Exponential decay weights: most recent observation gets highest weight
        decay = np.exp(-np.log(2) / ewm_halflife * np.arange(T - 1, -1, -1))
        # Normalize so sum = T (preserves effective scale for shrinkage estimator)
        decay = decay / decay.sum() * T
        # Apply sqrt(weights) to returns so cov(weighted) = EWM covariance
        mat = mat * np.sqrt(decay[:, None])

    T_obs = mat.shape[0]
    # Require T > 2.5N for LW/OAS — near T≈N ratios still yield near-singular matrices.
    lw_threshold = max(n_assets + 1, int(2.5 * n_assets))
    if _HAS_SKLEARN and T_obs >= lw_threshold:
        try:
            estimator = _OAS() if method == "oas" else _LedoitWolf()
            estimator.fit(mat)
            return estimator.covariance_ * 252
        except Exception:
            pass
    cov = np.cov(mat.T) * 252
    # Ridge scales inversely with T/N — small ratios → near-singular → need larger ridge
    tn_ratio = T_obs / n_assets if n_assets > 0 else 1.0
    ridge = 1e-6 if tn_ratio >= 2.5 else max(1e-4, 1e-4 / tn_ratio)
    cov += ridge * np.eye(n_assets)
    return cov


# ── Per-rebalancing computation (thread-safe, all inputs read-only) ───────────

def _compute_rebal_weights(
    rebal_idx: int,
    dates: pd.DatetimeIndex,
    train_days: int,
    valid_tickers: list[str],
    ohlcv_data: dict[str, pd.DataFrame],
    edgar_data: dict[str, dict],
    market_returns: pd.Series,
    returns_df: pd.DataFrame,
    returns_df_real: pd.DataFrame,
    signal_alpha: float,
    hist_mean_weight: float,
    max_weight: float,
    max_sector_weight: float,
    max_turnover: float,
    sector_map: dict[str, str],
    cov_method: str = "ledoit_wolf",
    lambda_risk: float = 1.0,
    risk_objective: str = "variance",
    cvar_alpha: float = 0.05,
    turnover_penalty: float = 0.0,
    target_vol: float = 0.0,
    max_leverage: float = 1.0,
    pre_computed_regime: tuple[str, dict] | None = None,
) -> tuple[int, pd.Timestamp, np.ndarray, str]:
    """Compute optimized weights for one rebalancing date.

    All inputs are read-only references shared across threads.  No mutable shared
    state is written — thread-safe for use with ThreadPoolExecutor.

    prev_weights is always None in parallel mode; this slightly overestimates
    transaction costs on the first rebalancing of each independent fold (~0.005%
    aggregate drag) in exchange for full parallelism.

    When ``pre_computed_regime`` is provided (a (regime_name, adjustment_dict) tuple),
    the regime adjustments are applied to signal_alpha, max_weight, target_vol,
    lambda_risk, covariance halflife, and factor category weights.  Regime detection
    is pre-computed sequentially in the caller to ensure BOCPD cooldown state is
    properly threaded across rebalancing dates.

    Returns (rebal_idx, rebal_date, weights, regime_state).
    """
    N = len(valid_tickers)
    rebal_date = dates[rebal_idx]
    regime_state = "Bull"

    # ── Apply pre-computed regime adjustments ──────────────────────────────
    category_weights: dict[str, float] | None = None
    cov_halflife = 0
    if pre_computed_regime is not None:
        regime_state, regime_adj = pre_computed_regime
        signal_alpha = signal_alpha * regime_adj["signal_alpha"]
        max_weight = min(max_weight * regime_adj["max_weight"], 1.0)
        target_vol = target_vol * regime_adj["target_vol"]
        lambda_risk = lambda_risk * regime_adj.get("lambda_scale", 1.0)
        cov_halflife = regime_adj.get("cov_halflife", 0)
        category_weights = regime_adj.get("category_weights")
        logger.debug(f"Regime at {rebal_date.date()}: {regime_state}")

    train_start_idx = max(0, rebal_idx - train_days)
    train_dates = dates[train_start_idx : rebal_idx + 1]

    # OHLCV slice for this training window
    reb_ohlcv: dict[str, pd.DataFrame] = {
        t: ohlcv_data[t][
            (ohlcv_data[t].index >= train_dates[0]) & (ohlcv_data[t].index <= rebal_date)
        ]
        for t in valid_tickers if t in ohlcv_data
    }

    reb_market_ret = market_returns[
        (market_returns.index >= train_dates[0]) & (market_returns.index <= rebal_date)
    ]

    reb_edgar = _slice_edgar_by_cutoff(edgar_data, rebal_date)

    # Factor computation
    try:
        factors = compute_all_factors(reb_ohlcv, reb_edgar, reb_market_ret)
    except Exception as e:
        logger.warning(f"Factor computation failed at {rebal_date}: {e}")
        factors = pd.DataFrame()

    # Category-balanced composite signal (regime-conditional weights)
    if not factors.empty and len(factors) >= 3:
        factor_z = (factors - factors.mean()) / factors.std().replace(0, 1)
        factor_z = factor_z.replace([np.inf, -np.inf], 0).fillna(0)
        composite = _category_balanced_composite(factor_z, category_weights=category_weights)
        composite_aligned = composite.reindex(valid_tickers).fillna(0).values
    else:
        composite_aligned = np.zeros(N)

    # Expected returns: dampened historical mean + signal composite.
    # Using only hist_mean_weight (30%) of the realized mean reduces overfitting
    # to noisy historical drift estimates (Merton 1980).
    train_dates_in_idx = train_dates[train_dates.isin(returns_df.index)]
    train_ret_df = returns_df.loc[train_dates_in_idx]

    train_ret_real_mu = returns_df_real.loc[
        (returns_df_real.index >= train_dates[0]) & (returns_df_real.index <= rebal_date),
        valid_tickers,
    ]
    mu_hist = train_ret_real_mu.mean().reindex(valid_tickers).fillna(0).values * 252
    mu = mu_hist * hist_mean_weight + signal_alpha * composite_aligned

    # Covariance: shrinkage on raw (non-ffill'd) returns, EWM-weighted in stress
    train_ret_real = returns_df_real.loc[
        (returns_df_real.index >= train_dates[0]) & (returns_df_real.index <= rebal_date),
        valid_tickers,
    ]
    min_coverage = int(0.8 * N)
    row_mask = train_ret_real.notna().sum(axis=1) >= min_coverage
    lw_input = train_ret_real.loc[row_mask]
    if len(lw_input) < max(N + 5, 30):
        lw_input = train_ret_df[valid_tickers]
        lw_input = lw_input.fillna(lw_input.mean())
    else:
        # Fill NaN with 0 (not column mean) to avoid conditional-mean bias for
        # recently-IPO'd stocks whose column mean is estimated from few observations,
        # which compresses variance and underestimates covariance.
        lw_input = lw_input.fillna(0.0)
    cov = _shrinkage_cov(lw_input.values, N, method=cov_method, ewm_halflife=cov_halflife)

    # Build returns matrix for CVaR / turnover penalty
    train_returns_matrix = lw_input.values if risk_objective == "cvar" or turnover_penalty > 0 else None

    # Optimize (prev_weights=None in parallel mode)
    try:
        weights, _ = optimize_portfolio(
            mu, cov, None,
            max_weight, max_sector_weight, max_turnover,
            sector_map, valid_tickers,
            lambda_risk=lambda_risk,
            risk_objective=risk_objective,
            cvar_alpha=cvar_alpha,
            returns_matrix=train_returns_matrix,
            turnover_penalty=turnover_penalty,
        )
    except Exception as e:
        logger.warning(f"Optimisation failed at {rebal_date}: {e}. Using equal-weight.")
        weights = np.ones(N) / N

    # Volatility targeting: scale weights so ex-ante vol matches target
    if target_vol > 0:
        port_vol = np.sqrt(weights @ cov @ weights)
        if port_vol > 1e-10:
            scale = min(target_vol / port_vol, max_leverage)
            weights = weights * scale
            # Iterative clip-renormalize to ensure max_weight is respected
            for _ in range(5):
                weights = np.clip(weights, 0, max_weight)
                s = weights.sum()
                if s > 0:
                    weights /= s
                if np.all(weights <= max_weight + 1e-10):
                    break

    return rebal_idx, rebal_date, weights, regime_state


# ── Strategy class ────────────────────────────────────────────────────────────

class SecondTowerStrategy:
    def __init__(self, config: dict, risk_level: str = "medium"):
        self.risk_level = risk_level
        risk = RISK_CONFIG.get(risk_level, RISK_CONFIG["medium"])

        self.max_weight = config.get("max_weight", risk["max_weight"])
        self.max_sector_weight = config.get("max_sector_weight", risk["max_sector_weight"])
        self.max_turnover = config.get("max_turnover", risk["max_turnover"])
        self.signal_alpha = config.get("signal_alpha", risk["signal_alpha"])
        self.hist_mean_weight = config.get("hist_mean_weight", risk["hist_mean_weight"])
        self.train_years = int(config.get("train_years", 3))
        self.test_years = float(config.get("test_years", 1))
        self.rebalance_days = int(config.get("rebalance_days", 21))
        # step_days controls how frequently we recompute weights.  Default equals
        # rebalance_days (non-overlapping).  Set step_days < rebalance_days to get
        # overlapping hold windows; daily returns are averaged across concurrent windows.
        self.step_days = int(config.get("step_days", self.rebalance_days))
        # Number of parallel workers for fold computation.
        self.max_workers = int(config.get("max_workers", min(4, os.cpu_count() or 1)))
        # ── Monte Carlo SL calibration ────────────────────────────────────────
        # use_mc_sl: enable bootstrap SL calibration (default True)
        # hard_multiplier: sl_hard = sl_soft × hard_multiplier (default 2.0)
        # recovery_factor: fraction of soft_sl gap needed to cancel warning (default 0.30)
        # mc_n_paths: bootstrap paths per (sl_soft, wait_days) candidate (default 2000)
        self.use_mc_sl = bool(config.get("use_mc_sl", True))
        self.hard_multiplier = float(config.get("hard_multiplier", 2.0))
        self.recovery_factor = float(config.get("recovery_factor", 0.30))
        self.mc_n_paths = int(config.get("mc_n_paths", 2000))
        # mc_recalibrate_days: re-run MC calibration every N trading days (default 252 = ~annual)
        self.mc_recalibrate_days = int(config.get("mc_recalibrate_days", 252))
        # ── Take-profit liquidity release ─────────────────────────────────────
        # tp_threshold: % gain from entry to trigger TP (0 = disabled)
        # tp_reduce_fraction: fraction of weight to release (default 0.5)
        # min_cash_pct: TP only fires when cash allocation below this (default 0.05)
        self.tp_threshold = float(config.get("tp_threshold", 0.0)) or None
        self.tp_reduce_fraction = float(config.get("tp_reduce_fraction", 0.5))
        self.min_cash_pct = float(config.get("min_cash_pct", 0.05))
        # ── Covariance method ─────────────────────────────────────────────────
        self.cov_method = str(config.get("cov_method", "ledoit_wolf"))
        # ── Risk objective ────────────────────────────────────────────────────
        self.risk_objective = str(config.get("risk_objective", "variance"))
        self.cvar_alpha = float(config.get("cvar_alpha", 0.05))
        # ── Volatility targeting ──────────────────────────────────────────────
        self.target_vol = float(config.get("target_vol", 0.0))
        self.max_leverage = float(config.get("max_leverage", 1.0))
        # ── Lambda risk (risk aversion coefficient) ────────────────────────
        self.lambda_risk = float(config.get("lambda_risk", 1.0))
        # ── Soft turnover penalty ─────────────────────────────────────────────
        self.turnover_penalty = float(config.get("turnover_penalty", 0.0))
        # ── Regime detection ──────────────────────────────────────────────────
        self.use_regime_detection = bool(config.get("use_regime_detection", False))
        self.regime_lookback = int(config.get("regime_lookback", 1500))
        self.regime_model = str(config.get("regime_model", "jump"))
        self.regime_n_states = int(config.get("regime_n_states", 4))
        self.regime_use_macro = bool(config.get("regime_use_macro", True))
        self.regime_blending = bool(config.get("regime_blending", True))
        self.regime_confirmation = int(config.get("regime_confirmation", 1))
        self.regime_jump_penalty = float(config.get("regime_jump_penalty", 20.0))
        self.regime_use_sparse = bool(config.get("regime_use_sparse", True))
        self.regime_use_bocpd = bool(config.get("regime_use_bocpd", False))
        self.regime_max_feats = float(config.get("regime_max_feats", 2.0))
        # ── Tiered deleveraging ──────────────────────────────────────────────
        self.use_tiered_deleveraging = bool(config.get("use_tiered_deleveraging", False))
        # ── Signal decay exit ─────────────────────────────────────────────────
        self.use_signal_decay = bool(config.get("use_signal_decay", False))
        self.signal_check_days = int(config.get("signal_check_days", 7))
        self.soft_exit_z = float(config.get("soft_exit_z", -0.5))
        self.hard_exit_z = float(config.get("hard_exit_z", -1.5))
        self.signal_grace_days = int(config.get("signal_grace_days", 5))

    def run(
        self,
        tickers: list[str],
        ohlcv_data: dict[str, pd.DataFrame],
        edgar_data: dict[str, dict],
        market_ohlcv: pd.DataFrame,
        progress_cb: Callable[[int], None] | None = None,
    ) -> dict:
        """Run walk-forward second-tower optimization.

        Parameters
        ----------
        tickers      : list of ticker symbols
        ohlcv_data   : {ticker: OHLCV DataFrame} — full date range
        edgar_data   : {ticker: {'balance': df, 'income': df, 'cashflow': df}}
        market_ohlcv : SPY (or benchmark) OHLCV DataFrame
        progress_cb  : optional callback(int 0-100)

        Returns
        -------
        dict with keys: weights, returns, dates, tickers, fold_details,
                        factor_contributions, weight_history, trade_log
        """
        if progress_cb:
            progress_cb(5)

        # ── Ticker filtering with 1-year grace period ─────────────────────────
        # A ticker is valid if it has ≥ 60 days of data AND its first available
        # date is within 1 year of the earliest data in the universe.  This allows
        # stocks that IPO'd just after the simulation start to participate, while
        # excluding tickers with genuinely insufficient history.
        all_first_dates = [
            ohlcv_data[t].index.min()
            for t in tickers if t in ohlcv_data and len(ohlcv_data[t]) > 0
        ]
        if not all_first_dates:
            raise ValueError("No OHLCV data found for any ticker.")
        data_start = min(all_first_dates)
        grace_cutoff = data_start + pd.DateOffset(years=1)

        valid_tickers = [
            t for t in tickers
            if t in ohlcv_data
            and len(ohlcv_data[t]) >= 60
            and ohlcv_data[t].index.min() <= grace_cutoff
        ]
        if len(valid_tickers) < 5:
            raise ValueError(f"Only {len(valid_tickers)} tickers have sufficient data (need ≥ 5)")

        logger.info(f"Second-tower: {len(valid_tickers)} valid tickers out of {len(tickers)}")

        # ── Build aligned close / returns ─────────────────────────────────────
        close_dict = {t: ohlcv_data[t]["Close"] for t in valid_tickers}
        close_raw = pd.DataFrame(close_dict).dropna(how="all")
        close_ffill = close_raw.ffill(limit=5)
        # Keep tickers that have data in the first 5 rows.
        has_initial_data = close_ffill.iloc[:5].notna().any()
        close_filtered = close_raw.loc[:, has_initial_data]
        # NaN-out prices after last valid close so delisted tickers exit the factor
        # universe once they stop trading. Without this, stale flat prices persist
        # for up to a year, contaminating cross-sectional factor z-score ranks.
        for col in close_filtered.columns:
            last_valid = close_filtered[col].last_valid_index()
            if last_valid is not None and last_valid < close_filtered.index[-1]:
                close_filtered.loc[close_filtered.index > last_valid, col] = np.nan
        # Only fill short intra-series gaps (holidays, data gaps ≤5 days)
        close_df = close_filtered.ffill(limit=5).dropna(axis=1)
        valid_tickers = list(close_df.columns)
        N = len(valid_tickers)

        returns_df = close_df.pct_change().dropna()
        # Un-ffill'd returns for LW covariance and mu_hist estimation
        returns_df_real = close_raw[valid_tickers].pct_change()
        # Align returns_df_real to returns_df's index so iloc positions map to
        # the same calendar dates across both DataFrames (ST-M3 fix).
        returns_df_real = returns_df_real.reindex(returns_df.index)
        dates = returns_df.index

        market_returns = market_ohlcv["Close"].pct_change().dropna()
        sector_map = get_sector_map(valid_tickers)

        if progress_cb:
            progress_cb(10)

        train_days = self.train_years * 252
        total_days = len(dates)

        if total_days < train_days + self.step_days:
            raise ValueError(
                f"Not enough data: have {total_days} days, need ≥ {train_days + self.step_days}"
            )

        # Rebalancing / signal-computation dates (step by step_days)
        rebal_indices = list(range(train_days, total_days, self.step_days))
        if not rebal_indices:
            raise ValueError("No rebalancing dates found after the training warm-up period.")

        logger.info(
            f"Walk-forward: {len(rebal_indices)} rebalancing dates, "
            f"train={train_days}d, hold={self.rebalance_days}d, step={self.step_days}d, "
            f"workers={self.max_workers}"
        )

        # ── Parallel weight computation ───────────────────────────────────────
        progress_counter = [0]
        progress_lock = threading.Lock()

        # ── Regime pre-computation ────────────────────────────────────────
        # Regime detection depends only on market_returns (SPY), not per-ticker
        # factors, so it runs independently.  When BOCPD is disabled (default),
        # calls are fully independent and can be parallelised.  When BOCPD is
        # enabled, sequential execution is required so the cooldown state
        # (bocpd_last_fired) threads forward across rebalancing dates.
        pre_computed_regimes: dict[int, tuple[str, dict]] = {}
        regime_labels: dict[pd.Timestamp, str] = {}
        if self.use_regime_detection:
            from .regime import (
                detect_regime, build_all_features, _REGIME_ADJUSTMENTS as _RA,
            )

            # Pre-compute all features once (bulk macro fetch + rolling features)
            all_features = build_all_features(
                market_returns,
                [dates[ri] for ri in rebal_indices],
                lookback=self.regime_lookback,
                use_macro=self.regime_use_macro,
            )
            if progress_cb:
                progress_cb(12)

            _regime_kwargs = dict(
                n_states=self.regime_n_states,
                model_type=self.regime_model,
                use_macro=self.regime_use_macro,
                blending=self.regime_blending,
                confirmation_bars=self.regime_confirmation,
                jump_penalty=self.regime_jump_penalty,
                use_sparse=self.regime_use_sparse,
                max_feats=self.regime_max_feats,
            )

            if self.regime_use_bocpd:
                # BOCPD requires sequential execution for cooldown state
                bocpd_last_fired = None
                for idx, ri in enumerate(rebal_indices):
                    rebal_date = dates[ri]
                    if progress_cb:
                        pct = 12 + int(18 * idx / len(rebal_indices))
                        progress_cb(min(pct, 29))
                    try:
                        regime_state, regime_adj, bocpd_last_fired = detect_regime(
                            market_returns, rebal_date, self.regime_lookback,
                            use_bocpd=True,
                            bocpd_last_fired=bocpd_last_fired,
                            precomputed_features=all_features,
                            **_regime_kwargs,
                        )
                        pre_computed_regimes[ri] = (regime_state, regime_adj)
                        regime_labels[rebal_date] = regime_state
                    except Exception as e:
                        logger.warning(f"Regime pre-computation failed at {rebal_date}: {e}")
                        pre_computed_regimes[ri] = ("Bull", _RA["Bull"])
                        regime_labels[rebal_date] = "Bull"
            else:
                # No BOCPD — calls are independent, parallelise
                def _detect_regime(ri: int):
                    rebal_date = dates[ri]
                    try:
                        state, adj, _ = detect_regime(
                            market_returns, rebal_date, self.regime_lookback,
                            use_bocpd=False, bocpd_last_fired=None,
                            precomputed_features=all_features,
                            **_regime_kwargs,
                        )
                        return ri, rebal_date, state, adj
                    except Exception as e:
                        logger.warning(f"Regime detection failed at {rebal_date}: {e}")
                        return ri, rebal_date, "Bull", _RA["Bull"]

                regime_counter = 0
                with ThreadPoolExecutor(max_workers=self.max_workers) as regime_pool:
                    for ri, rebal_date, state, adj in regime_pool.map(
                        _detect_regime, rebal_indices
                    ):
                        pre_computed_regimes[ri] = (state, adj)
                        regime_labels[rebal_date] = state
                        regime_counter += 1
                        if progress_cb:
                            pct = 12 + int(18 * regime_counter / len(rebal_indices))
                            progress_cb(min(pct, 29))

            if progress_cb:
                progress_cb(30)

            logger.info(
                f"Regime pre-computed for {len(pre_computed_regimes)} rebalancing dates"
                f" (parallel={not self.regime_use_bocpd})"
            )

        def _submit(rebal_idx: int):
            return _compute_rebal_weights(
                rebal_idx, dates, train_days, valid_tickers,
                ohlcv_data, edgar_data, market_returns,
                returns_df, returns_df_real,
                self.signal_alpha, self.hist_mean_weight,
                self.max_weight, self.max_sector_weight, self.max_turnover,
                sector_map,
                cov_method=self.cov_method,
                lambda_risk=self.lambda_risk,
                risk_objective=self.risk_objective,
                cvar_alpha=self.cvar_alpha,
                turnover_penalty=self.turnover_penalty,
                target_vol=self.target_vol,
                max_leverage=self.max_leverage,
                pre_computed_regime=pre_computed_regimes.get(rebal_idx),
            )

        raw_results: dict[int, tuple[pd.Timestamp, np.ndarray]] = {}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_map = {executor.submit(_submit, ri): ri for ri in rebal_indices}
            for future in as_completed(future_map):
                ri = future_map[future]
                try:
                    _, rebal_date, weights, regime_name = future.result()
                    raw_results[ri] = (rebal_date, weights)
                    # Update regime_labels for non-regime-detection runs
                    if dates[ri] not in regime_labels:
                        regime_labels[dates[ri]] = regime_name
                except Exception as e:
                    logger.warning(f"Rebalancing {ri} failed: {e} — using equal-weight.")
                    raw_results[ri] = (dates[ri], np.ones(N) / N)
                    if dates[ri] not in regime_labels:
                        regime_labels[dates[ri]] = "Bull"

                with progress_lock:
                    progress_counter[0] += 1
                    pct = 30 + int(60 * progress_counter[0] / len(rebal_indices))
                    if progress_cb:
                        progress_cb(min(pct, 90))

        # ── Sequential turnover enforcement ──────────────────────────────────
        # prev_weights is always None in parallel mode, so hard turnover
        # constraints and soft L2 penalties are inoperative during optimization.
        # This post-processing pass enforces max_turnover by blending toward
        # the prior rebalancing's weights when turnover exceeds the budget.
        if self.max_turnover > 0:
            sorted_ri = sorted(raw_results.keys())
            prev_w = None
            for ri in sorted_ri:
                rebal_date, w = raw_results[ri]
                if prev_w is not None:
                    turnover = float(np.sum(np.abs(w - prev_w)))
                    if turnover > self.max_turnover:
                        alpha = self.max_turnover / max(turnover, 1e-10)
                        w = alpha * w + (1.0 - alpha) * prev_w
                        # Iterative clip-renormalize
                        for _ in range(5):
                            w = np.clip(w, 0, self.max_weight)
                            s = w.sum()
                            if s > 0:
                                w /= s
                            if np.all(w <= self.max_weight + 1e-10):
                                break
                        raw_results[ri] = (rebal_date, w)
                prev_w = w

        # ── Assemble all_weights with optional overlap averaging ──────────────
        # Each rebalancing date's weights are held for exactly rebalance_days.
        # When step_days < rebalance_days, consecutive windows overlap; any day
        # covered by multiple weight vectors is assigned their average.
        all_weights_lists: dict[pd.Timestamp, list[np.ndarray]] = {}

        for ri in sorted(raw_results.keys()):
            rebal_date, weights = raw_results[ri]
            hold_end_idx = min(ri + self.rebalance_days, total_days - 1)
            hold_dates = dates[ri + 1 : hold_end_idx + 1]
            for d in hold_dates:
                if d not in all_weights_lists:
                    all_weights_lists[d] = []
                all_weights_lists[d].append(weights)

        if not all_weights_lists:
            raise ValueError("No valid rebalancing points produced weights.")

        # Average overlapping weight vectors (identity when no overlap)
        all_weights: dict[pd.Timestamp, np.ndarray] = {}
        for d, ws in all_weights_lists.items():
            avg = np.mean(np.stack(ws), axis=0)
            s = avg.sum()
            all_weights[d] = avg / s if s > 0 else np.ones(N) / N

        # ── Build output arrays ───────────────────────────────────────────────
        all_dates_sorted = sorted(all_weights.keys())
        weights_matrix = np.array([all_weights[d] for d in all_dates_sorted])
        returns_matrix = returns_df.reindex(all_dates_sorted)[valid_tickers].values
        dates_index = pd.DatetimeIndex(all_dates_sorted)

        # ── Monte Carlo SL calibration ────────────────────────────────────────
        # Calibrate per-ticker stop-loss thresholds using bootstrap simulation.
        # Re-calibrates periodically (every mc_recalibrate_days) so thresholds
        # adapt to changing volatility regimes across the backtest.
        sl_config: dict[int, tuple[float, float, int, float]] | None = None
        sl_config_schedule: list[tuple[int, dict[int, tuple[float, float, int, float]]]] | None = None
        if self.use_mc_sl:
            try:
                from .monte_carlo import calibrate_portfolio

                # Build calibration points: first rebalancing, then every mc_recalibrate_days
                cal_points = [rebal_indices[0]]
                last_cal = rebal_indices[0]
                for ri in rebal_indices[1:]:
                    if ri - last_cal >= self.mc_recalibrate_days:
                        cal_points.append(ri)
                        last_cal = ri

                sl_config_schedule = []
                for cal_idx in cal_points:
                    mc_returns_df = returns_df_real[valid_tickers].iloc[
                        max(0, cal_idx - train_days):cal_idx
                    ]
                    mc_results = calibrate_portfolio(
                        mc_returns_df,
                        horizon=self.rebalance_days,
                        hard_multiplier=self.hard_multiplier,
                        recovery_factor=self.recovery_factor,
                        n_paths=self.mc_n_paths,
                    )
                    # Apply regime-conditional SL scaling if regime detection is active.
                    # Find the closest preceding rebalancing date's regime for this
                    # calibration point (cal_idx may not be a rebal date itself).
                    sl_soft_scale = 1.0
                    sl_hard_scale = 1.0
                    sl_wait_scale = 1.0
                    if pre_computed_regimes:
                        for ri_key in sorted(pre_computed_regimes.keys(), reverse=True):
                            if ri_key <= cal_idx:
                                _, regime_adj = pre_computed_regimes[ri_key]
                                sl_soft_scale = regime_adj.get("sl_soft_scale", 1.0)
                                sl_hard_scale = regime_adj.get("sl_hard_scale", 1.0)
                                sl_wait_scale = regime_adj.get("sl_wait_scale", 1.0)
                                break

                    cal_config = {
                        j: (
                            mc_results[ticker].sl_soft * sl_soft_scale,
                            mc_results[ticker].sl_hard * sl_hard_scale,
                            max(1, round(mc_results[ticker].wait_days * sl_wait_scale)),
                            mc_results[ticker].recovery_factor,
                        )
                        for j, ticker in enumerate(valid_tickers)
                        if ticker in mc_results
                    }
                    sl_config_schedule.append((cal_idx, cal_config))

                # Backward compat: flat sl_config from first calibration point
                sl_config = sl_config_schedule[0][1] if sl_config_schedule else None

                logger.info(
                    f"Second-tower MC SL calibrated at {len(cal_points)} points "
                    f"for {len(valid_tickers)} tickers"
                )
            except Exception as e:
                logger.warning(f"Second-tower MC SL calibration failed: {e}; running without SL")

        # ── Trade log ─────────────────────────────────────────────────────────
        trade_log = _build_trade_log(
            raw_results, rebal_indices, dates, valid_tickers, ohlcv_data,
            self.rebalance_days, total_days,
        )

        # ── Fold performance summary ──────────────────────────────────────────
        fold_details = _build_fold_details(
            weights_matrix, returns_matrix, dates_index,
            close_df[valid_tickers], train_days, self.rebalance_days,
        )

        # ── Factor contributions at last rebalancing date ─────────────────────
        last_ri = rebal_indices[-1]
        last_rebal_date = dates[last_ri]
        last_reb_edgar = _slice_edgar_by_cutoff(edgar_data, last_rebal_date)
        train_start_last = max(0, last_ri - train_days)
        last_reb_ohlcv = {
            t: ohlcv_data[t][
                (ohlcv_data[t].index >= dates[train_start_last]) &
                (ohlcv_data[t].index <= last_rebal_date)
            ]
            for t in valid_tickers if t in ohlcv_data
        }
        last_reb_mkt = market_returns[
            (market_returns.index >= dates[train_start_last]) &
            (market_returns.index <= last_rebal_date)
        ]
        factor_contributions = _compute_factor_contributions(
            last_reb_ohlcv, last_reb_edgar, last_reb_mkt
        )

        weight_history = _build_weight_history(weights_matrix, dates_index, valid_tickers)

        if progress_cb:
            progress_cb(95)

        # ── Signal decay config (passed to backtester) ─────────────────────
        signal_decay_config = None
        if self.use_signal_decay:
            # Build regime-adaptive z-score schedule: at each rebalancing date,
            # scale the base soft/hard exit z by the regime's sd_z_scale.
            z_schedule: list[tuple[int, float, float]] = []
            if pre_computed_regimes:
                for ri in sorted(pre_computed_regimes.keys()):
                    _, regime_adj = pre_computed_regimes[ri]
                    sd_z_scale = regime_adj.get("sd_z_scale", 1.0)
                    z_schedule.append((
                        ri,
                        self.soft_exit_z * sd_z_scale,
                        self.hard_exit_z * sd_z_scale,
                    ))

            signal_decay_config = {
                "signal_check_days": self.signal_check_days,
                "soft_exit_z": self.soft_exit_z,
                "hard_exit_z": self.hard_exit_z,
                "signal_grace_days": self.signal_grace_days,
                "ohlcv_data": ohlcv_data,
                "market_returns": market_returns,
                "tickers": valid_tickers,
                "z_schedule": z_schedule,
            }

        # ── Regime-conditional cash buffer schedule ─────────────────────────
        min_cash_pct_schedule: list[tuple[int, float]] = []
        if pre_computed_regimes:
            for ri in sorted(pre_computed_regimes.keys()):
                _, regime_adj = pre_computed_regimes[ri]
                min_cash_pct_schedule.append((
                    ri, regime_adj.get("min_cash_pct", self.min_cash_pct),
                ))

        # ── Per-regime weight snapshots (for stress testing) ─────────────
        # Collect the last weight vector observed in each regime label.
        regime_weight_snapshots: dict[str, np.ndarray] = {}
        if regime_labels and raw_results:
            # Build bar_idx → regime_label mapping from Timestamp-based regime_labels
            bar_regime: dict[int, str] = {}
            for ri in rebal_indices:
                d = dates[ri]
                if d in regime_labels:
                    bar_regime[ri] = regime_labels[d]
            # For each regime, find the latest rebalancing weight vector
            for ri in sorted(bar_regime.keys()):
                label = bar_regime[ri]
                if ri in raw_results:
                    regime_weight_snapshots[label] = raw_results[ri][1]

        return {
            "weights": weights_matrix,
            "returns": returns_matrix,
            "dates": dates_index,
            "tickers": valid_tickers,
            "fold_details": fold_details,
            "factor_contributions": factor_contributions,
            "weight_history": weight_history,
            "trade_log": trade_log,
            "sl_config": sl_config,
            "sl_config_schedule": sl_config_schedule,
            "tp_threshold": self.tp_threshold,
            "tp_reduce_fraction": self.tp_reduce_fraction,
            "min_cash_pct": self.min_cash_pct,
            "min_cash_pct_schedule": min_cash_pct_schedule,
            "signal_decay_config": signal_decay_config,
            "use_tiered_deleveraging": self.use_tiered_deleveraging,
            "regime_labels": regime_labels,
            "regime_weight_snapshots": regime_weight_snapshots,
        }


# ── Helper functions ──────────────────────────────────────────────────────────

def _build_trade_log(
    raw_results: dict[int, tuple[pd.Timestamp, np.ndarray]],
    rebal_indices: list[int],
    dates: pd.DatetimeIndex,
    valid_tickers: list[str],
    ohlcv_data: dict[str, pd.DataFrame],
    rebalance_days: int,
    total_days: int,
) -> list[dict]:
    """Build per-position trade records for each rebalancing period."""
    trades = []
    for ri in sorted(raw_results.keys()):
        rebal_date, weights = raw_results[ri]
        exit_idx = min(ri + rebalance_days, total_days - 1)
        exit_date = dates[exit_idx]

        for k, ticker in enumerate(valid_tickers):
            w = float(weights[k])
            if w < 1e-4:
                continue
            ticker_df = ohlcv_data.get(ticker)
            if ticker_df is None or ticker_df.empty:
                continue
            try:
                entry_next_idx = min(ri + 1, total_days - 1)
                entry_next_date = dates[entry_next_idx]
                entry_px = float(ticker_df["Close"].asof(entry_next_date))
                exit_px = float(ticker_df["Close"].asof(exit_date))
            except Exception:
                continue
            if not (np.isfinite(entry_px) and np.isfinite(exit_px) and entry_px > 0):
                continue
            hold_ret = exit_px / entry_px - 1.0
            exit_reason = (
                "End of period" if ri + rebalance_days >= total_days - 1
                else "Rebalance"
            )
            trades.append({
                "ticker": ticker,
                "entry_date": str(rebal_date.date()),
                "exit_date": str(exit_date.date()),
                "weight": round(w, 4),
                "entry_price": round(entry_px, 4),
                "exit_price": round(exit_px, 4),
                "holding_return": round(hold_ret, 6),
                "pnl_weighted": round(w * hold_ret, 6),
                "exit_reason": exit_reason,
            })
    return trades


def _build_year_end_snapshots(
    weights: np.ndarray,
    dates: pd.DatetimeIndex,
    tickers: list[str],
    equity_curve: list[dict],
) -> list[dict]:
    """Build year-end portfolio snapshots with per-ticker dollar amounts."""
    # Build a quick lookup from date string to portfolio value
    equity_lookup = {rec["date"]: rec["value"] for rec in equity_curve}

    # Group dates by year and find the last trading day per year
    dates_series = pd.Series(range(len(dates)), index=dates)
    yearly_last = dates_series.groupby(dates_series.index.year).last()

    snapshots = []
    for year, idx in yearly_last.items():
        date = dates[idx]
        date_str = str(date.date())
        total_value = equity_lookup.get(date_str, 0.0)
        if total_value <= 0:
            continue

        year_weights = weights[idx]
        holdings = []
        for k, ticker in enumerate(tickers):
            w = float(year_weights[k])
            if w < 1e-4:
                continue
            holdings.append({
                "ticker": ticker,
                "weight": round(w, 6),
                "dollar_amount": round(w * total_value, 2),
            })
        holdings.sort(key=lambda h: h["dollar_amount"], reverse=True)

        snapshots.append({
            "year": int(year),
            "date": date_str,
            "total_value": round(total_value, 2),
            "holdings": holdings,
        })

    return snapshots


def _build_fold_details(
    weights_matrix: np.ndarray,
    returns_matrix: np.ndarray,
    dates_index: pd.DatetimeIndex,
    close_df: pd.DataFrame,
    train_days: int,
    rebalance_days: int,
) -> list[dict]:
    """Break the backtest into yearly reporting folds and compute per-fold metrics."""
    T = len(dates_index)
    fold_size = 252
    details = []
    fold_num = 0

    for start in range(0, T, fold_size):
        end = min(start + fold_size, T)
        fold_weights = weights_matrix[start:end]
        fold_returns = returns_matrix[start:end]
        fold_dates = dates_index[start:end]

        if len(fold_dates) < 2:
            continue

        port_ret = (fold_weights * fold_returns).sum(axis=1)
        fold_return = float(np.prod(1 + port_ret) - 1)
        fold_sharpe = 0.0
        if np.std(port_ret) > 0:
            fold_sharpe = float(np.mean(port_ret) / np.std(port_ret) * np.sqrt(252))

        wt_changes = np.abs(np.diff(fold_weights, axis=0)).sum(axis=1)  # per-day turnover
        fold_turnover = float(wt_changes.mean() * 252 * 0.5)  # annualized one-way turnover

        first_close_date = close_df.index[0]
        test_start = fold_dates[0]
        train_end_approx = test_start
        if test_start in close_df.index:
            pos = close_df.index.searchsorted(test_start)
            train_start_approx = close_df.index[max(0, pos - train_days)]
        else:
            train_start_approx = first_close_date

        fold_num += 1
        details.append({
            "fold": fold_num,
            "train_start": str(train_start_approx.date()),
            "train_end": str(train_end_approx.date()),
            "test_start": str(fold_dates[0].date()),
            "test_end": str(fold_dates[-1].date()),
            "sharpe": round(fold_sharpe, 3),
            "return_pct": round(fold_return * 100, 2),
            "annual_turnover": round(fold_turnover, 4),
        })

    return details


def _compute_factor_contributions(
    ohlcv_data: dict[str, pd.DataFrame],
    edgar_data: dict[str, dict],
    market_returns: pd.Series,
) -> list[dict]:
    """Compute factor group signal strength at the last rebalancing date."""
    categories = _FACTOR_CATEGORIES
    try:
        factors = compute_all_factors(ohlcv_data, edgar_data, market_returns)
        if factors.empty:
            return [{"category": cat, "avg_weight": 0.0} for cat in categories]

        factor_z = (factors - factors.mean()) / factors.std().replace(0, 1)
        factor_z = factor_z.fillna(0)

        result = []
        for cat, cols in categories.items():
            present = [c for c in cols if c in factor_z.columns]
            if present:
                avg_abs_z = float(factor_z[present].abs().mean().mean())
            else:
                avg_abs_z = 0.0
            result.append({"category": cat, "avg_weight": round(avg_abs_z, 3)})
        return result
    except Exception:
        return [{"category": cat, "avg_weight": 0.0} for cat in categories]


def _build_weight_history(
    weights: np.ndarray,
    dates: pd.DatetimeIndex,
    tickers: list[str],
    max_snapshots: int = 50,
) -> list[dict]:
    """Build weight snapshots for UI visualization."""
    T = len(dates)
    step = max(1, T // max_snapshots)
    snapshots = []
    for t in range(0, T, step):
        w = weights[t]
        top_indices = np.argsort(w)[::-1][:10]
        weight_dict: dict[str, float] = {}
        for i in top_indices:
            if w[i] > 0.001:
                weight_dict[tickers[i]] = round(float(w[i]), 4)
        other = float(np.sum(w) - sum(weight_dict.values()))
        if other > 0.001:
            weight_dict["Other"] = round(other, 4)
        snapshots.append({"date": str(dates[t].date()), "weights": weight_dict})
    return snapshots

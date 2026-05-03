"""Regime detection via multi-feature models (Jump Model / HMM).

Supports 3 or 4 states, probability blending, confirmation filtering,
macro feature integration via FRED + yfinance, SparseJumpModel with
automatic feature selection, and BOCPD fast-trigger layer.

Fallback chain: SparseJM → JumpModel → HMM → Bull default.
"""
from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# ── Regime adjustment multipliers ────────────────────────────────────────────

_REGIME_ADJUSTMENTS: dict[str, dict] = {
    "Bull": {
        "max_weight": 1.0, "target_vol": 1.0, "signal_alpha": 1.0,
        "lambda_scale": 1.0, "cov_halflife": 0,
        # Regime-conditional risk management (research-backed multipliers)
        # SL scales: Lo & Remorov (2017) — tighter SL justified in trending (Bear),
        # wider in mean-reverting (Crisis) to avoid whipsaw stop-outs.
        "sl_soft_scale": 1.0, "sl_hard_scale": 1.0, "sl_wait_scale": 1.0,
        # Signal decay: Daniel & Moskowitz (2016) — momentum crash risk rises in
        # panic states; tighter exit z-scores reduce tail exposure.
        "sd_z_scale": 1.0,
        # Cash buffer: Shu, Yu & Mulvey (2024) — higher cash in stress regimes.
        "min_cash_pct": 0.05,
        "category_weights": {
            "Momentum": 1.0, "Value": 1.0, "Quality": 1.0,
            "Technical": 1.0, "Risk": 1.0, "Accruals": 1.0,
        },
    },
    "Bear": {
        "max_weight": 0.75, "target_vol": 0.75, "signal_alpha": 0.6,
        "lambda_scale": 2.0, "cov_halflife": 42,
        "sl_soft_scale": 0.80, "sl_hard_scale": 0.85, "sl_wait_scale": 0.80,
        "sd_z_scale": 0.75,
        "min_cash_pct": 0.08,
        "category_weights": {
            "Momentum": 0.3, "Value": 1.2, "Quality": 1.5,
            "Technical": 0.5, "Risk": 1.5, "Accruals": 0.8,
        },
    },
    "WalkingOnIce": {
        "max_weight": 0.60, "target_vol": 0.60, "signal_alpha": 0.8,
        "lambda_scale": 1.5, "cov_halflife": 63,
        "sl_soft_scale": 0.90, "sl_hard_scale": 0.90, "sl_wait_scale": 0.85,
        "sd_z_scale": 0.80,
        "min_cash_pct": 0.10,
        "category_weights": {
            "Momentum": 0.6, "Value": 1.0, "Quality": 1.3,
            "Technical": 0.7, "Risk": 1.3, "Accruals": 0.9,
        },
    },
    "Crisis": {
        "max_weight": 0.50, "target_vol": 0.50, "signal_alpha": 0.3,
        "lambda_scale": 3.0, "cov_halflife": 21,
        # Wider soft SL in Crisis (mean-reverting: sharp drops → sharp rebounds)
        # but shorter patience (wait_scale) before forced exit.
        "sl_soft_scale": 1.10, "sl_hard_scale": 1.0, "sl_wait_scale": 0.70,
        "sd_z_scale": 0.60,
        "min_cash_pct": 0.15,
        "category_weights": {
            "Momentum": 0.1, "Value": 0.8, "Quality": 1.8,
            "Technical": 0.3, "Risk": 1.8, "Accruals": 0.6,
        },
    },
}

_STATE_LABELS_4 = ["Bull", "Bear", "WalkingOnIce", "Crisis"]
_STATE_LABELS_3 = ["Bull", "Bear", "Crisis"]

_DEFAULT_ADJ = _REGIME_ADJUSTMENTS["Bull"]

# Fast features used for alert scoring
_FAST_FEATURE_COLS = ["vol_ratio_5_63", "vix_relative", "credit_delta"]


def detect_regime(
    market_returns: pd.Series,
    as_of_date: pd.Timestamp,
    lookback: int = 1500,
    *,
    n_states: int = 4,
    model_type: str = "jump",
    use_macro: bool = True,
    blending: bool = True,
    confirmation_bars: int = 1,
    jump_penalty: float = 20.0,
    use_sparse: bool = True,
    use_bocpd: bool = True,
    bocpd_hazard: float = 1 / 252,
    max_feats: float = 2.0,
    bocpd_last_fired: pd.Timestamp | None = None,
    precomputed_features: pd.DataFrame | None = None,
) -> tuple[str, dict[str, float], pd.Timestamp | None]:
    """Detect market regime using multi-feature models + BOCPD fast-trigger.

    Args:
        market_returns: SPY daily returns series (DatetimeIndex)
        as_of_date: date to classify
        lookback: trailing trading days for model fitting
        n_states: number of regime states (3 or 4)
        model_type: "jump" (Statistical Jump Model) or "hmm" (Gaussian HMM)
        use_macro: whether to fetch FRED/yfinance macro features
        blending: use probability-weighted adjustments instead of hard labels
        confirmation_bars: consecutive bars in new state before switching
        jump_penalty: jump model penalty parameter
        use_sparse: use SparseJumpModel (auto feature selection + mode_loss)
        use_bocpd: run BOCPD fast-trigger layer
        bocpd_hazard: BOCPD prior hazard rate (expected changepoints per day)
        max_feats: SparseJM L1-norm budget for feature selection (must be >= 1.0)
        bocpd_last_fired: timestamp of last BOCPD fire (for cooldown)
        precomputed_features: if provided, skip ``_build_features()`` and slice
            this DataFrame per-window instead (from ``build_all_features()``).
            The strict ``index <= as_of_date`` filter prevents look-ahead.

    Returns:
        (regime_name, adjustment_dict, bocpd_fired_at) — adjustment dict has
        keys max_weight, target_vol, signal_alpha. bocpd_fired_at is the
        as_of_date if BOCPD fired this call, else the passed-in value.
    """
    n_states = max(2, min(n_states, 4))
    state_labels = _STATE_LABELS_4[:n_states] if n_states == 4 else _STATE_LABELS_3[:n_states]

    # ── Build feature matrix ─────────────────────────────────────────────
    if precomputed_features is not None:
        features = _slice_precomputed(precomputed_features, as_of_date, lookback)
    else:
        features = _build_features(market_returns, as_of_date, lookback, use_macro)
    if features is None or len(features) < 100:
        logger.info(
            f"Regime: insufficient data ({0 if features is None else len(features)}/100), "
            f"defaulting to Bull"
        )
        return "Bull", _DEFAULT_ADJ.copy(), bocpd_last_fired

    X_raw = features.values
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    # ── Compute returns aligned to features for jump model sorting ───────
    aligned_returns = market_returns.reindex(features.index).fillna(0)

    # ── Fit model ────────────────────────────────────────────────────────
    states, probs = _fit_and_predict(
        X, aligned_returns, n_states, model_type, X_raw, features.columns.tolist(),
        jump_penalty=jump_penalty,
        use_sparse=use_sparse,
        max_feats=max_feats,
    )

    if states is None:
        return "Bull", _DEFAULT_ADJ.copy(), bocpd_last_fired

    # ── State labeling ───────────────────────────────────────────────────
    label_map = _build_label_map(
        states, X_raw, features.columns.tolist(), state_labels, model_type=model_type,
    )

    # ── BOCPD fast-trigger + graduated response ──────────────────────────
    bocpd_fired_at = bocpd_last_fired
    if use_bocpd and probs is not None and blending:
        alert_score = _compute_fast_alert_score(features)
        bocpd_prob = 0.0

        # Check cooldown: 10 trading days (~14 calendar days)
        bocpd_in_cooldown = (
            bocpd_last_fired is not None
            and (as_of_date - bocpd_last_fired).days < 14
        )

        if not bocpd_in_cooldown:
            # Run BOCPD on raw SPY returns
            mask = market_returns.index <= as_of_date
            recent_returns = market_returns[mask].iloc[-min(lookback, 504):]
            if len(recent_returns) >= 50:
                bocpd_prob = _bocpd_changepoint_prob(
                    recent_returns.values, hazard_rate=bocpd_hazard,
                )

        bocpd_fired = bocpd_prob > 0.5 and not bocpd_in_cooldown

        # Determine tier
        tier = _compute_alert_tier(alert_score, bocpd_fired)

        if tier >= 2:
            # Shift probability toward stress states
            probs = _apply_fast_trigger_shift(probs, label_map, state_labels, tier)
            if bocpd_fired:
                bocpd_fired_at = as_of_date
            logger.info(
                f"Regime fast-trigger at {as_of_date.date()}: tier={tier}, "
                f"alert_score={alert_score}, bocpd_prob={bocpd_prob:.3f}"
            )

    # ── Confirmation filter ──────────────────────────────────────────────
    labeled_states = [label_map.get(s, "Bull") for s in states]
    regime = _apply_confirmation(labeled_states, confirmation_bars)

    # ── Compute adjustment dict ──────────────────────────────────────────
    if blending and probs is not None:
        adj = _blend_adjustments(probs[-1], label_map)
    else:
        adj = _REGIME_ADJUSTMENTS.get(regime, _DEFAULT_ADJ).copy()

    logger.info(
        f"Regime at {as_of_date.date()}: {regime} "
        f"(model={model_type}, states={n_states}, features={len(features.columns)}, "
        f"blending={blending}, sparse={use_sparse})"
    )

    return regime, adj, bocpd_fired_at


# ── Feature construction ─────────────────────────────────────────────────────

def _build_features(
    market_returns: pd.Series,
    as_of_date: pd.Timestamp,
    lookback: int,
    use_macro: bool,
) -> pd.DataFrame | None:
    """Build the feature matrix from SPY returns + optional macro features."""
    mask = market_returns.index <= as_of_date
    trailing = market_returns[mask].iloc[-(lookback + 50):]  # extra for rolling warmup

    if len(trailing) < 100:
        return None

    # Core features (always available)
    roll_vol = trailing.rolling(21).std().dropna() * np.sqrt(252)
    roll_ret = trailing.rolling(21).mean().dropna() * 252
    # Drawdown from rolling 63-day high — captures sustained declines
    # (e.g., 2022 rate shock) that don't spike short-term vol
    cum_ret = (1 + trailing).cumprod()
    rolling_max = cum_ret.rolling(63, min_periods=1).max()
    drawdown = (cum_ret / rolling_max - 1).reindex(roll_vol.index)
    core = pd.DataFrame({
        "roll_vol": roll_vol,
        "roll_ret": roll_ret,
        "drawdown_63": drawdown,
    }).dropna()

    if use_macro:
        try:
            from data.macro_fetcher import fetch_macro_features
            macro = fetch_macro_features(market_returns, as_of_date, lookback)
            if macro is not None and not macro.empty:
                core = core.join(macro, how="left")
        except Exception as e:
            logger.warning(f"Macro feature fetch failed: {e}")

    # Trim to lookback window
    core = core[core.index <= as_of_date].iloc[-lookback:]

    # Drop columns with >50% NaN
    nan_pct = core.isna().mean()
    keep = nan_pct[nan_pct <= 0.5].index.tolist()
    core = core[keep]

    # Forward-fill then drop remaining NaN rows
    core = core.ffill().dropna()

    if len(core) < 100:
        return None

    return core


# ── Bulk feature pre-computation ────────────────────────────────────────────

def build_all_features(
    market_returns: pd.Series,
    rebal_dates: list[pd.Timestamp],
    lookback: int = 1500,
    use_macro: bool = True,
) -> pd.DataFrame | None:
    """Pre-compute the full feature matrix once for all rebalancing dates.

    Computes core rolling features and bulk-fetches macro data for the widest
    date range.  The result must be sliced per-window via ``_slice_precomputed``
    to prevent look-ahead bias.

    Rolling/EWM ops are backward-looking so computing on the full series then
    slicing ``[:as_of_date]`` is mathematically identical to computing on a
    pre-sliced series.  The drawdown ratio is scale-invariant w.r.t. cumprod
    starting point.

    Returns None if insufficient data.
    """
    if not rebal_dates:
        return None

    # Core features on the full market_returns series
    roll_vol = market_returns.rolling(21).std().dropna() * np.sqrt(252)
    roll_ret = market_returns.rolling(21).mean().dropna() * 252

    cum_ret = (1 + market_returns).cumprod()
    rolling_max = cum_ret.rolling(63, min_periods=1).max()
    drawdown = (cum_ret / rolling_max - 1).reindex(roll_vol.index)

    core = pd.DataFrame({
        "roll_vol": roll_vol,
        "roll_ret": roll_ret,
        "drawdown_63": drawdown,
    }).dropna()

    if len(core) < 100:
        return None

    # Macro features (bulk fetch — single network round-trip)
    if use_macro:
        try:
            from data.macro_fetcher import fetch_macro_features_bulk
            earliest = min(rebal_dates)
            start_date = earliest - pd.Timedelta(days=lookback + 50)
            end_date = max(rebal_dates)
            macro = fetch_macro_features_bulk(market_returns, start_date, end_date)
            if macro is not None and not macro.empty:
                core = core.join(macro, how="left")
        except Exception as e:
            logger.warning(f"Bulk macro feature fetch failed: {e}")

    return core


def _slice_precomputed(
    full_features: pd.DataFrame,
    as_of_date: pd.Timestamp,
    lookback: int,
) -> pd.DataFrame | None:
    """Slice pre-computed features for a single as_of_date window.

    Replicates the per-window post-processing of ``_build_features()``
    (lines 254-268): causal trim, NaN column drop, ffill, row drop.

    The ``index <= as_of_date`` filter is the primary defense against
    look-ahead bias.
    """
    # Strict causal filter — no data after as_of_date
    window = full_features[full_features.index <= as_of_date].iloc[-lookback:]

    if len(window) < 100:
        return None

    # Defensive assertion against look-ahead
    assert window.index.max() <= as_of_date, (
        f"Look-ahead detected: feature date {window.index.max()} > as_of_date {as_of_date}"
    )

    # Drop columns with >50% NaN (per-window — availability varies by date)
    nan_pct = window.isna().mean()
    keep = nan_pct[nan_pct <= 0.5].index.tolist()
    window = window[keep]

    # Forward-fill then drop remaining NaN rows
    window = window.ffill().dropna()

    if len(window) < 100:
        return None

    return window


# ── Model fitting ────────────────────────────────────────────────────────────

def _fit_and_predict(
    X: np.ndarray,
    returns: pd.Series,
    n_states: int,
    model_type: str,
    X_raw: np.ndarray,
    feature_names: list[str],
    *,
    jump_penalty: float = 50.0,
    use_sparse: bool = True,
    max_feats: float = 2.0,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Fit the regime model and return (state_sequence, probability_matrix).

    Returns (None, None) on failure.
    """
    # Try jump model first (sparse variant preferred)
    if model_type == "jump":
        if use_sparse:
            result = _fit_sparse_jump_model(
                X, returns, n_states,
                jump_penalty=jump_penalty, max_feats=max_feats,
            )
            if result is not None:
                return result
            logger.info("SparseJM failed/unavailable, falling back to standard JM")

        result = _fit_jump_model(X, returns, n_states, jump_penalty=jump_penalty)
        if result is not None:
            return result

        logger.info("Jump model failed/unavailable, falling back to HMM")

    # Try HMM
    result = _fit_hmm(X, n_states)
    if result is not None:
        return result

    logger.warning("All regime models failed, defaulting to Bull")
    return None, None


def _fit_sparse_jump_model(
    X: np.ndarray,
    returns: pd.Series,
    n_states: int,
    *,
    jump_penalty: float = 20.0,
    max_feats: float = 2.0,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Fit Sparse Jump Model with auto feature selection + mode loss."""
    try:
        from jumpmodels.sparse_jump import SparseJumpModel
    except ImportError:
        logger.info("SparseJumpModel not available in jumpmodels package")
        return None

    try:
        X_df = pd.DataFrame(X, index=returns.index)
        ret_ser = returns.copy()

        sjm = SparseJumpModel(
            n_components=n_states,
            jump_penalty=jump_penalty,
            cont=False,
            mode_loss=True,
            max_feats=max_feats,
        )
        sjm.fit(X_df, ret_ser=ret_ser, sort_by="cumret")

        states = sjm.predict_online(X_df)
        try:
            probs = sjm.predict_proba_online(X_df)
        except Exception:
            probs = None

        if states is None or len(states) == 0:
            return None

        states = np.asarray(states, dtype=int)

        if probs is not None:
            probs = np.asarray(probs)
            if probs.ndim != 2 or probs.shape[0] != len(states):
                probs = None

        return states, probs

    except Exception as e:
        logger.warning(f"Sparse jump model fitting failed: {e}")
        return None


def _fit_jump_model(
    X: np.ndarray,
    returns: pd.Series,
    n_states: int,
    *,
    jump_penalty: float = 50.0,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Fit Statistical Jump Model (Nystrup et al.)."""
    try:
        from jumpmodels.jump import JumpModel
    except ImportError:
        logger.info("jumpmodels not installed; jump model unavailable")
        return None

    try:
        X_df = pd.DataFrame(X, index=returns.index)
        ret_ser = returns.copy()

        jm = JumpModel(n_components=n_states, jump_penalty=jump_penalty, cont=False)
        jm.fit(X_df, ret_ser=ret_ser, sort_by="cumret")

        # Use online predict to avoid lookahead bias
        states = jm.predict_online(X_df)
        try:
            probs = jm.predict_proba_online(X_df)
        except Exception:
            probs = None

        if states is None or len(states) == 0:
            return None

        states = np.asarray(states, dtype=int)

        if probs is not None:
            probs = np.asarray(probs)
            if probs.ndim != 2 or probs.shape[0] != len(states):
                probs = None

        return states, probs

    except Exception as e:
        logger.warning(f"Jump model fitting failed: {e}")
        return None


def _fit_hmm(
    X: np.ndarray,
    n_states: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Fit Gaussian HMM via hmmlearn."""
    try:
        from hmmlearn.hmm import GaussianHMM
    except ImportError:
        logger.info("hmmlearn not installed; HMM unavailable")
        return None

    try:
        model = GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            n_iter=300,
            random_state=42,
            tol=1e-4,
        )
        # Fit on all-but-last observation to avoid lookahead bias for
        # the current date's classification (which is the only one used).
        if len(X) > 2:
            model.fit(X[:-1])
        else:
            model.fit(X)
            states = model.predict(X)
            probs = model.predict_proba(X)
            return np.asarray(states, dtype=int), np.asarray(probs)

        # Forward-only state estimation: run Viterbi on X[:-1] (causal),
        # then compute last state via one-step forward update to avoid
        # the bidirectional look-ahead inherent in full-sequence Viterbi.
        states_hist = model.predict(X[:-1])
        probs_hist = model.predict_proba(X[:-1])

        # P(s_T) = sum_{s_{T-1}} P(s_T | s_{T-1}) * P(s_{T-1} | x_{1..T-1})
        pred_prob = probs_hist[-1] @ model.transmat_

        # P(x_T | s_T) for each state — Gaussian emission
        n_states_local = model.n_components
        emission_prob = np.zeros(n_states_local)
        for k_state in range(n_states_local):
            diff = X[-1].ravel() - model.means_[k_state].ravel()
            cov = model.covars_[k_state]
            # Use log-space for numerical stability
            sign, logdet = np.linalg.slogdet(cov)
            if sign <= 0:
                emission_prob[k_state] = 1e-300
                continue
            inv_cov = np.linalg.inv(cov)
            log_p = -0.5 * (diff @ inv_cov @ diff + logdet + len(diff) * np.log(2 * np.pi))
            emission_prob[k_state] = np.exp(np.clip(log_p, -700, 0))

        # P(s_T | x_{1..T}) ~ P(x_T | s_T) * P(s_T)
        posterior = pred_prob * emission_prob
        denom = posterior.sum()
        if denom > 0:
            posterior /= denom
        else:
            posterior = np.ones(n_states_local) / n_states_local

        last_state = int(np.argmax(posterior))
        states = np.append(states_hist, last_state)
        probs = np.vstack([probs_hist, posterior.reshape(1, -1)])

        return np.asarray(states, dtype=int), np.asarray(probs)

    except Exception as e:
        logger.warning(f"HMM fitting failed: {e}")
        return None


# ── State labeling ───────────────────────────────────────────────────────────

def _build_label_map(
    states: np.ndarray,
    X_raw: np.ndarray,
    feature_names: list[str],
    state_labels: list[str],
    *,
    model_type: str = "jump",
) -> dict[int, str]:
    """Map numeric state IDs to regime labels.

    For Jump Model: trusts the library's sort_by="cumret" ordering.
    State 0 = highest cumulative return (Bull), state N-1 = lowest (Crisis).

    For HMM: uses composite risk score = roll_vol - roll_ret + downside_dev.
    Lowest risk → Bull, highest → Crisis.
    """
    unique_states = sorted(set(states))
    n_labels = len(state_labels)

    if model_type == "jump":
        # JumpModel sorts states by decreasing cumulative return via
        # sort_by="cumret".  State 0 = best returns (Bull), state N-1 =
        # worst returns (Crisis).  Trust this ordering directly.
        label_map: dict[int, str] = {}
        for i, s in enumerate(unique_states):
            label_idx = min(i, n_labels - 1)
            label_map[s] = state_labels[label_idx]
        return label_map

    # ── HMM path: sort by composite risk score ───────────────────────
    vol_idx = feature_names.index("roll_vol") if "roll_vol" in feature_names else 0
    ret_idx = feature_names.index("roll_ret") if "roll_ret" in feature_names else None
    dd_indices = [
        i for i, name in enumerate(feature_names)
        if "downside_dev" in name
    ]

    risk_scores: dict[int, float] = {}
    for s in unique_states:
        s_mask = states == s
        if not s_mask.any():
            risk_scores[s] = float("inf")
            continue
        # Higher vol = riskier
        score = float(X_raw[s_mask, vol_idx].mean())
        # Lower return = riskier (subtract so negative returns increase score)
        if ret_idx is not None:
            score -= float(X_raw[s_mask, ret_idx].mean())
        # Higher downside deviation = riskier
        for dd_idx in dd_indices:
            score += float(X_raw[s_mask, dd_idx].mean())
        risk_scores[s] = score

    # Sort by risk score (lowest = safest regime)
    sorted_states = sorted(risk_scores.keys(), key=lambda s: risk_scores[s])

    label_map = {}
    for i, s in enumerate(sorted_states):
        label_idx = min(i, n_labels - 1)
        label_map[s] = state_labels[label_idx]

    # Log separation quality
    if len(sorted_states) >= 2:
        spread = risk_scores[sorted_states[-1]] - risk_scores[sorted_states[0]]
        if spread < 0.05 and spread != float("inf"):
            logger.warning(
                f"Regime: low risk spread ({spread:.4f}) — states may not be well separated"
            )

    return label_map


# ── BOCPD fast-trigger ───────────────────────────────────────────────────────

def _bocpd_changepoint_prob(
    returns: np.ndarray,
    hazard_rate: float = 1 / 252,
    run_length_threshold: int = 5,
) -> float:
    """Bayesian Online Changepoint Detection on return series.

    Uses Normal-Inverse-Gamma conjugate prior for computational efficiency.
    Returns P(current run length <= threshold), i.e., the probability that
    a changepoint occurred within the last `run_length_threshold` days.

    Based on Adams & MacKay (2007).
    """
    T = len(returns)
    if T < 10:
        return 0.0

    # Prior hyperparameters (weakly informative)
    mu0 = 0.0
    kappa0 = 1.0
    alpha0 = 1.0
    beta0 = 1e-4

    # Hazard function: constant prior probability of changepoint
    log_H = math.log(hazard_rate)
    log_1mH = math.log(1.0 - hazard_rate)

    # Run length log-probabilities: R[r] = log P(run_length = r)
    # We maintain a vector of size (current_t + 1) at each step
    max_rl = min(T, 300)  # cap for efficiency

    # Sufficient statistics for each run length
    # (mu, kappa, alpha, beta) per run length
    mus = np.full(max_rl + 1, mu0)
    kappas = np.full(max_rl + 1, kappa0)
    alphas = np.full(max_rl + 1, alpha0)
    betas = np.full(max_rl + 1, beta0)

    # Log-probabilities of run lengths
    log_R = np.full(max_rl + 1, -np.inf)
    log_R[0] = 0.0  # start with run length 0

    for t in range(T):
        x = returns[t]
        n_rl = min(t + 1, max_rl)

        # Predictive probability: Student-t with updated params
        # P(x | run_length = r) for each active run length
        log_pred = np.full(n_rl, -np.inf)
        for r in range(n_rl):
            # Student-t predictive
            nu = 2.0 * alphas[r]
            sigma2 = betas[r] * (kappas[r] + 1.0) / (alphas[r] * kappas[r])
            if sigma2 <= 0 or nu <= 0:
                continue
            # Log PDF of Student-t
            z = (x - mus[r]) ** 2 / sigma2
            log_pred[r] = (
                math.lgamma((nu + 1) / 2) - math.lgamma(nu / 2)
                - 0.5 * math.log(nu * math.pi * sigma2)
                - ((nu + 1) / 2) * math.log(1 + z / nu)
            )

        # Growth probabilities: extend each run length
        log_growth = log_R[:n_rl] + log_pred + log_1mH

        # Changepoint probability: sum over all run lengths
        log_cp = np.logaddexp.reduce(log_R[:n_rl] + log_pred + log_H)

        # Update run length distribution
        new_log_R = np.full(max_rl + 1, -np.inf)
        new_log_R[0] = log_cp
        limit = min(n_rl + 1, max_rl + 1)
        new_log_R[1:limit] = log_growth[:limit - 1]

        # Normalize
        log_evidence = np.logaddexp.reduce(new_log_R[:limit])
        new_log_R[:limit] -= log_evidence

        # Update sufficient statistics for each run length
        new_mus = np.full(max_rl + 1, mu0)
        new_kappas = np.full(max_rl + 1, kappa0)
        new_alphas = np.full(max_rl + 1, alpha0)
        new_betas = np.full(max_rl + 1, beta0)

        for r in range(1, limit):
            prev_r = r - 1
            k_old = kappas[prev_r]
            mu_old = mus[prev_r]
            a_old = alphas[prev_r]
            b_old = betas[prev_r]

            k_new = k_old + 1
            mu_new = (k_old * mu_old + x) / k_new
            a_new = a_old + 0.5
            b_new = b_old + k_old * (x - mu_old) ** 2 / (2 * k_new)

            new_mus[r] = mu_new
            new_kappas[r] = k_new
            new_alphas[r] = a_new
            new_betas[r] = b_new

        log_R = new_log_R
        mus = new_mus
        kappas = new_kappas
        alphas = new_alphas
        betas = new_betas

    # P(run_length <= threshold) = P(changepoint in last `threshold` days)
    n_active = min(T + 1, max_rl + 1)
    threshold = min(run_length_threshold, n_active - 1)
    if threshold < 0:
        return 0.0

    log_short = np.logaddexp.reduce(log_R[:threshold + 1])
    return float(np.exp(log_short))


def _compute_fast_alert_score(features: pd.DataFrame) -> int:
    """Count how many fast features are elevated (z-score > 2.0 relative
    to their own 252-day rolling history).

    Adaptive thresholds: what's 'elevated' in low-vol 2017 is different
    from high-vol 2022.
    """
    score = 0
    for col in _FAST_FEATURE_COLS:
        if col not in features.columns:
            continue
        series = features[col].dropna()
        if len(series) < 63:
            continue
        # Use up to 252 days of history for z-score baseline
        history = series.iloc[-min(252, len(series)):]
        val = series.iloc[-1]
        mu = history.mean()
        sigma = history.std()
        if sigma > 0 and (val - mu) / sigma > 2.0:
            score += 1
    return score


def _compute_alert_tier(alert_score: int, bocpd_fired: bool) -> int:
    """Determine the graduated response tier.

    Tier 0: quiet — no action
    Tier 1: mild (1 fast feature) — log only
    Tier 2: moderate (2+ features OR BOCPD alone) — 15% prob shift
    Tier 3: strong (2+ features AND BOCPD) — 30% prob shift
    """
    if alert_score >= 2 and bocpd_fired:
        return 3
    if alert_score >= 2 or bocpd_fired:
        return 2
    if alert_score >= 1:
        return 1
    return 0


def _apply_fast_trigger_shift(
    probs: np.ndarray,
    label_map: dict[int, str],
    state_labels: list[str],
    tier: int,
) -> np.ndarray:
    """Shift the final probability vector toward stress states.

    Tier 2: shift 15% of Bull probability mass
    Tier 3: shift 30% of Bull probability mass
    """
    probs = probs.copy()
    shift_pct = 0.15 if tier == 2 else 0.30

    # Find Bull and stress state indices in the probability vector
    bull_idx = None
    crisis_idx = None
    bear_idx = None
    for state_id, label in label_map.items():
        if state_id >= probs.shape[1]:
            continue
        if label == "Bull":
            bull_idx = state_id
        elif label == "Crisis":
            crisis_idx = state_id
        elif label == "Bear":
            bear_idx = state_id

    if bull_idx is None:
        return probs

    shift = probs[-1, bull_idx] * shift_pct
    probs[-1, bull_idx] -= shift

    if crisis_idx is not None and bear_idx is not None:
        probs[-1, crisis_idx] += shift * 0.6
        probs[-1, bear_idx] += shift * 0.4
    elif crisis_idx is not None:
        probs[-1, crisis_idx] += shift
    elif bear_idx is not None:
        probs[-1, bear_idx] += shift

    # Renormalize
    row_sum = probs[-1].sum()
    if row_sum > 0:
        probs[-1] /= row_sum

    return probs


# ── Confirmation filter ──────────────────────────────────────────────────────

def _apply_confirmation(
    labeled_states: list[str],
    confirmation_bars: int,
) -> str:
    """Apply confirmation filter: only switch regime if last N bars agree.

    Thread-safe — uses only the decoded state sequence, no external state.
    """
    if confirmation_bars <= 1 or len(labeled_states) == 0:
        return labeled_states[-1] if labeled_states else "Bull"

    if len(labeled_states) < confirmation_bars:
        return labeled_states[-1]

    tail = labeled_states[-confirmation_bars:]
    if all(s == tail[0] for s in tail):
        return tail[0]

    # Not confirmed — find the most recent confirmed regime by scanning backward
    for i in range(len(labeled_states) - confirmation_bars, -1, -1):
        window = labeled_states[i : i + confirmation_bars]
        if all(s == window[0] for s in window):
            return window[0]

    # No confirmed regime found — use the most common label
    return max(set(labeled_states), key=labeled_states.count)


# ── Probability blending ────────────────────────────────────────────────────

def _blend_adjustments(
    probs: np.ndarray,
    label_map: dict[int, str],
) -> dict:
    """Blend regime adjustments using probability weights.

    Produces a weighted average of adjustment dicts across all states,
    eliminating hard regime switching.
    """
    _SCALAR_KEYS = (
        "max_weight", "target_vol", "signal_alpha", "lambda_scale", "cov_halflife",
        "sl_soft_scale", "sl_hard_scale", "sl_wait_scale", "sd_z_scale", "min_cash_pct",
    )
    _CAT_NAMES = list(_REGIME_ADJUSTMENTS["Bull"]["category_weights"].keys())

    adj: dict = {k: 0.0 for k in _SCALAR_KEYS}
    adj["category_weights"] = {cat: 0.0 for cat in _CAT_NAMES}

    total_prob = 0.0
    for state_id, label in label_map.items():
        if state_id >= len(probs):
            continue
        p = float(probs[state_id])
        state_adj = _REGIME_ADJUSTMENTS.get(label, _DEFAULT_ADJ)
        for key in _SCALAR_KEYS:
            adj[key] += p * state_adj[key]
        state_cat_w = state_adj.get("category_weights", {})
        for cat in _CAT_NAMES:
            adj["category_weights"][cat] += p * state_cat_w.get(cat, 1.0)
        total_prob += p

    # Normalize if probabilities don't sum to 1
    if total_prob > 0 and abs(total_prob - 1.0) > 0.01:
        for key in _SCALAR_KEYS:
            adj[key] /= total_prob
        for cat in _CAT_NAMES:
            adj["category_weights"][cat] /= total_prob

    # Safety clamps
    adj["max_weight"] = max(0.1, min(1.0, adj["max_weight"]))
    adj["target_vol"] = max(0.1, min(1.5, adj["target_vol"]))
    adj["signal_alpha"] = max(0.1, min(2.0, adj["signal_alpha"]))
    adj["lambda_scale"] = max(0.5, min(5.0, adj["lambda_scale"]))
    # Blended halflife < 10 is too short to be meaningful — treat as 0 (standard LW)
    adj["cov_halflife"] = max(0, int(adj["cov_halflife"])) if adj["cov_halflife"] >= 10 else 0
    # Regime-conditional risk management clamps
    adj["sl_soft_scale"] = max(0.5, min(1.5, adj["sl_soft_scale"]))
    adj["sl_hard_scale"] = max(0.5, min(1.5, adj["sl_hard_scale"]))
    adj["sl_wait_scale"] = max(0.3, min(1.5, adj["sl_wait_scale"]))
    adj["sd_z_scale"] = max(0.3, min(1.5, adj["sd_z_scale"]))
    adj["min_cash_pct"] = max(0.02, min(0.30, adj["min_cash_pct"]))

    return adj

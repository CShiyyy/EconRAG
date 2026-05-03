"""Portfolio-level backtester for CNN+Transformer and Second Tower weight outputs."""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any

from .metrics import compute_metrics, compute_drawdown_series, compute_rolling_sharpe
from .backtester import RISK_CONFIG


def run_portfolio_backtest(
    weights: np.ndarray,       # (T, N) daily weights
    returns: np.ndarray,       # (T, N) daily asset returns aligned to weights
    dates: pd.DatetimeIndex,   # length T
    initial_capital: float = 100_000.0,
    risk_level: str = "medium",
    fee_rate: float = 0.001,
    slippage: float = 0.001,
    volume: np.ndarray | None = None,        # (T, N) daily share volumes
    close_prices: np.ndarray | None = None,  # (T, N) daily close prices
    initial_weights: np.ndarray | None = None,  # prior period's ending weights
    benchmark_returns: pd.Series | None = None,
    use_drawdown_stop: bool = True,
    # ── Monte Carlo soft/hard SL ─────────────────────────────────────────────
    # sl_config: {ticker_index → OptimalSL} or None (disabled)
    # Passed as a plain dict of ints→(sl_soft, sl_hard, wait_days, recovery_factor)
    # to avoid importing OptimalSL here.
    sl_config: dict[int, tuple[float, float, int, float]] | None = None,
    # sl_config_schedule: time-varying SL — list of (calibration_bar, per_ticker_config)
    # When provided, the backtester uses the most recent calibration point for each bar.
    sl_config_schedule: list[tuple[int, dict[int, tuple[float, float, int, float]]]] | None = None,
    # ── Take-profit liquidity release ────────────────────────────────────────
    tp_threshold: float | None = None,   # e.g. 0.15 → release when position up 15%
    tp_reduce_fraction: float = 0.5,     # fraction of weight to release on TP
    min_cash_pct: float = 0.05,          # TP only fires when cash below this
    # ── Market impact model ────────────────────────────────────────────────
    impact_model: str = "linear",         # "linear" (default) or "sqrt" (Almgren et al. 2005)
    # ── Signal decay exit ──────────────────────────────────────────────────
    signal_decay_config: dict | None = None,
    # ── Regime-conditional cash buffer ───────────────────────────────────
    # list of (bar_index, min_cash_pct) — overrides scalar min_cash_pct per regime
    min_cash_pct_schedule: list[tuple[int, float]] | None = None,
    # ── Tiered deleveraging ──────────────────────────────────────────────
    use_tiered_deleveraging: bool = False,
    deleverage_regime_aware: bool = False,
    regime_labels: dict[int, str] | None = None,
) -> dict[str, Any]:
    """
    Run a portfolio backtest given daily weight and return arrays.

    Daily portfolio return = sum(w_i * r_i) - tc
    tc = (fee_rate + slippage) * sum(|w_i[t] - w_i[t-1]|)

    Per-position exits (when sl_config is provided):
      Soft SL: if price drops below entry × (1 + sl_soft), start a wait_days timer.
               If price recovers by recovery_factor of the gap before timer expires, cancel.
               If timer expires without recovery, exit the position.
      Hard SL: if price drops below entry × (1 + sl_hard), exit immediately.
      Resets on rebalancing (when optimizer changes a position's target weight).

    Take-profit liquidity release (when tp_threshold is set):
      Only fires when portfolio cash allocation < min_cash_pct (fully invested).
      Reduces position weight by tp_reduce_fraction when position is up tp_threshold.
      Freed weight goes to cash until the next rebalancing.

    Returns a dict in the same format as run_backtest():
      - metrics: dict of performance metrics
      - equity: list of {date, value} dicts
      - drawdown: list of {date, value} dicts
      - rolling_sharpe: list of {date, value} dicts
      - signals: list of {date, weight_sum} dicts
    """
    risk = RISK_CONFIG.get(risk_level, RISK_CONFIG["medium"])
    drawdown_stop = risk["drawdown_stop"]

    T, N = weights.shape
    assert len(dates) == T, f"dates length {len(dates)} != weights rows {T}"
    assert returns.shape == (T, N), f"returns shape {returns.shape} != ({T}, {N})"

    # NaN guard: zero out weights where returns are NaN.
    # Do NOT rescale — when a position NaNs out (delisting), total allocation
    # drops proportionally. Rescaling to preserve the original weight sum causes
    # unintended leverage spikes (e.g. a 40% long + 10% short where short delists
    # would rescale the long to 50%). The optimizer handles rebalancing at the
    # next rebalance date.
    nan_mask = np.isnan(returns)
    if nan_mask.any():
        clean_returns = np.where(nan_mask, 0.0, returns)
        clean_weights = np.where(nan_mask, 0.0, weights)
    else:
        clean_returns = returns
        clean_weights = weights

    # ── Next-bar execution: weights computed at bar T apply at bar T+1 ────────
    # Mirrors the shift(1) in the single-asset backtester (backtester.py line 153).
    # Without this, the portfolio earns bar T's return using weights decided at
    # bar T's close — a look-ahead bias.
    shifted = np.zeros_like(clean_weights)
    shifted[1:] = clean_weights[:-1]
    if initial_weights is not None and initial_weights.shape == (N,):
        shifted[0] = initial_weights
    clean_weights = shifted

    # ── Pre-compute ADV matrix for market impact ──────────────────────────────
    adv: np.ndarray | None = None
    if volume is not None and close_prices is not None:
        safe_close = np.where(np.isnan(close_prices), 0.0, close_prices)
        safe_volume = np.where(np.isnan(volume), 0.0, volume)
        dollar_volume = safe_close * safe_volume
        adv = np.zeros_like(dollar_volume)
        for j in range(N):
            s = pd.Series(dollar_volume[:, j])
            adv[:, j] = s.rolling(20, min_periods=1).mean().values
        adv = np.where(adv < 1e-10, 1e-10, adv)

    WEIGHT_THRESHOLD = 1e-6
    use_sl = (sl_config is not None or sl_config_schedule is not None) and close_prices is not None
    use_tp = tp_threshold is not None and tp_threshold > 0 and close_prices is not None

    # ── Signal decay exit pre-computation ──────────────────────────────────────
    use_signal_decay = signal_decay_config is not None
    sd_scores: dict[int, float] | None = None  # bar → per-ticker z-score cache
    sd_grace_start = np.full(N, -1, dtype=int)
    sd_soft_exit_z = signal_decay_config["soft_exit_z"] if use_signal_decay else -0.5
    sd_hard_exit_z = signal_decay_config["hard_exit_z"] if use_signal_decay else -1.5
    sd_grace_days = signal_decay_config["signal_grace_days"] if use_signal_decay else 5
    sd_check_days = signal_decay_config["signal_check_days"] if use_signal_decay else 7

    # Regime-adaptive z-score schedule: list of (bar_idx, soft_z, hard_z)
    sd_z_schedule: list[tuple[int, float, float]] = (
        signal_decay_config.get("z_schedule", []) if use_signal_decay else []
    )
    # Active z thresholds (updated when we pass a z_schedule entry)
    sd_active_soft_z = sd_soft_exit_z
    sd_active_hard_z = sd_hard_exit_z

    # Regime-conditional cash buffer schedule: list of (bar_idx, min_cash_pct)
    active_min_cash_pct = min_cash_pct
    mcp_schedule = min_cash_pct_schedule or []

    # Pre-compute signal decay z-scores at check intervals
    if use_signal_decay:
        from strategies.factors import compute_fast_factors
        from strategies.second_tower import _category_balanced_composite, _FACTOR_CATEGORIES
        sd_ohlcv = signal_decay_config["ohlcv_data"]
        sd_tickers_list = None  # will be set from caller context
        sd_scores_by_bar: dict[int, dict[str, float]] = {}

        # Compute fast factor scores at each signal_check_days interval
        for t_idx in range(0, T, sd_check_days):
            check_date = dates[t_idx]
            # Slice OHLCV up to check_date for each ticker
            sliced = {}
            for ticker_name in (sd_ohlcv or {}):
                df = sd_ohlcv[ticker_name]
                if df is not None and not df.empty:
                    sliced[ticker_name] = df[df.index <= check_date]
            if not sliced:
                continue
            try:
                factors = compute_fast_factors(sliced)
                if factors.empty or len(factors) < 3:
                    continue
                fz = (factors - factors.mean()) / factors.std().replace(0, 1)
                fz = fz.replace([np.inf, -np.inf], 0).fillna(0)
                # Filter to only Momentum + Technical categories
                mom_tech_cols = []
                for cat in ("Momentum", "Technical"):
                    cat_cols = _FACTOR_CATEGORIES.get(cat, [])
                    mom_tech_cols.extend([c for c in cat_cols if c in fz.columns])
                if mom_tech_cols:
                    composite = fz[mom_tech_cols].mean(axis=1)
                else:
                    composite = fz.mean(axis=1)
                sd_scores_by_bar[t_idx] = composite.to_dict()
            except Exception:
                continue

    # ── Per-position SL state ─────────────────────────────────────────────────
    entry_prices = np.full(N, np.nan)        # price when position opened
    warning_active = np.zeros(N, dtype=bool)
    warning_start_t = np.full(N, -1, dtype=int)

    # ── Per-position TP state ─────────────────────────────────────────────────
    tp_triggered = np.zeros(N, dtype=bool)   # weight already partially reduced by TP

    prev_effective_w = (
        initial_weights.copy()
        if initial_weights is not None and initial_weights.shape == (N,)
        else np.zeros(N)
    )

    equity_vals = np.zeros(T)
    effective_w_history = np.zeros((T, N))  # track actual holdings for trade log
    capital = initial_capital
    peak = initial_capital
    portfolio_stopped = False

    # ── Tiered deleveraging controller ─────────────────────────────────────
    delev_controller = None
    delev_exposure_mult = 1.0  # T+1 model: mult set at bar T, applied at bar T+1
    if use_tiered_deleveraging:
        from engine.deleverage import DeleverageController
        delev_controller = DeleverageController(
            risk_level=risk_level,
            regime_aware=deleverage_regime_aware,
        )

    # Track actual SL/TP exits for trade log enrichment
    sl_exits: list[dict] = []

    for t in range(T):
        if portfolio_stopped:
            equity_vals[t] = capital
            continue

        target_w = clean_weights[t]
        prev_target_w = clean_weights[t - 1] if t > 0 else np.zeros(N)

        # ── Detect rebalancing events, reset per-position state ───────────────
        # Notify deleverage controller of regime changes at rebalancing
        if delev_controller is not None and regime_labels is not None:
            is_rebal = any(abs(target_w[j] - prev_target_w[j]) > 1e-6 for j in range(N))
            if is_rebal:
                for ri in sorted(regime_labels.keys(), reverse=True):
                    if ri <= t:
                        delev_controller.update_regime(regime_labels[ri])
                        break

        for j in range(N):
            w_change = abs(target_w[j] - prev_target_w[j])
            # Only reset SL state on meaningful weight changes (new position or >5% relative)
            is_new_position = prev_target_w[j] <= WEIGHT_THRESHOLD and target_w[j] > WEIGHT_THRESHOLD
            is_significant = w_change > max(0.05 * prev_target_w[j], 1e-6) if prev_target_w[j] > WEIGHT_THRESHOLD else w_change > 1e-6
            if is_new_position or is_significant:
                # Optimizer issued new/significantly changed weight → fresh position start
                warning_active[j] = False
                warning_start_t[j] = -1
                tp_triggered[j] = False
                if target_w[j] > WEIGHT_THRESHOLD and close_prices is not None:
                    cp_j = close_prices[t, j]
                    entry_prices[j] = cp_j if not np.isnan(cp_j) else np.nan
                else:
                    entry_prices[j] = np.nan
            elif target_w[j] > WEIGHT_THRESHOLD and np.isnan(entry_prices[j]):
                # First bar for this position (t=0 case)
                if close_prices is not None:
                    cp_j = close_prices[t, j]
                    if not np.isnan(cp_j):
                        entry_prices[j] = cp_j

        # ── Update regime-conditional schedules at current bar ────────────────
        # Signal decay z-score schedule: find the latest entry <= t
        for _zs_bar, _zs_soft, _zs_hard in sd_z_schedule:
            if _zs_bar <= t:
                sd_active_soft_z = _zs_soft
                sd_active_hard_z = _zs_hard
            else:
                break
        # Min cash percentage schedule: find the latest entry <= t
        for _mcp_bar, _mcp_val in mcp_schedule:
            if _mcp_bar <= t:
                active_min_cash_pct = _mcp_val
            else:
                break

        # ── Build effective weights (starts from target, may be modified) ─────
        effective_w = target_w.copy()

        # ── Tiered deleveraging (T+1: mult was computed at previous bar) ─────
        if delev_controller is not None and delev_exposure_mult < 1.0:
            effective_w = effective_w * delev_exposure_mult

        # ── Soft / Hard SL checks ─────────────────────────────────────────────
        if use_sl:
            for j in range(N):
                if target_w[j] <= WEIGHT_THRESHOLD or np.isnan(entry_prices[j]):
                    continue

                # Look up SL params — prefer time-varying schedule, fall back to flat config
                sl_params = None
                if sl_config_schedule is not None:
                    active_cfg = None
                    for cal_bar, cal_cfg in sl_config_schedule:
                        if cal_bar <= t:
                            active_cfg = cal_cfg
                        else:
                            break
                    if active_cfg is not None and j in active_cfg:
                        sl_params = active_cfg[j]
                if sl_params is None and sl_config is not None and j in sl_config:
                    sl_params = sl_config[j]
                if sl_params is None:
                    continue

                sl_soft, sl_hard, wait_days, recovery_factor = sl_params
                cur_price = float(close_prices[t, j])
                if np.isnan(cur_price):
                    continue

                entry_p = entry_prices[j]
                soft_level = entry_p * (1.0 + sl_soft)    # e.g. entry * 0.95
                hard_level = entry_p * (1.0 + sl_hard)    # e.g. entry * 0.90
                # Recovery cancel: price recovers (1 - recovery_factor) of the drop
                recovery_level = soft_level * (1.0 + abs(sl_soft) * recovery_factor)

                # Hard SL — immediate exit
                if cur_price < hard_level:
                    effective_w[j] = 0.0
                    warning_active[j] = False
                    entry_prices[j] = np.nan
                    sl_exits.append({"ticker_idx": j, "bar": t, "exit_reason": "Hard Stop Loss", "exit_price": cur_price})
                    continue

                # Soft SL — start warning
                if cur_price < soft_level and not warning_active[j]:
                    warning_active[j] = True
                    warning_start_t[j] = t

                # Recovery — cancel warning
                if warning_active[j] and cur_price > recovery_level:
                    warning_active[j] = False
                    warning_start_t[j] = -1

                # Warning expired — forced exit
                if warning_active[j] and (t - warning_start_t[j]) >= wait_days:
                    effective_w[j] = 0.0
                    warning_active[j] = False
                    entry_prices[j] = np.nan
                    sl_exits.append({"ticker_idx": j, "bar": t, "exit_reason": "Soft Stop Loss", "exit_price": cur_price})

        # ── TP liquidity release ──────────────────────────────────────────────
        if use_tp:
            # Estimate current cash as 1 - sum(effective weights)
            current_invested = float(np.sum(effective_w))
            cash_pct = max(0.0, 1.0 - current_invested)

            if cash_pct < active_min_cash_pct:
                for j in range(N):
                    if effective_w[j] <= WEIGHT_THRESHOLD or tp_triggered[j]:
                        continue
                    if np.isnan(entry_prices[j]):
                        continue
                    cur_price = float(close_prices[t, j])
                    if np.isnan(cur_price):
                        continue
                    gain = cur_price / entry_prices[j] - 1.0
                    if gain >= tp_threshold:
                        # Reduce position by tp_reduce_fraction, free rest to cash
                        effective_w[j] *= (1.0 - tp_reduce_fraction)
                        tp_triggered[j] = True  # don't fire again until next rebalance
                        sl_exits.append({"ticker_idx": j, "bar": t, "exit_reason": "Take Profit", "exit_price": cur_price})

        # ── Regime-conditional cash floor enforcement ──────────────────────
        # If regime demands higher cash buffer, proportionally scale down weights.
        if mcp_schedule:
            total_w = float(np.sum(effective_w))
            max_invested = 1.0 - active_min_cash_pct
            if total_w > max_invested + 1e-8 and total_w > 1e-8:
                effective_w *= (max_invested / total_w)

        # ── Signal decay exit ───────────────────────────────────────────────
        # Use floor-division to look up the most recent pre-computed check bar.
        # sd_scores_by_bar is populated only at sd_check_days intervals; checking
        # `t in sd_scores_by_bar` would silently skip all inter-interval bars.
        check_t = (t // sd_check_days) * sd_check_days if use_signal_decay else 0
        if use_signal_decay and check_t in sd_scores_by_bar:
            scores = sd_scores_by_bar[check_t]
            for j in range(N):
                if effective_w[j] <= WEIGHT_THRESHOLD:
                    sd_grace_start[j] = -1
                    continue
                # Find closest ticker name — use index from caller
                # tickers list not directly available; use clean_weights shape
                # sd_scores keyed by ticker name, need mapping
                # For now, scores are keyed by ticker name; we need the ticker list
                # which was passed via signal_decay_config indirectly
                # The tickers are the keys of sd_ohlcv that match ordering
                # Skip if no score available for this ticker index
                pass  # handled below

            # Look up by ticker name if we have a ticker mapping
            if signal_decay_config and "tickers" in signal_decay_config:
                sd_ticker_list = signal_decay_config["tickers"]
                for j in range(N):
                    if effective_w[j] <= WEIGHT_THRESHOLD:
                        sd_grace_start[j] = -1
                        continue
                    if j >= len(sd_ticker_list):
                        continue
                    ticker_name = sd_ticker_list[j]
                    z = scores.get(ticker_name, 0.0)

                    # Hard exit: immediate (threshold may be regime-adjusted)
                    if z < sd_active_hard_z:
                        effective_w[j] = 0.0
                        sd_grace_start[j] = -1
                        sl_exits.append({"ticker_idx": j, "bar": t, "exit_reason": "Signal Decay (hard)", "exit_price": float(close_prices[t, j]) if close_prices is not None else 0.0})
                        continue

                    # Soft exit: start grace period (threshold may be regime-adjusted)
                    if z < sd_active_soft_z:
                        if sd_grace_start[j] < 0:
                            sd_grace_start[j] = t
                        elif (t - sd_grace_start[j]) >= sd_grace_days:
                            effective_w[j] = 0.0
                            sd_grace_start[j] = -1
                            sl_exits.append({"ticker_idx": j, "bar": t, "exit_reason": "Signal Decay (soft)", "exit_price": float(close_prices[t, j]) if close_prices is not None else 0.0})
                    else:
                        sd_grace_start[j] = -1  # recovered

        # ── Transaction costs on actual weight changes ────────────────────────
        w_change = effective_w - prev_effective_w

        if adv is not None:
            k = 0.1
            trade_dollar = np.abs(w_change) * capital
            if impact_model == "sqrt":
                # Almgren et al. 2005: square-root market impact replaces flat slippage.
                impact = k * np.sqrt(trade_dollar / adv[t])
                tc = (
                    fee_rate * np.abs(w_change).sum()
                    + (impact * np.abs(w_change)).sum()
                )
            else:
                # Linear market impact (original model)
                impact = 1.0 + k * trade_dollar / adv[t]
                tc = (
                    fee_rate * np.abs(w_change).sum()
                    + slippage * (np.abs(w_change) * impact).sum()
                )
        else:
            tc = (fee_rate + slippage) * np.abs(w_change).sum()

        # ── Daily portfolio return ────────────────────────────────────────────
        port_ret_t = float((effective_w * clean_returns[t]).sum()) - tc
        capital = capital * (1.0 + port_ret_t)
        equity_vals[t] = capital

        if capital > peak:
            peak = capital

        dd = (capital - peak) / peak
        if delev_controller is not None:
            delev_exposure_mult = delev_controller.step(dd)
            if delev_controller.is_hard_stopped:
                portfolio_stopped = True
        elif use_drawdown_stop and dd < -drawdown_stop:
            portfolio_stopped = True

        effective_w_history[t] = effective_w
        prev_effective_w = effective_w.copy()

    equity_series = pd.Series(equity_vals, index=dates)
    returns_series = equity_series.pct_change().fillna(0)

    # ── Trade-level metrics (from weight transitions) ─────────────────────────
    trades_records = []
    WTHRESH = 1e-6
    for j in range(N):
        in_trade = False
        entry_idx = 0
        for t in range(T):
            has_pos = abs(effective_w_history[t, j]) > WTHRESH
            if not in_trade and has_pos:
                in_trade = True
                entry_idx = t
            elif in_trade and not has_pos:
                cum_ret = float(np.prod(1 + clean_returns[entry_idx:t, j]) - 1)
                trades_records.append({
                    "pnl": cum_ret,
                    "duration": (
                        (dates[t] - dates[entry_idx]).days
                        if hasattr(dates[entry_idx], "day")
                        else int(t - entry_idx)
                    ),
                })
                in_trade = False
        if in_trade:
            cum_ret = float(np.prod(1 + clean_returns[entry_idx:T, j]) - 1)
            trades_records.append({
                "pnl": cum_ret,
                "duration": (
                    (dates[T - 1] - dates[entry_idx]).days
                    if hasattr(dates[entry_idx], "day")
                    else int(T - 1 - entry_idx)
                ),
            })

    trades_df = pd.DataFrame(trades_records) if trades_records else None

    metrics = compute_metrics(
        equity_series, returns_series,
        trades_df=trades_df, benchmark_returns=benchmark_returns,
    )
    drawdown_series = compute_drawdown_series(equity_series)
    rolling_sharpe_series = compute_rolling_sharpe(returns_series)

    def s2r(s: pd.Series, vk: str = "value") -> list[dict]:
        s = s.replace([float("inf"), float("-inf")], float("nan")).dropna()
        return [
            {
                "date": str(idx_.date()) if hasattr(idx_, "date") else str(idx_),
                vk: round(float(v), 6),
            }
            for idx_, v in s.items()
        ]

    SIGNAL_DELTA = 0.01   # 1% weight change threshold to fire a signal
    signals_records = []
    for t in range(T):
        date_str = str(dates[t].date()) if hasattr(dates[t], "date") else str(dates[t])
        has_buy = has_sell = False
        for j in range(N):
            if t == 0:
                if clean_weights[t, j] > WTHRESH:
                    has_buy = True
            else:
                delta = clean_weights[t, j] - clean_weights[t - 1, j]
                if delta > SIGNAL_DELTA:
                    has_buy = True
                elif delta < -SIGNAL_DELTA:
                    has_sell = True
        if has_buy:
            signals_records.append({
                "date": date_str,
                "close": round(float(equity_vals[t]), 4),
                "signal": 1,
            })
        if has_sell:
            signals_records.append({
                "date": date_str,
                "close": round(float(equity_vals[t]), 4),
                "signal": -1,
            })

    return {
        "metrics": metrics,
        "equity": s2r(equity_series, "value"),
        "drawdown": s2r(drawdown_series, "value"),
        "rolling_sharpe": s2r(rolling_sharpe_series, "value"),
        "signals": signals_records,
        "sl_exits": sl_exits,
        "deleverage_transitions": delev_controller.transitions if delev_controller else [],
    }

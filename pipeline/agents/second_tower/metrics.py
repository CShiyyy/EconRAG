"""Performance metrics computation."""
import math
import numpy as np
import pandas as pd
from typing import Any


def compute_metrics(
    equity: pd.Series,
    returns: pd.Series,
    trades_df: pd.DataFrame | None = None,
    rf_annual: float = 0.04,
    benchmark_returns: pd.Series | None = None,
) -> dict[str, Any]:
    """
    Compute all performance metrics from an equity curve and returns series.

    Args:
        equity: Portfolio value series (Date-indexed).
        returns: Daily returns series.
        trades_df: Optional trades DataFrame with entry/exit info.
        rf_annual: Annual risk-free rate (default 4%).

    Returns:
        Dictionary of metric name -> value.
    """
    metrics: dict[str, Any] = {}

    # Clean data
    equity = equity.dropna()
    returns = returns.dropna()

    if len(equity) < 2:
        return {k: None for k in [
            "cagr", "sharpe", "sortino", "calmar", "max_drawdown",
            "max_drawdown_duration", "win_rate", "profit_factor",
            "avg_win_loss_ratio", "num_trades", "avg_holding_period",
            "var_95", "cvar_95",
        ]}

    # CAGR — use actual calendar time to handle gaps correctly
    n_years = (equity.index[-1] - equity.index[0]).days / 365.25
    total_return = equity.iloc[-1] / equity.iloc[0]
    metrics["cagr"] = float(total_return ** (1 / n_years) - 1) if n_years > 0 else 0.0

    # Sharpe
    rf_daily = (1 + rf_annual) ** (1 / 252) - 1
    excess = returns - rf_daily
    std = returns.std()
    metrics["sharpe"] = float(excess.mean() / std * np.sqrt(252)) if std > 0 else 0.0

    # Sortino — proper downside deviation (all days, min(excess, 0))
    # If no returns fall below rf (perfect bull period), downside_std is 0;
    # return None rather than inf so the UI can show "N/A" instead of Infinity.
    downside_diff = np.minimum(returns - rf_daily, 0)
    downside_std = np.sqrt((downside_diff ** 2).mean())
    if downside_std > 0:
        metrics["sortino"] = float(excess.mean() / downside_std * np.sqrt(252))
    else:
        metrics["sortino"] = None

    # Max Drawdown
    rolling_max = equity.cummax()
    drawdown = (equity - rolling_max) / rolling_max
    metrics["max_drawdown"] = float(drawdown.min())

    # Max Drawdown Duration
    underwater = drawdown < 0
    if underwater.any():
        max_dur = 0
        cur_dur = 0
        for val in underwater:
            if val:
                cur_dur += 1
                max_dur = max(max_dur, cur_dur)
            else:
                cur_dur = 0
        metrics["max_drawdown_duration"] = int(max_dur)
    else:
        metrics["max_drawdown_duration"] = 0

    # Calmar
    mdd = abs(metrics["max_drawdown"])
    metrics["calmar"] = float(metrics["cagr"] / mdd) if mdd > 0 else 0.0

    # VaR and CVaR (95%, 1-day, historical)
    if len(returns) > 0:
        var_95 = float(np.percentile(returns, 5))
        metrics["var_95"] = var_95
        cvar_mask = returns <= var_95
        metrics["cvar_95"] = float(returns[cvar_mask].mean()) if cvar_mask.any() else var_95
    else:
        metrics["var_95"] = 0.0
        metrics["cvar_95"] = 0.0

    # Trade-level metrics (from trades_df if provided)
    if trades_df is not None and not trades_df.empty:
        _compute_trade_metrics(metrics, trades_df)
    else:
        metrics["win_rate"] = None
        metrics["profit_factor"] = None
        metrics["avg_win_loss_ratio"] = None
        metrics["num_trades"] = 0
        metrics["avg_holding_period"] = None

    # Benchmark-relative metrics (information ratio, up/down capture, alpha, beta)
    if benchmark_returns is not None and len(benchmark_returns) > 1:
        _compute_benchmark_metrics(metrics, returns, benchmark_returns)

    return metrics


def _compute_trade_metrics(metrics: dict, trades_df: pd.DataFrame) -> None:
    """Extract win rate, profit factor, etc. from a trades DataFrame."""
    # Trade dicts from the pure-pandas backtester have columns: pnl, duration, etc.
    if "pnl" in trades_df.columns:
        pnl = trades_df["pnl"].dropna()
    elif "PnL" in trades_df.columns:
        pnl = trades_df["PnL"].dropna()
    else:
        pnl = pd.Series(dtype=float)

    if len(pnl) == 0:
        metrics["win_rate"] = None
        metrics["profit_factor"] = None
        metrics["avg_win_loss_ratio"] = None
        metrics["num_trades"] = 0
        metrics["avg_holding_period"] = None
        return

    metrics["num_trades"] = int(len(pnl))

    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]

    metrics["win_rate"] = float(len(wins) / len(pnl)) if len(pnl) > 0 else 0.0
    gross_profit = wins.sum()
    gross_loss = abs(losses.sum())
    metrics["profit_factor"] = float(gross_profit / gross_loss) if gross_loss > 0 else None
    avg_win = wins.mean() if len(wins) > 0 else 0.0
    avg_loss = abs(losses.mean()) if len(losses) > 0 else 0.0
    metrics["avg_win_loss_ratio"] = float(avg_win / avg_loss) if avg_loss > 0 else None

    # Holding period
    if "duration" in trades_df.columns:
        dur = trades_df["duration"].dropna()
        if hasattr(dur.iloc[0], "days") if len(dur) > 0 else False:
            metrics["avg_holding_period"] = float(dur.apply(lambda d: d.days).mean())
        else:
            metrics["avg_holding_period"] = float(dur.mean())
    else:
        metrics["avg_holding_period"] = None

    # Sanitize any remaining inf/nan before returning
    for k, v in list(metrics.items()):
        if isinstance(v, float) and not math.isfinite(v):
            metrics[k] = None


def _compute_benchmark_metrics(
    metrics: dict,
    returns: pd.Series,
    benchmark_returns: pd.Series,
) -> None:
    """Add information ratio, up/down capture, alpha, and beta vs benchmark."""
    try:
        aligned = pd.DataFrame({"p": returns, "b": benchmark_returns}).dropna()
        if len(aligned) < 20:
            return
        p, b = aligned["p"], aligned["b"]

        # Information ratio: annualised active return / tracking error
        active = p - b
        te = float(active.std())
        metrics["information_ratio"] = float(active.mean() / te * np.sqrt(252)) if te > 0 else None

        # Up-capture: mean portfolio return on benchmark up-days / mean benchmark return on those days
        up_mask = b > 0
        if up_mask.sum() > 5:
            metrics["up_capture"] = float(p[up_mask].mean() / b[up_mask].mean()) if b[up_mask].mean() != 0 else None
        else:
            metrics["up_capture"] = None

        # Down-capture: mean portfolio return on benchmark down-days / mean benchmark return on those days
        dn_mask = b < 0
        if dn_mask.sum() > 5:
            metrics["down_capture"] = float(p[dn_mask].mean() / b[dn_mask].mean()) if b[dn_mask].mean() != 0 else None
        else:
            metrics["down_capture"] = None

        # Beta and annualised alpha via OLS
        cov_mat = np.cov(p.values, b.values)
        beta = float(cov_mat[0, 1] / cov_mat[1, 1]) if cov_mat[1, 1] > 0 else 0.0
        alpha = float((p.mean() - beta * b.mean()) * 252)
        metrics["beta_vs_benchmark"] = beta
        metrics["alpha_vs_benchmark"] = alpha
    except Exception:
        pass

    # Sanitize
    for k in ("information_ratio", "up_capture", "down_capture", "beta_vs_benchmark", "alpha_vs_benchmark"):
        v = metrics.get(k)
        if isinstance(v, float) and not math.isfinite(v):
            metrics[k] = None


def compute_concentration_metrics(weights: np.ndarray) -> dict[str, float]:
    """Compute time-averaged HHI and Effective N Bets from a (T, N) weight matrix.

    HHI = sum(w_i^2), ranges from 1/N (diversified) to 1 (concentrated).
    Effective N Bets = 1 / HHI — intuitive count of independent positions.
    """
    if weights.ndim != 2 or weights.shape[0] == 0:
        return {"hhi": 0.0, "effective_n_bets": 0.0}
    daily_hhi = (weights ** 2).sum(axis=1)
    avg_hhi = float(np.mean(daily_hhi))
    enb = float(1.0 / avg_hhi) if avg_hhi > 1e-10 else 0.0
    return {"hhi": round(avg_hhi, 4), "effective_n_bets": round(enb, 1)}


def compute_rolling_alpha(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    rf_annual: float = 0.04,
    window: int = 63,
) -> pd.Series:
    """Rolling Jensen's alpha (annualized) via trailing OLS regression.

    At each date, regresses trailing `window` days of portfolio excess returns
    against benchmark excess returns and extracts the annualized intercept.
    Positive = portfolio earned more than CAPM predicts; negative = underperformed.
    Uses rolling covariance/variance for O(n) efficiency.
    """
    rf_daily = (1 + rf_annual) ** (1 / 252) - 1
    aligned = pd.DataFrame({"p": portfolio_returns, "b": benchmark_returns}).dropna()
    if len(aligned) < window + 5:
        return pd.Series(dtype=float)

    p_excess = aligned["p"] - rf_daily
    b_excess = aligned["b"] - rf_daily

    roll_cov = p_excess.rolling(window).cov(b_excess)
    roll_var = b_excess.rolling(window).var()
    roll_beta = roll_cov / roll_var.replace(0, np.nan)
    residuals = p_excess - roll_beta * b_excess
    roll_alpha = residuals.rolling(window).mean() * 252  # annualize

    return roll_alpha.dropna().replace([np.inf, -np.inf], np.nan).dropna()


def compute_signal_ic(
    signals: list[dict],
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    rf_annual: float = 0.04,
    horizons: list[int] | None = None,
) -> dict:
    """Information Coefficient: how well signals predict forward alpha.

    For each buy/sell signal date, computes the portfolio's CAPM-adjusted
    excess return over the next h days. IC = Spearman corr(signal, fwd_alpha).
    Directional accuracy = fraction of signals with correct alpha direction (21d).

    Returns dict with ic_5d, ic_21d, ic_63d, directional_accuracy.
    """
    if horizons is None:
        horizons = [5, 21, 63]

    empty: dict = {f"ic_{h}d": None for h in horizons}
    empty["directional_accuracy"] = None

    if not signals or len(portfolio_returns) < max(horizons) + 10:
        return empty

    rf_daily = (1 + rf_annual) ** (1 / 252) - 1
    aligned = pd.DataFrame({"p": portfolio_returns, "b": benchmark_returns}).dropna()
    if len(aligned) < max(horizons) + 10:
        return empty

    p = aligned["p"]
    b = aligned["b"]

    # Full-period beta (simplified; avoids per-signal regression)
    cov_mat = np.cov(p.values, b.values)
    beta_full = float(cov_mat[0, 1] / cov_mat[1, 1]) if cov_mat[1, 1] > 0 else 1.0

    result: dict = {}
    primary_sigs: list[float] = []
    primary_alphas: list[float] = []

    for h in horizons:
        sigs: list[float] = []
        fwd_alphas: list[float] = []

        for s_rec in signals:
            sig = s_rec.get("signal", 0)
            if sig == 0:
                continue
            try:
                ts = pd.Timestamp(s_rec["date"])
                loc = p.index.get_indexer([ts], method="nearest")[0]
            except Exception:
                continue

            if loc < 0 or loc + h >= len(p):
                continue

            p_w = p.iloc[loc : loc + h]
            b_w = b.iloc[loc : loc + h]
            if len(p_w) < h:
                continue

            p_ret = float((1 + p_w).prod() - 1)
            b_ret = float((1 + b_w).prod() - 1)
            rf_h = float((1 + rf_daily) ** h - 1)
            fwd_alpha = (p_ret - rf_h) - beta_full * (b_ret - rf_h)

            sigs.append(float(sig))
            fwd_alphas.append(fwd_alpha)

        if len(sigs) < 5:
            result[f"ic_{h}d"] = None
        else:
            ic = float(pd.Series(sigs).corr(pd.Series(fwd_alphas), method="spearman"))
            result[f"ic_{h}d"] = round(ic, 4) if math.isfinite(ic) else None

        if h == 21:
            primary_sigs = sigs
            primary_alphas = fwd_alphas

    # Directional accuracy (using 21d horizon)
    if len(primary_sigs) >= 5:
        correct = sum(
            1 for s, a in zip(primary_sigs, primary_alphas)
            if (s > 0 and a > 0) or (s < 0 and a < 0)
        )
        result["directional_accuracy"] = round(correct / len(primary_sigs), 4)
    else:
        result["directional_accuracy"] = None

    return result


def compute_drawdown_series(equity: pd.Series) -> pd.Series:
    """Return percentage drawdown series (0 to -1)."""
    rolling_max = equity.cummax()
    return (equity - rolling_max) / rolling_max


def compute_rolling_sharpe(returns: pd.Series, window: int = 252, rf_annual: float = 0.04) -> pd.Series:
    """Rolling annualized Sharpe ratio."""
    rf_daily = (1 + rf_annual) ** (1 / 252) - 1
    excess = returns - rf_daily
    rolling_mean = excess.rolling(window).mean()
    rolling_std = returns.rolling(window).std()
    sharpe = rolling_mean / rolling_std * np.sqrt(252)
    sharpe = sharpe.replace([np.inf, -np.inf], np.nan)
    return sharpe

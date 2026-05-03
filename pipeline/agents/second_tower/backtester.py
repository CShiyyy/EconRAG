"""Pure-pandas backtesting engine (replaces vectorbt for Python 3.14 compat)."""
import logging

import numpy as np
import pandas as pd
from .metrics import (
    compute_metrics,
    compute_drawdown_series,
    compute_rolling_sharpe,
)
from typing import Any

logger = logging.getLogger(__name__)

RISK_CONFIG = {
    "low":    {"kelly_fraction": 0.25, "max_position": 0.05,  "drawdown_stop": 0.10},
    "medium": {"kelly_fraction": 0.50, "max_position": 0.10,  "drawdown_stop": 0.20},
    "high":   {"kelly_fraction": 1.00, "max_position": 0.25,  "drawdown_stop": 0.40},
}

MIN_TRADES_FOR_KELLY = 10


def _kelly_position_size(
    trade_records: list[dict],
    kelly_fraction: float,
    max_position: float,
    min_trades: int = MIN_TRADES_FOR_KELLY,
) -> float:
    """
    Compute position size using Kelly Criterion based on completed trades.

    f* = (p * b - q) / b
    Final size = min(kelly_fraction * max(f*, 0), max_position)

    Falls back to max_position when fewer than min_trades have been completed.
    """
    if len(trade_records) < min_trades:
        return max_position

    wins = [t["pnl"] for t in trade_records if t["pnl"] > 0]
    losses = [t["pnl"] for t in trade_records if t["pnl"] <= 0]

    if not losses:
        # All winners — f* = 1.0, cap at max_position
        return max_position

    p = len(wins) / len(trade_records)
    q = 1 - p

    if p == 0:
        # No edge — use minimum floor
        return max_position * 0.1

    avg_win = sum(wins) / len(wins)
    avg_loss = abs(sum(losses) / len(losses))

    if avg_loss < 1e-12:
        return max_position

    b = avg_win / avg_loss
    f_star = (p * b - q) / b

    if f_star <= 0:
        # Negative edge — trade at minimum floor to allow recovery
        return max_position * 0.1

    return min(kelly_fraction * f_star, max_position)

DEFAULT_FEE_RATE = 0.001   # 0.1% commission per trade
DEFAULT_SLIPPAGE = 0.001   # 0.1% slippage per trade


def _effective_slippage(
    base_slippage: float,
    trade_value: float,
    volume: pd.Series | None,
    idx: int,
    price: float,
    k: float = 0.1,
) -> float:
    """Scale slippage by market impact: slippage * (1 + k * trade_value / avg_daily_dollar_volume).

    If volume is not provided, returns base_slippage unchanged.
    """
    if volume is None:
        return base_slippage
    window = volume.iloc[max(0, idx - 20):idx + 1]
    avg_vol = float(window.mean()) if len(window) > 0 else 0.0
    dollar_vol = avg_vol * price
    if dollar_vol < 1e-10:
        return base_slippage
    return base_slippage * (1.0 + k * trade_value / dollar_vol)


def run_backtest(
    price: pd.Series,
    entries: pd.Series,
    exits: pd.Series,
    short_entries: pd.Series | None = None,
    short_exits: pd.Series | None = None,
    initial_capital: float = 100_000.0,
    risk_level: str = "medium",
    freq: str = "D",
    fee_rate: float = DEFAULT_FEE_RATE,
    slippage: float = DEFAULT_SLIPPAGE,
    full_allocation: bool = False,
    use_drawdown_stop: bool = True,
    cash_return_rate: float = 0.0,
    rf_annual: float = 0.04,
    volume: pd.Series | None = None,
) -> dict[str, Any]:
    """
    Run a pure-pandas backtest and return metrics + time series data.

    Returns dict with:
      - metrics: dict of performance metrics
      - equity: list of {date, value} dicts
      - drawdown: list of {date, value} dicts
      - rolling_sharpe: list of {date, value} dicts
      - signals: list of {date, close, signal} dicts
    """
    risk = RISK_CONFIG.get(risk_level, RISK_CONFIG["medium"])
    kelly_fraction = risk["kelly_fraction"]
    max_position = risk["max_position"]
    drawdown_stop = risk["drawdown_stop"]
    daily_cash_rate = (1 + cash_return_rate) ** (1 / 252) - 1 if cash_return_rate > 0 else 0.0

    price = price.dropna()
    idx = price.index
    n = len(price)

    # Align volume to price index for market impact model
    if volume is not None:
        volume = volume.reindex(idx)

    if n < 2:
        empty: dict[str, Any] = {
            "metrics": compute_metrics(pd.Series(dtype=float), pd.Series(dtype=float)),
            "equity": [], "drawdown": [], "rolling_sharpe": [], "signals": [],
        }
        return empty

    # Align signals to price index
    entries = entries.reindex(idx).fillna(False).astype(bool)
    exits = exits.reindex(idx).fillna(False).astype(bool)
    use_short = short_entries is not None
    if use_short:
        short_entries = short_entries.reindex(idx).fillna(False).astype(bool)
        short_exits = short_exits.reindex(idx).fillna(False).astype(bool)

    # --- Next-bar execution: signal on bar T → trade on bar T+1 ---
    entries = entries.shift(1).fillna(False).astype(bool)
    exits = exits.shift(1).fillna(False).astype(bool)
    if use_short:
        short_entries = short_entries.shift(1).fillna(False).astype(bool)
        short_exits = short_exits.shift(1).fillna(False).astype(bool)

    # Simulate position: 0=flat, 1=long, -1=short
    cash = initial_capital
    holdings = 0.0          # number of shares held (positive=long, negative=short)
    position = 0            # 0, 1, -1
    equity_vals = np.zeros(n)
    trade_records = []       # (entry_date, exit_date, pnl, entry_price, exit_price)

    # Peak tracking for drawdown stop
    peak_equity = initial_capital
    stopped_out = False
    entry_price = 0.0
    entry_date = None

    for i in range(n):
        # Accrue interest on idle cash
        if position == 0 and daily_cash_rate > 0:
            cash *= (1 + daily_cash_rate)

        p = float(price.iloc[i])
        cur_equity = cash + holdings * p
        equity_vals[i] = cur_equity

        # Update peak
        if cur_equity > peak_equity:
            peak_equity = cur_equity

        # Check drawdown stop
        dd = (cur_equity - peak_equity) / peak_equity
        if use_drawdown_stop and dd < -drawdown_stop and position != 0:
            liq_value = abs(holdings) * p
            eff_slip = _effective_slippage(slippage, liq_value, volume, i, p)
            if position == 1:
                # Liquidate long
                proceeds = holdings * p * (1 - fee_rate) * (1 - eff_slip)
                pnl = proceeds - (holdings * entry_price)
                cash += proceeds
            else:
                # Liquidate short (buy to cover)
                shares_short = -holdings
                buy_cost = shares_short * p * (1 + fee_rate) * (1 + eff_slip)
                pnl = shares_short * entry_price * (1 - fee_rate) - buy_cost
                cash -= buy_cost
            trade_records.append({
                "entry_date": entry_date,
                "exit_date": idx[i],
                "pnl": pnl,
                "entry_price": entry_price,
                "exit_price": p,
                "duration": idx[i] - entry_date if entry_date else None,
            })
            holdings = 0.0
            position = 0
            stopped_out = True
            continue

        # Check exits first
        if position == 1 and exits.iloc[i]:
            # Close long
            exit_value = holdings * p
            eff_slip = _effective_slippage(slippage, exit_value, volume, i, p)
            proceeds = holdings * p * (1 - fee_rate) * (1 - eff_slip)
            pnl = proceeds - (holdings * entry_price)
            trade_records.append({
                "entry_date": entry_date,
                "exit_date": idx[i],
                "pnl": pnl,
                "entry_price": entry_price,
                "exit_price": p,
                "duration": idx[i] - entry_date if entry_date else None,
            })
            cash += proceeds
            holdings = 0.0
            position = 0

        elif use_short and position == -1 and short_exits.iloc[i]:
            # Close short (buy to cover)
            shares_short = -holdings
            exit_value = shares_short * p
            eff_slip = _effective_slippage(slippage, exit_value, volume, i, p)
            buy_cost = shares_short * p * (1 + fee_rate) * (1 + eff_slip)
            pnl = shares_short * entry_price * (1 - fee_rate) - buy_cost
            trade_records.append({
                "entry_date": entry_date,
                "exit_date": idx[i],
                "pnl": pnl,
                "entry_price": entry_price,
                "exit_price": p,
                "duration": idx[i] - entry_date if entry_date else None,
            })
            cash -= buy_cost
            holdings = 0.0
            position = 0

        # Reset stopped_out when equity recovers past half the stop threshold
        if stopped_out and cur_equity >= peak_equity * (1 - drawdown_stop * 0.5):
            stopped_out = False

        # Check entries
        if position == 0 and entries.iloc[i] and not stopped_out:
            # Enter long
            pos_frac = 1.0 if full_allocation else _kelly_position_size(trade_records, kelly_fraction, max_position)
            invest = cash * pos_frac
            eff_slip = _effective_slippage(slippage, invest, volume, i, p)
            exec_price = p * (1 + eff_slip)
            shares = invest / (exec_price * (1 + fee_rate))  # fee reduces shares purchased
            cash -= invest  # full allocation deducted from cash
            holdings = shares
            position = 1
            entry_price = exec_price
            entry_date = idx[i]

        elif use_short and position == 0 and short_entries.iloc[i] and not stopped_out:
            # Enter short — compute shares at mid price so slippage reduces proceeds
            pos_frac = 1.0 if full_allocation else _kelly_position_size(trade_records, kelly_fraction, max_position)
            invest = cash * pos_frac
            eff_slip = _effective_slippage(slippage, invest, volume, i, p)
            exec_price = p * (1 - eff_slip)
            shares = invest / p  # shares at mid price; slippage reduces proceeds
            cash += shares * exec_price * (1 - fee_rate)  # receive proceeds
            holdings = -shares
            position = -1
            entry_price = exec_price
            entry_date = idx[i]

    # Final equity
    equity_series = pd.Series(equity_vals, index=idx)
    returns_series = equity_series.pct_change().fillna(0)

    trades_df = pd.DataFrame(trade_records) if trade_records else None
    metrics = compute_metrics(equity_series, returns_series, trades_df, rf_annual=rf_annual)
    drawdown_series = compute_drawdown_series(equity_series)
    rolling_sharpe_series = compute_rolling_sharpe(returns_series, rf_annual=rf_annual)

    # Build signal points for chart
    signals_records = []
    for i, date in enumerate(idx):
        date_str = str(date.date()) if hasattr(date, "date") else str(date)
        sig = 0
        if entries.iloc[i]:
            sig = 1
        elif exits.iloc[i]:
            sig = -1
        elif use_short and short_entries.iloc[i]:
            sig = -1
        if sig != 0:
            signals_records.append({
                "date": date_str,
                "close": round(float(price.iloc[i]), 4),
                "signal": sig,
            })

    def s2r(s: pd.Series, vk: str = "value") -> list[dict]:
        s = s.replace([float("inf"), float("-inf")], float("nan")).dropna()
        return [
            {"date": str(idx_.date()) if hasattr(idx_, "date") else str(idx_), vk: round(float(v), 6)}
            for idx_, v in s.items()
        ]

    return {
        "metrics": metrics,
        "equity": s2r(equity_series, "value"),
        "drawdown": s2r(drawdown_series, "value"),
        "rolling_sharpe": s2r(rolling_sharpe_series, "value"),
        "signals": signals_records,
    }


def _liquidate_pairs(
    cash: float, shares_a: float, shares_b: float,
    pa: float, pb: float, fee_rate: float, slippage: float,
) -> float:
    """Close both legs of a pairs position and return resulting cash."""
    # Close leg A
    if shares_a > 0:
        cash += shares_a * pa * (1 - fee_rate) * (1 - slippage)
    elif shares_a < 0:
        cash -= (-shares_a) * pa * (1 + fee_rate) * (1 + slippage)
    # Close leg B
    if shares_b > 0:
        cash += shares_b * pb * (1 - fee_rate) * (1 - slippage)
    elif shares_b < 0:
        cash -= (-shares_b) * pb * (1 + fee_rate) * (1 + slippage)
    return cash


def run_pairs_backtest(
    price_a: pd.Series,
    price_b: pd.Series,
    entries: pd.Series,
    exits: pd.Series,
    short_entries: pd.Series | None = None,
    short_exits: pd.Series | None = None,
    beta_series: pd.Series | None = None,
    initial_capital: float = 100_000.0,
    risk_level: str = "medium",
    fee_rate: float = DEFAULT_FEE_RATE,
    slippage: float = DEFAULT_SLIPPAGE,
    rf_annual: float = 0.04,
    volume_a: pd.Series | None = None,
    volume_b: pd.Series | None = None,
    use_drawdown_stop: bool = True,
) -> dict[str, Any]:
    """
    Two-leg pairs trading backtest.

    Long spread = long A + short beta*B.
    Short spread = short A + long beta*B.
    Equity tracks both legs simultaneously.
    """
    risk = RISK_CONFIG.get(risk_level, RISK_CONFIG["medium"])
    kelly_fraction = risk["kelly_fraction"]
    max_position = risk["max_position"]
    drawdown_stop = risk["drawdown_stop"]

    price_a = price_a.dropna()
    price_b = price_b.reindex(price_a.index).ffill(limit=5)
    # Keep only dates where both tickers have valid prices
    valid_mask = price_a.notna() & price_b.notna()
    price_a = price_a[valid_mask]
    price_b = price_b[valid_mask]
    idx = price_a.index
    n = len(idx)

    if n < 2:
        return {
            "metrics": compute_metrics(pd.Series(dtype=float), pd.Series(dtype=float)),
            "equity": [], "drawdown": [], "rolling_sharpe": [], "signals": [],
        }

    if beta_series is None:
        beta_series = pd.Series(1.0, index=idx)
    else:
        beta_series = beta_series.reindex(idx).ffill().fillna(1.0)

    # Align volume to pairs index for market impact
    if volume_a is not None:
        volume_a = volume_a.reindex(idx)
    if volume_b is not None:
        volume_b = volume_b.reindex(idx)

    # Align signals
    entries = entries.reindex(idx).fillna(False).astype(bool)
    exits = exits.reindex(idx).fillna(False).astype(bool)
    use_short = short_entries is not None
    if use_short:
        short_entries = short_entries.reindex(idx).fillna(False).astype(bool)
        short_exits = short_exits.reindex(idx).fillna(False).astype(bool)

    # Next-bar execution
    entries = entries.shift(1).fillna(False).astype(bool)
    exits = exits.shift(1).fillna(False).astype(bool)
    if use_short:
        short_entries = short_entries.shift(1).fillna(False).astype(bool)
        short_exits = short_exits.shift(1).fillna(False).astype(bool)

    cash = initial_capital
    shares_a = 0.0   # positive = long, negative = short
    shares_b = 0.0   # positive = long, negative = short
    position = 0     # 0=flat, 1=long spread, -1=short spread
    equity_vals = np.zeros(n)
    trade_records = []
    peak_equity = initial_capital
    stopped_out = False
    entry_equity = 0.0
    entry_date = None
    entry_spread = 0.0

    for i in range(n):
        pa = float(price_a.iloc[i])
        pb = float(price_b.iloc[i])
        beta = float(beta_series.iloc[i])

        # Mark-to-market equity: cash + long_value - short_liability
        cur_equity = cash + shares_a * pa + shares_b * pb
        equity_vals[i] = cur_equity

        if cur_equity > peak_equity:
            peak_equity = cur_equity

        # Drawdown stop — liquidate both legs
        dd = (cur_equity - peak_equity) / peak_equity
        if use_drawdown_stop and dd < -drawdown_stop and position != 0:
            cash = _liquidate_pairs(cash, shares_a, shares_b, pa, pb, fee_rate, slippage)
            trade_records.append({
                "entry_date": entry_date, "exit_date": idx[i],
                "pnl": cash - entry_equity,
                "entry_price": entry_spread,
                "exit_price": pa - beta * pb,
                "duration": idx[i] - entry_date if entry_date else None,
            })
            shares_a = 0.0
            shares_b = 0.0
            position = 0
            stopped_out = True
            continue

        # Exit signals — close both legs
        if position == 1 and exits.iloc[i]:
            cash = _liquidate_pairs(cash, shares_a, shares_b, pa, pb, fee_rate, slippage)
            trade_records.append({
                "entry_date": entry_date, "exit_date": idx[i],
                "pnl": cash - entry_equity,
                "entry_price": entry_spread,
                "exit_price": pa - beta * pb,
                "duration": idx[i] - entry_date if entry_date else None,
            })
            shares_a = 0.0
            shares_b = 0.0
            position = 0

        elif use_short and position == -1 and short_exits.iloc[i]:
            cash = _liquidate_pairs(cash, shares_a, shares_b, pa, pb, fee_rate, slippage)
            trade_records.append({
                "entry_date": entry_date, "exit_date": idx[i],
                "pnl": cash - entry_equity,
                "entry_price": entry_spread,
                "exit_price": pa - beta * pb,
                "duration": idx[i] - entry_date if entry_date else None,
            })
            shares_a = 0.0
            shares_b = 0.0
            position = 0

        # Reset stopped_out
        if stopped_out and cur_equity >= peak_equity * (1 - drawdown_stop * 0.5):
            stopped_out = False

        # Entry signals
        abs_beta = abs(beta) if abs(beta) > 1e-8 else 1.0

        if position == 0 and entries.iloc[i] and not stopped_out:
            # Long spread: long A, short beta*B
            pos_frac = _kelly_position_size(trade_records, kelly_fraction, max_position)
            invest = cash * pos_frac
            invest_a = invest / (1 + abs_beta)
            invest_b = invest * abs_beta / (1 + abs_beta)

            # Buy A (long)
            eff_slip_a = _effective_slippage(slippage, invest_a, volume_a, i, pa)
            exec_pa = pa * (1 + eff_slip_a)
            sa = (invest_a * (1 - fee_rate)) / exec_pa
            cash -= sa * exec_pa

            # Short B (receive proceeds minus fees)
            eff_slip_b = _effective_slippage(slippage, invest_b, volume_b, i, pb)
            exec_pb = pb * (1 - eff_slip_b)
            sb = invest_b / exec_pb
            cash += sb * exec_pb * (1 - fee_rate)

            shares_a = sa
            shares_b = -sb if beta > 0 else sb
            position = 1
            entry_equity = cash + shares_a * pa + shares_b * pb  # total MtM equity at entry
            entry_date = idx[i]
            entry_spread = pa - beta * pb

        elif use_short and position == 0 and short_entries.iloc[i] and not stopped_out:
            # Short spread: short A, long beta*B
            pos_frac = _kelly_position_size(trade_records, kelly_fraction, max_position)
            invest = cash * pos_frac
            invest_a = invest / (1 + abs_beta)
            invest_b = invest * abs_beta / (1 + abs_beta)

            # Short A (receive proceeds minus fees)
            eff_slip_a = _effective_slippage(slippage, invest_a, volume_a, i, pa)
            exec_pa = pa * (1 - eff_slip_a)
            sa = invest_a / exec_pa
            cash += sa * exec_pa * (1 - fee_rate)

            # Buy B (long)
            eff_slip_b = _effective_slippage(slippage, invest_b, volume_b, i, pb)
            exec_pb = pb * (1 + eff_slip_b)
            sb = (invest_b * (1 - fee_rate)) / exec_pb
            cash -= sb * exec_pb

            shares_a = -sa
            shares_b = sb if beta > 0 else -sb
            position = -1
            entry_equity = cash + shares_a * pa + shares_b * pb  # total MtM equity at entry
            entry_date = idx[i]
            entry_spread = pa - beta * pb

    # Build results
    equity_series = pd.Series(equity_vals, index=idx)
    returns_series = equity_series.pct_change().fillna(0)

    trades_df = pd.DataFrame(trade_records) if trade_records else None
    metrics = compute_metrics(equity_series, returns_series, trades_df, rf_annual=rf_annual)
    drawdown_series = compute_drawdown_series(equity_series)
    rolling_sharpe_series = compute_rolling_sharpe(returns_series, rf_annual=rf_annual)

    # Signals chart: show spread value (A - beta*B) instead of single price
    spread_values = price_a - beta_series * price_b
    signals_records = []
    for i, date in enumerate(idx):
        date_str = str(date.date()) if hasattr(date, "date") else str(date)
        sig = 0
        if entries.iloc[i]:
            sig = 1
        elif exits.iloc[i]:
            sig = -1
        elif use_short and short_entries.iloc[i]:
            sig = -1
        if sig != 0:
            signals_records.append({
                "date": date_str,
                "close": round(float(spread_values.iloc[i]), 4),
                "signal": sig,
            })

    def s2r(s: pd.Series, vk: str = "value") -> list[dict]:
        s = s.replace([float("inf"), float("-inf")], float("nan")).dropna()
        return [
            {"date": str(idx_.date()) if hasattr(idx_, "date") else str(idx_), vk: round(float(v), 6)}
            for idx_, v in s.items()
        ]

    return {
        "metrics": metrics,
        "equity": s2r(equity_series, "value"),
        "drawdown": s2r(drawdown_series, "value"),
        "rolling_sharpe": s2r(rolling_sharpe_series, "value"),
        "signals": signals_records,
    }


def run_rotation_backtest(
    rotation_df: pd.DataFrame,
    price_dict: dict[str, pd.Series],
    initial_capital: float = 100_000.0,
    risk_level: str = "medium",
    fee_rate: float = DEFAULT_FEE_RATE,
    slippage: float = DEFAULT_SLIPPAGE,
    volume_dict: dict[str, pd.Series] | None = None,
    use_drawdown_stop: bool = True,
) -> dict[str, Any]:
    """
    Backtest a rotation/GEM strategy where 100% capital is always in one asset.

    Args:
        rotation_df: DataFrame with 'allocation' (ticker str) and 'signal' (bool) columns.
        price_dict: {ticker: Close price Series} for all possible assets.
        initial_capital: Starting capital.
        risk_level: Only drawdown_stop is used (no Kelly/position sizing — always 100%).
        fee_rate: Commission per trade.
        slippage: Slippage per trade.

    Returns same dict format as run_backtest().
    """
    risk = RISK_CONFIG.get(risk_level, RISK_CONFIG["medium"])
    drawdown_stop = risk["drawdown_stop"]

    idx = rotation_df.index
    n = len(idx)

    if n < 2:
        return {
            "metrics": compute_metrics(pd.Series(dtype=float), pd.Series(dtype=float)),
            "equity": [], "drawdown": [], "rolling_sharpe": [], "signals": [],
        }

    # Align all price series to rotation index
    prices = {}
    for ticker, series in price_dict.items():
        aligned = series.reindex(idx).ffill(limit=5)
        prices[ticker] = aligned
        last_valid = series.last_valid_index()
        if last_valid is not None and last_valid < idx[-1]:
            logger.warning(
                f"run_rotation_backtest: {ticker} price data ends "
                f"{last_valid.date()}, simulation ends {idx[-1].date()}"
            )

    equity_vals = np.zeros(n)
    cash = initial_capital
    shares = 0.0
    current_asset = None
    peak_equity = initial_capital
    stopped_out = False
    trade_records = []
    entry_price = 0.0
    entry_date = None
    signals_records = []

    # Next-bar execution: signals computed at close of bar i apply at bar i+1
    rotation_df = rotation_df.copy()
    rotation_df["allocation"] = rotation_df["allocation"].shift(1)
    rotation_df["signal"] = rotation_df["signal"].shift(1).fillna(False)

    for i in range(n):
        date = idx[i]
        target_asset = rotation_df["allocation"].iloc[i]
        is_signal = bool(rotation_df["signal"].iloc[i])

        # Current equity — handle delisting (NaN price)
        if current_asset is not None and shares > 0:
            cur_price_raw = prices[current_asset].iloc[i]
            if pd.isna(cur_price_raw):
                # Ticker delisted — liquidate at last known price
                last_known = prices[current_asset].iloc[:i].dropna()
                if not last_known.empty:
                    lp = float(last_known.iloc[-1])
                    proceeds = shares * lp * (1 - fee_rate) * (1 - slippage)
                    trade_records.append({
                        "entry_date": entry_date, "exit_date": date,
                        "pnl": proceeds - shares * entry_price,
                        "entry_price": entry_price, "exit_price": lp,
                        "duration": date - entry_date if entry_date else None,
                    })
                    cash += proceeds
                logger.warning(f"Delisting forced liquidation of {current_asset} on {date.date()}")
                shares = 0.0
                current_asset = None
                cur_equity = cash
            else:
                cur_price = float(cur_price_raw)
                cur_equity = cash + shares * cur_price
        else:
            cur_equity = cash
        equity_vals[i] = cur_equity

        # Peak tracking
        if cur_equity > peak_equity:
            peak_equity = cur_equity

        # Drawdown stop check
        dd = (cur_equity - peak_equity) / peak_equity
        if use_drawdown_stop and dd < -drawdown_stop and current_asset is not None and shares > 0:
            sell_price = float(prices[current_asset].iloc[i])
            vol_s = volume_dict.get(current_asset) if volume_dict else None
            eff_slip = _effective_slippage(slippage, shares * sell_price, vol_s, i, sell_price)
            proceeds = shares * sell_price * (1 - fee_rate) * (1 - eff_slip)
            trade_records.append({
                "entry_date": entry_date, "exit_date": date,
                "pnl": proceeds - shares * entry_price,
                "entry_price": entry_price, "exit_price": sell_price,
                "duration": date - entry_date if entry_date else None,
            })
            cash += proceeds
            shares = 0.0
            current_asset = None
            stopped_out = True
            date_str = str(date.date()) if hasattr(date, "date") else str(date)
            signals_records.append({"date": date_str, "close": round(sell_price, 4), "signal": -1})
            continue

        # Passive recovery from drawdown stop: if equity has recovered enough, allow re-entry
        if stopped_out and not is_signal and cur_equity >= peak_equity * (1 - drawdown_stop * 0.5):
            stopped_out = False

        # Rebalance on signal days (or resume after drawdown stop)
        if is_signal and not stopped_out:
            stopped_out = False

            # Sell current holding
            if current_asset is not None and shares > 0:
                sell_price = float(prices[current_asset].iloc[i])
                vol_s = volume_dict.get(current_asset) if volume_dict else None
                eff_slip = _effective_slippage(slippage, shares * sell_price, vol_s, i, sell_price)
                proceeds = shares * sell_price * (1 - fee_rate) * (1 - eff_slip)
                trade_records.append({
                    "entry_date": entry_date, "exit_date": date,
                    "pnl": proceeds - shares * entry_price,
                    "entry_price": entry_price, "exit_price": sell_price,
                    "duration": date - entry_date if entry_date else None,
                })
                cash += proceeds
                shares = 0.0

            # Buy new asset with 100% of capital
            if target_asset is not None and target_asset in prices:
                raw_price = float(prices[target_asset].iloc[i])
                vol_s = volume_dict.get(target_asset) if volume_dict else None
                eff_slip = _effective_slippage(slippage, cash, vol_s, i, raw_price)
                buy_price = raw_price * (1 + eff_slip)
                invest = cash * (1 - fee_rate)
                shares = invest / buy_price
                cash = 0.0
                current_asset = target_asset
                entry_price = buy_price
                entry_date = date

                date_str = str(date.date()) if hasattr(date, "date") else str(date)
                signals_records.append({
                    "date": date_str,
                    "close": round(float(prices[target_asset].iloc[i]), 4),
                    "signal": 1,
                })

    # Final equity series
    equity_series = pd.Series(equity_vals, index=idx)
    returns_series = equity_series.pct_change().fillna(0)

    trades_df = pd.DataFrame(trade_records) if trade_records else None
    metrics = compute_metrics(equity_series, returns_series, trades_df)
    drawdown_series = compute_drawdown_series(equity_series)
    rolling_sharpe_series = compute_rolling_sharpe(returns_series)

    def s2r(s: pd.Series, vk: str = "value") -> list[dict]:
        s = s.replace([float("inf"), float("-inf")], float("nan")).dropna()
        return [
            {"date": str(idx_.date()) if hasattr(idx_, "date") else str(idx_), vk: round(float(v), 6)}
            for idx_, v in s.items()
        ]

    return {
        "metrics": metrics,
        "equity": s2r(equity_series, "value"),
        "drawdown": s2r(drawdown_series, "value"),
        "rolling_sharpe": s2r(rolling_sharpe_series, "value"),
        "signals": signals_records,
    }

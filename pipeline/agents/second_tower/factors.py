"""63 cross-sectional factor computations for Second-tower strategy.

All fundamental factors (value, quality, accruals) use point-in-time SEC EDGAR
data that has already been filtered by a 90-day filing lag by the caller.
Technical and risk factors use the OHLCV slice passed in (no lookahead).
"""
from __future__ import annotations

import logging
import numpy as np
import pandas as pd
import ta

logger = logging.getLogger(__name__)


# ── EDGAR data helpers ────────────────────────────────────────────────────────

def _edgar_val(stmts: dict, stmt_name: str, *cols) -> float | None:
    """Return the most recent (iloc[0]) value for the first matching column."""
    df = stmts.get(stmt_name)
    if not isinstance(df, pd.DataFrame) or df.empty:
        return None
    df = df.sort_index(ascending=False)
    for col in cols:
        if col in df.columns:
            val = df[col].iloc[0]
            return float(val) if pd.notna(val) else None
    return None


def _edgar_val_prior(stmts: dict, stmt_name: str, *cols) -> float | None:
    """Return the prior-year (iloc[1]) value for the first matching column."""
    df = stmts.get(stmt_name)
    if not isinstance(df, pd.DataFrame) or len(df) < 2:
        return None
    df = df.sort_index(ascending=False)
    for col in cols:
        if col in df.columns:
            val = df[col].iloc[1]
            return float(val) if pd.notna(val) else None
    return None


def _mcap(stmts: dict, close_price: float) -> float | None:
    """Compute market cap from EDGAR shares outstanding × current close price."""
    shares = _edgar_val(stmts, "balance", "CommonStockSharesOutstanding")
    if shares is not None and shares > 0 and close_price > 0:
        return shares * close_price
    return None


# ── Momentum factors (12) ─────────────────────────────────────────────────────

def compute_momentum_factors(ohlcv_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Compute 12 momentum factors cross-sectionally from the sliced OHLCV window."""
    # Compute cross-sectional average 3M return once (O(N), not O(N²))
    _cross_3m = [
        float(df["Close"].iloc[-1] / df["Close"].iloc[-63] - 1)
        for df in ohlcv_data.values() if len(df) >= 63
    ]
    market_ret_3m = float(np.nanmean(_cross_3m)) if _cross_3m else 0.0

    records = []
    for ticker, df in ohlcv_data.items():
        close = df["Close"]
        if len(close) < 252:
            continue

        rec = {"ticker": ticker}

        rec["MOM1M"] = float(close.iloc[-1] / close.iloc[-21] - 1) if len(close) >= 21 else np.nan
        rec["MOM3M"] = float(close.iloc[-1] / close.iloc[-63] - 1) if len(close) >= 63 else np.nan
        rec["MOM6M"] = float(close.iloc[-1] / close.iloc[-126] - 1) if len(close) >= 126 else np.nan
        rec["MOM12M"] = float(close.iloc[-1] / close.iloc[-252] - 1) if len(close) >= 252 else np.nan

        # 12-1 month momentum (skip most recent month)
        rec["MOM12M1"] = float(close.iloc[-21] / close.iloc[-252] - 1) if len(close) >= 252 else np.nan

        # 52-week high ratio
        high52 = close.iloc[-252:].max() if len(close) >= 252 else close.max()
        rec["HIGH52W_RATIO"] = float(close.iloc[-1] / high52)

        # Cross-sectional-relative momentum (stock 3M return minus universe average)
        rec["IND_MOM"] = rec["MOM3M"] - market_ret_3m if not np.isnan(rec.get("MOM3M", np.nan)) else np.nan

        # Short-term reversal (5-day) — NEGATED: recent winners tend to mean-revert
        rec["STR"] = -float(close.iloc[-1] / close.iloc[-5] - 1) if len(close) >= 5 else np.nan

        # Long-term reversal (year 2 vs year 3 ago) — NEGATED: long-term winners tend to reverse
        rec["LTR"] = -float(close.iloc[-252] / close.iloc[-504] - 1) if len(close) >= 504 else np.nan

        # Consecutive up days (last 20)
        daily_ret = close.pct_change().iloc[-20:]
        rec["CONSEC_UP"] = float((daily_ret > 0).sum()) / 20.0

        # Time-series momentum — Moskowitz, Ooi & Pedersen (2012):
        # sign(12M return) × realized annual volatility.
        # The sign term is direction; vol-scaling risk-normalises across assets.
        ret_12m = close.pct_change().iloc[-252:]
        sign_12m = np.sign(float(close.iloc[-1] / close.iloc[-252] - 1))
        rvol_12m = float(ret_12m.std() * np.sqrt(252))
        rec["TSMOM"] = float(sign_12m * rvol_12m) if rvol_12m > 0 else 0.0

        # Price acceleration: 6-month return - prior 6-month return
        mom6m_prev = float(close.iloc[-126] / close.iloc[-252] - 1) if len(close) >= 252 else np.nan
        rec["PRICE_ACCEL"] = rec["MOM6M"] - mom6m_prev if not np.isnan(mom6m_prev) else np.nan

        records.append(rec)

    return pd.DataFrame(records).set_index("ticker") if records else pd.DataFrame()


# ── Value factors (9) ─────────────────────────────────────────────────────────

def compute_value_factors(
    edgar_data: dict[str, dict],
    ohlcv_data: dict[str, pd.DataFrame],
    fundamentals_data: dict[str, dict] | None = None,
) -> pd.DataFrame:
    """Compute 9 value factors from point-in-time EDGAR data.

    edgar_data:      {ticker: {'balance': df, 'income': df, 'cashflow': df}}
                     already filtered to the rebalancing date (filed_date <= rebal_date).
    ohlcv_data:      {ticker: df} sliced to the training window (for current price).
    fundamentals_data: optional {ticker: yfinance_info_dict}; enables FWD_EP when provided.
    """
    records = []
    all_tickers = set(edgar_data) | set(ohlcv_data)
    fundamentals_data = fundamentals_data or {}

    for ticker in all_tickers:
        stmts = edgar_data.get(ticker.upper(), {})
        df = ohlcv_data.get(ticker)
        if df is None or df.empty:
            continue
        close_price = float(df["Close"].iloc[-1])
        if close_price <= 0:
            continue

        rec = {"ticker": ticker}

        # Market cap proxy from EDGAR shares × current price
        mcap = _mcap(stmts, close_price)

        equity = _edgar_val(stmts, "balance", "StockholdersEquityNet", "StockholdersEquity",
                            "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest")
        net_income = _edgar_val(stmts, "income", "NetIncomeLoss")
        ocf = _edgar_val(stmts, "cashflow", "NetCashProvidedByOperatingActivities",
                         "NetCashProvidedByUsedInOperatingActivities",
                         "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations")
        revenue = _edgar_val(stmts, "income", "Revenue", "Revenues",
                             "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet")
        assets = _edgar_val(stmts, "balance", "Assets")
        lt_debt = _edgar_val(stmts, "balance", "LongTermDebt") or 0.0
        st_debt = _edgar_val(stmts, "balance", "ShortTermBorrowings") or 0.0
        cash = _edgar_val(stmts, "balance", "CashAndCashEquivalentsAtCarryingValue") or 0.0
        op_income = _edgar_val(stmts, "income", "OperatingIncomeLoss")
        da = _edgar_val(stmts, "cashflow", "DepreciationDepletionAndAmortization",
                        "DepreciationAndAmortization") or 0.0

        # ── Original 5 value factors ──────────────────────────────────────────

        # Book-to-Market
        rec["BM"] = float(equity / mcap) if equity is not None and mcap else np.nan

        # Earnings-to-Price (allows negative)
        rec["EP"] = float(net_income / mcap) if net_income is not None and mcap else np.nan

        # Cash flow to price
        rec["CFP"] = float(ocf / mcap) if ocf is not None and mcap else np.nan

        # Sales to price
        rec["SP"] = float(revenue / mcap) if revenue is not None and mcap else np.nan

        # EBITDA yield = EBITDA / EV (capital-structure neutral; higher = cheaper)
        if mcap:
            ev = mcap + lt_debt + st_debt - cash
            ebitda = (op_income or 0) + da
            rec["EV_EBITDA"] = float(ebitda / ev) if ev > 0 and ebitda is not None else np.nan
        else:
            ev = None
            rec["EV_EBITDA"] = np.nan

        # ── 4 additional value factors ────────────────────────────────────────

        # Trailing P/E inverse — like EP but NaN for negative earnings (cleaner signal).
        # EP handles negative-earnings firms; TRAILING_PE_INV explicitly excludes them.
        rec["TRAILING_PE_INV"] = float(net_income / mcap) if net_income is not None and net_income > 0 and mcap else np.nan

        # Dividend yield = cash dividends paid / market cap (from EDGAR cashflow).
        # PaymentsOfDividendsCommonStock is a cashflow outflow (positive = dividends paid).
        div_paid = _edgar_val(stmts, "cashflow", "PaymentsOfDividendsCommonStock",
                              "PaymentsOfDividends")
        rec["DIV_YIELD"] = float(div_paid / mcap) if div_paid is not None and div_paid > 0 and mcap else np.nan

        # Payout yield = (dividends + buybacks) / market cap.
        # Captures the full shareholder return including share repurchases.
        buybacks = _edgar_val(stmts, "cashflow", "PaymentsForRepurchaseOfCommonStock",
                              "RepaymentsOfDebt") or 0.0
        total_payout = (div_paid or 0.0) + abs(buybacks)
        rec["PAYOUT_YIELD"] = float(total_payout / mcap) if total_payout > 0 and mcap else np.nan

        # Forward EPS yield = forward EPS / price (from yfinance forwardEps when available).
        # This is only populated when fundamentals_data is provided by the caller.
        fwd_eps = fundamentals_data.get(ticker, {}).get("forwardEps")
        rec["FWD_EP"] = float(fwd_eps / close_price) if fwd_eps is not None and close_price > 0 else np.nan

        records.append(rec)

    return pd.DataFrame(records).set_index("ticker") if records else pd.DataFrame()


# ── Quality factors (12) ──────────────────────────────────────────────────────

def compute_quality_factors(
    edgar_data: dict[str, dict],
    ohlcv_data: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Compute 12 quality factors from point-in-time EDGAR data."""
    records = []

    for ticker in set(edgar_data) | set(ohlcv_data):
        stmts = edgar_data.get(ticker.upper(), {})
        df = ohlcv_data.get(ticker)
        if df is None or df.empty:
            continue

        rec = {"ticker": ticker}

        assets = _edgar_val(stmts, "balance", "Assets")
        assets_prior = _edgar_val_prior(stmts, "balance", "Assets")
        equity = _edgar_val(stmts, "balance", "StockholdersEquityNet", "StockholdersEquity",
                            "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest")
        net_income = _edgar_val(stmts, "income", "NetIncomeLoss")
        revenue = _edgar_val(stmts, "income", "Revenue", "Revenues",
                             "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet")
        revenue_prior = _edgar_val_prior(stmts, "income", "Revenue", "Revenues",
                                         "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet")
        cogs = _edgar_val(stmts, "income", "CostOfGoods", "CostOfGoodsAndServicesSold", "CostOfRevenue")
        op_income = _edgar_val(stmts, "income", "OperatingIncomeLoss")
        ocf = _edgar_val(stmts, "cashflow", "NetCashProvidedByOperatingActivities",
                         "NetCashProvidedByUsedInOperatingActivities",
                         "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations")
        lt_debt = _edgar_val(stmts, "balance", "LongTermDebt") or 0.0
        current_assets = _edgar_val(stmts, "balance", "AssetsCurrent")
        current_liabs = _edgar_val(stmts, "balance", "LiabilitiesCurrent")

        # ROA
        rec["ROA"] = float(net_income / assets) if net_income is not None and assets is not None and assets > 0 else np.nan

        # ROE
        rec["ROE"] = float(net_income / equity) if net_income is not None and equity is not None and equity != 0 else np.nan

        # Gross profitability (Novy-Marx)
        gross_profit = (revenue - cogs) if revenue is not None and cogs is not None else None
        rec["GP_A"] = float(gross_profit / assets) if gross_profit is not None and assets is not None and assets > 0 else np.nan

        # Operating margin
        rec["OP_MARGIN"] = float(op_income / revenue) if op_income is not None and revenue is not None and revenue != 0 else np.nan

        # Net margin
        rec["NET_MARGIN"] = float(net_income / revenue) if net_income is not None and revenue is not None and revenue != 0 else np.nan

        # Asset growth (year-over-year)
        rec["ASSET_GROWTH"] = float(assets / assets_prior - 1) if assets is not None and assets_prior is not None and assets_prior > 0 else np.nan

        # Capex growth (year-over-year)
        capex = _edgar_val(stmts, "cashflow", "PaymentsToAcquirePropertyPlantAndEquipment")
        capex_prior = _edgar_val_prior(stmts, "cashflow", "PaymentsToAcquirePropertyPlantAndEquipment")
        rec["CAPEX_GROWTH"] = float(capex / capex_prior - 1) if capex is not None and capex_prior is not None and capex_prior != 0 else np.nan

        # Sales growth
        rec["SALES_GROWTH"] = float(revenue / revenue_prior - 1) if revenue is not None and revenue_prior is not None and revenue_prior > 0 else np.nan

        # Additional prior-year values for full Piotroski F-score
        net_income_prior = _edgar_val_prior(stmts, "income", "NetIncomeLoss")
        current_assets_prior = _edgar_val_prior(stmts, "balance", "AssetsCurrent")
        current_liabs_prior = _edgar_val_prior(stmts, "balance", "LiabilitiesCurrent")
        shares_now = _edgar_val(stmts, "balance", "CommonStockSharesOutstanding")
        shares_prior = _edgar_val_prior(stmts, "balance", "CommonStockSharesOutstanding")

        # Piotroski F-score (all 9 signals per Piotroski 2000)
        f_score = 0
        # Profitability signals (4)
        roa_now = (net_income / assets) if net_income is not None and assets is not None and assets > 0 else None
        roa_prior = (net_income_prior / assets_prior) if net_income_prior is not None and assets_prior is not None and assets_prior > 0 else None
        if roa_now is not None and roa_now > 0:
            f_score += 1  # F1: positive ROA
        if ocf is not None and ocf > 0:
            f_score += 1  # F2: positive operating cash flow
        if roa_now is not None and roa_prior is not None and roa_now > roa_prior:
            f_score += 1  # F3: improving ROA
        if ocf is not None and net_income is not None and (ocf / assets if assets is not None and assets > 0 else 0) > (net_income / assets if assets is not None and assets > 0 else 1):
            f_score += 1  # F4: accruals (OCF/assets > ROA)
        # Leverage / liquidity signals (3)
        if assets is not None and assets_prior is not None and assets_prior > 0:
            lev_now = lt_debt / assets if lt_debt else 0.0
            lev_prior = (_edgar_val_prior(stmts, "balance", "LongTermDebt") or 0.0) / assets_prior
            if lev_now < lev_prior:
                f_score += 1  # F5: decreasing leverage
        cur_now = (current_assets / current_liabs) if current_assets is not None and current_liabs is not None and current_liabs > 0 else None
        cur_prior = (current_assets_prior / current_liabs_prior) if current_assets_prior is not None and current_liabs_prior is not None and current_liabs_prior > 0 else None
        if cur_now is not None and cur_prior is not None and cur_now > cur_prior:
            f_score += 1  # F6: improving current ratio
        if shares_now is not None and shares_prior is not None and shares_now <= shares_prior:
            f_score += 1  # F7: no share dilution
        # Operating efficiency signals (2)
        gm_now = ((revenue - cogs) / revenue) if revenue is not None and cogs is not None and revenue > 0 else None
        gm_prior = ((revenue_prior - (_edgar_val_prior(stmts, "income", "CostOfGoods", "CostOfGoodsAndServicesSold", "CostOfRevenue") or 0)) / revenue_prior) if revenue_prior is not None and revenue_prior > 0 else None
        if gm_now is not None and gm_prior is not None and gm_now > gm_prior:
            f_score += 1  # F8: improving gross margin
        at_now = (revenue / assets) if revenue is not None and assets is not None and assets > 0 else None
        at_prior = (revenue_prior / assets_prior) if revenue_prior is not None and assets_prior is not None and assets_prior > 0 else None
        if at_now is not None and at_prior is not None and at_now > at_prior:
            f_score += 1  # F9: improving asset turnover
        rec["PIOTROSKI_F"] = float(f_score)

        # Accruals ratio: NEGATED so positive = low accruals = high earnings quality
        rec["ACCRUALS_RATIO"] = float(-((net_income - (ocf or 0)) / assets)) if net_income is not None and assets is not None and assets > 0 else np.nan

        # Earnings quality (CFO / Net Income)
        rec["EARN_QUALITY"] = float(ocf / net_income) if ocf is not None and net_income is not None and net_income > 0 else np.nan

        # Leverage ratio — negated: higher leverage predicts lower expected returns.
        rec["LEV_RATIO"] = float(-(lt_debt / assets)) if lt_debt is not None and assets is not None and assets > 0 else np.nan

        records.append(rec)

    return pd.DataFrame(records).set_index("ticker") if records else pd.DataFrame()


# ── Technical factors (15) ────────────────────────────────────────────────────

def compute_technical_factors(ohlcv_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Compute 15 technical factors using the ta library.

    All computed on the passed OHLCV slice — .iloc[-1] picks the last bar
    of the training window, which is the current rebalancing date.
    """
    records = []
    for ticker, df in ohlcv_data.items():
        if len(df) < 200:
            continue
        rec = {"ticker": ticker}
        close = df["Close"]
        high = df["High"]
        low = df["Low"]
        volume = df["Volume"]

        rsi = ta.momentum.RSIIndicator(close, window=14).rsi()
        # Normalize RSI from [0, 100] to [-1, 1]: (RSI − 50) / 50
        # Prevents outlier raw values dominating cross-sectional z-scoring.
        rec["RSI14"] = float((rsi.iloc[-1] - 50.0) / 50.0) if not rsi.empty else np.nan

        macd_ind = ta.trend.MACD(close)
        macd_diff = macd_ind.macd_diff()
        # Price-normalize MACD so it is comparable across tickers with different price levels.
        cur_px = float(close.iloc[-1])
        rec["MACD"] = float(macd_diff.iloc[-1] / cur_px) if not macd_diff.empty and cur_px > 0 else np.nan

        bb = ta.volatility.BollingerBands(close, window=20)
        bb_high = bb.bollinger_hband().iloc[-1]
        bb_low = bb.bollinger_lband().iloc[-1]
        # Map Bollinger %B from [0, 1] to [-1, 1]: 2 × %B − 1
        raw_pos = float((close.iloc[-1] - bb_low) / (bb_high - bb_low)) if (bb_high - bb_low) > 0 else 0.5
        rec["BB_POS"] = float(2.0 * raw_pos - 1.0)

        atr = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()
        rec["ATR_NORM"] = float(atr.iloc[-1] / close.iloc[-1]) if close.iloc[-1] > 0 else np.nan

        obv = ta.volume.OnBalanceVolumeIndicator(close, volume).on_balance_volume()
        obv_20 = obv.iloc[-20:]
        if len(obv_20) >= 20 and obv_20.std() > 0:
            rec["OBV"] = float((obv_20.iloc[-1] - obv_20.iloc[0]) / obv_20.std())
        else:
            rec["OBV"] = 0.0

        ma50 = close.rolling(50).mean().iloc[-1]
        ma200 = close.rolling(200).mean().iloc[-1]
        last_px = float(close.iloc[-1])
        # Difference of price's relative distance from each MA: captures whether
        # price is both above its 200d MA AND diverging upward from its 50d MA.
        rec["MA50_200"] = float(last_px / ma50 - last_px / ma200) if ma50 > 0 and ma200 > 0 else np.nan

        ma50_prev = close.rolling(50).mean().iloc[-2]
        ma200_prev = close.rolling(200).mean().iloc[-2]
        rec["CROSS_SIGNAL"] = 1.0 if (ma50 > ma200 and ma50_prev <= ma200_prev) else (
            -1.0 if (ma50 < ma200 and ma50_prev >= ma200_prev) else 0.0
        )

        vol_20 = volume.iloc[-20:].mean()
        vol_60 = volume.iloc[-60:].mean()
        rec["VOL_MOM"] = float(vol_20 / vol_60) if vol_60 > 0 else np.nan

        stoch = ta.momentum.StochasticOscillator(high, low, close).stoch()
        rec["STOCH"] = float(stoch.iloc[-1]) if not stoch.empty else np.nan

        wr = ta.momentum.WilliamsRIndicator(high, low, close).williams_r()
        rec["WILLIAMS_R"] = float(wr.iloc[-1]) if not wr.empty else np.nan

        cci = ta.trend.CCIIndicator(high, low, close).cci()
        rec["CCI"] = float(cci.iloc[-1]) if not cci.empty else np.nan

        adx = ta.trend.ADXIndicator(high, low, close).adx()
        rec["ADX"] = float(adx.iloc[-1]) if not adx.empty else np.nan

        low52 = low.iloc[-252:].min() if len(low) >= 252 else low.min()
        rec["LOW52W_DIST"] = float(close.iloc[-1] / low52 - 1)

        vol_5 = volume.iloc[-5:].mean()
        rec["VOL_RATIO"] = float(vol_5 / vol_60) if vol_60 > 0 else np.nan

        rec["DOLLAR_VOL"] = float(np.log1p(close.iloc[-1] * volume.iloc[-1]))

        records.append(rec)

    return pd.DataFrame(records).set_index("ticker") if records else pd.DataFrame()


# ── Risk factors (8) ─────────────────────────────────────────────────────────

def compute_risk_factors(ohlcv_data: dict[str, pd.DataFrame], market_returns: pd.Series) -> pd.DataFrame:
    """Compute 8 risk factors from the OHLCV training window."""
    records = []
    for ticker, df in ohlcv_data.items():
        if len(df) < 252:
            continue
        rec = {"ticker": ticker}
        close = df["Close"]
        returns = close.pct_change().dropna()

        common_idx = returns.index.intersection(market_returns.index)
        if len(common_idx) < 60:
            continue
        ret_aligned = returns.loc[common_idx]
        mkt_aligned = market_returns.loc[common_idx]

        # Cap beta estimation window to 60 months (1260 trading days) — more stable
        # than using the full training window which can be 1–3 years and makes beta
        # estimates noisy for short simulations.
        if len(ret_aligned) > 1260:
            ret_aligned = ret_aligned.iloc[-1260:]
            mkt_aligned = mkt_aligned.reindex(ret_aligned.index)
            _pair = pd.concat([ret_aligned, mkt_aligned], axis=1).dropna()
            ret_aligned = _pair.iloc[:, 0]
            mkt_aligned = _pair.iloc[:, 1]

        cov_matrix = np.cov(ret_aligned.values, mkt_aligned.values)
        rec["TRAILING_BETA"] = float(cov_matrix[0, 1] / cov_matrix[1, 1]) if cov_matrix[1, 1] > 0 else np.nan

        beta = rec["TRAILING_BETA"]
        if beta and not np.isnan(beta):
            residual = ret_aligned.values - beta * mkt_aligned.values
            rec["IVOL"] = float(np.std(residual) * np.sqrt(252))
        else:
            rec["IVOL"] = np.nan

        rec["TVOL"] = float(returns.std() * np.sqrt(252))
        rec["MAX_RET"] = float(-returns.max())  # negated: lottery stocks underperform
        # Negated: positive skew (lottery preference) predicts lower future returns
        # (Bali et al. 2011; Boyer et al. 2010).
        rec["SKEW"] = float(-returns.skew())

        down_mask = mkt_aligned < 0
        if down_mask.sum() > 20:
            down_ret = ret_aligned[down_mask]
            down_mkt = mkt_aligned[down_mask]
            cov_down = np.cov(down_ret.values, down_mkt.values)
            rec["DOWN_BETA"] = float(cov_down[0, 1] / cov_down[1, 1]) if cov_down[1, 1] > 0 else np.nan
        else:
            rec["DOWN_BETA"] = np.nan

        rec["KURT"] = float(returns.kurtosis())

        volume = df["Volume"]
        dv = close * volume
        amihud_vals = returns.abs() / dv
        amihud_vals = amihud_vals.replace([np.inf, -np.inf], np.nan).dropna()
        # Negated: higher AMIHUD = more illiquid = lower expected return cross-sectionally
        # (Amihud 2002; as a cross-sectional signal, liquid stocks are preferred).
        rec["AMIHUD"] = float(-amihud_vals.mean() * 1e6) if len(amihud_vals) >= 60 else np.nan

        records.append(rec)

    return pd.DataFrame(records).set_index("ticker") if records else pd.DataFrame()


# ── Accrual factors (6) ───────────────────────────────────────────────────────

def compute_accrual_factors(edgar_data: dict[str, dict]) -> pd.DataFrame:
    """Compute 4 accrual-related factors from point-in-time EDGAR data."""
    records = []

    for ticker, stmts in edgar_data.items():
        if not stmts:
            continue
        rec = {"ticker": ticker}

        assets = _edgar_val(stmts, "balance", "Assets")
        assets_prior = _edgar_val_prior(stmts, "balance", "Assets")
        net_income = _edgar_val(stmts, "income", "NetIncomeLoss")
        ocf = _edgar_val(stmts, "cashflow", "NetCashProvidedByOperatingActivities",
                         "NetCashProvidedByUsedInOperatingActivities",
                         "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations")

        # ── Balance sheet accruals: delta-NOA / avg_assets (NEGATED: low accruals = better) ──
        # NOA = (Assets - Cash) - (Liabilities - Debt)
        # Distinct from ACCRUALS_RATIO in quality factors (which uses NI - OCF approach).
        liab = _edgar_val(stmts, "balance", "Liabilities")
        liab_prior = _edgar_val_prior(stmts, "balance", "Liabilities")
        ltd = _edgar_val(stmts, "balance", "LongTermDebt") or 0.0
        ltd_prior = _edgar_val_prior(stmts, "balance", "LongTermDebt") or 0.0
        std_bs = _edgar_val(stmts, "balance", "ShortTermBorrowings") or 0.0
        std_bs_prior = _edgar_val_prior(stmts, "balance", "ShortTermBorrowings") or 0.0
        cash_bs = _edgar_val(stmts, "balance", "CashAndCashEquivalentsAtCarryingValue") or 0.0
        cash_bs_prior = _edgar_val_prior(stmts, "balance", "CashAndCashEquivalentsAtCarryingValue") or 0.0
        if assets and assets_prior and assets_prior > 0 and liab is not None and liab_prior is not None:
            noa_now = (assets - cash_bs) - (liab - ltd - std_bs)
            noa_prev = (assets_prior - cash_bs_prior) - (liab_prior - ltd_prior - std_bs_prior)
            avg_assets = (assets + assets_prior) / 2.0
            rec["BS_ACCRUALS"] = float(-((noa_now - noa_prev) / avg_assets)) if avg_assets > 0 else np.nan
        else:
            rec["BS_ACCRUALS"] = np.nan

        # ── Working capital accruals: (ΔWC) / Assets ──
        # WC = CurrentAssets - CurrentLiabilities - Cash
        # Returns NaN when any required prior-year field is unavailable (no silent zero-fill).
        ca_raw = _edgar_val(stmts, "balance", "AssetsCurrent")
        ca_prior_raw = _edgar_val_prior(stmts, "balance", "AssetsCurrent")
        cl_raw = _edgar_val(stmts, "balance", "LiabilitiesCurrent")
        cl_prior_raw = _edgar_val_prior(stmts, "balance", "LiabilitiesCurrent")
        if (ca_raw is not None and ca_prior_raw is not None
                and cl_raw is not None and cl_prior_raw is not None
                and assets and assets > 0):
            cash_wc = _edgar_val(stmts, "balance", "CashAndCashEquivalentsAtCarryingValue") or 0.0
            cash_wc_prior = _edgar_val_prior(stmts, "balance", "CashAndCashEquivalentsAtCarryingValue") or 0.0
            delta_wc = (ca_raw - cl_raw - cash_wc) - (ca_prior_raw - cl_prior_raw - cash_wc_prior)
            rec["WC_ACCRUALS"] = float(delta_wc / assets)
        else:
            rec["WC_ACCRUALS"] = np.nan

        # ── Depreciation ratio: D&A / Assets ──
        da = _edgar_val(stmts, "cashflow", "DepreciationDepletionAndAmortization",
                        "DepreciationAndAmortization")
        rec["DEPR_RATIO"] = float(da / assets) if da is not None and assets is not None and assets > 0 else np.nan

        # ── SUE: Standardized Unexpected Earnings (Earnings Surprise / σ_EPS) ──
        # Uses annual 10-K EPS series.  Positive SUE → recent earnings beat expectations
        # and predicts continuation (post-earnings announcement drift, PEAD).
        # Requires ≥ 4 annual observations to estimate a prior earnings change series.
        inc_df = stmts.get("income") if stmts else None
        try:
            if (isinstance(inc_df, pd.DataFrame) and not inc_df.empty
                    and "EarningsPerShareDiluted" in inc_df.columns):
                eps_s = inc_df["EarningsPerShareDiluted"].dropna()
                if len(eps_s) >= 4:
                    # DataFrame is sorted newest-first (iloc[0] = most recent filing).
                    # diff(-1) on that order gives current_year - prior_year for each row.
                    eps_changes = eps_s.diff(-1).dropna()
                    latest_surprise = float(eps_changes.iloc[0])
                    prior_std = float(eps_changes.iloc[1:].std())
                    rec["SUE"] = float(np.clip(latest_surprise / prior_std, -5, 5)) if prior_std > 1e-8 else np.nan
                else:
                    rec["SUE"] = np.nan
            else:
                rec["SUE"] = np.nan
        except Exception:
            rec["SUE"] = np.nan

        # ── Jones model intermediate values for DISC_ACCRUALS (cross-sectional OLS) ──
        # These columns are used by compute_all_factors() to run the cross-sectional
        # Jones model regression and compute discretionary accruals as the residual.
        # They are dropped from the final factor matrix before it is returned.
        revenue_now = _edgar_val(stmts, "income", "Revenue", "Revenues",
                                 "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet")
        revenue_lag = _edgar_val_prior(stmts, "income", "Revenue", "Revenues",
                                       "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet")
        ppe = _edgar_val(stmts, "balance", "PropertyPlantAndEquipmentNet")
        ocf_jones = _edgar_val(stmts, "cashflow", "NetCashProvidedByOperatingActivities",
                               "NetCashProvidedByUsedInOperatingActivities",
                               "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations")
        ni_jones = _edgar_val(stmts, "income", "NetIncomeLoss")
        avg_assets_j = (assets + assets_prior) / 2.0 if assets is not None and assets_prior is not None and assets_prior > 0 else None
        if avg_assets_j and avg_assets_j > 0:
            ta_val = ((ni_jones or 0) - (ocf_jones or 0)) / avg_assets_j
            delta_rev = ((revenue_now or 0) - (revenue_lag or 0)) / avg_assets_j
            ppe_scaled = (ppe or 0) / avg_assets_j
            rec["_JONES_TA"] = float(ta_val)
            rec["_JONES_DREV"] = float(delta_rev)
            rec["_JONES_PPE"] = float(ppe_scaled)
            rec["_JONES_1A"] = float(1.0 / avg_assets_j)
        else:
            rec["_JONES_TA"] = np.nan
            rec["_JONES_DREV"] = np.nan
            rec["_JONES_PPE"] = np.nan
            rec["_JONES_1A"] = np.nan

        records.append(rec)

    return pd.DataFrame(records).set_index("ticker") if records else pd.DataFrame()


# ── Orchestrator ─────────────────────────────────────────────────────────────

def _jones_disc_accruals(combined: pd.DataFrame) -> pd.Series:
    """Cross-sectional Jones (1991) model: compute discretionary accruals residual.

    Fits OLS across all tickers at this rebalancing date:
        TA / AvgAssets  ~  1/AvgAssets  +  ΔRev/AvgAssets  +  PPE/AvgAssets

    Returns Series of discretionary accruals (residual), negated so higher = cleaner
    earnings (lower discretionary inflation).  Returns all-NaN if < 10 valid tickers.
    """
    jones_cols = ["_JONES_TA", "_JONES_1A", "_JONES_DREV", "_JONES_PPE"]
    if not all(c in combined.columns for c in jones_cols):
        return pd.Series(np.nan, index=combined.index, name="DISC_ACCRUALS")

    sub = combined[jones_cols].dropna()
    if len(sub) < 10:
        return pd.Series(np.nan, index=combined.index, name="DISC_ACCRUALS")

    try:
        X = sub[["_JONES_1A", "_JONES_DREV", "_JONES_PPE"]].values
        y = sub["_JONES_TA"].values
        # OLS via normal equations with small ridge for stability
        XtX = X.T @ X + 1e-8 * np.eye(X.shape[1])
        Xty = X.T @ y
        beta = np.linalg.solve(XtX, Xty)
        fitted = X @ beta
        residuals = y - fitted
        # Negate: high discretionary accruals predict lower returns
        disc = pd.Series(-residuals, index=sub.index, name="DISC_ACCRUALS")
        return disc.reindex(combined.index)
    except Exception as e:
        logger.debug(f"Jones model OLS failed: {e}")
        return pd.Series(np.nan, index=combined.index, name="DISC_ACCRUALS")


def compute_all_factors(
    ohlcv_data: dict[str, pd.DataFrame],
    edgar_data: dict[str, dict],
    market_returns: pd.Series,
    fundamentals_data: dict[str, dict] | None = None,
) -> pd.DataFrame:
    """Compute all cross-sectional factors and return (N_tickers, N_factors) DataFrame.

    All data must be pre-sliced to the rebalancing date by the caller:
    - ohlcv_data:        only data up to and including the rebalancing date
    - edgar_data:        already filtered by filed_date <= rebal_date (point-in-time)
    - market_returns:    only data up to and including the rebalancing date
    - fundamentals_data: optional yfinance info dicts; enables FWD_EP when provided

    Drops factors with <30% coverage; fills remaining NaNs with cross-sectional median.
    """
    dfs = []

    logger.debug("Computing momentum factors...")
    mom = compute_momentum_factors(ohlcv_data)
    if not mom.empty:
        dfs.append(mom)

    logger.debug("Computing value factors...")
    val = compute_value_factors(edgar_data, ohlcv_data, fundamentals_data)
    if not val.empty:
        dfs.append(val)

    logger.debug("Computing quality factors...")
    qual = compute_quality_factors(edgar_data, ohlcv_data)
    if not qual.empty:
        dfs.append(qual)

    logger.debug("Computing technical factors...")
    tech = compute_technical_factors(ohlcv_data)
    if not tech.empty:
        dfs.append(tech)

    logger.debug("Computing risk factors...")
    risk = compute_risk_factors(ohlcv_data, market_returns)
    if not risk.empty:
        dfs.append(risk)

    logger.debug("Computing accrual factors...")
    accr = compute_accrual_factors(edgar_data)
    if not accr.empty:
        dfs.append(accr)

    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs, axis=1)
    combined = combined.loc[:, ~combined.columns.duplicated()]

    # ── Jones model DISC_ACCRUALS (cross-sectional; requires the full matrix) ──
    disc = _jones_disc_accruals(combined)
    combined["DISC_ACCRUALS"] = disc

    # Drop Jones model intermediate columns (not factors themselves)
    jones_intermediates = ["_JONES_TA", "_JONES_1A", "_JONES_DREV", "_JONES_PPE"]
    combined = combined.drop(columns=[c for c in jones_intermediates if c in combined.columns])

    # Drop factors with <30% cross-sectional coverage
    coverage = combined.notna().mean()
    low_coverage = coverage[coverage < 0.30].index.tolist()
    if low_coverage:
        logger.debug(f"Dropping {len(low_coverage)} low-coverage factors: {low_coverage[:10]}...")
        combined = combined.drop(columns=low_coverage)

    # Fill NaNs with cross-sectional median
    for col in combined.columns:
        median_val = combined[col].median()
        combined[col] = combined[col].fillna(median_val)

    combined = combined.replace([np.inf, -np.inf], np.nan).fillna(0)

    logger.debug(f"Factor matrix: {combined.shape[0]} tickers × {combined.shape[1]} factors")
    return combined


def compute_fast_factors(ohlcv_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Compute only Momentum + Technical factors (27/63) for signal decay checks.

    These factors require only OHLCV data (no EDGAR), so they can be recomputed
    cheaply between rebalancing dates to detect signal reversals.
    """
    dfs = []
    mom = compute_momentum_factors(ohlcv_data)
    if not mom.empty:
        dfs.append(mom)
    tech = compute_technical_factors(ohlcv_data)
    if not tech.empty:
        dfs.append(tech)
    if not dfs:
        return pd.DataFrame()
    combined = pd.concat(dfs, axis=1)
    combined = combined.loc[:, ~combined.columns.duplicated()]
    coverage = combined.notna().mean()
    low_cov = coverage[coverage < 0.30].index.tolist()
    if low_cov:
        combined = combined.drop(columns=low_cov)
    for col in combined.columns:
        combined[col] = combined[col].fillna(combined[col].median())
    combined = combined.replace([np.inf, -np.inf], np.nan).fillna(0)
    return combined

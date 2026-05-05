"""Human-readable display names for the Second Tower factor codes.

Used by Agent C when formatting Agent B output for the LLM prompt and in the
generated rationale, so user-facing text reads as plain English instead of
internal code identifiers.
"""

from __future__ import annotations


FACTOR_DISPLAY_NAMES: dict[str, str] = {
    # Momentum
    "MOM1M": "1-month price momentum",
    "MOM3M": "3-month price momentum",
    "MOM6M": "6-month price momentum",
    "MOM12M": "12-month price momentum",
    "MOM12M1": "12-minus-1 month momentum",
    "HIGH52W_RATIO": "52-week high ratio",
    "IND_MOM": "industry-relative momentum",
    "STR": "short-term reversal",
    "LTR": "long-term reversal",
    "CONSEC_UP": "consecutive up days (20-day)",
    "TSMOM": "time-series momentum",
    "PRICE_ACCEL": "price acceleration",
    # Value
    "BM": "book-to-market ratio",
    "EP": "earnings yield",
    "CFP": "cash-flow yield",
    "SP": "sales yield",
    "EV_EBITDA": "EV-to-EBITDA",
    "TRAILING_PE_INV": "inverse trailing price-to-earnings",
    "DIV_YIELD": "dividend yield",
    "PAYOUT_YIELD": "shareholder payout yield",
    "FWD_EP": "forward earnings yield",
    # Quality
    "ROA": "return on assets",
    "ROE": "return on equity",
    "GP_A": "gross profitability",
    "OP_MARGIN": "operating margin",
    "NET_MARGIN": "net margin",
    "ASSET_GROWTH": "asset growth",
    "CAPEX_GROWTH": "capital-expenditure growth",
    "SALES_GROWTH": "sales growth",
    "PIOTROSKI_F": "Piotroski F-score",
    "ACCRUALS_RATIO": "accruals ratio",
    "EARN_QUALITY": "earnings quality",
    "LEV_RATIO": "leverage ratio",
    # Technical
    "RSI14": "14-day Relative Strength Index",
    "MACD": "MACD histogram",
    "BB_POS": "Bollinger Band position",
    "ATR_NORM": "normalized Average True Range",
    "OBV": "On-Balance Volume trend",
    "MA50_200": "50-day vs 200-day moving-average spread",
    "CROSS_SIGNAL": "moving-average crossover signal",
    "VOL_MOM": "volume momentum",
    "STOCH": "Stochastic Oscillator",
    "WILLIAMS_R": "Williams %R",
    "CCI": "Commodity Channel Index",
    "ADX": "Average Directional Index",
    "LOW52W_DIST": "distance from 52-week low",
    "VOL_RATIO": "recent volume ratio",
    "DOLLAR_VOL": "dollar volume",
    # Risk
    "TRAILING_BETA": "trailing beta",
    "IVOL": "idiosyncratic volatility",
    "TVOL": "total volatility",
    "MAX_RET": "maximum daily return",
    "SKEW": "return skewness",
    "DOWN_BETA": "downside beta",
    "KURT": "return kurtosis",
    "AMIHUD": "Amihud illiquidity",
    # Accruals
    "BS_ACCRUALS": "balance-sheet accruals",
    "WC_ACCRUALS": "working-capital accruals",
    "DEPR_RATIO": "depreciation ratio",
    "SUE": "standardized unexpected earnings",
    "DISC_ACCRUALS": "discretionary accruals",
}


def factor_display_name(code: str) -> str:
    """Return the human-readable name for a factor code, falling back to the code itself."""
    return FACTOR_DISPLAY_NAMES.get(code, code)

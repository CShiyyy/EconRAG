"""SEC EDGAR fetcher using data.sec.gov XBRL API.

Provides point-in-time financial statement data from annual 10-K filings only,
with a 90-day filing lag applied by the caller to avoid lookahead bias.

Ported from quant_bb2/data/fetchers/sec_edgar_fetcher.py and adapted for
TradeModem's SQLite-backed cache (data/cache.py).
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

from pipeline.config import DATA_DIR

logger = logging.getLogger(__name__)

CACHE_DB = DATA_DIR / "edgar_cache.db"
EDGAR_CACHE_TTL_DAYS = 90
EDGAR_RATE_LIMIT_RPS = 8.0

# ETFs and non-operating entities that have no EDGAR filings
_SKIP_TICKERS = frozenset({
    "SPY", "QQQ", "IWM", "DIA", "GLD", "SLV", "TLT", "IEF", "SHY", "AGG",
    "BND", "LQD", "HYG", "VTI", "VEA", "VWO", "EEM", "EFA", "IAU", "USO",
    "UNG", "VNQ", "XLF", "XLK", "XLE", "XLV", "XLI", "XLB", "XLU", "XLP",
    "XLY", "XLC", "XLRE", "BIL", "SHV", "ACWX", "VXX", "UVXY", "SVXY",
    "^VIX", "^GSPC", "^DJI", "^IXIC", "^TNX", "^IRX",
})


# ── Rate limiter ─────────────────────────────────────────────────────────────

class TokenBucket:
    """Thread-safe token bucket rate limiter."""

    def __init__(self, rate: float = 8.0, capacity: float = 8.0):
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.last_refill = time.monotonic()
        self._lock = threading.Lock()

    def consume(self, tokens: float = 1.0) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                elapsed = now - self.last_refill
                self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
                self.last_refill = now
                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return
            time.sleep(0.05)


# ── SQLite cache helpers ──────────────────────────────────────────────────────

def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(CACHE_DB))
    conn.execute(
        """CREATE TABLE IF NOT EXISTS edgar_cache (
            cache_key TEXT PRIMARY KEY,
            data TEXT NOT NULL,
            created_at TEXT DEFAULT (datetime('now'))
        )"""
    )
    conn.commit()
    return conn


def _cache_get(key: str) -> str | None:
    conn = _get_conn()
    try:
        row = conn.execute(
            "SELECT data, created_at FROM edgar_cache WHERE cache_key = ?", (key,)
        ).fetchone()
        if row is None:
            return None
        data, created_at = row
        # TTL check
        try:
            created_dt = datetime.fromisoformat(created_at)
            if created_dt.tzinfo is None:
                created_dt = created_dt.replace(tzinfo=timezone.utc)
            age = datetime.now(timezone.utc) - created_dt
            if age > timedelta(days=EDGAR_CACHE_TTL_DAYS):
                conn.execute("DELETE FROM edgar_cache WHERE cache_key = ?", (key,))
                conn.commit()
                return None
        except Exception:
            pass
        return data
    finally:
        conn.close()


def _cache_set(key: str, data: str) -> None:
    conn = _get_conn()
    try:
        conn.execute(
            "INSERT OR REPLACE INTO edgar_cache (cache_key, data) VALUES (?, ?)",
            (key, data),
        )
        conn.commit()
    finally:
        conn.close()


# ── XBRL concept lists ────────────────────────────────────────────────────────

_BALANCE_CONCEPTS = [
    ("Assets", "USD"),
    ("CashAndCashEquivalentsAtCarryingValue", "USD"),
    ("Liabilities", "USD"),
    ("LongTermDebt", "USD"),
    ("ShortTermBorrowings", "USD"),
    ("AssetsCurrent", "USD"),
    ("LiabilitiesCurrent", "USD"),
    ("StockholdersEquity", "USD"),
    ("StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest", "USD"),
    ("PropertyPlantAndEquipmentNet", "USD"),
    ("CommonStockSharesOutstanding", "shares"),
]

_INCOME_CONCEPTS = [
    ("NetIncomeLoss", "USD"),
    ("Revenues", "USD"),
    ("RevenueFromContractWithCustomerExcludingAssessedTax", "USD"),
    ("SalesRevenueNet", "USD"),
    ("CostOfGoodsAndServicesSold", "USD"),
    ("CostOfRevenue", "USD"),
    ("GrossProfit", "USD"),
    ("OperatingIncomeLoss", "USD"),
    ("EarningsPerShareBasic", "USD/shares"),
    ("EarningsPerShareDiluted", "USD/shares"),
    ("DepreciationAndAmortization", "USD"),
    ("InterestExpense", "USD"),
]

_CASHFLOW_CONCEPTS = [
    ("NetCashProvidedByUsedInOperatingActivities", "USD"),
    ("NetCashProvidedByUsedInOperatingActivitiesContinuingOperations", "USD"),
    ("PaymentsToAcquirePropertyPlantAndEquipment", "USD"),
    ("DepreciationDepletionAndAmortization", "USD"),
    # Shareholder return flows — used by DIV_YIELD and PAYOUT_YIELD value factors.
    ("PaymentsOfDividendsCommonStock", "USD"),
    ("PaymentsOfDividends", "USD"),
    ("PaymentsForRepurchaseOfCommonStock", "USD"),
]

# Alias: signal code expects "NetCashProvidedByOperatingActivities"
# but EDGAR stores it under the longer name.
_CASHFLOW_ALIASES = {
    "NetCashProvidedByOperatingActivities": [
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    ],
    "Revenue": [
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "SalesRevenueNet",
    ],
    "CostOfGoods": [
        "CostOfGoodsAndServicesSold",
        "CostOfRevenue",
    ],
    "StockholdersEquityNet": [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ],
}


# ── Core fetcher class ────────────────────────────────────────────────────────

class SECEDGARFetcher:
    """Fetches point-in-time annual financials from SEC EDGAR XBRL API."""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "TradeModem/1.0 research@example.com",
            "Accept-Encoding": "gzip, deflate",
        })
        self.bucket = TokenBucket(rate=EDGAR_RATE_LIMIT_RPS, capacity=EDGAR_RATE_LIMIT_RPS)
        self._cik_map: dict[str, str] | None = None
        self._cik_lock = threading.Lock()

    def _load_cik_map(self) -> dict[str, str]:
        with self._cik_lock:
            if self._cik_map is not None:
                return self._cik_map

            # Hold lock through entire fetch to prevent concurrent HTTP requests
            cached = _cache_get("cik_map")
            if cached:
                try:
                    data = json.loads(cached)
                    self._cik_map = data
                    logger.debug(f"Loaded CIK map from cache ({len(data)} entries)")
                    return data
                except Exception:
                    pass

            logger.info("Fetching CIK map from SEC EDGAR...")
            self.bucket.consume()
            try:
                resp = self.session.get(
                    "https://www.sec.gov/files/company_tickers.json", timeout=30
                )
                resp.raise_for_status()
                raw = resp.json()
                cik_map = {}
                for entry in raw.values():
                    ticker = entry.get("ticker", "").upper()
                    cik = str(entry.get("cik_str", ""))
                    if ticker and cik:
                        cik_map[ticker] = cik.zfill(10)
                _cache_set("cik_map", json.dumps(cik_map))
                self._cik_map = cik_map
                logger.info(f"Cached CIK map ({len(cik_map)} tickers)")
                return cik_map
            except Exception as e:
                logger.error(f"Failed to fetch CIK map: {e}")
                self._cik_map = {}
                return {}

    def get_cik(self, ticker: str) -> str | None:
        cik_map = self._load_cik_map()
        return cik_map.get(ticker.upper())

    def fetch_company_facts(self, ticker: str) -> dict:
        """Fetch raw XBRL company facts from EDGAR. Returns {} for ETFs or on error."""
        ticker = ticker.upper()
        if ticker in _SKIP_TICKERS:
            return {}

        cache_key = f"edgar_facts_{ticker}"
        cached = _cache_get(cache_key)
        if cached:
            try:
                return json.loads(cached)
            except Exception:
                pass

        cik = self.get_cik(ticker)
        if not cik:
            logger.debug(f"No CIK found for {ticker}")
            return {}

        url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
        self.bucket.consume()
        try:
            resp = self.session.get(url, timeout=30)
            if resp.status_code == 404:
                logger.debug(f"No EDGAR facts for {ticker} (CIK {cik})")
                return {}
            resp.raise_for_status()
            facts = resp.json()
            _cache_set(cache_key, json.dumps(facts))
            return facts
        except Exception as e:
            logger.warning(f"Failed to fetch EDGAR facts for {ticker}: {e}")
            return {}


def _extract_one_concept(us_gaap: dict, concept: str, unit: str) -> pd.Series:
    """Extract annual 10-K time series for one XBRL concept.

    Returns a Series indexed by the SEC filing date (the date the 10-K was
    actually filed with the SEC).  This is the correct point-in-time index
    for lookahead-free filtering: a filing is only usable on or after the
    day it was submitted.

    If the 'filed' field is absent from an entry (rare in practice) we fall
    back to fiscal-year-end + 90 days as a conservative surrogate.

    Only 10-K and 10-K/A filings are included.
    """
    entries = us_gaap.get(concept, {}).get("units", {}).get(unit, [])
    records = []
    for e in entries:
        if e.get("form") not in ("10-K", "10-K/A"):
            continue
        fy = e.get("fy")
        fp = e.get("fp", "")
        end_date = e.get("end")
        filed_date = e.get("filed")
        val = e.get("val")
        if fy and end_date and val is not None:
            if filed_date:
                idx_date = pd.Timestamp(filed_date)
            else:
                # Conservative fallback: assume filing arrived 90 days after FY-end
                idx_date = pd.Timestamp(end_date) + pd.Timedelta(days=90)
            records.append({
                "fy": fy, "fp": fp,
                "idx_date": idx_date,
                "val": float(val),
            })
    if not records:
        return pd.Series(dtype=float, name=concept)
    df = (
        pd.DataFrame(records)
        .sort_values("idx_date")  # chronological order → keep="last" retains the most-recently-filed version
        .drop_duplicates(subset=["fy", "fp"], keep="last")
    )
    s = df.set_index("idx_date")["val"].sort_index().rename(concept)
    # Two different (fy, fp) pairs can share the same filed date (e.g. a 10-K and a 10-K/A
    # both filed on the same day). Keep the last value per date so the index stays unique.
    s = s[~s.index.duplicated(keep="last")]
    return s


def _build_statement_df(us_gaap: dict, concepts: list[tuple[str, str]]) -> pd.DataFrame:
    """Build a financial statement DataFrame from a list of (concept, unit) tuples.

    Indexed by SEC filing date, sorted DESCENDING (most recently filed first →
    iloc[0] = latest available, iloc[1] = prior year).

    After concat, different concepts may have different filing dates (e.g. a
    10-K/A amendment that updates only one concept creates a new row where all
    other columns are NaN).  We sort ascending and forward-fill so that any
    amendment-only row carries the latest non-NaN value for every column.
    This prevents _edgar_val(iloc[0]) from silently returning None for concepts
    that are valid but were filed on a slightly earlier date.
    """
    series_list = []
    for concept, unit in concepts:
        s = _extract_one_concept(us_gaap, concept, unit)
        if len(s) > 0:
            series_list.append(s)
    if not series_list:
        return pd.DataFrame()
    df = pd.concat(series_list, axis=1)
    # Sort ascending → ffill (fills NaN gaps with the most recent prior value)
    # → sort descending so iloc[0] = most recently filed row with all columns populated.
    df = df.sort_index(ascending=True).ffill().sort_index(ascending=False)
    return df


def fetch_all_financials(ticker: str) -> dict:
    """Return point-in-time structured financials for backtesting.

    Returns dict: {'balance': df, 'income': df, 'cashflow': df}
      - Each DataFrame is indexed by SEC filing date, sorted DESCENDING
      - iloc[0] = most recently filed annual report, iloc[1] = prior year
      - Only 10-K/10-K/A filings — no quarterly data
      - Caller filters by rebalancing date (df.index <= rebal_date) to
        enforce point-in-time correctness — no artificial lag required since
        the index IS the actual availability date

    Returns {} for ETFs or when no EDGAR data is available.
    """
    facts = _fetcher.fetch_company_facts(ticker)
    if not facts:
        return {}
    us_gaap = facts.get("facts", {}).get("us-gaap", {})
    if not us_gaap:
        return {}

    balance_df = _build_statement_df(us_gaap, _BALANCE_CONCEPTS)
    income_df = _build_statement_df(us_gaap, _INCOME_CONCEPTS)
    cashflow_df = _build_statement_df(us_gaap, _CASHFLOW_CONCEPTS)

    # Add alias columns so downstream code can use canonical names
    for target_col, source_cols in _CASHFLOW_ALIASES.items():
        for df in [balance_df, income_df, cashflow_df]:
            if target_col not in df.columns:
                for src in source_cols:
                    if src in df.columns:
                        df[target_col] = df[src]
                        break

    return {"balance": balance_df, "income": income_df, "cashflow": cashflow_df}


def fetch_edgar_batch(tickers: list[str], max_workers: int = 4) -> dict[str, dict]:
    """Batch-fetch EDGAR financials for multiple tickers using a thread pool.

    Returns dict: ticker -> {'balance': df, 'income': df, 'cashflow': df}
    Missing/ETF tickers map to {}.
    """
    result: dict[str, dict] = {}
    to_fetch = [t for t in tickers if t.upper() not in _SKIP_TICKERS]

    logger.info(f"Fetching EDGAR data for {len(to_fetch)} tickers ({max_workers} workers)...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_all_financials, t): t for t in to_fetch}
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                result[ticker.upper()] = future.result()
            except Exception as e:
                logger.warning(f"EDGAR fetch failed for {ticker}: {e}")
                result[ticker.upper()] = {}

    # ETFs get empty dict
    for t in tickers:
        if t.upper() not in result:
            result[t.upper()] = {}

    have_data = sum(1 for v in result.values() if v)
    logger.info(f"EDGAR data loaded: {have_data}/{len(tickers)} tickers have filings")
    return result


# Module-level singleton fetcher
_fetcher = SECEDGARFetcher()

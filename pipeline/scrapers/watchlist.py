import logging
import re
from io import StringIO

import pandas as pd
import requests

from pipeline.config import UNIVERSE_URLS, VALID_UNIVERSES

_HEADERS = {
    "User-Agent": "EconRAG/0.1 (https://github.com; educational project)"
}

logger = logging.getLogger(__name__)

WatchlistEntry = tuple[str, str, str]  # (ticker, company_name, sector)


def _clean_ticker(raw: str) -> str:
    """Strip whitespace and remove bracketed footnote markers like [1]."""
    return re.sub(r"\[.*?\]", "", str(raw)).strip()


def _find_table(tables: list[pd.DataFrame], required_cols: set[str]) -> pd.DataFrame:
    """Find the first table whose columns contain all required_cols."""
    for df in tables:
        cols = {str(c).strip() for c in df.columns}
        if required_cols.issubset(cols):
            return df
    raise RuntimeError(
        f"No table found with required columns {required_cols}. "
        f"Available tables have columns: {[list(df.columns) for df in tables]}"
    )


def _fetch_html(url: str) -> str:
    """Fetch HTML from a URL with a proper User-Agent header."""
    resp = requests.get(url, headers=_HEADERS, timeout=30)
    resp.raise_for_status()
    return resp.text


def _scrape_sp500() -> list[WatchlistEntry]:
    url = UNIVERSE_URLS["sp500"]
    try:
        html = _fetch_html(url)
        tables = pd.read_html(StringIO(html), header=0)
    except Exception as e:
        raise RuntimeError(f"Failed to parse sp500 watchlist from Wikipedia: {e}") from e

    df = _find_table(tables, {"Symbol", "Security", "GICS Sector"})
    results = []
    for _, row in df.iterrows():
        ticker = _clean_ticker(row["Symbol"])
        name = str(row["Security"]).strip()
        sector = str(row["GICS Sector"]).strip()
        if ticker:
            results.append((ticker, name, sector))

    if not (480 <= len(results) <= 520):
        logger.warning("S&P 500 returned %d entries (expected ~500)", len(results))
    return results


def _scrape_nasdaq100() -> list[WatchlistEntry]:
    url = UNIVERSE_URLS["nasdaq100"]
    try:
        html = _fetch_html(url)
        tables = pd.read_html(StringIO(html), header=0)
    except Exception as e:
        raise RuntimeError(f"Failed to parse nasdaq100 watchlist from Wikipedia: {e}") from e

    df = _find_table(tables, {"Ticker", "Company"})
    # Sector column varies: "GICS Sector", "ICB Industry", or "GICS Sub-Industry"
    sector_col = None
    for candidate in ("GICS Sector", "GICS Sub-Industry", "ICB Industry"):
        if candidate in df.columns:
            sector_col = candidate
            break
    if sector_col is None:
        sector_col = df.columns[-1]  # fallback to last column

    results = []
    for _, row in df.iterrows():
        ticker = _clean_ticker(row["Ticker"])
        name = str(row["Company"]).strip()
        sector = str(row[sector_col]).strip()
        if ticker:
            results.append((ticker, name, sector))

    if not (95 <= len(results) <= 110):
        logger.warning("Nasdaq-100 returned %d entries (expected ~100)", len(results))
    return results


def _scrape_djia30() -> list[WatchlistEntry]:
    url = UNIVERSE_URLS["djia30"]
    try:
        html = _fetch_html(url)
        tables = pd.read_html(StringIO(html), header=0)
    except Exception as e:
        raise RuntimeError(f"Failed to parse djia30 watchlist from Wikipedia: {e}") from e

    df = _find_table(tables, {"Symbol", "Company"})
    # Sector column may be "Industry" or "Sector"
    sector_col = None
    for candidate in ("Industry", "Sector", "GICS Sector"):
        if candidate in df.columns:
            sector_col = candidate
            break
    if sector_col is None:
        sector_col = df.columns[-1]

    results = []
    for _, row in df.iterrows():
        ticker = _clean_ticker(row["Symbol"])
        name = str(row["Company"]).strip()
        sector = str(row[sector_col]).strip()
        if ticker:
            results.append((ticker, name, sector))

    if not (25 <= len(results) <= 35):
        logger.warning("DJIA returned %d entries (expected ~30)", len(results))
    return results


_SCRAPERS = {
    "sp500": _scrape_sp500,
    "nasdaq100": _scrape_nasdaq100,
    "djia30": _scrape_djia30,
}


def scrape_universe(universe: str) -> list[WatchlistEntry]:
    """Scrape Wikipedia for index constituents.

    Args:
        universe: One of 'sp500', 'nasdaq100', 'djia30'.

    Returns:
        List of (ticker, company_name, sector) tuples.
    """
    if universe not in VALID_UNIVERSES:
        raise ValueError(f"Unknown universe '{universe}'. Must be one of {VALID_UNIVERSES}")
    return _SCRAPERS[universe]()

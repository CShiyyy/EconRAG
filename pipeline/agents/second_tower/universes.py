"""Universe presets for multi-asset strategies."""
import io
import logging
import sqlite3
import time
from datetime import datetime, timezone
from functools import lru_cache

import pandas as pd
import requests
from requests.exceptions import HTTPError
import yfinance as yf

from pipeline.config import DATA_DIR
CACHE_DB = DATA_DIR / "edgar_cache.db"

logger = logging.getLogger(__name__)

_HEADERS = {
    "User-Agent": "TradeModem/1.0 (local quant sim; https://github.com/CShiyyy/TradeModem) Python-requests/2.x",
    "Accept-Language": "en-US,en;q=0.5",
}

# ---------------------------------------------------------------------------
# SQLite-backed Wikipedia page cache (survives server restarts)
# ---------------------------------------------------------------------------
_WIKI_CACHE_TTL = 7 * 24 * 3600  # 7 days — index changes are quarterly
_MAX_RETRIES = 3
_BACKOFF_BASE = 2.0  # seconds: 2, 4, 8


def _get_wiki_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(CACHE_DB))
    conn.execute("""CREATE TABLE IF NOT EXISTS wiki_page_cache (
        cache_key TEXT PRIMARY KEY,
        html_text TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""")
    conn.commit()
    return conn


def _load_wiki_html(key: str) -> str | None:
    """Load cached HTML from SQLite if fresh enough."""
    conn = _get_wiki_conn()
    try:
        row = conn.execute(
            "SELECT html_text, created_at FROM wiki_page_cache WHERE cache_key = ?",
            (key,),
        ).fetchone()
        if row is None:
            return None
        created_at = datetime.fromisoformat(row[1])
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)
        age = (datetime.now(timezone.utc) - created_at).total_seconds()
        if age > _WIKI_CACHE_TTL:
            return None
        return row[0]
    finally:
        conn.close()


def _save_wiki_html(key: str, html_text: str) -> None:
    """Persist HTML to SQLite."""
    conn = _get_wiki_conn()
    try:
        conn.execute(
            "INSERT OR REPLACE INTO wiki_page_cache (cache_key, html_text, created_at) "
            "VALUES (?, ?, ?)",
            (key, html_text, datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
    finally:
        conn.close()


def _fetch_wikipedia_url(url: str) -> str:
    """Fetch a Wikipedia page with retry + exponential backoff on 403/429.

    Falls back to MediaWiki API and Wikimedia REST API if direct fetch fails.
    """
    last_exc = None
    for attempt in range(_MAX_RETRIES):
        try:
            response = requests.get(url, headers=_HEADERS, timeout=15)
            response.raise_for_status()
            return response.text
        except HTTPError as e:
            last_exc = e
            if e.response is not None and e.response.status_code in (403, 429):
                wait = _BACKOFF_BASE * (2 ** attempt)
                logger.warning(
                    f"Wikipedia returned {e.response.status_code} for {url}, "
                    f"retrying in {wait:.0f}s (attempt {attempt + 1}/{_MAX_RETRIES})"
                )
                time.sleep(wait)
            else:
                raise  # non-retryable HTTP error

    # Fallback 1: MediaWiki API (action=parse)
    page_title = url.rsplit("/wiki/", 1)[-1] if "/wiki/" in url else None
    if page_title:
        try:
            api_url = f"https://en.wikipedia.org/w/api.php?action=parse&page={page_title}&prop=text&format=json"
            resp = requests.get(api_url, headers=_HEADERS, timeout=20)
            resp.raise_for_status()
            html = resp.json().get("parse", {}).get("text", {}).get("*", "")
            if html:
                logger.info(f"Wikipedia fallback: fetched {page_title} via MediaWiki API")
                return html
        except Exception as e2:
            logger.warning(f"MediaWiki API fallback failed for {page_title}: {e2}")

        # Fallback 2: Wikimedia REST API
        try:
            rest_url = f"https://en.wikipedia.org/api/rest_v1/page/html/{page_title}"
            resp = requests.get(rest_url, headers=_HEADERS, timeout=20)
            resp.raise_for_status()
            logger.info(f"Wikipedia fallback: fetched {page_title} via Wikimedia REST API")
            return resp.text
        except Exception as e3:
            logger.warning(f"Wikimedia REST API fallback failed for {page_title}: {e3}")

    raise last_exc  # all fallbacks exhausted

SECTOR_ETFS = [
    "XLK", "XLF", "XLV", "XLE", "XLI", "XLC", "XLY", "XLP",
    "XLB", "XLRE", "XLU", "VGT", "VHT", "VNQ",
]

# ---------------------------------------------------------------------------
# S&P 500 Wikipedia page cache (shared between current + historical lookups)
# ---------------------------------------------------------------------------
_sp500_page_cache: dict[str, object] = {}
_SP500_CACHE_TTL = 3600  # 1 hour


def _fetch_sp500_page() -> list[pd.DataFrame]:
    """Fetch S&P 500 page tables: memory -> SQLite -> Wikipedia (with retry)."""
    now = time.time()
    if (
        _sp500_page_cache
        and (now - _sp500_page_cache["fetched_at"]) < _SP500_CACHE_TTL  # type: ignore[operator]
    ):
        return _sp500_page_cache["tables"]  # type: ignore[return-value]

    # Tier 2: SQLite disk cache
    html = _load_wiki_html("sp500")
    if html is not None:
        logger.info("Loaded S&P 500 page from disk cache")
        tables = pd.read_html(io.StringIO(html))
        _sp500_page_cache["tables"] = tables
        _sp500_page_cache["fetched_at"] = now
        return tables

    # Tier 3: fetch from Wikipedia with retry
    logger.info("Fetching S&P 500 page from Wikipedia...")
    html = _fetch_wikipedia_url(
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    )
    tables = pd.read_html(io.StringIO(html))
    _save_wiki_html("sp500", html)
    _sp500_page_cache["tables"] = tables
    _sp500_page_cache["fetched_at"] = now
    return tables


def get_sp500_tickers() -> list[str]:
    """Download current S&P 500 ticker list from Wikipedia."""
    try:
        tables = _fetch_sp500_page()
        tickers = tables[0]["Symbol"].tolist()
        tickers = [t.replace(".", "-") for t in tickers]
        logger.info(f"Found {len(tickers)} S&P 500 tickers")
        return tickers
    except Exception as e:
        logger.error(f"Failed to fetch S&P 500 tickers: {e}")
        raise


def _get_sp500_changes() -> list[tuple[pd.Timestamp, str | None, str | None]]:
    """Parse the S&P 500 historical changes table from Wikipedia.

    Returns list of (date, added_ticker, removed_ticker) sorted by date ascending.
    """
    tables = _fetch_sp500_page()
    if len(tables) < 2:
        logger.warning("S&P 500 Wikipedia page has no changes table")
        return []

    changes_df = tables[1]

    # The changes table has multi-level columns (tuples) or flat strings.
    # Use the original column objects for DataFrame access, string repr for matching.
    raw_cols = list(changes_df.columns)

    # Find date column
    date_col = None
    for c in raw_cols:
        if "date" in str(c).lower():
            date_col = c
            break
    if date_col is None:
        date_col = raw_cols[0]  # first column is typically the date

    # Find added/removed ticker columns by matching against string representation.
    added_col = None
    removed_col = None
    for c in raw_cols:
        cl = str(c).lower()
        if "added" in cl and ("ticker" in cl or "symbol" in cl):
            added_col = c
        elif "removed" in cl and ("ticker" in cl or "symbol" in cl):
            removed_col = c

    # Fallback: positional heuristic (Date | Added Ticker | Added Security | Removed Ticker | ...)
    if added_col is None and len(raw_cols) >= 3:
        added_col = raw_cols[1]
    if removed_col is None and len(raw_cols) >= 5:
        removed_col = raw_cols[3]

    if added_col is None and removed_col is None:
        logger.warning("Could not identify added/removed columns in S&P 500 changes table")
        return []

    changes: list[tuple[pd.Timestamp, str | None, str | None]] = []
    for _, row in changes_df.iterrows():
        try:
            date = pd.to_datetime(row[date_col])
        except Exception:
            continue
        if pd.isna(date):
            continue

        added = str(row[added_col]).strip() if added_col and pd.notna(row[added_col]) else None
        removed = str(row[removed_col]).strip() if removed_col and pd.notna(row[removed_col]) else None

        # Normalize tickers
        if added and added.lower() in ("nan", ""):
            added = None
        if removed and removed.lower() in ("nan", ""):
            removed = None
        if added:
            added = added.replace(".", "-")
        if removed:
            removed = removed.replace(".", "-")

        if added or removed:
            changes.append((date, added, removed))

    changes.sort(key=lambda x: x[0])
    logger.info(f"Parsed {len(changes)} S&P 500 historical changes")
    return changes


def get_sp500_tickers_as_of(as_of_date: str) -> list[str]:
    """Reconstruct S&P 500 membership as of a historical date.

    Works backwards from current members, undoing changes that occurred after
    as_of_date to reconstruct the index composition at that point in time.
    """
    try:
        current = set(get_sp500_tickers())
        changes = _get_sp500_changes()
    except Exception as e:
        logger.error(f"Failed to reconstruct historical S&P 500: {e}")
        raise

    if not changes:
        logger.warning("No historical changes available; returning current S&P 500 members")
        return sorted(current)

    cutoff = pd.Timestamp(as_of_date)

    # Survivorship bias warning: reconstruction works backward from CURRENT
    # members.  Companies that left the index before the Wikipedia change table
    # begins (or whose removal was never recorded) will be silently omitted.
    # For long look-back periods this overstates backtest returns.
    earliest_change = changes[0][0]
    years_before_table = (earliest_change - cutoff).days / 365.25
    if cutoff < earliest_change:
        logger.warning(
            f"as_of_date {as_of_date} is {years_before_table:.1f} years before "
            f"earliest Wikipedia change record ({earliest_change.date()}). "
            f"Reconstruction is incomplete — companies that left the index before "
            f"that date are missing, which introduces survivorship bias."
        )

    # Filter to changes AFTER as_of_date, walk in reverse chronological order
    future_changes = [(d, a, r) for d, a, r in changes if d > cutoff]
    members = set(current)

    for _, added, removed in reversed(future_changes):
        # Undo this change: if a ticker was added after our date, remove it
        if added and added in members:
            members.discard(added)
        # If a ticker was removed after our date, add it back
        if removed:
            members.add(removed)

    logger.info(
        f"Reconstructed S&P 500 as of {as_of_date}: {len(members)} members "
        f"(current: {len(current)}, undid {len(future_changes)} changes)"
    )
    return sorted(members)


# ---------------------------------------------------------------------------
# Nasdaq-100 Wikipedia page cache
# ---------------------------------------------------------------------------
_nasdaq100_page_cache: dict[str, object] = {}
_dow30_page_cache: dict[str, object] = {}
_INDEX_PAGE_TTL = 3600  # 1 hour


def _fetch_nasdaq100_page() -> list[pd.DataFrame]:
    """Fetch Nasdaq-100 page tables: memory -> SQLite -> Wikipedia (with retry)."""
    now = time.time()
    if (
        _nasdaq100_page_cache
        and (now - _nasdaq100_page_cache["fetched_at"]) < _INDEX_PAGE_TTL  # type: ignore[operator]
    ):
        return _nasdaq100_page_cache["tables"]  # type: ignore[return-value]

    # Tier 2: SQLite disk cache
    html = _load_wiki_html("nasdaq100")
    if html is not None:
        logger.info("Loaded Nasdaq-100 page from disk cache")
        tables = pd.read_html(io.StringIO(html))
        _nasdaq100_page_cache["tables"] = tables
        _nasdaq100_page_cache["fetched_at"] = now
        return tables

    # Tier 3: fetch from Wikipedia with retry
    logger.info("Fetching Nasdaq-100 page from Wikipedia...")
    html = _fetch_wikipedia_url("https://en.wikipedia.org/wiki/Nasdaq-100")
    tables = pd.read_html(io.StringIO(html))
    _save_wiki_html("nasdaq100", html)
    _nasdaq100_page_cache["tables"] = tables
    _nasdaq100_page_cache["fetched_at"] = now
    return tables


def _fetch_dow30_page() -> list[pd.DataFrame]:
    """Fetch Dow 30 page tables: memory -> SQLite -> Wikipedia (with retry)."""
    now = time.time()
    if (
        _dow30_page_cache
        and (now - _dow30_page_cache["fetched_at"]) < _INDEX_PAGE_TTL  # type: ignore[operator]
    ):
        return _dow30_page_cache["tables"]  # type: ignore[return-value]

    # Tier 2: SQLite disk cache
    html = _load_wiki_html("dow30")
    if html is not None:
        logger.info("Loaded Dow 30 page from disk cache")
        tables = pd.read_html(io.StringIO(html))
        _dow30_page_cache["tables"] = tables
        _dow30_page_cache["fetched_at"] = now
        return tables

    # Tier 3: fetch from Wikipedia with retry
    logger.info("Fetching Dow 30 page from Wikipedia...")
    html = _fetch_wikipedia_url(
        "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average"
    )
    tables = pd.read_html(io.StringIO(html))
    _save_wiki_html("dow30", html)
    _dow30_page_cache["tables"] = tables
    _dow30_page_cache["fetched_at"] = now
    return tables


def get_nasdaq100_tickers() -> list[str]:
    """Download Nasdaq-100 ticker list from Wikipedia."""
    logger.info("Fetching Nasdaq-100 ticker list from Wikipedia...")
    try:
        tables = _fetch_nasdaq100_page()
        for tbl in tables:
            for col in ("Ticker", "Symbol"):
                if col in tbl.columns:
                    tickers = tbl[col].tolist()
                    tickers = [t.replace(".", "-") for t in tickers]
                    logger.info(f"Found {len(tickers)} Nasdaq-100 tickers")
                    return tickers
        raise ValueError("Could not find ticker column in Nasdaq-100 Wikipedia tables")
    except Exception as e:
        logger.error(f"Failed to fetch Nasdaq-100 tickers: {e}")
        raise


def _get_nasdaq100_changes() -> list[tuple[pd.Timestamp, str | None, str | None]]:
    """Parse the Nasdaq-100 historical changes table from Wikipedia.

    Returns list of (date, added_ticker, removed_ticker) sorted by date ascending.
    """
    try:
        tables = _fetch_nasdaq100_page()
    except Exception as e:
        logger.warning(f"Failed to fetch Nasdaq-100 page for changes: {e}")
        return []

    # Look for a changes table — typically contains "date", "added", "removed"
    changes_df = None
    for tbl in tables:
        raw_cols = [str(c).lower() for c in tbl.columns]
        has_date = any("date" in c for c in raw_cols)
        has_added = any("added" in c for c in raw_cols)
        has_removed = any("removed" in c for c in raw_cols)
        if has_date and (has_added or has_removed):
            changes_df = tbl
            break

    if changes_df is None:
        logger.warning("Could not find Nasdaq-100 changes table on Wikipedia page")
        return []

    raw_cols = list(changes_df.columns)

    # Find date column
    date_col = None
    for c in raw_cols:
        if "date" in str(c).lower():
            date_col = c
            break
    if date_col is None:
        date_col = raw_cols[0]

    # Find added/removed ticker columns
    added_col = None
    removed_col = None
    for c in raw_cols:
        cl = str(c).lower()
        if "added" in cl and ("ticker" in cl or "symbol" in cl):
            added_col = c
        elif "removed" in cl and ("ticker" in cl or "symbol" in cl):
            removed_col = c

    # Fallback positional heuristic
    if added_col is None and len(raw_cols) >= 3:
        added_col = raw_cols[1]
    if removed_col is None and len(raw_cols) >= 5:
        removed_col = raw_cols[3]

    if added_col is None and removed_col is None:
        logger.warning("Could not identify added/removed columns in Nasdaq-100 changes table")
        return []

    changes: list[tuple[pd.Timestamp, str | None, str | None]] = []
    for _, row in changes_df.iterrows():
        try:
            date = pd.to_datetime(row[date_col])
        except Exception:
            continue
        if pd.isna(date):
            continue

        added = str(row[added_col]).strip() if added_col and pd.notna(row[added_col]) else None
        removed = str(row[removed_col]).strip() if removed_col and pd.notna(row[removed_col]) else None

        if added and added.lower() in ("nan", ""):
            added = None
        if removed and removed.lower() in ("nan", ""):
            removed = None
        if added:
            added = added.replace(".", "-")
        if removed:
            removed = removed.replace(".", "-")

        if added or removed:
            changes.append((date, added, removed))

    changes.sort(key=lambda x: x[0])
    if len(changes) < 10:
        logger.warning(f"Nasdaq-100 changes table has only {len(changes)} records — may be incomplete")
    logger.info(f"Parsed {len(changes)} Nasdaq-100 historical changes")
    return changes


def get_nasdaq100_tickers_as_of(as_of_date: str) -> list[str]:
    """Reconstruct Nasdaq-100 membership as of a historical date."""
    try:
        current = set(get_nasdaq100_tickers())
        changes = _get_nasdaq100_changes()
    except Exception as e:
        logger.error(f"Failed to reconstruct historical Nasdaq-100: {e}")
        raise

    if not changes:
        logger.warning("No historical Nasdaq-100 changes available; returning current members")
        return sorted(current)

    cutoff = pd.Timestamp(as_of_date)

    earliest_change = changes[0][0]
    if cutoff < earliest_change:
        years_before = (earliest_change - cutoff).days / 365.25
        logger.warning(
            f"as_of_date {as_of_date} is {years_before:.1f} years before "
            f"earliest Nasdaq-100 change record ({earliest_change.date()}). "
            f"Reconstruction is incomplete — survivorship bias likely."
        )

    future_changes = [(d, a, r) for d, a, r in changes if d > cutoff]
    members = set(current)

    for _, added, removed in reversed(future_changes):
        if added and added in members:
            members.discard(added)
        if removed:
            members.add(removed)

    logger.info(
        f"Reconstructed Nasdaq-100 as of {as_of_date}: {len(members)} members "
        f"(current: {len(current)}, undid {len(future_changes)} changes)"
    )
    return sorted(members)


def get_dow30_tickers() -> list[str]:
    """Download Dow 30 ticker list from Wikipedia."""
    logger.info("Fetching Dow 30 ticker list from Wikipedia...")
    try:
        tables = _fetch_dow30_page()
        for tbl in tables:
            for col in ("Symbol", "Ticker"):
                if col in tbl.columns:
                    tickers = tbl[col].tolist()
                    tickers = [t.replace(".", "-") for t in tickers]
                    logger.info(f"Found {len(tickers)} Dow 30 tickers")
                    return tickers
        raise ValueError("Could not find ticker column in Dow 30 Wikipedia tables")
    except Exception as e:
        logger.error(f"Failed to fetch Dow 30 tickers: {e}")
        raise


def _get_dow30_changes() -> list[tuple[pd.Timestamp, str | None, str | None]]:
    """Parse the Dow 30 historical changes table from Wikipedia.

    Returns list of (date, added_ticker, removed_ticker) sorted by date ascending.
    """
    try:
        tables = _fetch_dow30_page()
    except Exception as e:
        logger.warning(f"Failed to fetch Dow 30 page for changes: {e}")
        return []

    # Look for a changes table with date + added/removed columns
    changes_df = None
    for tbl in tables:
        raw_cols = [str(c).lower() for c in tbl.columns]
        has_date = any("date" in c for c in raw_cols)
        has_added = any("added" in c for c in raw_cols)
        has_removed = any("removed" in c for c in raw_cols)
        if has_date and (has_added or has_removed):
            changes_df = tbl
            break

    if changes_df is None:
        logger.warning("Could not find Dow 30 changes table on Wikipedia page")
        return []

    raw_cols = list(changes_df.columns)

    # Guard against very large tables with unclear structure
    if len(changes_df) > 200:
        date_cols = [c for c in raw_cols if "date" in str(c).lower()]
        if not date_cols:
            logger.warning(
                f"Dow 30 changes table has {len(changes_df)} rows but no clear date column; "
                "falling back to current members"
            )
            return []

    # Find date column
    date_col = None
    for c in raw_cols:
        if "date" in str(c).lower():
            date_col = c
            break
    if date_col is None:
        date_col = raw_cols[0]

    # Find added/removed ticker columns
    added_col = None
    removed_col = None
    for c in raw_cols:
        cl = str(c).lower()
        if "added" in cl and ("ticker" in cl or "symbol" in cl):
            added_col = c
        elif "removed" in cl and ("ticker" in cl or "symbol" in cl):
            removed_col = c

    # Fallback positional heuristic
    if added_col is None and len(raw_cols) >= 3:
        added_col = raw_cols[1]
    if removed_col is None and len(raw_cols) >= 5:
        removed_col = raw_cols[3]

    if added_col is None and removed_col is None:
        logger.warning("Could not identify added/removed columns in Dow 30 changes table")
        return []

    changes: list[tuple[pd.Timestamp, str | None, str | None]] = []
    for _, row in changes_df.iterrows():
        try:
            date = pd.to_datetime(row[date_col])
        except Exception:
            continue
        if pd.isna(date):
            continue

        added = str(row[added_col]).strip() if added_col and pd.notna(row[added_col]) else None
        removed = str(row[removed_col]).strip() if removed_col and pd.notna(row[removed_col]) else None

        if added and added.lower() in ("nan", ""):
            added = None
        if removed and removed.lower() in ("nan", ""):
            removed = None
        if added:
            added = added.replace(".", "-")
        if removed:
            removed = removed.replace(".", "-")

        if added or removed:
            changes.append((date, added, removed))

    changes.sort(key=lambda x: x[0])
    logger.info(f"Parsed {len(changes)} Dow 30 historical changes")
    return changes


def get_dow30_tickers_as_of(as_of_date: str) -> list[str]:
    """Reconstruct Dow 30 membership as of a historical date."""
    try:
        current = set(get_dow30_tickers())
        changes = _get_dow30_changes()
    except Exception as e:
        logger.error(f"Failed to reconstruct historical Dow 30: {e}")
        raise

    if not changes:
        logger.warning("No historical Dow 30 changes available; returning current members")
        return sorted(current)

    cutoff = pd.Timestamp(as_of_date)

    earliest_change = changes[0][0]
    if cutoff < earliest_change:
        years_before = (earliest_change - cutoff).days / 365.25
        logger.warning(
            f"as_of_date {as_of_date} is {years_before:.1f} years before "
            f"earliest Dow 30 change record ({earliest_change.date()}). "
            f"Reconstruction is incomplete — survivorship bias likely."
        )

    future_changes = [(d, a, r) for d, a, r in changes if d > cutoff]
    members = set(current)

    for _, added, removed in reversed(future_changes):
        if added and added in members:
            members.discard(added)
        if removed:
            members.add(removed)

    logger.info(
        f"Reconstructed Dow 30 as of {as_of_date}: {len(members)} members "
        f"(current: {len(current)}, undid {len(future_changes)} changes)"
    )
    return sorted(members)


def get_universe_tickers(
    preset: str,
    custom_tickers: list[str] | None = None,
    as_of_date: str | None = None,
) -> list[str]:
    """Dispatch universe preset to ticker list.

    For S&P 500, if as_of_date is provided, reconstructs historical membership.
    """
    preset_lower = preset.lower().replace(" ", "_")
    if preset_lower == "sp500":
        if as_of_date:
            return get_sp500_tickers_as_of(as_of_date)
        return get_sp500_tickers()
    elif preset_lower == "nasdaq100":
        if as_of_date:
            return get_nasdaq100_tickers_as_of(as_of_date)
        return get_nasdaq100_tickers()
    elif preset_lower == "dow30":
        if as_of_date:
            return get_dow30_tickers_as_of(as_of_date)
        return get_dow30_tickers()
    elif preset_lower == "sector_etfs":
        return list(SECTOR_ETFS)
    elif preset_lower == "custom":
        if not custom_tickers:
            raise ValueError("Custom universe requires a list of tickers.")
        return [t.upper().strip() for t in custom_tickers if t.strip()]
    else:
        raise ValueError(f"Unknown universe preset: {preset}")


# Cache sector lookups for the session
_sector_cache: dict[str, str] = {}


def get_sector_map(tickers: list[str]) -> dict[str, str]:
    """Fetch sector for each ticker via yfinance, with caching.

    Reuses data from the fundamentals cache (data.fundamentals._fundamentals_cache)
    when available, avoiding redundant HTTP calls.
    """
    _fundamentals_cache: dict = {}  # fundamentals cache not available in this context

    result = {}
    to_fetch = []

    for t in tickers:
        if t in _sector_cache:
            continue
        # Try fundamentals cache first (populated by fetch_fundamentals)
        if t in _fundamentals_cache and _fundamentals_cache[t]:
            _sector_cache[t] = _fundamentals_cache[t].get("sector", "Unknown")
        else:
            to_fetch.append(t)

    # Only make HTTP calls for tickers not in any cache
    for t in to_fetch:
        try:
            info = yf.Ticker(t).info
            sector = info.get("sector", "Unknown")
            _sector_cache[t] = sector
        except Exception:
            _sector_cache[t] = "Unknown"

    for t in tickers:
        result[t] = _sector_cache.get(t, "Unknown")
    return result

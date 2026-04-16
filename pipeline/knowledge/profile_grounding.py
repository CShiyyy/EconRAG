"""Profile grounding: fetch live fact packs from yfinance before LLM synthesis.

All I/O side-effects (yfinance calls) are isolated here so the seeder can be
tested without network access and the grounding can be tested without the LLM.
No LLM calls. No DB writes.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

import yfinance as yf

logger = logging.getLogger(__name__)

_MACRO_KEYWORDS = frozenset({
    "fed", "federal reserve", "fomc", "inflation", "interest rate",
    "rate hike", "rate cut", "cpi", "pce", "gdp", "recession",
})
_MAX_HEADLINES = 10
_BUSINESS_SUMMARY_MAX_CHARS = 4000
_MAX_PEER_TICKERS = 5
_MACRO_NEWS_TICKERS = ("SPY", "QQQ", "^TNX")

# Sector ETFs for 1-month return snapshot in MacroFactPack.
_SECTOR_ETFS = {
    "Technology": "XLK",
    "Financials": "XLF",
    "Energy": "XLE",
    "Healthcare": "XLV",
    "Industrials": "XLI",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Utilities": "XLU",
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TickerFactPack:
    """Structured facts for one watchlist ticker, fetched at grounding time."""
    ticker: str
    company_name: str
    sector: str
    industry: str | None = None
    business_summary: str | None = None
    ceo_name: str | None = None
    employees: int | None = None
    market_cap: float | None = None
    trailing_pe: float | None = None
    week_52_high: float | None = None
    week_52_low: float | None = None
    current_price: float | None = None
    revenue: float | None = None
    net_income: float | None = None
    free_cash_flow: float | None = None
    dividend_yield: float | None = None
    beta: float | None = None
    debt_to_equity: float | None = None
    peer_tickers: list[str] = field(default_factory=list)
    recent_headlines: list[str] = field(default_factory=list)
    fetched_at: str = ""
    grounding_available: bool = True


@dataclass
class MacroFactPack:
    """Structured market-level facts fetched at grounding time."""
    treasury_10y: float | None = None
    treasury_3m: float | None = None
    vix: float | None = None
    dxy: float | None = None
    spy_level: float | None = None
    qqq_level: float | None = None
    gold_price: float | None = None
    oil_price: float | None = None
    sector_performance: dict[str, float] = field(default_factory=dict)
    macro_headlines: list[str] = field(default_factory=list)
    fetched_at: str = ""
    grounding_available: bool = True


# ---------------------------------------------------------------------------
# Sync helpers (run via asyncio.to_thread)
# ---------------------------------------------------------------------------

def _extract_ceo(officers: list[dict]) -> str | None:
    """Return the name of the first officer whose title contains 'CEO' or 'Chief Executive'."""
    for officer in officers or []:
        title = (officer.get("title") or "").lower()
        if "chief executive" in title or " ceo" in title or title == "ceo":
            return officer.get("name")
    return None


def _fetch_info_sync(ticker: str) -> dict | None:
    """Fetch yfinance Ticker.info. Returns None on any failure."""
    try:
        info = yf.Ticker(ticker).info
        return info if info else None
    except Exception as exc:
        logger.warning("yfinance info failed for %s: %s", ticker, exc)
        return None


def _fetch_news_headlines_sync(ticker: str) -> list[str]:
    """Return up to _MAX_HEADLINES headline strings from yfinance ticker.news."""
    try:
        news = yf.Ticker(ticker).news or []
        return [
            a["title"] for a in news[:_MAX_HEADLINES]
            if a.get("title")
        ]
    except Exception as exc:
        logger.warning("yfinance news failed for %s: %s", ticker, exc)
        return []


def _fetch_macro_price_sync(yf_ticker: str) -> float | None:
    """Fetch current price for a macro yfinance symbol."""
    try:
        info = yf.Ticker(yf_ticker).info
        return info.get("regularMarketPrice") or info.get("currentPrice")
    except Exception as exc:
        logger.warning("yfinance macro price failed for %s: %s", yf_ticker, exc)
        return None


def _fetch_sector_return_sync(etf: str) -> float | None:
    """Fetch approximate 1-month return for a sector ETF."""
    try:
        hist = yf.Ticker(etf).history(period="1mo")
        if hist.empty or len(hist) < 2:
            return None
        start = hist["Close"].iloc[0]
        end = hist["Close"].iloc[-1]
        if start and start > 0:
            return round((end - start) / start * 100, 2)
    except Exception as exc:
        logger.warning("yfinance sector return failed for %s: %s", etf, exc)
    return None


def _fetch_macro_headlines_sync() -> list[str]:
    """Return macro-relevant headlines by scanning news for Fed/inflation keywords."""
    headlines: list[str] = []
    try:
        for sym in _MACRO_NEWS_TICKERS:
            for article in yf.Ticker(sym).news or []:
                title = article.get("title", "")
                if any(kw in title.lower() for kw in _MACRO_KEYWORDS):
                    headlines.append(title)
                    if len(headlines) >= _MAX_HEADLINES:
                        return headlines
    except Exception as exc:
        logger.warning("yfinance macro news scan failed: %s", exc)
    return headlines[:_MAX_HEADLINES]


# ---------------------------------------------------------------------------
# Watchlist peer lookup (sync SQL, cheap)
# ---------------------------------------------------------------------------

def _get_sector_peers(conn, ticker: str, sector: str) -> list[str]:
    """Return up to _MAX_PEER_TICKERS watchlist tickers in the same sector, excluding self."""
    rows = conn.execute(
        "SELECT ticker FROM watchlist WHERE sector = ? AND ticker != ? "
        "ORDER BY ticker LIMIT ?",
        (sector, ticker, _MAX_PEER_TICKERS),
    ).fetchall()
    return [r[0] for r in rows]


# ---------------------------------------------------------------------------
# Public async API
# ---------------------------------------------------------------------------

async def fetch_ticker_fact_pack(
    conn,
    ticker: str,
    company_name: str,
    sector: str,
) -> TickerFactPack:
    """Fetch a TickerFactPack for one watchlist ticker.

    Peer tickers and news are always attempted; yfinance .info failure degrades
    the pack to grounding_available=False but does not raise.
    """
    fetched_at = datetime.now(timezone.utc).isoformat()
    peer_tickers = _get_sector_peers(conn, ticker, sector)

    # Run info + news concurrently
    info, headlines = await asyncio.gather(
        asyncio.to_thread(_fetch_info_sync, ticker),
        asyncio.to_thread(_fetch_news_headlines_sync, ticker),
    )

    if info is None:
        logger.warning(
            "Profile grounding unavailable for %s — falling back to minimal fact pack", ticker
        )
        return TickerFactPack(
            ticker=ticker,
            company_name=company_name,
            sector=sector,
            peer_tickers=peer_tickers,
            recent_headlines=headlines,
            fetched_at=fetched_at,
            grounding_available=False,
        )

    summary = info.get("longBusinessSummary")
    if summary and len(summary) > _BUSINESS_SUMMARY_MAX_CHARS:
        summary = summary[:_BUSINESS_SUMMARY_MAX_CHARS] + "\u2026"

    return TickerFactPack(
        ticker=ticker,
        company_name=company_name,
        sector=sector,
        industry=info.get("industry"),
        business_summary=summary,
        ceo_name=_extract_ceo(info.get("companyOfficers") or []),
        employees=info.get("fullTimeEmployees"),
        market_cap=info.get("marketCap"),
        trailing_pe=info.get("trailingPE"),
        week_52_high=info.get("fiftyTwoWeekHigh"),
        week_52_low=info.get("fiftyTwoWeekLow"),
        current_price=info.get("currentPrice") or info.get("regularMarketPrice"),
        revenue=info.get("totalRevenue"),
        net_income=info.get("netIncomeToCommon"),
        free_cash_flow=info.get("freeCashflow"),
        dividend_yield=info.get("dividendYield"),
        beta=info.get("beta"),
        debt_to_equity=info.get("debtToEquity"),
        peer_tickers=peer_tickers,
        recent_headlines=headlines,
        fetched_at=fetched_at,
        grounding_available=True,
    )


async def fetch_macro_fact_pack() -> MacroFactPack:
    """Fetch a MacroFactPack from yfinance macro tickers and news.

    grounding_available is False only if all price fetches fail.
    """
    fetched_at = datetime.now(timezone.utc).isoformat()

    sector_etf_tasks = [
        asyncio.to_thread(_fetch_sector_return_sync, etf)
        for etf in _SECTOR_ETFS.values()
    ]

    results = await asyncio.gather(
        asyncio.to_thread(_fetch_macro_price_sync, "^TNX"),
        asyncio.to_thread(_fetch_macro_price_sync, "^IRX"),
        asyncio.to_thread(_fetch_macro_price_sync, "^VIX"),
        asyncio.to_thread(_fetch_macro_price_sync, "DX-Y.NYB"),
        asyncio.to_thread(_fetch_macro_price_sync, "SPY"),
        asyncio.to_thread(_fetch_macro_price_sync, "QQQ"),
        asyncio.to_thread(_fetch_macro_price_sync, "GLD"),
        asyncio.to_thread(_fetch_macro_price_sync, "USO"),
        asyncio.to_thread(_fetch_macro_headlines_sync),
        *sector_etf_tasks,
    )

    treasury_10y, treasury_3m, vix, dxy, spy_level, qqq_level, gold_price, oil_price, macro_headlines = results[:9]
    sector_returns_raw = results[9:]

    sector_performance: dict[str, float] = {}
    for name, ret in zip(_SECTOR_ETFS.keys(), sector_returns_raw):
        if ret is not None:
            sector_performance[name] = ret

    grounding_available = any(
        v is not None for v in (treasury_10y, vix, spy_level)
    )

    return MacroFactPack(
        treasury_10y=treasury_10y,
        treasury_3m=treasury_3m,
        vix=vix,
        dxy=dxy,
        spy_level=spy_level,
        qqq_level=qqq_level,
        gold_price=gold_price,
        oil_price=oil_price,
        sector_performance=sector_performance,
        macro_headlines=macro_headlines,
        fetched_at=fetched_at,
        grounding_available=grounding_available,
    )

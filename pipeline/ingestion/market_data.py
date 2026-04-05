"""yfinance market data scraper."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import yfinance as yf

from pipeline.ingestion.health import timed_health
from pipeline.ingestion.models import MarketDataPoint, SourceHealth


def _fetch_single_ticker(ticker_str: str) -> MarketDataPoint | None:
    """Sync helper: fetch data for one ticker via yfinance."""
    try:
        ticker = yf.Ticker(ticker_str)
        info = ticker.info
        if not info or (not info.get("currentPrice") and not info.get("regularMarketPrice")):
            return None

        hist = ticker.history(period="1mo")
        price_history = hist["Close"].tolist() if not hist.empty else []

        return MarketDataPoint(
            ticker=ticker_str,
            current_price=info.get("currentPrice") or info.get("regularMarketPrice", 0.0),
            open_price=info.get("open") or info.get("regularMarketOpen", 0.0),
            previous_close=info.get("previousClose") or info.get("regularMarketPreviousClose", 0.0),
            pe_ratio=info.get("trailingPE"),
            market_cap=info.get("marketCap"),
            price_history_30d=price_history,
            fetched_at=datetime.now(timezone.utc).isoformat(),
        )
    except Exception:
        return None


async def fetch_market_data(
    tickers: list[str],
) -> tuple[dict[str, MarketDataPoint | None], SourceHealth]:
    """Fetch market data for all tickers via yfinance.

    Returns:
        Tuple of (data dict keyed by ticker, SourceHealth).
        Tickers that fail return None in the dict.
    """
    async with timed_health("yfinance") as ctx:
        results: dict[str, MarketDataPoint | None] = {}
        for ticker_str in tickers:
            mdp = await asyncio.to_thread(_fetch_single_ticker, ticker_str)
            results[ticker_str] = mdp
        ctx.items_fetched = sum(1 for v in results.values() if v is not None)

    return results, ctx.health

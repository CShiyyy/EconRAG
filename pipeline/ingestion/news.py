"""News scraper: yfinance news + NewsData.io."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone, timedelta

import httpx
import yfinance as yf

from pipeline.ingestion.health import timed_health
from pipeline.ingestion.models import NewsHit, SourceHealth
from pipeline.config import NEWS_WINDOW_HOURS


def _fetch_yfinance_news(tickers: list[str], cutoff: datetime) -> list[NewsHit]:
    """Sync: fetch news from yfinance for all tickers."""
    hits: list[NewsHit] = []
    for ticker_str in tickers:
        try:
            ticker = yf.Ticker(ticker_str)
            for article in (ticker.news or []):
                pub_ts = article.get("providerPublishTime", 0)
                pub_dt = datetime.fromtimestamp(pub_ts, tz=timezone.utc)
                if pub_dt < cutoff:
                    continue
                hits.append(NewsHit(
                    ticker=ticker_str,
                    headline=article.get("title", ""),
                    url=article.get("link", ""),
                    source="yfinance",
                    published_at=pub_dt.isoformat(),
                ))
        except Exception:
            continue
    return hits


async def _fetch_newsdata(
    tickers: list[str], api_key: str, cutoff: datetime,
) -> tuple[list[NewsHit], SourceHealth]:
    """Async: fetch news from NewsData.io."""
    hits: list[NewsHit] = []

    async with timed_health("newsdata") as ctx:
        async with httpx.AsyncClient(timeout=30.0) as client:
            query = " OR ".join(tickers)
            resp = await client.get(
                "https://newsdata.io/api/1/latest",
                params={"apikey": api_key, "q": query, "language": "en"},
            )

            if resp.status_code == 429:
                raise Exception("429 rate limit exceeded")
            if resp.status_code == 401:
                raise Exception("401 invalid API key")
            resp.raise_for_status()

            data = resp.json()
            for article in data.get("results", []):
                pub_str = article.get("pubDate", "")
                try:
                    pub_dt = datetime.fromisoformat(pub_str.replace(" ", "T"))
                    if pub_dt.tzinfo is None:
                        pub_dt = pub_dt.replace(tzinfo=timezone.utc)
                except (ValueError, AttributeError):
                    pub_dt = datetime.now(timezone.utc)

                if pub_dt < cutoff:
                    continue

                url = article.get("link", "")
                title = (article.get("title") or "").upper()
                for ticker_str in tickers:
                    if ticker_str.upper() in title or ticker_str.upper() in (article.get("description") or "").upper():
                        hits.append(NewsHit(
                            ticker=ticker_str,
                            headline=article.get("title", ""),
                            url=url,
                            source="newsdata",
                            published_at=pub_dt.isoformat(),
                        ))
                        break
                else:
                    if tickers:
                        hits.append(NewsHit(
                            ticker=tickers[0],
                            headline=article.get("title", ""),
                            url=url,
                            source="newsdata",
                            published_at=pub_dt.isoformat(),
                        ))

            ctx.items_fetched = len(hits)

    return hits, ctx.health


async def fetch_news(
    tickers: list[str], api_key: str | None,
) -> tuple[list[NewsHit], list[SourceHealth]]:
    """Fetch news from yfinance and optionally NewsData.io.

    Returns:
        Tuple of (deduplicated NewsHit list, list of SourceHealth per sub-source).
    """
    cutoff = datetime.now(timezone.utc) - timedelta(hours=NEWS_WINDOW_HOURS)
    healths: list[SourceHealth] = []

    # yfinance news (always runs)
    async with timed_health("yfinance_news") as yf_ctx:
        yf_hits = await asyncio.to_thread(_fetch_yfinance_news, tickers, cutoff)
        yf_ctx.items_fetched = len(yf_hits)
    healths.append(yf_ctx.health)

    all_hits = list(yf_hits)

    # NewsData.io (only if key provided)
    if api_key:
        nd_hits, nd_health = await _fetch_newsdata(tickers, api_key, cutoff)
        healths.append(nd_health)
        all_hits.extend(nd_hits)

    # Deduplicate by URL (keep first seen)
    seen_urls: set[str] = set()
    deduped: list[NewsHit] = []
    for hit in all_hits:
        if hit.url and hit.url not in seen_urls:
            seen_urls.add(hit.url)
            deduped.append(hit)

    return deduped, healths

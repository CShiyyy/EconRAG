"""Ingestion orchestrator — runs all scrapers, applies volume caps and dedup."""

from __future__ import annotations

import asyncio

from pipeline.config import (
    DEFAULT_SUBREDDITS,
    MAX_NEWS_PER_TICKER,
    MAX_SOCIAL_PER_TICKER,
    NEWSDATA_API_KEY,
    REDDIT_CLIENT_ID,
    REDDIT_CLIENT_SECRET,
    REDDIT_USER_AGENT,
)
from pipeline.ingestion.market_data import fetch_market_data
from pipeline.ingestion.models import (
    IngestionResult,
    NewsHit,
    SocialHit,
    SourceHealth,
)
from pipeline.ingestion.news import fetch_news
from pipeline.ingestion.parser import parse_urls
from pipeline.ingestion.social import fetch_social


def _apply_volume_caps(
    news_hits: list[NewsHit],
    social_hits: list[SocialHit],
    max_news: int = MAX_NEWS_PER_TICKER,
    max_social: int = MAX_SOCIAL_PER_TICKER,
) -> dict:
    """Apply per-ticker volume caps and collect unique URLs.

    Returns:
        {"news": capped news, "social": capped social, "unique_urls": deduped URL list}
    """
    # Cap news per ticker — sort by recency (most recent first)
    news_by_ticker: dict[str, list[NewsHit]] = {}
    for hit in news_hits:
        news_by_ticker.setdefault(hit.ticker, []).append(hit)

    capped_news: list[NewsHit] = []
    for ticker, hits in news_by_ticker.items():
        sorted_hits = sorted(hits, key=lambda h: h.published_at, reverse=True)
        capped_news.extend(sorted_hits[:max_news])

    # Cap social per ticker — sort by score desc, then recency
    social_by_ticker: dict[str, list[SocialHit]] = {}
    for hit in social_hits:
        social_by_ticker.setdefault(hit.ticker, []).append(hit)

    capped_social: list[SocialHit] = []
    for ticker, hits in social_by_ticker.items():
        sorted_hits = sorted(hits, key=lambda h: (-h.score, h.posted_at), reverse=False)
        capped_social.extend(sorted_hits[:max_social])

    # Collect unique URLs for Crawl4AI
    seen: set[str] = set()
    unique_urls: list[str] = []
    for hit in capped_news:
        if hit.url and hit.url not in seen:
            seen.add(hit.url)
            unique_urls.append(hit.url)
    for hit in capped_social:
        if hit.url and hit.url not in seen:
            seen.add(hit.url)
            unique_urls.append(hit.url)

    return {"news": capped_news, "social": capped_social, "unique_urls": unique_urls}


async def run_ingestion(tickers: list[str]) -> IngestionResult:
    """Run the full ingestion pipeline.

    1. Fetch market data, news, and social concurrently
    2. Apply volume caps per ticker
    3. Deduplicate URLs
    4. Parse unique URLs via Crawl4AI
    5. Return bundled IngestionResult
    """
    market_task = fetch_market_data(tickers)
    news_task = fetch_news(tickers, api_key=NEWSDATA_API_KEY)
    social_task = fetch_social(
        tickers,
        subreddits=DEFAULT_SUBREDDITS,
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT,
    )

    market_result, news_result, social_result = await asyncio.gather(
        market_task, news_task, social_task
    )

    market_data, market_health = market_result
    news_hits, news_healths = news_result
    social_hits, social_health = social_result

    all_health: list[SourceHealth] = [market_health] + news_healths + [social_health]

    capped = _apply_volume_caps(news_hits, social_hits)

    parsed_content, parse_health = await parse_urls(capped["unique_urls"])
    all_health.append(parse_health)

    return IngestionResult(
        market_data=market_data,
        news_hits=capped["news"],
        social_hits=capped["social"],
        parsed_content=parsed_content,
        health=all_health,
    )

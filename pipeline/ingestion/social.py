"""Reddit social scraper via PRAW."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

import praw

from pipeline.ingestion.health import timed_health
from pipeline.ingestion.models import SocialHit, SourceHealth

logger = logging.getLogger(__name__)


def _fetch_reddit_posts(
    tickers: list[str],
    subreddits: list[str],
    client_id: str,
    client_secret: str,
    user_agent: str,
) -> list[SocialHit]:
    """Sync: search Reddit for ticker mentions."""
    reddit = praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
    )

    hits: list[SocialHit] = []
    for sub_name in subreddits:
        try:
            subreddit = reddit.subreddit(sub_name)
            for ticker_str in tickers:
                query = f"${ticker_str} OR {ticker_str}"
                try:
                    for submission in subreddit.search(query, sort="relevance", time_filter="week", limit=25):
                        hits.append(SocialHit(
                            ticker=ticker_str,
                            title=submission.title,
                            body=getattr(submission, "selftext", ""),
                            url=submission.url,
                            score=submission.score,
                            subreddit=sub_name,
                            posted_at=datetime.fromtimestamp(
                                submission.created_utc, tz=timezone.utc
                            ).isoformat(),
                        ))
                except Exception as exc:
                    logger.warning("Reddit search failed for %s in r/%s: %s", ticker_str, sub_name, exc)
                    continue
        except Exception as exc:
            logger.warning("Reddit subreddit access failed for r/%s: %s", sub_name, exc)
            continue

    return hits


async def fetch_social(
    tickers: list[str],
    subreddits: list[str],
    client_id: str | None = None,
    client_secret: str | None = None,
    user_agent: str = "EconRAG/1.0",
) -> tuple[list[SocialHit], SourceHealth]:
    """Fetch social posts from Reddit.

    Returns:
        Tuple of (SocialHit list, SourceHealth).
        If credentials are missing, returns empty list + error health.
    """
    if not client_id or not client_secret:
        logger.warning("Reddit credentials not configured — skipping social fetch (set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET in .env)")
        return [], SourceHealth(
            source="reddit",
            status="error",
            items_fetched=0,
            duration_ms=0,
            error_detail="Reddit credentials not configured",
        )

    async with timed_health("reddit") as ctx:
        hits = await asyncio.to_thread(
            _fetch_reddit_posts, tickers, subreddits, client_id, client_secret, user_agent
        )
        ctx.items_fetched = len(hits)

    return hits, ctx.health

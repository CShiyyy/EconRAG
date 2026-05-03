"""Crawl4AI URL-to-Markdown parser."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from crawl4ai import AsyncWebCrawler

from pipeline.ingestion.health import timed_health
from pipeline.ingestion.models import ParsedContent, SourceHealth
from pipeline.config import CRAWL4AI_CONCURRENCY, CRAWL4AI_TIMEOUT_SECONDS

logger = logging.getLogger(__name__)


async def parse_urls(
    urls: list[str],
) -> tuple[list[ParsedContent], SourceHealth]:
    """Parse a list of URLs into clean Markdown via Crawl4AI.

    Uses a semaphore to limit concurrency. Each URL has an individual timeout.

    Returns:
        Tuple of (list of ParsedContent, SourceHealth).
    """
    if not urls:
        return [], SourceHealth(
            source="crawl4ai",
            status="success",
            items_fetched=0,
            duration_ms=0,
        )

    semaphore = asyncio.Semaphore(CRAWL4AI_CONCURRENCY)
    results: list[ParsedContent] = []

    async with timed_health("crawl4ai") as ctx:
        try:
            async with AsyncWebCrawler() as crawler:

                async def _parse_one(url: str) -> ParsedContent:
                    async with semaphore:
                        try:
                            result = await asyncio.wait_for(
                                crawler.arun(url=url),
                                timeout=CRAWL4AI_TIMEOUT_SECONDS,
                            )
                            if result.success:
                                return ParsedContent(
                                    url=url,
                                    markdown=result.markdown,
                                    fetched_at=datetime.now(timezone.utc).isoformat(),
                                    success=True,
                                    error=None,
                                )
                            else:
                                return ParsedContent(
                                    url=url,
                                    markdown="",
                                    fetched_at=datetime.now(timezone.utc).isoformat(),
                                    success=False,
                                    error=getattr(result, "error_message", "Unknown error"),
                                )
                        except asyncio.TimeoutError:
                            return ParsedContent(
                                url=url,
                                markdown="",
                                fetched_at=datetime.now(timezone.utc).isoformat(),
                                success=False,
                                error=f"Timeout after {CRAWL4AI_TIMEOUT_SECONDS}s",
                            )
                        except Exception as exc:
                            return ParsedContent(
                                url=url,
                                markdown="",
                                fetched_at=datetime.now(timezone.utc).isoformat(),
                                success=False,
                                error=str(exc),
                            )

                results = list(await asyncio.gather(*[_parse_one(url) for url in urls]))
                ctx.items_fetched = sum(1 for r in results if r.success)

        except (NotImplementedError, OSError) as exc:
            logger.warning("Crawl4AI browser launch failed (skipping URL parsing): %s", exc)
            ctx.items_fetched = 0

    return results, ctx.health

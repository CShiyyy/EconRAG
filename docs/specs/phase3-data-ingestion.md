# Phase 3 Design: Data Ingestion Layer

**Date:** 2026-04-06
**Status:** Approved

## Overview

Build the data ingestion layer that fetches market data, news, and social content for the watchlist universe, parses raw URLs into clean Markdown via Crawl4AI, and tracks source health. All scrapers run async with `asyncio`. No single source failure blocks the pipeline.

## Module Structure

```
pipeline/ingestion/
    __init__.py
    models.py          — dataclasses for all ingestion types
    market_data.py     — yfinance scraper
    news.py            — yfinance news + NewsData.io
    social.py          — Reddit/PRAW scraper
    parser.py          — Crawl4AI (async headless Chromium)
    health.py          — source health tracking utilities
    orchestrator.py    — runs all scrapers, volume caps, dedup
```

## Data Models (`models.py`)

```python
@dataclass
class MarketDataPoint:
    ticker: str
    current_price: float
    open_price: float
    previous_close: float
    pe_ratio: float | None
    market_cap: float | None
    price_history_30d: list[float]
    fetched_at: str

@dataclass
class NewsHit:
    ticker: str
    headline: str
    url: str
    source: str            # "yfinance" or "newsdata"
    published_at: str

@dataclass
class SocialHit:
    ticker: str
    title: str
    body: str
    url: str
    score: int
    subreddit: str
    posted_at: str

@dataclass
class ParsedContent:
    url: str
    markdown: str
    fetched_at: str
    success: bool
    error: str | None

@dataclass
class SourceHealth:
    source: str
    status: str            # "success" | "timeout" | "rate_limited" | "error"
    items_fetched: int
    duration_ms: int
    error_detail: str | None = None

@dataclass
class IngestionResult:
    market_data: dict[str, MarketDataPoint]
    news_hits: list[NewsHit]
    social_hits: list[SocialHit]
    parsed_content: list[ParsedContent]
    health: list[SourceHealth]
```

## Config Additions (`pipeline/config.py`)

```python
NEWSDATA_API_KEY = os.getenv("NEWSDATA_API_KEY")
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT", "EconRAG/1.0")
DEFAULT_SUBREDDITS: list[str] = ["wallstreetbets", "stocks", "investing"]
MAX_NEWS_PER_TICKER: int = 20
MAX_SOCIAL_PER_TICKER: int = 30
NEWS_WINDOW_HOURS: int = 48
```

API keys stored in `.env` only, never hardcoded. `.env` must be in `.gitignore`.

## Scrapers

### Market Data (`market_data.py`)

- `async fetch_market_data(tickers: list[str]) -> tuple[dict[str, MarketDataPoint], SourceHealth]`
- Uses `asyncio.to_thread` to wrap yfinance's sync `yf.Tickers()` batch call
- Fetches `info` dict for price/fundamentals, `history(period="1mo")` for 30-day price series
- Invalid/missing tickers return `None` in the dict, logged in health

### News (`news.py`)

- `async fetch_news(tickers: list[str], api_key: str | None) -> tuple[list[NewsHit], list[SourceHealth]]`
- Two sub-sources merged:
  1. **yfinance news** — `ticker.news` property. Always runs (no key needed). Wrapped in `asyncio.to_thread`.
  2. **NewsData.io** — runs only if `api_key` is provided. Uses `httpx.AsyncClient`. Searches by ticker symbol and company name.
- Deduplicate by URL across both sub-sources
- Filter to 48-hour window
- Returns separate `SourceHealth` entries for each sub-source ("yfinance_news", "newsdata")
- Handles: 401 (bad key), 429 (rate limit), timeouts

### Social (`social.py`)

- `async fetch_social(tickers: list[str], subreddits: list[str]) -> tuple[list[SocialHit], SourceHealth]`
- PRAW is sync — wrapped in `asyncio.to_thread`
- Searches each subreddit for cashtag (`$AAPL`) and ticker name
- Missing credentials: returns empty list + `SourceHealth(status="error", error_detail="Reddit credentials not configured")`
- Handles quarantined/private subreddits gracefully

### Parser (`parser.py`)

- `async parse_urls(urls: list[str]) -> tuple[list[ParsedContent], SourceHealth]`
- Crawl4AI `AsyncWebCrawler` with headless Chromium
- Semaphore-limited to 5 concurrent fetches
- Per-URL timeout: 30 seconds
- Failed URLs: `ParsedContent(success=False, error=...)`, does not stop others
- Aggregate health based on success rate

## Orchestrator (`orchestrator.py`)

- `async run_ingestion(tickers: list[str]) -> IngestionResult`
- Runs market data, news, and social scrapers concurrently via `asyncio.gather`
- Applies volume caps per ticker after scrapers return:
  - News: top 20 per ticker. Priority: source tier (major outlets > others), then recency.
  - Social: top 30 per ticker. Priority: score descending, then recency.
- URL-level dedup: unique URL set across all capped hits. Each URL processed by Crawl4AI exactly once.
- Passes deduped URLs to `parse_urls()`
- Maps parsed content back to hits by URL
- Aggregates all `SourceHealth` objects

## Health Tracking (`health.py`)

- Utility functions for `SourceHealth`
- `aggregate_health(health_list: list[SourceHealth]) -> dict` — produces JSON blob for `run_log.source_status`
- `timed_health(source: str)` — async context manager that wraps scraper calls, catches exceptions, measures duration, returns appropriate `SourceHealth`

## Error Handling

- No single scraper failure blocks the pipeline
- Orchestrator catches exceptions per-scraper, records failure in `SourceHealth`
- Crawl4AI failures are per-URL
- Missing credentials produce clean `SourceHealth(status="error")`, not exceptions
- `IngestionResult` always returns; downstream checks `health` for degraded data

## Testing (`tests/test_ingestion.py`)

All external APIs mocked. No live network calls.

- **Market data:** mock yfinance `Ticker`, verify `MarketDataPoint` fields. Test invalid ticker.
- **News:** mock yfinance `.news` + httpx for NewsData.io. Verify 48h filter. Verify cross-source dedup. Verify yfinance-only mode (no API key). Verify 429/timeout health.
- **Social:** mock PRAW. Verify cashtag search. Verify missing credentials handling. Verify subreddit unavailability.
- **Parser:** mock `AsyncWebCrawler`. Verify markdown output. Verify per-URL timeout. Verify semaphore concurrency limit.
- **Orchestrator:** mock all scrapers. Verify volume cap (50 news in, 20 kept). Verify URL dedup (same URL under 2 tickers, Crawl4AI processes once). Verify health aggregation.
- **Integration:** mock external APIs, run full orchestrator, verify `IngestionResult` shape.

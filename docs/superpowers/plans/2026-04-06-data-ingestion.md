# Phase 3: Data Ingestion Layer — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the async data ingestion layer that fetches market data (yfinance), news (yfinance + NewsData.io), social posts (Reddit/PRAW), parses URLs to Markdown (Crawl4AI), and tracks source health — all with volume caps, URL dedup, and graceful degradation.

**Architecture:** All scrapers are async coroutines coordinated by an orchestrator via `asyncio.gather`. Sync libraries (yfinance, PRAW) are wrapped with `asyncio.to_thread`. Crawl4AI runs natively async with a concurrency semaphore. No single source failure blocks the pipeline — each scraper reports `SourceHealth` and the orchestrator always returns an `IngestionResult`.

**Tech Stack:** Python 3.12+, asyncio, yfinance, httpx, praw, crawl4ai, pytest, pytest-asyncio

**Spec:** `docs/superpowers/specs/2026-04-06-data-ingestion-design.md`

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `pipeline/ingestion/__init__.py` | Package init, re-exports |
| Create | `pipeline/ingestion/models.py` | All dataclasses |
| Create | `pipeline/ingestion/health.py` | `SourceHealth` utilities, `timed_health` context manager, `aggregate_health` |
| Create | `pipeline/ingestion/market_data.py` | yfinance scraper |
| Create | `pipeline/ingestion/news.py` | yfinance news + NewsData.io scraper |
| Create | `pipeline/ingestion/social.py` | Reddit/PRAW scraper |
| Create | `pipeline/ingestion/parser.py` | Crawl4AI URL-to-Markdown parser |
| Create | `pipeline/ingestion/orchestrator.py` | Runs scrapers, volume caps, dedup, Crawl4AI dispatch |
| Modify | `pipeline/config.py` | Add API key env vars, ingestion constants |
| Modify | `.env` | Add `NEWSDATA_API_KEY` (never committed) |
| Create | `tests/test_ingestion.py` | All ingestion tests |

---

## Task 1: Config Additions & `.env` Setup

**Files:**
- Modify: `pipeline/config.py`
- Modify: `.env`

- [ ] **Step 1: Add ingestion config to `pipeline/config.py`**

Add after the existing `DEFAULT_CONSTRAINTS` block:

```python
# --- Ingestion config ---
NEWSDATA_API_KEY: str | None = os.getenv("NEWSDATA_API_KEY")
REDDIT_CLIENT_ID: str | None = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET: str | None = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT: str = os.getenv("REDDIT_USER_AGENT", "EconRAG/1.0")

DEFAULT_SUBREDDITS: list[str] = ["wallstreetbets", "stocks", "investing"]
MAX_NEWS_PER_TICKER: int = 20
MAX_SOCIAL_PER_TICKER: int = 30
NEWS_WINDOW_HOURS: int = 48
CRAWL4AI_CONCURRENCY: int = 5
CRAWL4AI_TIMEOUT_SECONDS: int = 30
```

- [ ] **Step 2: Add keys to `.env`**

Append to the existing `.env` file:

```
# Ingestion API keys
NEWSDATA_API_KEY=pub_544b7354e4754edf8dde9673a7a272ec
# Reddit — fill in when available
# REDDIT_CLIENT_ID=
# REDDIT_CLIENT_SECRET=
```

- [ ] **Step 3: Verify `.env` is in `.gitignore`**

Run: `grep -q "\.env" .gitignore && echo "OK" || echo "MISSING"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add pipeline/config.py
git commit -m "feat(ingestion): add ingestion config constants and API key env vars"
```

Do NOT `git add .env` — it must never be committed.

---

## Task 2: Data Models

**Files:**
- Create: `pipeline/ingestion/__init__.py`
- Create: `pipeline/ingestion/models.py`
- Create: `tests/test_ingestion.py` (initial structure)

- [ ] **Step 1: Create package init**

```python
# pipeline/ingestion/__init__.py
```

Empty file — just establishes the package.

- [ ] **Step 2: Write the test for models**

```python
# tests/test_ingestion.py
"""Tests for the Data Ingestion Layer (Phase 3)."""

from pipeline.ingestion.models import (
    IngestionResult,
    MarketDataPoint,
    NewsHit,
    ParsedContent,
    SocialHit,
    SourceHealth,
)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class TestModels:
    def test_market_data_point_fields(self):
        mdp = MarketDataPoint(
            ticker="AAPL",
            current_price=175.50,
            open_price=174.00,
            previous_close=173.80,
            pe_ratio=28.5,
            market_cap=2.8e12,
            price_history_30d=[170.0, 171.0, 172.0],
            fetched_at="2026-04-06T16:00:00Z",
        )
        assert mdp.ticker == "AAPL"
        assert mdp.pe_ratio == 28.5
        assert len(mdp.price_history_30d) == 3

    def test_market_data_point_nullable_fields(self):
        mdp = MarketDataPoint(
            ticker="FAKE",
            current_price=0.0,
            open_price=0.0,
            previous_close=0.0,
            pe_ratio=None,
            market_cap=None,
            price_history_30d=[],
            fetched_at="2026-04-06T16:00:00Z",
        )
        assert mdp.pe_ratio is None
        assert mdp.market_cap is None

    def test_news_hit_fields(self):
        hit = NewsHit(
            ticker="MSFT",
            headline="Microsoft beats earnings",
            url="https://example.com/msft",
            source="yfinance",
            published_at="2026-04-06T10:00:00Z",
        )
        assert hit.source == "yfinance"

    def test_social_hit_fields(self):
        hit = SocialHit(
            ticker="NVDA",
            title="NVDA to the moon",
            body="DD on NVDA earnings...",
            url="https://reddit.com/r/stocks/123",
            score=500,
            subreddit="stocks",
            posted_at="2026-04-06T08:00:00Z",
        )
        assert hit.score == 500

    def test_parsed_content_success(self):
        pc = ParsedContent(
            url="https://example.com/article",
            markdown="# Article\nContent here",
            fetched_at="2026-04-06T16:00:00Z",
            success=True,
            error=None,
        )
        assert pc.success is True

    def test_parsed_content_failure(self):
        pc = ParsedContent(
            url="https://example.com/broken",
            markdown="",
            fetched_at="2026-04-06T16:00:00Z",
            success=False,
            error="Timeout after 30s",
        )
        assert pc.success is False
        assert "Timeout" in pc.error

    def test_source_health_defaults(self):
        sh = SourceHealth(
            source="yfinance",
            status="success",
            items_fetched=30,
            duration_ms=1200,
        )
        assert sh.error_detail is None

    def test_ingestion_result_bundle(self):
        result = IngestionResult(
            market_data={},
            news_hits=[],
            social_hits=[],
            parsed_content=[],
            health=[],
        )
        assert isinstance(result.market_data, dict)
        assert isinstance(result.health, list)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_ingestion.py::TestModels -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pipeline.ingestion.models'`

- [ ] **Step 4: Write `models.py`**

```python
# pipeline/ingestion/models.py
"""Data models for the ingestion layer."""

from __future__ import annotations

from dataclasses import dataclass, field


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
    source: str  # "yfinance" or "newsdata"
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
    status: str  # "success" | "timeout" | "rate_limited" | "error"
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

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_ingestion.py::TestModels -v`
Expected: All 8 tests PASS

- [ ] **Step 6: Commit**

```bash
git add pipeline/ingestion/__init__.py pipeline/ingestion/models.py tests/test_ingestion.py
git commit -m "feat(ingestion): add data models for ingestion layer"
```

---

## Task 3: Health Tracking Utilities

**Files:**
- Create: `pipeline/ingestion/health.py`
- Modify: `tests/test_ingestion.py`

- [ ] **Step 1: Write the tests**

Append to `tests/test_ingestion.py`:

```python
import asyncio
import time

from pipeline.ingestion.health import aggregate_health, timed_health
from pipeline.ingestion.models import SourceHealth


# ---------------------------------------------------------------------------
# Health utilities
# ---------------------------------------------------------------------------

class TestHealth:
    def test_aggregate_health_success(self):
        health_list = [
            SourceHealth("yfinance", "success", 30, 1200),
            SourceHealth("newsdata", "success", 50, 800),
        ]
        result = aggregate_health(health_list)
        assert result["yfinance"]["status"] == "success"
        assert result["newsdata"]["items_fetched"] == 50

    def test_aggregate_health_mixed(self):
        health_list = [
            SourceHealth("yfinance", "success", 30, 1200),
            SourceHealth("reddit", "error", 0, 0, "Reddit credentials not configured"),
        ]
        result = aggregate_health(health_list)
        assert result["reddit"]["status"] == "error"
        assert "credentials" in result["reddit"]["error_detail"]

    def test_timed_health_success(self):
        async def _run():
            async with timed_health("test_source") as ctx:
                ctx.items_fetched = 10
            return ctx.health

        health = asyncio.run(_run())
        assert health.source == "test_source"
        assert health.status == "success"
        assert health.items_fetched == 10
        assert health.duration_ms >= 0

    def test_timed_health_exception(self):
        async def _run():
            async with timed_health("failing_source") as ctx:
                raise TimeoutError("connection timed out")
            return ctx.health  # noqa: unreachable for clarity

        health = asyncio.run(_run())
        assert health.status == "timeout"
        assert health.items_fetched == 0

    def test_timed_health_generic_exception(self):
        async def _run():
            async with timed_health("broken_source") as ctx:
                raise RuntimeError("something broke")
            return ctx.health

        health = asyncio.run(_run())
        assert health.status == "error"
        assert "something broke" in health.error_detail
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_ingestion.py::TestHealth -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pipeline.ingestion.health'`

- [ ] **Step 3: Write `health.py`**

```python
# pipeline/ingestion/health.py
"""Source health tracking utilities."""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import AsyncIterator

from pipeline.ingestion.models import SourceHealth


def aggregate_health(health_list: list[SourceHealth]) -> dict:
    """Convert a list of SourceHealth into a JSON-serializable dict keyed by source name.

    This dict is stored in run_log.source_status.
    """
    result: dict[str, dict] = {}
    for h in health_list:
        result[h.source] = {
            "status": h.status,
            "items_fetched": h.items_fetched,
            "duration_ms": h.duration_ms,
        }
        if h.error_detail:
            result[h.source]["error_detail"] = h.error_detail
    return result


@dataclass
class _HealthContext:
    """Mutable context passed into the timed_health block."""

    source: str
    items_fetched: int = 0
    health: SourceHealth | None = None


@asynccontextmanager
async def timed_health(source: str) -> AsyncIterator[_HealthContext]:
    """Async context manager that wraps a scraper call.

    Measures duration, catches exceptions, and produces a SourceHealth.

    Usage:
        async with timed_health("yfinance") as ctx:
            data = await fetch(...)
            ctx.items_fetched = len(data)
        health = ctx.health  # SourceHealth object
    """
    ctx = _HealthContext(source=source)
    start = time.monotonic()
    try:
        yield ctx
        elapsed_ms = int((time.monotonic() - start) * 1000)
        ctx.health = SourceHealth(
            source=source,
            status="success",
            items_fetched=ctx.items_fetched,
            duration_ms=elapsed_ms,
        )
    except TimeoutError as exc:
        elapsed_ms = int((time.monotonic() - start) * 1000)
        ctx.health = SourceHealth(
            source=source,
            status="timeout",
            items_fetched=0,
            duration_ms=elapsed_ms,
            error_detail=str(exc),
        )
    except Exception as exc:
        elapsed_ms = int((time.monotonic() - start) * 1000)
        status = "rate_limited" if "429" in str(exc) or "rate" in str(exc).lower() else "error"
        ctx.health = SourceHealth(
            source=source,
            status=status,
            items_fetched=0,
            duration_ms=elapsed_ms,
            error_detail=str(exc),
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_ingestion.py::TestHealth -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add pipeline/ingestion/health.py tests/test_ingestion.py
git commit -m "feat(ingestion): add source health tracking utilities"
```

---

## Task 4: Market Data Scraper

**Files:**
- Create: `pipeline/ingestion/market_data.py`
- Modify: `tests/test_ingestion.py`

- [ ] **Step 1: Write the tests**

Append to `tests/test_ingestion.py`:

```python
from unittest.mock import MagicMock, patch
import pandas as pd

from pipeline.ingestion.market_data import fetch_market_data


# ---------------------------------------------------------------------------
# Market Data
# ---------------------------------------------------------------------------

class TestMarketData:
    def test_fetch_valid_tickers(self):
        """Mock yfinance to return structured data for known tickers."""
        mock_ticker = MagicMock()
        mock_ticker.info = {
            "currentPrice": 175.50,
            "open": 174.00,
            "previousClose": 173.80,
            "trailingPE": 28.5,
            "marketCap": 2_800_000_000_000,
        }
        mock_ticker.history.return_value = pd.DataFrame(
            {"Close": [170.0, 171.0, 172.0, 173.0, 174.0]},
            index=pd.date_range("2026-03-01", periods=5),
        )

        with patch("pipeline.ingestion.market_data.yf.Ticker", return_value=mock_ticker):
            data, health = asyncio.run(fetch_market_data(["AAPL"]))

        assert "AAPL" in data
        assert data["AAPL"].current_price == 175.50
        assert data["AAPL"].pe_ratio == 28.5
        assert len(data["AAPL"].price_history_30d) == 5
        assert health.status == "success"
        assert health.items_fetched == 1

    def test_fetch_invalid_ticker(self):
        """Invalid ticker returns None in dict, health still success."""
        mock_ticker = MagicMock()
        mock_ticker.info = {}
        mock_ticker.history.return_value = pd.DataFrame()

        with patch("pipeline.ingestion.market_data.yf.Ticker", return_value=mock_ticker):
            data, health = asyncio.run(fetch_market_data(["FAKETICKER"]))

        assert data.get("FAKETICKER") is None
        assert health.status == "success"

    def test_fetch_exception_produces_error_health(self):
        """If yfinance throws, health reports error."""
        with patch("pipeline.ingestion.market_data.yf.Ticker", side_effect=Exception("network down")):
            data, health = asyncio.run(fetch_market_data(["AAPL"]))

        assert data == {}
        assert health.status == "error"
        assert "network down" in health.error_detail
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_ingestion.py::TestMarketData -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write `market_data.py`**

```python
# pipeline/ingestion/market_data.py
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
        if not info or not info.get("currentPrice") and not info.get("regularMarketPrice"):
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_ingestion.py::TestMarketData -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add pipeline/ingestion/market_data.py tests/test_ingestion.py
git commit -m "feat(ingestion): add yfinance market data scraper"
```

---

## Task 5: News Scraper (yfinance + NewsData.io)

**Files:**
- Create: `pipeline/ingestion/news.py`
- Modify: `tests/test_ingestion.py`

- [ ] **Step 1: Write the tests**

Append to `tests/test_ingestion.py`:

```python
from unittest.mock import AsyncMock, patch
import httpx

from pipeline.ingestion.news import fetch_news


# ---------------------------------------------------------------------------
# News
# ---------------------------------------------------------------------------

class TestNews:
    def test_yfinance_news_only(self):
        """When no API key, only yfinance news is fetched."""
        mock_ticker = MagicMock()
        mock_ticker.news = [
            {
                "title": "Apple beats earnings",
                "link": "https://example.com/aapl-earnings",
                "publisher": "Reuters",
                "providerPublishTime": 1743955200,  # 2025-04-06T16:00:00Z
            },
        ]

        with patch("pipeline.ingestion.news.yf.Ticker", return_value=mock_ticker):
            hits, healths = asyncio.run(fetch_news(["AAPL"], api_key=None))

        assert len(hits) >= 1
        assert hits[0].source == "yfinance"
        assert any(h.source == "yfinance_news" for h in healths)
        # No newsdata health entry when key is None
        assert not any(h.source == "newsdata" for h in healths)

    def test_newsdata_integration(self):
        """When API key provided, both sources run and results merge."""
        mock_ticker = MagicMock()
        mock_ticker.news = [
            {
                "title": "AAPL from yfinance",
                "link": "https://example.com/yf-aapl",
                "publisher": "AP",
                "providerPublishTime": 1743955200,
            },
        ]

        newsdata_response = httpx.Response(
            200,
            json={
                "status": "success",
                "results": [
                    {
                        "title": "AAPL from newsdata",
                        "link": "https://example.com/nd-aapl",
                        "source_id": "reuters",
                        "pubDate": "2026-04-06 10:00:00",
                    },
                ],
            },
            request=httpx.Request("GET", "https://newsdata.io/api/1/latest"),
        )

        with (
            patch("pipeline.ingestion.news.yf.Ticker", return_value=mock_ticker),
            patch("pipeline.ingestion.news.httpx.AsyncClient") as mock_client_cls,
        ):
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=newsdata_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            hits, healths = asyncio.run(fetch_news(["AAPL"], api_key="test_key"))

        sources = {h.source for h in hits}
        assert "yfinance" in sources
        assert "newsdata" in sources
        assert len(healths) == 2

    def test_newsdata_url_dedup(self):
        """Same URL from both sources is deduplicated."""
        shared_url = "https://example.com/shared-article"
        mock_ticker = MagicMock()
        mock_ticker.news = [
            {
                "title": "Shared article",
                "link": shared_url,
                "publisher": "AP",
                "providerPublishTime": 1743955200,
            },
        ]

        newsdata_response = httpx.Response(
            200,
            json={
                "status": "success",
                "results": [
                    {
                        "title": "Shared article",
                        "link": shared_url,
                        "source_id": "ap",
                        "pubDate": "2026-04-06 10:00:00",
                    },
                ],
            },
            request=httpx.Request("GET", "https://newsdata.io/api/1/latest"),
        )

        with (
            patch("pipeline.ingestion.news.yf.Ticker", return_value=mock_ticker),
            patch("pipeline.ingestion.news.httpx.AsyncClient") as mock_client_cls,
        ):
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=newsdata_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            hits, _ = asyncio.run(fetch_news(["AAPL"], api_key="test_key"))

        urls = [h.url for h in hits]
        assert urls.count(shared_url) == 1

    def test_newsdata_rate_limit(self):
        """429 response produces rate_limited health."""
        mock_ticker = MagicMock()
        mock_ticker.news = []

        newsdata_response = httpx.Response(
            429,
            json={"status": "error", "results": {"message": "Rate limit exceeded"}},
            request=httpx.Request("GET", "https://newsdata.io/api/1/latest"),
        )

        with (
            patch("pipeline.ingestion.news.yf.Ticker", return_value=mock_ticker),
            patch("pipeline.ingestion.news.httpx.AsyncClient") as mock_client_cls,
        ):
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=newsdata_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            hits, healths = asyncio.run(fetch_news(["AAPL"], api_key="test_key"))

        newsdata_health = next(h for h in healths if h.source == "newsdata")
        assert newsdata_health.status == "rate_limited"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_ingestion.py::TestNews -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write `news.py`**

```python
# pipeline/ingestion/news.py
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
                # Map article to relevant tickers
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
                    # If no specific ticker match, assign to first ticker in query
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
        try:
            nd_hits, nd_health = await _fetch_newsdata(tickers, api_key, cutoff)
            healths.append(nd_health)
            all_hits.extend(nd_hits)
        except Exception:
            # _fetch_newsdata already handles errors via timed_health
            pass

    # Deduplicate by URL (keep first seen)
    seen_urls: set[str] = set()
    deduped: list[NewsHit] = []
    for hit in all_hits:
        if hit.url and hit.url not in seen_urls:
            seen_urls.add(hit.url)
            deduped.append(hit)

    return deduped, healths
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_ingestion.py::TestNews -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add pipeline/ingestion/news.py tests/test_ingestion.py
git commit -m "feat(ingestion): add news scraper (yfinance + NewsData.io)"
```

---

## Task 6: Social Scraper (Reddit/PRAW)

**Files:**
- Create: `pipeline/ingestion/social.py`
- Modify: `tests/test_ingestion.py`

- [ ] **Step 1: Write the tests**

Append to `tests/test_ingestion.py`:

```python
from pipeline.ingestion.social import fetch_social


# ---------------------------------------------------------------------------
# Social
# ---------------------------------------------------------------------------

class TestSocial:
    def test_missing_credentials(self):
        """No credentials returns empty + error health."""
        hits, health = asyncio.run(
            fetch_social(["AAPL"], subreddits=["stocks"], client_id=None, client_secret=None)
        )
        assert hits == []
        assert health.status == "error"
        assert "credentials" in health.error_detail.lower()

    def test_fetch_social_posts(self):
        """Mock PRAW to return posts."""
        mock_submission = MagicMock()
        mock_submission.title = "$NVDA earnings beat"
        mock_submission.selftext = "DD on NVDA..."
        mock_submission.url = "https://reddit.com/r/stocks/abc"
        mock_submission.score = 250
        mock_submission.created_utc = 1743955200.0

        mock_subreddit = MagicMock()
        mock_subreddit.search.return_value = [mock_submission]

        mock_reddit = MagicMock()
        mock_reddit.subreddit.return_value = mock_subreddit

        with patch("pipeline.ingestion.social.praw.Reddit", return_value=mock_reddit):
            hits, health = asyncio.run(
                fetch_social(
                    ["NVDA"],
                    subreddits=["stocks"],
                    client_id="test_id",
                    client_secret="test_secret",
                )
            )

        assert len(hits) >= 1
        assert hits[0].ticker == "NVDA"
        assert hits[0].subreddit == "stocks"
        assert health.status == "success"

    def test_subreddit_unavailable(self):
        """Unavailable subreddit is skipped, not crash."""
        mock_subreddit = MagicMock()
        mock_subreddit.search.side_effect = Exception("subreddit is quarantined")

        mock_reddit = MagicMock()
        mock_reddit.subreddit.return_value = mock_subreddit

        with patch("pipeline.ingestion.social.praw.Reddit", return_value=mock_reddit):
            hits, health = asyncio.run(
                fetch_social(
                    ["AAPL"],
                    subreddits=["wallstreetbets"],
                    client_id="test_id",
                    client_secret="test_secret",
                )
            )

        # Should not crash, health may still be success with 0 items
        assert isinstance(hits, list)
        assert health.status == "success"
        assert health.items_fetched == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_ingestion.py::TestSocial -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write `social.py`**

```python
# pipeline/ingestion/social.py
"""Reddit social scraper via PRAW."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import praw

from pipeline.ingestion.health import timed_health
from pipeline.ingestion.models import SocialHit, SourceHealth


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
                except Exception:
                    continue
        except Exception:
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_ingestion.py::TestSocial -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add pipeline/ingestion/social.py tests/test_ingestion.py
git commit -m "feat(ingestion): add Reddit social scraper via PRAW"
```

---

## Task 7: Crawl4AI Parser

**Files:**
- Create: `pipeline/ingestion/parser.py`
- Modify: `tests/test_ingestion.py`

- [ ] **Step 1: Write the tests**

Append to `tests/test_ingestion.py`:

```python
from pipeline.ingestion.parser import parse_urls


# ---------------------------------------------------------------------------
# Parser (Crawl4AI)
# ---------------------------------------------------------------------------

class TestParser:
    def test_parse_single_url_success(self):
        """Mock Crawl4AI to return markdown for a URL."""
        mock_result = MagicMock()
        mock_result.markdown = "# Article Title\n\nArticle content here."
        mock_result.success = True

        mock_crawler = AsyncMock()
        mock_crawler.arun.return_value = mock_result
        mock_crawler.__aenter__ = AsyncMock(return_value=mock_crawler)
        mock_crawler.__aexit__ = AsyncMock(return_value=False)

        with patch("pipeline.ingestion.parser.AsyncWebCrawler", return_value=mock_crawler):
            results, health = asyncio.run(parse_urls(["https://example.com/article"]))

        assert len(results) == 1
        assert results[0].success is True
        assert "Article Title" in results[0].markdown
        assert health.status == "success"
        assert health.items_fetched == 1

    def test_parse_url_failure(self):
        """Failed URL produces ParsedContent with success=False."""
        mock_result = MagicMock()
        mock_result.markdown = ""
        mock_result.success = False
        mock_result.error_message = "Connection timeout"

        mock_crawler = AsyncMock()
        mock_crawler.arun.return_value = mock_result
        mock_crawler.__aenter__ = AsyncMock(return_value=mock_crawler)
        mock_crawler.__aexit__ = AsyncMock(return_value=False)

        with patch("pipeline.ingestion.parser.AsyncWebCrawler", return_value=mock_crawler):
            results, health = asyncio.run(parse_urls(["https://example.com/broken"]))

        assert len(results) == 1
        assert results[0].success is False
        assert results[0].error is not None

    def test_parse_empty_list(self):
        """Empty URL list returns empty results + success health."""
        results, health = asyncio.run(parse_urls([]))
        assert results == []
        assert health.status == "success"
        assert health.items_fetched == 0

    def test_parse_mixed_success_failure(self):
        """Mix of successful and failed URLs."""
        success_result = MagicMock()
        success_result.markdown = "# Good"
        success_result.success = True

        fail_result = MagicMock()
        fail_result.markdown = ""
        fail_result.success = False
        fail_result.error_message = "404"

        mock_crawler = AsyncMock()
        mock_crawler.arun.side_effect = [success_result, fail_result]
        mock_crawler.__aenter__ = AsyncMock(return_value=mock_crawler)
        mock_crawler.__aexit__ = AsyncMock(return_value=False)

        with patch("pipeline.ingestion.parser.AsyncWebCrawler", return_value=mock_crawler):
            results, health = asyncio.run(
                parse_urls(["https://example.com/good", "https://example.com/bad"])
            )

        assert len(results) == 2
        assert results[0].success is True
        assert results[1].success is False
        assert health.items_fetched == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_ingestion.py::TestParser -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write `parser.py`**

```python
# pipeline/ingestion/parser.py
"""Crawl4AI URL-to-Markdown parser."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from crawl4ai import AsyncWebCrawler

from pipeline.ingestion.health import timed_health
from pipeline.ingestion.models import ParsedContent, SourceHealth
from pipeline.config import CRAWL4AI_CONCURRENCY, CRAWL4AI_TIMEOUT_SECONDS


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

            results = await asyncio.gather(*[_parse_one(url) for url in urls])
            results = list(results)
            ctx.items_fetched = sum(1 for r in results if r.success)

    return results, ctx.health
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_ingestion.py::TestParser -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add pipeline/ingestion/parser.py tests/test_ingestion.py
git commit -m "feat(ingestion): add Crawl4AI URL parser with concurrency control"
```

---

## Task 8: Orchestrator

**Files:**
- Create: `pipeline/ingestion/orchestrator.py`
- Modify: `tests/test_ingestion.py`

- [ ] **Step 1: Write the tests**

Append to `tests/test_ingestion.py`:

```python
from pipeline.ingestion.orchestrator import run_ingestion, _apply_volume_caps


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class TestOrchestrator:
    def test_volume_cap_news(self):
        """50 news hits for one ticker capped to 20."""
        hits = [
            NewsHit(
                ticker="AAPL",
                headline=f"Article {i}",
                url=f"https://example.com/article-{i}",
                source="yfinance",
                published_at=f"2026-04-06T{10 + (i % 12):02d}:00:00Z",
            )
            for i in range(50)
        ]
        capped = _apply_volume_caps(news_hits=hits, social_hits=[], max_news=20, max_social=30)
        news_for_aapl = [h for h in capped["news"] if h.ticker == "AAPL"]
        assert len(news_for_aapl) == 20

    def test_volume_cap_social(self):
        """40 social hits for one ticker capped to 30."""
        hits = [
            SocialHit(
                ticker="NVDA",
                title=f"Post {i}",
                body="",
                url=f"https://reddit.com/r/stocks/{i}",
                score=100 - i,
                subreddit="stocks",
                posted_at=f"2026-04-06T{10 + (i % 12):02d}:00:00Z",
            )
            for i in range(40)
        ]
        capped = _apply_volume_caps(news_hits=[], social_hits=hits, max_news=20, max_social=30)
        social_for_nvda = [h for h in capped["social"] if h.ticker == "NVDA"]
        assert len(social_for_nvda) == 30
        # Highest scores should be kept (sorted by score desc)
        scores = [h.score for h in social_for_nvda]
        assert scores == sorted(scores, reverse=True)

    def test_url_dedup(self):
        """Same URL under two tickers: Crawl4AI processes it once."""
        shared_url = "https://example.com/shared"
        news_hits = [
            NewsHit("AAPL", "Shared article", shared_url, "yfinance", "2026-04-06T10:00:00Z"),
            NewsHit("MSFT", "Shared article", shared_url, "yfinance", "2026-04-06T10:00:00Z"),
        ]
        social_hits = [
            SocialHit("AAPL", "Shared post", "", shared_url, 100, "stocks", "2026-04-06T10:00:00Z"),
        ]
        capped = _apply_volume_caps(news_hits=news_hits, social_hits=social_hits, max_news=20, max_social=30)
        unique_urls = capped["unique_urls"]
        assert unique_urls.count(shared_url) == 1

    def test_full_orchestrator(self):
        """Mock all scrapers, run orchestrator, verify IngestionResult shape."""
        mock_market = (
            {"AAPL": MarketDataPoint("AAPL", 175.0, 174.0, 173.0, 28.0, 2.8e12, [170.0], "2026-04-06T16:00:00Z")},
            SourceHealth("yfinance", "success", 1, 500),
        )
        mock_news = (
            [NewsHit("AAPL", "News", "https://example.com/1", "yfinance", "2026-04-06T10:00:00Z")],
            [SourceHealth("yfinance_news", "success", 1, 300)],
        )
        mock_social = (
            [],
            SourceHealth("reddit", "error", 0, 0, "Reddit credentials not configured"),
        )
        mock_parsed = (
            [ParsedContent("https://example.com/1", "# News", "2026-04-06T16:00:00Z", True, None)],
            SourceHealth("crawl4ai", "success", 1, 2000),
        )

        with (
            patch("pipeline.ingestion.orchestrator.fetch_market_data", AsyncMock(return_value=mock_market)),
            patch("pipeline.ingestion.orchestrator.fetch_news", AsyncMock(return_value=mock_news)),
            patch("pipeline.ingestion.orchestrator.fetch_social", AsyncMock(return_value=mock_social)),
            patch("pipeline.ingestion.orchestrator.parse_urls", AsyncMock(return_value=mock_parsed)),
        ):
            result = asyncio.run(run_ingestion(["AAPL"]))

        assert "AAPL" in result.market_data
        assert len(result.news_hits) == 1
        assert len(result.parsed_content) == 1
        assert len(result.health) == 4  # yfinance, yfinance_news, reddit, crawl4ai

    def test_orchestrator_scraper_failure_doesnt_block(self):
        """One scraper crashing doesn't block others."""
        mock_market = (
            {},
            SourceHealth("yfinance", "error", 0, 0, "crash"),
        )
        mock_news = (
            [NewsHit("AAPL", "News", "https://example.com/1", "yfinance", "2026-04-06T10:00:00Z")],
            [SourceHealth("yfinance_news", "success", 1, 300)],
        )
        mock_social = (
            [],
            SourceHealth("reddit", "error", 0, 0, "Reddit credentials not configured"),
        )
        mock_parsed = (
            [ParsedContent("https://example.com/1", "# News", "2026-04-06T16:00:00Z", True, None)],
            SourceHealth("crawl4ai", "success", 1, 2000),
        )

        with (
            patch("pipeline.ingestion.orchestrator.fetch_market_data", AsyncMock(return_value=mock_market)),
            patch("pipeline.ingestion.orchestrator.fetch_news", AsyncMock(return_value=mock_news)),
            patch("pipeline.ingestion.orchestrator.fetch_social", AsyncMock(return_value=mock_social)),
            patch("pipeline.ingestion.orchestrator.parse_urls", AsyncMock(return_value=mock_parsed)),
        ):
            result = asyncio.run(run_ingestion(["AAPL"]))

        # Pipeline still returns a result despite yfinance failure
        assert isinstance(result, IngestionResult)
        assert len(result.news_hits) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_ingestion.py::TestOrchestrator -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write `orchestrator.py`**

```python
# pipeline/ingestion/orchestrator.py
"""Ingestion orchestrator — runs all scrapers, applies volume caps and dedup."""

from __future__ import annotations

import asyncio

from pipeline.config import (
    CRAWL4AI_CONCURRENCY,
    DEFAULT_SUBREDDITS,
    MAX_NEWS_PER_TICKER,
    MAX_SOCIAL_PER_TICKER,
    NEWSDATA_API_KEY,
    REDDIT_CLIENT_ID,
    REDDIT_CLIENT_SECRET,
    REDDIT_USER_AGENT,
)
from pipeline.ingestion.health import aggregate_health
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
    # Run all scrapers concurrently
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

    # Collect all health entries
    all_health: list[SourceHealth] = [market_health] + news_healths + [social_health]

    # Apply volume caps and get unique URLs
    capped = _apply_volume_caps(news_hits, social_hits)

    # Parse URLs via Crawl4AI
    parsed_content, parse_health = await parse_urls(capped["unique_urls"])
    all_health.append(parse_health)

    return IngestionResult(
        market_data=market_data,
        news_hits=capped["news"],
        social_hits=capped["social"],
        parsed_content=parsed_content,
        health=all_health,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_ingestion.py::TestOrchestrator -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add pipeline/ingestion/orchestrator.py tests/test_ingestion.py
git commit -m "feat(ingestion): add orchestrator with volume caps and URL dedup"
```

---

## Task 9: Package Exports & Final Integration Test

**Files:**
- Modify: `pipeline/ingestion/__init__.py`
- Modify: `tests/test_ingestion.py`

- [ ] **Step 1: Update `__init__.py` with re-exports**

```python
# pipeline/ingestion/__init__.py
"""Data Ingestion Layer — Phase 3."""

from pipeline.ingestion.models import (
    IngestionResult,
    MarketDataPoint,
    NewsHit,
    ParsedContent,
    SocialHit,
    SourceHealth,
)
from pipeline.ingestion.orchestrator import run_ingestion

__all__ = [
    "IngestionResult",
    "MarketDataPoint",
    "NewsHit",
    "ParsedContent",
    "SocialHit",
    "SourceHealth",
    "run_ingestion",
]
```

- [ ] **Step 2: Write integration test**

Append to `tests/test_ingestion.py`:

```python
# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_import_from_package(self):
        """Verify all public types are importable from the package."""
        from pipeline.ingestion import (
            IngestionResult,
            MarketDataPoint,
            NewsHit,
            ParsedContent,
            SocialHit,
            SourceHealth,
            run_ingestion,
        )
        assert callable(run_ingestion)

    def test_health_aggregate_to_json(self):
        """Health list converts to JSON-serializable dict for run_log.source_status."""
        import json
        from pipeline.ingestion.health import aggregate_health

        healths = [
            SourceHealth("yfinance", "success", 30, 1200),
            SourceHealth("yfinance_news", "success", 15, 800),
            SourceHealth("newsdata", "rate_limited", 0, 100, "429 rate limit exceeded"),
            SourceHealth("reddit", "error", 0, 0, "Reddit credentials not configured"),
            SourceHealth("crawl4ai", "success", 12, 5000),
        ]
        blob = aggregate_health(healths)
        # Must be JSON-serializable (this is what goes into run_log.source_status)
        json_str = json.dumps(blob)
        parsed = json.loads(json_str)
        assert parsed["yfinance"]["status"] == "success"
        assert parsed["newsdata"]["error_detail"] == "429 rate limit exceeded"
        assert parsed["crawl4ai"]["items_fetched"] == 12
```

- [ ] **Step 3: Run full test suite**

Run: `pytest tests/test_ingestion.py -v`
Expected: All tests PASS

- [ ] **Step 4: Run existing tests to verify no regressions**

Run: `pytest tests/ -v`
Expected: All tests PASS (including Phase 1 and Phase 2 tests)

- [ ] **Step 5: Commit**

```bash
git add pipeline/ingestion/__init__.py tests/test_ingestion.py
git commit -m "feat(ingestion): finalize package exports and integration tests"
```

---

## Verification

After all tasks are complete:

1. **Unit tests:** `pytest tests/test_ingestion.py -v` — all pass
2. **Full suite:** `pytest tests/ -v` — no regressions from Phase 1/2
3. **Import check:** `python -c "from pipeline.ingestion import run_ingestion, IngestionResult; print('OK')"`
4. **Env check:** verify `.env` has `NEWSDATA_API_KEY` set, verify `.env` is NOT tracked by git (`git status` should not show `.env`)

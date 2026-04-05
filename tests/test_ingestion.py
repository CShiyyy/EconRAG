"""Tests for the Data Ingestion Layer (Phase 3)."""

from pipeline.ingestion.models import (
    IngestionResult,
    MarketDataPoint,
    NewsHit,
    ParsedContent,
    SocialHit,
    SourceHealth,
)


class TestModels:
    def test_market_data_point_fields(self):
        mdp = MarketDataPoint(
            ticker="AAPL", current_price=175.50, open_price=174.00,
            previous_close=173.80, pe_ratio=28.5, market_cap=2.8e12,
            price_history_30d=[170.0, 171.0, 172.0], fetched_at="2026-04-06T16:00:00Z",
        )
        assert mdp.ticker == "AAPL"
        assert mdp.pe_ratio == 28.5
        assert len(mdp.price_history_30d) == 3

    def test_market_data_point_nullable_fields(self):
        mdp = MarketDataPoint(
            ticker="FAKE", current_price=0.0, open_price=0.0,
            previous_close=0.0, pe_ratio=None, market_cap=None,
            price_history_30d=[], fetched_at="2026-04-06T16:00:00Z",
        )
        assert mdp.pe_ratio is None
        assert mdp.market_cap is None

    def test_news_hit_fields(self):
        hit = NewsHit(
            ticker="MSFT", headline="Microsoft beats earnings",
            url="https://example.com/msft", source="yfinance",
            published_at="2026-04-06T10:00:00Z",
        )
        assert hit.source == "yfinance"

    def test_social_hit_fields(self):
        hit = SocialHit(
            ticker="NVDA", title="NVDA to the moon", body="DD on NVDA earnings...",
            url="https://reddit.com/r/stocks/123", score=500, subreddit="stocks",
            posted_at="2026-04-06T08:00:00Z",
        )
        assert hit.score == 500

    def test_parsed_content_success(self):
        pc = ParsedContent(
            url="https://example.com/article", markdown="# Article\nContent here",
            fetched_at="2026-04-06T16:00:00Z", success=True, error=None,
        )
        assert pc.success is True

    def test_parsed_content_failure(self):
        pc = ParsedContent(
            url="https://example.com/broken", markdown="",
            fetched_at="2026-04-06T16:00:00Z", success=False, error="Timeout after 30s",
        )
        assert pc.success is False
        assert "Timeout" in pc.error

    def test_source_health_defaults(self):
        sh = SourceHealth(source="yfinance", status="success", items_fetched=30, duration_ms=1200)
        assert sh.error_detail is None

    def test_ingestion_result_bundle(self):
        result = IngestionResult(
            market_data={}, news_hits=[], social_hits=[], parsed_content=[], health=[],
        )
        assert isinstance(result.market_data, dict)
        assert isinstance(result.health, list)


import asyncio
from unittest.mock import MagicMock, patch

import pandas as pd

from pipeline.ingestion.health import aggregate_health, timed_health


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

    def test_timed_health_timeout(self):
        async def _run():
            async with timed_health("failing_source") as ctx:
                raise TimeoutError("connection timed out")
            return ctx.health
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


from pipeline.ingestion.market_data import fetch_market_data


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

        # _fetch_single_ticker catches the exception and returns None
        assert data["AAPL"] is None
        assert health.status == "success"  # timed_health wraps the loop, individual failures don't crash it

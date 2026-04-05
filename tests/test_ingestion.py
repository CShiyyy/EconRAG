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

from __future__ import annotations
from dataclasses import dataclass


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

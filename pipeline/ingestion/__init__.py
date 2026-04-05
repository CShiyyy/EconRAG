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

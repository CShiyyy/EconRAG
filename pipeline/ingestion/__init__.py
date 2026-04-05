"""Data Ingestion Layer — Phase 3."""

from pipeline.ingestion.models import (
    IngestionResult,
    MarketDataPoint,
    NewsHit,
    ParsedContent,
    SocialHit,
    SourceHealth,
)

__all__ = [
    "IngestionResult",
    "MarketDataPoint",
    "NewsHit",
    "ParsedContent",
    "SocialHit",
    "SourceHealth",
    "run_ingestion",
]


def __getattr__(name: str):
    if name == "run_ingestion":
        from pipeline.ingestion.orchestrator import run_ingestion
        return run_ingestion
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

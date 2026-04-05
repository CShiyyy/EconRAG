"""Source health tracking utilities."""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
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
        health = ctx.health
    """
    ctx = _HealthContext(source=source)
    start = time.monotonic()
    try:
        yield ctx
        elapsed_ms = int((time.monotonic() - start) * 1000)
        ctx.health = SourceHealth(
            source=source, status="success",
            items_fetched=ctx.items_fetched, duration_ms=elapsed_ms,
        )
    except TimeoutError as exc:
        elapsed_ms = int((time.monotonic() - start) * 1000)
        ctx.health = SourceHealth(
            source=source, status="timeout",
            items_fetched=0, duration_ms=elapsed_ms, error_detail=str(exc),
        )
    except Exception as exc:
        elapsed_ms = int((time.monotonic() - start) * 1000)
        status = "rate_limited" if "429" in str(exc) or "rate" in str(exc).lower() else "error"
        ctx.health = SourceHealth(
            source=source, status=status,
            items_fetched=0, duration_ms=elapsed_ms, error_detail=str(exc),
        )

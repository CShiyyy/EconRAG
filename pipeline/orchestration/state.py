"""Pipeline state schema shared across all subgraphs."""

from __future__ import annotations

from typing import Any
from typing_extensions import TypedDict


class PipelineState(TypedDict, total=False):
    run_id: int
    run_type: str                        # "scheduled" for new runs; legacy "pre_open"/"post_close"
    session_date: str                    # YYYY-MM-DD ET calendar date
    is_first_run: bool
    start_time: float                    # time.time() for wall_clock_seconds
    ingestion_result: dict | None        # IngestionResult serialized to dict
    source_health: list[dict]
    market_narrative: dict | None        # Agent A output
    quant_assessment: dict | None        # Agent B output (None only if Second Tower has insufficient history)
    risk_assessment: dict | None         # Agent C output
    previous_assessment: dict | None     # Most recent decision run recommendations
    standing_context: list[dict]         # Active standing events
    requery_count: int                   # 0 or 1
    requery_reason: str | None
    flagged_tickers: list[str]           # Tickers flagged for re-query
    sizing_result: dict | None
    trade_list: list[dict]
    db_path: str
    rag_storage_dir: str
    tickers: list[str]                   # Tracked tickers from watchlist
    profile_seed_result: dict | None     # SeedResult from seed_profiles_node
    last_close_dt: str | None            # ISO UTC of the most recent NYSE close at run time
    execution_result: dict | None        # Result from execute_pending_node

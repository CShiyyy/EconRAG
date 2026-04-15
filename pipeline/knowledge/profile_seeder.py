"""Profile seeder: generates LLM-synthesized ticker and macro profiles at init time.

Runs as a LangGraph node before the first ingest to give Agent A non-empty
LightRAG context on the first pipeline run. Also runs on subsequent runs to
seed profiles for any new tickers added via watchlist refresh.

Idempotency is tracked in the kg_seed_log SQLite table — tickers already
seeded are skipped, making the node a near-no-op after initialization.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from lightrag import LightRAG

from pipeline.config import (
    PROFILE_SEED_CONCURRENCY,
    PROFILE_SEED_ENABLED,
    PROFILE_SEED_TTL_HOURS,
)
from pipeline.knowledge.extraction import extract_from_markdown
from pipeline.knowledge.graph_ops import insert_validated_data
from pipeline.knowledge.validator import validate_extraction

if TYPE_CHECKING:
    from pipeline.agents.cloud_client import CloudLLMClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt constants
# ---------------------------------------------------------------------------

TICKER_PROFILE_SYSTEM_PROMPT = """\
You are a financial analyst writing concise company profiles for a knowledge graph seeding system.

Write a factual markdown profile (150-250 words) for the given company. Use the phrasing patterns \
below so a downstream entity-extraction model can identify relationships:

REQUIRED phrasing patterns (use the exact phrasing where applicable):
- "{TICKER} belongs to the {sector} sector"
- "{TICKER} is led by CEO {full name}"
- "{TICKER} is a constituent of the S&P 500" (include for S&P members; use "Nasdaq-100" or \
"Dow Jones Industrial Average" as appropriate)
- "{TICKER} competes with {TICKER_A} and {TICKER_B}" (use ticker symbols for competitors)
- "{TICKER} produces {product or service name}" (name 2-3 key products or services)
- End with a "**Current state.**" paragraph describing one or two dominant themes or pressures \
driving the stock right now (e.g. AI capex cycle, regulatory scrutiny, margin expansion).

Use plain, factual prose. Do not use bullet points. Do not hallucinate — if you are unsure of a \
specific CEO name or competitor ticker, omit that sentence rather than guess.
"""

MACRO_PROFILE_SYSTEM_PROMPT = """\
You are a macro analyst writing a concise market environment snapshot for a knowledge graph \
seeding system.

Write a factual markdown profile (300-400 words) describing the current macro environment. \
Use the phrasing patterns below so a downstream entity-extraction model can identify entities \
and relationships:

REQUIRED phrasing patterns (use the exact phrasing where applicable):
- Name institutions explicitly: "the Federal Reserve", "the European Central Bank", "the SEC"
- Name key officials: "Federal Reserve Chair {name}" (use the current chair)
- Name macro themes with explicit sluggable labels, e.g.: "the AI capital expenditure cycle", \
"China export restriction risks", "commercial real estate stress", "reshoring and tariff policy"
- "The {sector} sector is driven by {theme}" or "The {sector} sector is exposed to {theme}"
- "The Federal Reserve has {raised / held / cut} rates" with context on the current stance

REQUIRED sections (in order):
1. Monetary policy and interest rate environment
2. Inflation and growth context
3. Two to three dominant sector-level themes
4. Key geopolitical or regulatory forces

Use plain, factual prose. Do not use bullet points. Write in the present tense.
"""


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SeedResult:
    tickers_seeded: list[str] = field(default_factory=list)
    macro_seeded: bool = False
    tickers_failed: list[tuple[str, str]] = field(default_factory=list)  # (ticker, error)
    macro_error: str | None = None
    skipped_existing: int = 0


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _build_ticker_messages(ticker: str, company_name: str, sector: str) -> list[dict]:
    return [
        {"role": "system", "content": TICKER_PROFILE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Write a company profile for {company_name} (ticker: {ticker}). "
                f"It belongs to the {sector} sector."
            ),
        },
    ]


def _build_macro_messages() -> list[dict]:
    return [
        {"role": "system", "content": MACRO_PROFILE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Write a macro environment snapshot covering the current state of US equity "
                "markets: monetary policy, inflation/growth context, dominant sector-level "
                "narratives, and key geopolitical or regulatory forces."
            ),
        },
    ]


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

async def _generate_profile(client: "CloudLLMClient", messages: list[dict]) -> str | None:
    """Call cloud LLM and return markdown string. Returns None on any failure."""
    try:
        return await client.generate(messages, json_mode=False)
    except Exception as exc:
        logger.error("Profile generation LLM call failed: %s", exc)
        return None


async def _seed_single(
    conn: sqlite3.Connection,
    rag: LightRAG,
    client: "CloudLLMClient",
    seed_type: str,
    seed_key: str,
    messages: list[dict],
    source_id: str,
    run_id: int,
) -> bool:
    """Generate, extract, validate, and insert one profile. Write kg_seed_log on success.

    Steps:
    1. Cloud LLM generates markdown profile text.
    2. Local Ollama extractor parses entities/relationships from the markdown.
    3. Validator canonicalises results; Tier 2 edges receive extended TTL.
    4. insert_validated_data writes the result into LightRAG.
    5. kg_seed_log row is written only if all prior steps succeeded.

    Returns True on success, False on any failure.
    """
    # Stage 1: LLM generation
    markdown = await _generate_profile(client, messages)
    if not markdown or not markdown.strip():
        logger.warning("Empty profile from cloud LLM for %s:%s", seed_type, seed_key)
        return False

    # Stage 2: Local extraction
    raw_extraction = await extract_from_markdown(markdown)
    if raw_extraction is None:
        logger.warning("Ollama extraction failed for profile %s:%s", seed_type, seed_key)
        return False

    # Stage 3: Validate and canonicalise with extended TTL for Tier 2 edges
    validation = validate_extraction(
        raw_extraction,
        conn,
        run_id,
        base_ttl_hours=PROFILE_SEED_TTL_HOURS,
    )
    logger.debug(
        "Profile %s:%s validated — %d entities, %d relationships, %d rejected",
        seed_type, seed_key,
        len(validation.entities),
        len(validation.relationships),
        len(validation.rejected_entities) + len(validation.rejected_relationships),
    )

    if not validation.entities and not validation.relationships:
        logger.warning(
            "No valid entities/relationships extracted from profile %s:%s — skipping KG insert",
            seed_type, seed_key,
        )
        return False

    # Stage 4: Insert into LightRAG
    await insert_validated_data(rag, validation, markdown, source_id)

    # Stage 5: Record in seed log (only reached on full success)
    conn.execute(
        """
        INSERT OR REPLACE INTO kg_seed_log (seed_type, seed_key, seeded_at, run_id, source_id)
        VALUES (?, ?, ?, ?, ?)
        """,
        (seed_type, seed_key, datetime.now(timezone.utc).isoformat(), run_id, source_id),
    )
    conn.commit()
    logger.info("Seeded profile %s:%s (source_id=%s)", seed_type, seed_key, source_id)
    return True


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

async def seed_missing_profiles(
    conn: sqlite3.Connection,
    rag: LightRAG,
    client: "CloudLLMClient",
    run_id: int,
) -> SeedResult:
    """Seed LightRAG profiles for any watchlist tickers not yet in kg_seed_log.

    Idempotent: already-seeded tickers are skipped via a single SQL check.
    Concurrency is bounded by PROFILE_SEED_CONCURRENCY.
    Individual ticker failures are non-fatal and accumulated in SeedResult.
    """
    result = SeedResult()

    if not PROFILE_SEED_ENABLED:
        return result

    # Find tickers still needing a profile
    pending_rows = conn.execute(
        """
        SELECT w.ticker, w.company_name, w.sector
        FROM watchlist w
        WHERE w.ticker NOT IN (
            SELECT seed_key FROM kg_seed_log WHERE seed_type = 'ticker_profile'
        )
        ORDER BY w.ticker
        """,
    ).fetchall()

    # Check if the macro profile exists
    macro_missing = conn.execute(
        "SELECT 1 FROM kg_seed_log WHERE seed_type = 'macro_profile' AND seed_key = 'macro'",
    ).fetchone() is None

    if not pending_rows and not macro_missing:
        result.skipped_existing = conn.execute(
            "SELECT COUNT(*) FROM kg_seed_log"
        ).fetchone()[0]
        logger.info(
            "Profile seeder: all %d profiles already seeded, skipping",
            result.skipped_existing,
        )
        return result

    result.skipped_existing = conn.execute(
        "SELECT COUNT(*) FROM kg_seed_log"
    ).fetchone()[0]

    logger.info(
        "Profile seeder: %d ticker(s) pending, macro=%s",
        len(pending_rows),
        "pending" if macro_missing else "already seeded",
    )

    # Seed ticker profiles with bounded concurrency
    sem = asyncio.Semaphore(PROFILE_SEED_CONCURRENCY)

    async def _seed_ticker(ticker: str, company_name: str, sector: str) -> None:
        async with sem:
            try:
                ok = await _seed_single(
                    conn=conn,
                    rag=rag,
                    client=client,
                    seed_type="ticker_profile",
                    seed_key=ticker,
                    messages=_build_ticker_messages(ticker, company_name, sector),
                    source_id=f"profile:{ticker}",
                    run_id=run_id,
                )
                if ok:
                    result.tickers_seeded.append(ticker)
                else:
                    result.tickers_failed.append(
                        (ticker, "generation or extraction returned empty")
                    )
            except Exception as exc:
                logger.error("Unexpected error seeding ticker %s: %s", ticker, exc)
                result.tickers_failed.append((ticker, str(exc)))

    ticker_tasks = [_seed_ticker(t, n, s) for t, n, s in pending_rows]
    await asyncio.gather(*ticker_tasks)

    # Seed macro profile (after tickers, sequential)
    if macro_missing:
        try:
            ok = await _seed_single(
                conn=conn,
                rag=rag,
                client=client,
                seed_type="macro_profile",
                seed_key="macro",
                messages=_build_macro_messages(),
                source_id="profile:macro",
                run_id=run_id,
            )
            if ok:
                result.macro_seeded = True
            else:
                result.macro_error = "generation or extraction returned empty"
        except Exception as exc:
            logger.error("Unexpected error seeding macro profile: %s", exc)
            result.macro_error = str(exc)

    logger.info(
        "Profile seeder complete: %d ticker(s) seeded, %d failed, macro=%s",
        len(result.tickers_seeded),
        len(result.tickers_failed),
        "seeded" if result.macro_seeded else (
            f"failed: {result.macro_error}" if result.macro_error else "skipped"
        ),
    )
    return result

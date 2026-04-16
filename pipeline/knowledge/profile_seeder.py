"""Profile seeder: generates LLM-synthesized ticker and macro profiles at init time.

Runs as a LangGraph node before the first ingest to give Agent A non-empty
LightRAG context on the first pipeline run. Also runs on subsequent runs to
seed profiles for any new tickers added via watchlist refresh.

Idempotency is tracked in the kg_seed_log SQLite table — tickers already
seeded are skipped, making the node a near-no-op after initialization.

Each profile is grounded in a live TickerFactPack / MacroFactPack fetched from
yfinance immediately before the LLM call. If yfinance is unavailable for a
ticker, the seeder falls back to a minimal structural profile rather than
skipping the ticker entirely.
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
from pipeline.knowledge.profile_grounding import (
    MacroFactPack,
    TickerFactPack,
    fetch_macro_fact_pack,
    fetch_ticker_fact_pack,
)
from pipeline.knowledge.validator import validate_extraction

if TYPE_CHECKING:
    from pipeline.agents.cloud_client import CloudLLMClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt constants
# ---------------------------------------------------------------------------

TICKER_PROFILE_SYSTEM_PROMPT = """\
You are a financial analyst writing a company profile for a knowledge graph seeding system.

Output plain markdown prose only — no bullet points, no section headers inside the profile.
Every sentence that names the company must use the exact ticker symbol written in the FACT PACK \
(for example: "AAPL"). Never output template placeholder text — always write the real ticker symbol.
Use ONLY the facts provided in the FACT PACK. Do not recall or invent CEO names, product names, \
financial statistics, or current-state details that are not explicitly present in the FACT PACK.
The profile must open with the REQUIRED PHRASES listed in the user message, verbatim as written.
"""

MACRO_PROFILE_SYSTEM_PROMPT = """\
You are a macro analyst writing a market environment snapshot for a knowledge graph seeding system.

Output plain markdown prose only — no bullet points. Write in the present tense.
Use ONLY the data provided in the MARKET DATA section. Do not fabricate interest rate levels, \
index prices, or statistics not present in the data.
Name institutions explicitly: "the Federal Reserve", "the European Central Bank", "the SEC".
Follow the REQUIRED SECTIONS structure and end with the exact date stated in the user message.
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

def _build_ticker_messages(fact_pack: TickerFactPack) -> list[dict]:
    """Build the LLM message list for a ticker profile, grounded in a TickerFactPack.

    Required phrases in the user message are pre-rendered with the real ticker
    symbol, so any literal copying by the model produces correct output rather
    than placeholder leakage.
    """
    ticker = fact_pack.ticker
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Pre-render required phrases with real ticker (never use {TICKER} as a literal)
    required_phrases: list[str] = [
        f'"{ticker} belongs to the {fact_pack.sector} sector."',
    ]
    if fact_pack.ceo_name:
        required_phrases.append(f'"{ticker} is led by CEO {fact_pack.ceo_name}."')
    if fact_pack.peer_tickers:
        peers_str = " and ".join(fact_pack.peer_tickers[:2])
        required_phrases.append(f'"{ticker} competes with {peers_str}."')
    required_phrases.append(
        f'"{ticker} produces [describe the key products or services from the Business summary]."'
    )
    required_block = "\n".join(f"- {p}" for p in required_phrases)

    if fact_pack.grounding_available:
        fact_lines: list[str] = [
            f"- Ticker: {ticker}",
            f"- Company: {fact_pack.company_name}",
            f"- Sector: {fact_pack.sector}",
        ]
        if fact_pack.industry:
            fact_lines.append(f"- Industry: {fact_pack.industry}")
        if fact_pack.ceo_name:
            fact_lines.append(f"- CEO: {fact_pack.ceo_name}")
        if fact_pack.employees:
            fact_lines.append(f"- Employees: {fact_pack.employees:,}")
        if fact_pack.market_cap:
            fact_lines.append(f"- Market cap: ${fact_pack.market_cap / 1e9:.1f}B")
        if fact_pack.trailing_pe:
            fact_lines.append(f"- Trailing P/E: {fact_pack.trailing_pe:.1f}x")
        if fact_pack.week_52_high and fact_pack.week_52_low:
            fact_lines.append(
                f"- 52-week range: ${fact_pack.week_52_low:.2f}\u2013${fact_pack.week_52_high:.2f}"
            )
        if fact_pack.current_price:
            fact_lines.append(f"- Current price: ${fact_pack.current_price:.2f}")
        if fact_pack.peer_tickers:
            fact_lines.append(f"- Sector peers (watchlist): {', '.join(fact_pack.peer_tickers)}")
        if fact_pack.business_summary:
            fact_lines.append(f"- Business summary: {fact_pack.business_summary}")
        if fact_pack.recent_headlines:
            hl = "\n  ".join(
                f"{i + 1}. {h}" for i, h in enumerate(fact_pack.recent_headlines)
            )
            fact_lines.append(f"- Recent headlines:\n  {hl}")

        fact_block = "\n".join(fact_lines)

        user_content = (
            f"Write a company profile for {fact_pack.company_name} ({ticker}) "
            f"using the required phrases and FACT PACK below.\n\n"
            f"## REQUIRED PHRASES\n"
            f"Open your profile with these sentences, verbatim as written:\n"
            f"{required_block}\n\n"
            f"Follow the required phrases with a **Current state.** paragraph "
            f"(3\u20134 sentences). Ground it specifically in the 'Recent headlines' "
            f"and financial data from the FACT PACK. "
            f"The final sentence MUST say \"As of {today}.\"\n\n"
            f"## FACT PACK\n"
            f"{fact_block}\n\n"
            f"Today's date: {today}. Use this date in the \"Current state.\" paragraph."
        )
    else:
        # Minimal fallback — no real-time data; structural profile only
        minimal_lines = [
            f"- Ticker: {ticker}",
            f"- Company: {fact_pack.company_name}",
            f"- Sector: {fact_pack.sector}",
        ]
        if fact_pack.peer_tickers:
            minimal_lines.append(f"- Sector peers: {', '.join(fact_pack.peer_tickers)}")

        minimal_block = "\n".join(minimal_lines)

        user_content = (
            f"Write a brief structural company profile for {fact_pack.company_name} ({ticker}).\n\n"
            f"Note: No real-time financial data is available. "
            f"Use ONLY the facts below — do not invent products, financials, "
            f"CEO names, or a current-state paragraph.\n\n"
            f"## REQUIRED PHRASES\n"
            f"Include these sentences verbatim:\n"
            f"{required_block}\n\n"
            f"## FACTS\n"
            f"{minimal_block}\n\n"
            f"Today's date: {today}."
        )

    return [
        {"role": "system", "content": TICKER_PROFILE_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def _build_macro_messages(fact_pack: MacroFactPack) -> list[dict]:
    """Build the LLM message list for the macro profile, grounded in a MacroFactPack."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    if fact_pack.grounding_available:
        fact_lines: list[str] = []
        if fact_pack.treasury_10y is not None:
            fact_lines.append(f"- 10-year Treasury yield (^TNX): {fact_pack.treasury_10y:.2f}%")
        if fact_pack.treasury_3m is not None:
            fact_lines.append(f"- 3-month Treasury yield (^IRX): {fact_pack.treasury_3m:.2f}%")
        if fact_pack.vix is not None:
            fact_lines.append(f"- VIX (volatility index): {fact_pack.vix:.2f}")
        if fact_pack.dxy is not None:
            fact_lines.append(f"- US Dollar Index (DXY): {fact_pack.dxy:.2f}")
        if fact_pack.spy_level is not None:
            fact_lines.append(f"- S&P 500 (SPY): ${fact_pack.spy_level:.2f}")
        if fact_pack.qqq_level is not None:
            fact_lines.append(f"- Nasdaq-100 (QQQ): ${fact_pack.qqq_level:.2f}")
        if fact_pack.macro_headlines:
            hl = "\n  ".join(
                f"{i + 1}. {h}" for i, h in enumerate(fact_pack.macro_headlines)
            )
            fact_lines.append(f"- Recent macro headlines:\n  {hl}")

        market_data_block = "\n".join(fact_lines) if fact_lines else "(no data fetched)"

        user_content = (
            "Write a macro environment snapshot for US equity markets.\n\n"
            "## REQUIRED SECTIONS (in order)\n"
            "1. Monetary policy and interest rate environment (reference the yield data)\n"
            "2. Inflation and growth context\n"
            "3. Two to three dominant sector-level themes\n"
            "4. Key geopolitical or regulatory forces\n"
            f"5. Final sentence: \"As of {today}.\"\n\n"
            "## REQUIRED ENTITIES (use these exact names where applicable)\n"
            "- \"the Federal Reserve\"\n"
            "- \"the European Central Bank\" (if relevant)\n"
            "- \"the SEC\" (if relevant)\n"
            "- \"Federal Reserve Chair [name]\" — only if a name appears in the headlines; "
            "omit if uncertain\n"
            "- Macro theme labels: \"the AI capital expenditure cycle\", "
            "\"China export restriction risks\", \"commercial real estate stress\", "
            "\"reshoring and tariff policy\" — use only those relevant to the data\n"
            "- \"The [sector] sector is driven by [theme]\" or "
            "\"The [sector] sector is exposed to [theme]\"\n"
            "- \"The Federal Reserve has [raised / held / cut] rates\" with context\n\n"
            f"## MARKET DATA (as of {today})\n"
            f"{market_data_block}\n\n"
            f"Today's date: {today}. The snapshot must end with \"As of {today}.\""
        )
    else:
        # Minimal fallback — no live market data
        user_content = (
            "Write a macro environment snapshot for US equity markets.\n\n"
            "Note: No real-time market data is available. Write a structurally correct "
            "but brief snapshot covering:\n"
            "1. Monetary policy (present tense, no fabricated rate levels)\n"
            "2. Inflation and growth context\n"
            "3. Two dominant sector-level themes\n"
            "4. One geopolitical or regulatory force\n\n"
            "Do not fabricate specific interest rate levels, index prices, or statistics. "
            f"End with \"As of {today}.\"\n\n"
            f"Today's date: {today}."
        )

    return [
        {"role": "system", "content": MACRO_PROFILE_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
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
        INSERT OR REPLACE INTO kg_seed_log (seed_type, seed_key, seeded_at, run_id, source_id, seed_text)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (seed_type, seed_key, datetime.now(timezone.utc).isoformat(), run_id, source_id, markdown),
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
                fact_pack = await fetch_ticker_fact_pack(conn, ticker, company_name, sector)
                if not fact_pack.grounding_available:
                    logger.warning(
                        "Ticker %s: using minimal ungrounded profile (yfinance unavailable)",
                        ticker,
                    )
                ok = await _seed_single(
                    conn=conn,
                    rag=rag,
                    client=client,
                    seed_type="ticker_profile",
                    seed_key=ticker,
                    messages=_build_ticker_messages(fact_pack),
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
            macro_pack = await fetch_macro_fact_pack()
            if not macro_pack.grounding_available:
                logger.warning("Macro profile: using minimal ungrounded profile (yfinance unavailable)")
            ok = await _seed_single(
                conn=conn,
                rag=rag,
                client=client,
                seed_type="macro_profile",
                seed_key="macro",
                messages=_build_macro_messages(macro_pack),
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

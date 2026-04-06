"""Agent A — Local LLM Context Retriever.

Two-stage retrieval-then-synthesis pipeline:
  Stage 1: Query LightRAG knowledge graph for relevant context.
  Stage 2: Call local Ollama LLM to produce structured JSON analysis.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from typing import Any

import ollama

from pipeline.config import OLLAMA_BASE_URL, OLLAMA_MODEL
from pipeline.db.helpers import get_active_standing_events, store_agent_output
from pipeline.knowledge.graph_ops import query_graph

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Standing-event helpers
# ---------------------------------------------------------------------------

def _get_standing_context(
    conn: sqlite3.Connection,
) -> tuple[list[dict], dict[str, list[dict]]]:
    """Fetch active standing events and build a ticker-to-events mapping.

    Returns:
        (all_events, ticker_map) where ticker_map maps ticker -> list of events
        that include that ticker in affected_tickers.
    """
    events = get_active_standing_events(conn)
    ticker_map: dict[str, list[dict]] = {}
    for ev in events:
        for ticker in ev.get("affected_tickers", []):
            ticker_map.setdefault(ticker, []).append(ev)
    return events, ticker_map


# ---------------------------------------------------------------------------
# Context retrieval
# ---------------------------------------------------------------------------

async def _retrieve_context(rag: Any, query: str, mode: str) -> str:
    """Wrapper around query_graph with error handling."""
    try:
        return await query_graph(rag, query, mode=mode, only_context=True)
    except Exception:
        logger.exception("Failed to retrieve context for query: %s", query)
        return ""


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _build_macro_prompt(context: str) -> list[dict]:
    """Build messages for macro overview synthesis."""
    return [
        {
            "role": "system",
            "content": (
                "You are a financial analyst. Analyze the provided market context and "
                "produce a JSON object with a single key \"macro_overview\" containing "
                "a ~200 word summary of the dominant market narratives and themes. "
                "Respond with ONLY valid JSON."
            ),
        },
        {
            "role": "user",
            "content": f"Market context from the last 48 hours:\n\n{context}",
        },
    ]


def _build_standing_prompt(
    context: str,
    standing_events: list[dict],
) -> list[dict]:
    """Build messages for standing event assessment."""
    events_desc = "\n".join(
        f"- [{ev['canonical_id']}] {ev['summary']}"
        for ev in standing_events
    )
    canonical_ids = [ev["canonical_id"] for ev in standing_events]
    return [
        {
            "role": "system",
            "content": (
                "You are a financial analyst assessing ongoing market conditions. "
                "For each standing event listed, produce a JSON object where each key "
                "is the canonical_id of the event and the value is an object with: "
                "\"still_relevant\" (boolean), \"current_impact\" (string description), "
                "\"affected_tickers_update\" (list of ticker strings). "
                f"The canonical_ids are: {canonical_ids}. "
                "Respond with ONLY valid JSON."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Standing events:\n{events_desc}\n\n"
                f"Retrieved context:\n{context}"
            ),
        },
    ]


def _build_ticker_prompt(
    ticker: str,
    context: str,
    standing_summary: str | None,
) -> list[dict]:
    """Build messages for per-ticker analysis."""
    user_content = f"Context for {ticker}:\n\n{context}"
    if standing_summary:
        user_content += f"\n\nRelevant standing events:\n{standing_summary}"
    return [
        {
            "role": "system",
            "content": (
                f"You are a financial analyst evaluating {ticker}. "
                "Produce a JSON object with exactly these keys: "
                "\"sentiment\" (one of \"bullish\", \"bearish\", \"neutral\"), "
                "\"confidence\" (one of \"strong\", \"moderate\", \"weak\"), "
                "\"key_catalysts\" (list of short strings), "
                "\"narrative\" (string, 100-150 words summarizing the outlook). "
                "Respond with ONLY valid JSON."
            ),
        },
        {
            "role": "user",
            "content": user_content,
        },
    ]


# ---------------------------------------------------------------------------
# Ollama call with retry
# ---------------------------------------------------------------------------

async def _call_ollama(
    messages: list[dict],
    retry: bool = True,
    validator: Any | None = None,
) -> dict | None:
    """Call Ollama chat with JSON format, parse and optionally validate.

    Args:
        messages: Chat messages to send.
        retry: Whether to attempt a corrective retry on failure.
        validator: Optional callable(dict) -> bool. If provided and returns
                   False, the response is treated as invalid (triggers retry).
    """
    try:
        client = ollama.AsyncClient(host=OLLAMA_BASE_URL)
        response = await client.chat(
            model=OLLAMA_MODEL,
            messages=messages,
            format="json",
        )
        content = response["message"]["content"]
        data = json.loads(content)
        if validator and not validator(data):
            raise ValueError("Schema validation failed")
        return data
    except Exception:
        logger.warning("Ollama call failed or returned invalid response")
        if not retry:
            return None

    # Retry with corrective prompt
    corrective = {
        "role": "user",
        "content": (
            "Your previous response was not valid JSON or did not match the "
            "requested schema. Please respond with ONLY valid JSON matching "
            "the requested schema exactly."
        ),
    }
    return await _call_ollama(
        messages + [corrective], retry=False, validator=validator
    )


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------

def _validate_ticker_output(data: dict) -> bool:
    """Validate per-ticker output schema."""
    if not isinstance(data, dict):
        return False
    if data.get("sentiment") not in {"bullish", "bearish", "neutral"}:
        return False
    if data.get("confidence") not in {"strong", "moderate", "weak"}:
        return False
    if not isinstance(data.get("key_catalysts"), list):
        return False
    if not isinstance(data.get("narrative"), str):
        return False
    return True


def _validate_standing_output(data: dict) -> bool:
    """Validate standing context assessment output schema."""
    if not isinstance(data, dict):
        return False
    for value in data.values():
        if not isinstance(value, dict):
            return False
        if not isinstance(value.get("still_relevant"), bool):
            return False
        if not isinstance(value.get("current_impact"), str):
            return False
        if not isinstance(value.get("affected_tickers_update"), list):
            return False
    return True


# ---------------------------------------------------------------------------
# Degraded output
# ---------------------------------------------------------------------------

def _degraded_ticker_output(ticker: str) -> dict:
    """Return a safe neutral/weak fallback for a ticker."""
    return {
        "sentiment": "neutral",
        "confidence": "weak",
        "key_catalysts": [],
        "narrative": "Extraction failed — using degraded output.",
    }


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

async def run_agent_a(
    conn: sqlite3.Connection,
    rag: Any,
    tickers: list[str],
    run_id: int,
    flagged_tickers: list[str] | None = None,
) -> dict:
    """Run Agent A: retrieve context from LightRAG, synthesise via Ollama.

    Args:
        conn: SQLite connection.
        rag: Initialized LightRAG instance.
        tickers: Full list of tracked tickers.
        run_id: Current pipeline run ID.
        flagged_tickers: If provided (re-query mode), only re-run these tickers.

    Returns:
        Output dict with macro_overview, standing_context_assessment, per_ticker.
    """
    standing_events, ticker_event_map = _get_standing_context(conn)

    # ------------------------------------------------------------------
    # Re-query mode: only re-run flagged tickers, return partial update
    # ------------------------------------------------------------------
    if flagged_tickers is not None:
        per_ticker: dict[str, dict] = {}
        for ticker in flagged_tickers:
            standing_summary = _standing_summary_for_ticker(
                ticker, ticker_event_map
            )
            ctx = await _retrieve_context(
                rag,
                f"What events, catalysts, or sentiment shifts affect {ticker}?"
                + (f" {standing_summary}" if standing_summary else ""),
                mode="local",
            )
            data = await _call_ollama(
                _build_ticker_prompt(ticker, ctx, standing_summary),
                validator=_validate_ticker_output,
            )
            if data is not None:
                per_ticker[ticker] = data
            else:
                per_ticker[ticker] = _degraded_ticker_output(ticker)

        output = {"per_ticker": per_ticker}
        store_agent_output(conn, run_id, "A", output)
        return output

    # ------------------------------------------------------------------
    # Full run
    # ------------------------------------------------------------------

    # 1. Macro overview
    macro_ctx = await _retrieve_context(
        rag,
        "What are the dominant market narratives in the last 48 hours?",
        mode="global",
    )
    macro_data = await _call_ollama(_build_macro_prompt(macro_ctx))
    macro_overview = (
        macro_data.get("macro_overview", "")
        if isinstance(macro_data, dict)
        else ""
    )

    # 2. Standing context assessment
    standing_context_assessment: dict = {}
    if standing_events:
        events_query = (
            "How do the following ongoing conditions affect the current picture? "
            + " ".join(ev["summary"] for ev in standing_events)
        )
        standing_ctx = await _retrieve_context(rag, events_query, mode="global")
        standing_data = await _call_ollama(
            _build_standing_prompt(standing_ctx, standing_events),
            validator=_validate_standing_output,
        )
        if standing_data is not None:
            standing_context_assessment = standing_data
        else:
            standing_context_assessment = {}

    # 3. Per-ticker analysis
    per_ticker = {}
    for ticker in tickers:
        standing_summary = _standing_summary_for_ticker(ticker, ticker_event_map)
        query = f"What events, catalysts, or sentiment shifts affect {ticker}?"
        if standing_summary:
            query += f" {standing_summary}"
        ctx = await _retrieve_context(rag, query, mode="local")
        data = await _call_ollama(
            _build_ticker_prompt(ticker, ctx, standing_summary),
            validator=_validate_ticker_output,
        )
        if data is not None:
            per_ticker[ticker] = data
        else:
            per_ticker[ticker] = _degraded_ticker_output(ticker)

    # 4. Assemble and store
    output = {
        "macro_overview": macro_overview,
        "standing_context_assessment": standing_context_assessment,
        "per_ticker": per_ticker,
    }
    store_agent_output(conn, run_id, "A", output)
    return output


def _standing_summary_for_ticker(
    ticker: str,
    ticker_event_map: dict[str, list[dict]],
) -> str | None:
    """Build a short standing-event summary string for a ticker, or None."""
    events = ticker_event_map.get(ticker)
    if not events:
        return None
    return "; ".join(
        f"[{ev['canonical_id']}] {ev['summary']}" for ev in events
    )

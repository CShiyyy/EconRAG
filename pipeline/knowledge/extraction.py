"""Extraction orchestrator: markdown -> Ollama LLM call -> raw JSON -> validate -> insert."""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass, field

import httpx
from lightrag import LightRAG

from pipeline.config import (
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    EXTRACTION_TEMPERATURE,
    EXTRACTION_MAX_TOKENS,
)
from pipeline.ingestion.models import ParsedContent
from pipeline.knowledge.extraction_prompt import (
    CORRECTIVE_PROMPT,
    EXTRACTION_SYSTEM_PROMPT,
    EXTRACTION_USER_TEMPLATE,
)
from pipeline.knowledge.graph_ops import insert_validated_data
from pipeline.knowledge.validator import ValidationResult, validate_extraction

logger = logging.getLogger(__name__)


@dataclass
class BatchExtractionResult:
    documents_processed: int = 0
    documents_failed: int = 0
    total_entities: int = 0
    total_relationships: int = 0
    untyped_edges: int = 0
    new_canonical_entries: int = 0
    validation_results: list[ValidationResult] = field(default_factory=list)


async def _call_ollama(
    system_prompt: str,
    user_message: str,
) -> str:
    """Call Ollama chat API and return the raw response text."""
    async with httpx.AsyncClient(timeout=300) as client:
        resp = await client.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                "stream": False,
                "options": {
                    "temperature": EXTRACTION_TEMPERATURE,
                    "num_predict": EXTRACTION_MAX_TOKENS,
                },
                "format": "json",
            },
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]


def _parse_extraction_json(raw_text: str) -> dict | None:
    """Parse LLM output as JSON extraction result. Returns None on failure."""
    try:
        parsed = json.loads(raw_text)
        if "entities" in parsed and "relationships" in parsed:
            return parsed
        return None
    except (json.JSONDecodeError, TypeError):
        return None


async def extract_from_markdown(
    markdown: str,
) -> dict | None:
    """Run LLM extraction on a markdown chunk using the custom prompt.

    Calls Ollama with the extraction prompt. Retries once on malformed JSON.
    Returns parsed extraction dict or None on failure.
    """
    user_msg = EXTRACTION_USER_TEMPLATE.format(markdown_chunk=markdown)

    # First attempt
    raw = await _call_ollama(EXTRACTION_SYSTEM_PROMPT, user_msg)
    result = _parse_extraction_json(raw)
    if result is not None:
        return result

    # Retry with corrective prompt
    logger.warning("First extraction attempt returned malformed JSON, retrying")
    corrective_msg = CORRECTIVE_PROMPT.format(markdown_chunk=markdown)
    raw = await _call_ollama(EXTRACTION_SYSTEM_PROMPT, corrective_msg)
    result = _parse_extraction_json(raw)
    if result is None:
        logger.error("Extraction failed after retry: could not parse JSON")
    return result


async def process_content_batch(
    parsed_contents: list[ParsedContent],
    conn: sqlite3.Connection,
    rag: LightRAG,
    run_id: int,
) -> BatchExtractionResult:
    """Process a batch of parsed content through the full extraction pipeline.

    For each successful ParsedContent:
    1. extract_from_markdown() — LLM extraction via Ollama
    2. validate_extraction() — type checking, canonical resolution
    3. insert_validated_data() — graph insertion via LightRAG

    Args:
        parsed_contents: Output from Phase 3 ingestion (Crawl4AI parsed markdown).
        conn: SQLite connection for canonical registry lookups.
        rag: Initialized LightRAG instance.
        run_id: Current pipeline run ID.

    Returns:
        BatchExtractionResult with aggregate stats.
    """
    batch_result = BatchExtractionResult()

    for content in parsed_contents:
        if not content.success or not content.markdown:
            batch_result.documents_failed += 1
            continue

        # Stage A: LLM extraction
        raw_extraction = await extract_from_markdown(content.markdown)
        if raw_extraction is None:
            batch_result.documents_failed += 1
            continue

        # Stage B: Validate and canonicalize
        validation = validate_extraction(raw_extraction, conn, run_id)
        batch_result.validation_results.append(validation)
        batch_result.total_entities += len(validation.entities)
        batch_result.total_relationships += len(validation.relationships)
        batch_result.untyped_edges += len(validation.untyped_edges)

        # Stage C: Insert into graph
        await insert_validated_data(rag, validation, content.markdown, content.url)
        batch_result.documents_processed += 1

    conn.commit()
    return batch_result

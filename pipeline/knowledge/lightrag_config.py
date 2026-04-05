"""LightRAG instance factory and lifecycle management."""

from __future__ import annotations

from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path
from typing import AsyncIterator

from lightrag import LightRAG
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from lightrag.utils import EmbeddingFunc

from pipeline.config import (
    LIGHTRAG_STORAGE_DIR,
    OLLAMA_BASE_URL,
    OLLAMA_EMBED_DIM,
    OLLAMA_EMBED_MODEL,
    OLLAMA_MODEL,
)


async def get_rag_instance(working_dir: Path | None = None) -> LightRAG:
    """Create and initialize a LightRAG instance with NanoVectorDB + NetworkX.

    Args:
        working_dir: Override storage directory (use tmp_path in tests).
                     Defaults to LIGHTRAG_STORAGE_DIR from config.

    Returns:
        Initialized LightRAG instance. Caller must call finalize_storages() when done.
    """
    storage_dir = str(working_dir or LIGHTRAG_STORAGE_DIR)

    rag = LightRAG(
        working_dir=storage_dir,
        graph_storage="NetworkXStorage",
        vector_storage="NanoVectorDBStorage",
        kv_storage="JsonKVStorage",
        llm_model_func=ollama_model_complete,
        llm_model_name=OLLAMA_MODEL,
        llm_model_kwargs={
            "host": OLLAMA_BASE_URL,
            "options": {"num_ctx": 8192},
            "timeout": 300,
        },
        embedding_func=EmbeddingFunc(
            embedding_dim=OLLAMA_EMBED_DIM,
            max_token_size=8192,
            func=partial(
                ollama_embed.func,
                embed_model=OLLAMA_EMBED_MODEL,
                host=OLLAMA_BASE_URL,
            ),
        ),
        addon_params={
            "language": "English",
            "entity_types": [
                "company", "person", "sector", "index",
                "product", "event", "macro_theme", "institution",
            ],
        },
        entity_extract_max_gleaning=1,
        chunk_token_size=1200,
        chunk_overlap_token_size=100,
    )

    await rag.initialize_storages()
    return rag


@asynccontextmanager
async def rag_session(working_dir: Path | None = None) -> AsyncIterator[LightRAG]:
    """Async context manager for LightRAG lifecycle.

    Usage:
        async with rag_session() as rag:
            await rag.aquery(...)
    """
    rag = await get_rag_instance(working_dir)
    try:
        yield rag
    finally:
        await rag.finalize_storages()

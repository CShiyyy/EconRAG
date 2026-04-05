"""Knowledge graph: LightRAG integration, extraction, canonicalization, pruning."""

from pipeline.knowledge.canonical import resolve_entity, resolve_entities
from pipeline.knowledge.extraction import (
    BatchExtractionResult,
    extract_from_markdown,
    process_content_batch,
)
from pipeline.knowledge.graph_ops import (
    inject_correlation_edges,
    insert_validated_data,
    parse_edge_metadata,
    query_graph,
)
from pipeline.knowledge.lightrag_config import get_rag_instance, rag_session
from pipeline.knowledge.pruner import PruneResult, prune_expired_edges
from pipeline.knowledge.validator import (
    ValidatedEntity,
    ValidatedRelationship,
    ValidationResult,
    validate_extraction,
)

__all__ = [
    "resolve_entity",
    "resolve_entities",
    "extract_from_markdown",
    "process_content_batch",
    "BatchExtractionResult",
    "inject_correlation_edges",
    "insert_validated_data",
    "parse_edge_metadata",
    "query_graph",
    "get_rag_instance",
    "rag_session",
    "prune_expired_edges",
    "PruneResult",
    "validate_extraction",
    "ValidatedEntity",
    "ValidatedRelationship",
    "ValidationResult",
]

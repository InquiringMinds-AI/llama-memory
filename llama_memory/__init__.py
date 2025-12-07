"""
llama-memory: Local vector memory for AI assistants.

Uses llama.cpp for embeddings and sqlite-vec for similarity search.
Fully local, no external APIs required.

v2.5.0 adds:
- Auto-summarization on store
- Token-budgeted recall with dual-response pattern
- Named contexts
- Entity extraction (Phase 2)
- Document ingestion (Phase 3)
"""

__version__ = "2.5.0"

from .config import Config, get_config, ensure_directories
from .memory import (
    Memory,
    MemoryStore,
    get_store,
    store,
    search,
    list_memories,
    get_memory,
    update,
    supersede,
    stats,
)
from .embeddings import get_embedding, EmbeddingGenerator, EmbeddingError
from .database import Database, get_database

# v2.5 additions
from .summarizer import (
    Summarizer,
    HeuristicSummarizer,
    ExtractiveSummarizer,
    LLMSummarizer,
    auto_summarize,
    estimate_tokens,
    get_summarizer,
)
from .budget import (
    TokenBudgetManager,
    RecallResponse,
    MemorySummary,
    budgeted_recall,
)
from .entities import (
    Entity,
    EntityExtractor,
    EntityStore,
    extract_entities,
    get_extractor,
)
from .ingester import (
    Chunk,
    SmartChunker,
    Ingester,
    ingest_document,
    get_ingester,
)
from .capture import (
    CaptureConfig,
    CaptureEvent,
    CaptureEngine,
    get_capture_engine,
    set_capture_mode,
    get_capture_status,
)

__all__ = [
    # Config
    "Config",
    "get_config",
    "ensure_directories",
    # Memory
    "Memory",
    "MemoryStore",
    "get_store",
    "store",
    "search",
    "list_memories",
    "get_memory",
    "update",
    "supersede",
    "stats",
    # Embeddings
    "get_embedding",
    "EmbeddingGenerator",
    "EmbeddingError",
    # Database
    "Database",
    "get_database",
    # v2.5: Summarization
    "Summarizer",
    "HeuristicSummarizer",
    "ExtractiveSummarizer",
    "LLMSummarizer",
    "auto_summarize",
    "estimate_tokens",
    "get_summarizer",
    # v2.5: Token Budgeting
    "TokenBudgetManager",
    "RecallResponse",
    "MemorySummary",
    "budgeted_recall",
    # v2.5: Entity Extraction
    "Entity",
    "EntityExtractor",
    "EntityStore",
    "extract_entities",
    "get_extractor",
    # v2.5: Document Ingestion
    "Chunk",
    "SmartChunker",
    "Ingester",
    "ingest_document",
    "get_ingester",
    # v2.5: Auto-Capture
    "CaptureConfig",
    "CaptureEvent",
    "CaptureEngine",
    "get_capture_engine",
    "set_capture_mode",
    "get_capture_status",
]

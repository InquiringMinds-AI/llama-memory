"""
llama-memory: Local vector memory for AI assistants.

Uses llama.cpp for embeddings and sqlite-vec for similarity search.
Fully local, no external APIs required.

v2.6.0 adds:
- Full config.yaml system with configurable weights
- Enhanced entity extraction with name heuristics
- Entity match scoring bonus in hybrid ranking
- PDF ingestion support (optional: pip install llama-memory[pdf])

v2.5.0 added:
- Auto-summarization on store
- Token-budgeted recall with dual-response pattern
- Named contexts
- Entity extraction
- Document ingestion
"""

__version__ = "2.6.0"

from .config import (
    Config,
    get_config,
    reload_config,
    set_config,
    ensure_directories,
    ScoringConfig,
    EntityConfig,
    IngestionConfig,
    BudgetConfig,
    DecayConfig,
)
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
    ingest_pdf,
    is_pdf_available,
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
    "reload_config",
    "set_config",
    "ensure_directories",
    # v2.6 Config dataclasses
    "ScoringConfig",
    "EntityConfig",
    "IngestionConfig",
    "BudgetConfig",
    "DecayConfig",
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
    # v2.6: PDF Ingestion
    "ingest_pdf",
    "is_pdf_available",
    # v2.5: Auto-Capture
    "CaptureConfig",
    "CaptureEvent",
    "CaptureEngine",
    "get_capture_engine",
    "set_capture_mode",
    "get_capture_status",
]

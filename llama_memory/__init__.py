"""
llama-memory: Local vector memory for AI assistants.

Uses llama.cpp for embeddings and sqlite-vec for similarity search.
Fully local, no external APIs required.
"""

__version__ = "1.0.0"

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
]

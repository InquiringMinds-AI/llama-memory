"""
Database schema and migrations for llama-memory.
"""

from __future__ import annotations

import sqlite3
import struct
import json
from pathlib import Path
from typing import Optional
from datetime import datetime
from .config import Config, get_config

SCHEMA_VERSION = 1

SCHEMA = """
-- Main memories table
CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY,
    type TEXT NOT NULL DEFAULT 'fact',
    content TEXT NOT NULL,
    summary TEXT,
    project TEXT,
    tags TEXT,  -- JSON array

    created_at INTEGER NOT NULL,
    updated_at INTEGER,
    accessed_at INTEGER,
    access_count INTEGER DEFAULT 0,

    importance INTEGER DEFAULT 5 CHECK (importance >= 1 AND importance <= 10),
    retention TEXT DEFAULT 'long-term' CHECK (retention IN ('permanent', 'long-term', 'medium', 'short', 'session')),
    superseded_by INTEGER REFERENCES memories(id),
    archived INTEGER DEFAULT 0,

    embedding_model TEXT,

    CONSTRAINT valid_type CHECK (type IN ('fact', 'decision', 'event', 'entity', 'context', 'procedure'))
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(type);
CREATE INDEX IF NOT EXISTS idx_memories_project ON memories(project);
CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at);
CREATE INDEX IF NOT EXISTS idx_memories_archived ON memories(archived);
CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(importance);
CREATE INDEX IF NOT EXISTS idx_memories_retention ON memories(retention);
CREATE INDEX IF NOT EXISTS idx_memories_superseded ON memories(superseded_by);

-- Full-text search on content
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    content,
    summary,
    content='memories',
    content_rowid='id'
);

-- Triggers to keep FTS in sync
CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(rowid, content, summary) VALUES (new.id, new.content, new.summary);
END;

CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content, summary) VALUES ('delete', old.id, old.content, old.summary);
END;

CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content, summary) VALUES ('delete', old.id, old.content, old.summary);
    INSERT INTO memories_fts(rowid, content, summary) VALUES (new.id, new.content, new.summary);
END;

-- Metadata table
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT
);

-- Embedding model registry (for migrations)
CREATE TABLE IF NOT EXISTS embedding_models (
    version TEXT PRIMARY KEY,
    model_path TEXT,
    dimensions INTEGER,
    created_at INTEGER,
    is_current INTEGER DEFAULT 0
);

-- Audit log
CREATE TABLE IF NOT EXISTS memory_log (
    id INTEGER PRIMARY KEY,
    memory_id INTEGER,
    action TEXT NOT NULL,
    timestamp INTEGER NOT NULL,
    details TEXT  -- JSON
);
"""

VECTOR_TABLE_SQL = """
CREATE VIRTUAL TABLE IF NOT EXISTS memory_embeddings USING vec0(
    memory_id INTEGER PRIMARY KEY,
    embedding float[{dimensions}]
);
"""


class Database:
    """Database connection and operations."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or get_config()
        self._conn: Optional[sqlite3.Connection] = None

    def _get_connection(self) -> sqlite3.Connection:
        """Get or create database connection."""
        if self._conn is None:
            # Ensure directory exists
            self.config.database_path.parent.mkdir(parents=True, exist_ok=True)

            self._conn = sqlite3.connect(str(self.config.database_path))
            self._conn.row_factory = sqlite3.Row

            # Enable foreign keys
            self._conn.execute("PRAGMA foreign_keys = ON")

            # WAL mode for better concurrency
            self._conn.execute("PRAGMA journal_mode = WAL")

            # Load sqlite-vec extension
            if self.config.sqlite_vec_path and self.config.sqlite_vec_path.exists():
                self._conn.enable_load_extension(True)
                ext_path = str(self.config.sqlite_vec_path).replace(".so", "").replace(".dylib", "").replace(".dll", "")
                self._conn.load_extension(ext_path)
                self._conn.enable_load_extension(False)

        return self._conn

    @property
    def conn(self) -> sqlite3.Connection:
        return self._get_connection()

    def close(self):
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def initialize(self):
        """Initialize database schema."""
        conn = self.conn

        # Create main schema
        conn.executescript(SCHEMA)

        # Create vector table with correct dimensions
        vector_sql = VECTOR_TABLE_SQL.format(dimensions=self.config.embedding_dimensions)
        try:
            conn.execute(vector_sql)
        except sqlite3.OperationalError as e:
            if "already exists" not in str(e):
                raise

        # Set metadata
        now = int(datetime.now().timestamp())
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES ('schema_version', ?)",
            (str(SCHEMA_VERSION),)
        )
        conn.execute(
            "INSERT OR IGNORE INTO meta (key, value) VALUES ('created_at', ?)",
            (str(now),)
        )

        # Register current embedding model
        conn.execute("""
            INSERT OR REPLACE INTO embedding_models (version, model_path, dimensions, created_at, is_current)
            VALUES (?, ?, ?, ?, 1)
        """, (
            self.config.embedding_model_version,
            str(self.config.embedding_model) if self.config.embedding_model else None,
            self.config.embedding_dimensions,
            now
        ))

        # Unset is_current for other models
        conn.execute("""
            UPDATE embedding_models SET is_current = 0
            WHERE version != ? AND is_current = 1
        """, (self.config.embedding_model_version,))

        conn.commit()

    def get_schema_version(self) -> int:
        """Get current schema version."""
        try:
            result = self.conn.execute(
                "SELECT value FROM meta WHERE key = 'schema_version'"
            ).fetchone()
            return int(result[0]) if result else 0
        except:
            return 0

    def needs_migration(self) -> bool:
        """Check if schema needs migration."""
        return self.get_schema_version() < SCHEMA_VERSION

    def get_current_embedding_model(self) -> Optional[str]:
        """Get the current embedding model version."""
        result = self.conn.execute(
            "SELECT version FROM embedding_models WHERE is_current = 1"
        ).fetchone()
        return result[0] if result else None

    def needs_reembedding(self) -> bool:
        """Check if memories need re-embedding due to model change."""
        current = self.get_current_embedding_model()
        if not current:
            return False

        # Check if any memories were embedded with a different model
        result = self.conn.execute("""
            SELECT COUNT(*) FROM memories
            WHERE embedding_model IS NOT NULL
            AND embedding_model != ?
            AND archived = 0
        """, (current,)).fetchone()

        return result[0] > 0

    def get_stats(self) -> dict:
        """Get database statistics."""
        conn = self.conn

        total = conn.execute(
            "SELECT COUNT(*) FROM memories WHERE archived = 0"
        ).fetchone()[0]

        by_type = dict(conn.execute(
            "SELECT type, COUNT(*) FROM memories WHERE archived = 0 GROUP BY type"
        ).fetchall())

        by_project = dict(conn.execute(
            "SELECT COALESCE(project, 'global'), COUNT(*) FROM memories WHERE archived = 0 GROUP BY project"
        ).fetchall())

        archived = conn.execute(
            "SELECT COUNT(*) FROM memories WHERE archived = 1"
        ).fetchone()[0]

        # Database file size
        db_size = self.config.database_path.stat().st_size if self.config.database_path.exists() else 0

        return {
            "total": total,
            "archived": archived,
            "by_type": by_type,
            "by_project": by_project,
            "database_size_bytes": db_size,
            "schema_version": self.get_schema_version(),
            "embedding_model": self.get_current_embedding_model(),
        }

    def integrity_check(self) -> list[str]:
        """Run integrity checks, return list of issues."""
        issues = []

        # SQLite integrity check
        result = self.conn.execute("PRAGMA integrity_check").fetchone()
        if result[0] != "ok":
            issues.append(f"SQLite integrity check failed: {result[0]}")

        # Check for orphaned embeddings
        orphaned = self.conn.execute("""
            SELECT COUNT(*) FROM memory_embeddings e
            LEFT JOIN memories m ON e.memory_id = m.id
            WHERE m.id IS NULL
        """).fetchone()[0]
        if orphaned > 0:
            issues.append(f"Found {orphaned} orphaned embeddings")

        # Check for memories without embeddings
        missing = self.conn.execute("""
            SELECT COUNT(*) FROM memories m
            LEFT JOIN memory_embeddings e ON m.id = e.memory_id
            WHERE e.memory_id IS NULL AND m.archived = 0
        """).fetchone()[0]
        if missing > 0:
            issues.append(f"Found {missing} memories without embeddings")

        return issues


def embedding_to_blob(embedding: list[float]) -> bytes:
    """Convert embedding list to blob for sqlite-vec."""
    return struct.pack(f'{len(embedding)}f', *embedding)


def blob_to_embedding(blob: bytes, dimensions: int) -> list[float]:
    """Convert blob back to embedding list."""
    return list(struct.unpack(f'{dimensions}f', blob))


def get_database(config: Optional[Config] = None) -> Database:
    """Get database instance."""
    return Database(config)

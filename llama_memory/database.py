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

SCHEMA_VERSION = 8

# Decay system constants
DECAY_START_DAYS = 120  # Days without access before decay starts affecting ranking
DECAY_ARCHIVE_DAYS = 180  # Days without access before auto-archive (for decayable memories)

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
    session_id TEXT,  -- Track which session created this memory

    -- v4 additions
    source TEXT DEFAULT 'cli',  -- Where memory came from: cli, mcp, api, import, auto
    metadata TEXT,  -- JSON blob for future extensibility

    -- v5 additions
    confidence REAL DEFAULT 1.0 CHECK (confidence >= 0.0 AND confidence <= 1.0),  -- How certain is this memory (0.0-1.0)
    parent_id INTEGER REFERENCES memories(id),  -- Hierarchical parent memory
    topic TEXT,  -- Auto-assigned topic/cluster

    -- v6 additions
    protected INTEGER DEFAULT 0,  -- Manually protected from decay (1 = protected)

    -- v7 additions
    token_count INTEGER,  -- Estimated token count for budgeting
    context TEXT,  -- Named context (work, personal, etc.)
    entities TEXT,  -- JSON: extracted entities
    chunk_of INTEGER REFERENCES memories(id),  -- FK if chunk of larger document
    chunk_index INTEGER,  -- Position in document if chunked
    file_path TEXT,  -- Source file path if ingested

    CONSTRAINT valid_type CHECK (type IN ('fact', 'decision', 'event', 'entity', 'context', 'procedure', 'session'))
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(type);
CREATE INDEX IF NOT EXISTS idx_memories_project ON memories(project);
CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at);
CREATE INDEX IF NOT EXISTS idx_memories_archived ON memories(archived);
CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(importance);
CREATE INDEX IF NOT EXISTS idx_memories_retention ON memories(retention);
CREATE INDEX IF NOT EXISTS idx_memories_superseded ON memories(superseded_by);
CREATE INDEX IF NOT EXISTS idx_memories_session ON memories(session_id);
CREATE INDEX IF NOT EXISTS idx_memories_source ON memories(source);
CREATE INDEX IF NOT EXISTS idx_memories_parent ON memories(parent_id);
CREATE INDEX IF NOT EXISTS idx_memories_topic ON memories(topic);
CREATE INDEX IF NOT EXISTS idx_memories_context ON memories(context);
CREATE INDEX IF NOT EXISTS idx_memories_chunk_of ON memories(chunk_of);

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

-- Explicit links between memories
CREATE TABLE IF NOT EXISTS memory_links (
    id INTEGER PRIMARY KEY,
    source_id INTEGER NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    target_id INTEGER NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    link_type TEXT DEFAULT 'related',  -- related, depends_on, contradicts, etc.
    note TEXT,  -- Optional note about the relationship
    weight REAL DEFAULT 1.0,  -- Relationship strength (0.0-1.0)
    metadata TEXT,  -- JSON blob for future extensibility
    created_at INTEGER NOT NULL,
    UNIQUE(source_id, target_id)
);

CREATE INDEX IF NOT EXISTS idx_links_source ON memory_links(source_id);
CREATE INDEX IF NOT EXISTS idx_links_target ON memory_links(target_id);

-- Embedding versions table (for keeping old embeddings during migration)
CREATE TABLE IF NOT EXISTS embedding_versions (
    id INTEGER PRIMARY KEY,
    memory_id INTEGER NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    model_version TEXT NOT NULL,
    embedding BLOB NOT NULL,
    created_at INTEGER NOT NULL,
    is_current INTEGER DEFAULT 1,
    UNIQUE(memory_id, model_version)
);

CREATE INDEX IF NOT EXISTS idx_embedding_versions_memory ON embedding_versions(memory_id);
CREATE INDEX IF NOT EXISTS idx_embedding_versions_model ON embedding_versions(model_version);
CREATE INDEX IF NOT EXISTS idx_embedding_versions_current ON embedding_versions(is_current);

-- v5: Search history for tracking query patterns
CREATE TABLE IF NOT EXISTS search_history (
    id INTEGER PRIMARY KEY,
    query TEXT NOT NULL,
    query_type TEXT DEFAULT 'semantic',  -- semantic, text, recall
    result_count INTEGER,
    top_result_id INTEGER REFERENCES memories(id),
    session_id TEXT,
    created_at INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_search_history_query ON search_history(query);
CREATE INDEX IF NOT EXISTS idx_search_history_created ON search_history(created_at);
CREATE INDEX IF NOT EXISTS idx_search_history_session ON search_history(session_id);

-- v5: Topics/clusters for automatic categorization
CREATE TABLE IF NOT EXISTS topics (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    description TEXT,
    parent_topic_id INTEGER REFERENCES topics(id),
    memory_count INTEGER DEFAULT 0,
    created_at INTEGER NOT NULL,
    updated_at INTEGER
);

CREATE INDEX IF NOT EXISTS idx_topics_name ON topics(name);
CREATE INDEX IF NOT EXISTS idx_topics_parent ON topics(parent_topic_id);

-- v5: Potential conflicts between memories
CREATE TABLE IF NOT EXISTS memory_conflicts (
    id INTEGER PRIMARY KEY,
    memory1_id INTEGER NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    memory2_id INTEGER NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    conflict_type TEXT DEFAULT 'potential',  -- potential, confirmed, resolved
    similarity REAL,  -- How similar the content is
    note TEXT,
    resolved_by INTEGER REFERENCES memories(id),  -- Which memory was chosen as correct
    created_at INTEGER NOT NULL,
    resolved_at INTEGER,
    UNIQUE(memory1_id, memory2_id)
);

CREATE INDEX IF NOT EXISTS idx_conflicts_memory1 ON memory_conflicts(memory1_id);
CREATE INDEX IF NOT EXISTS idx_conflicts_memory2 ON memory_conflicts(memory2_id);
CREATE INDEX IF NOT EXISTS idx_conflicts_type ON memory_conflicts(conflict_type);

-- v7: Entities table for extracted entities
CREATE TABLE IF NOT EXISTS entities (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    normalized TEXT NOT NULL,
    type TEXT NOT NULL CHECK (type IN ('person', 'project', 'tool', 'concept', 'organization', 'location')),
    first_seen INTEGER NOT NULL,
    last_seen INTEGER NOT NULL,
    mention_count INTEGER DEFAULT 1,
    metadata TEXT,
    UNIQUE(normalized, type)
);

CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(type);
CREATE INDEX IF NOT EXISTS idx_entities_normalized ON entities(normalized);

-- v7: Junction table for memory-entity relationships
CREATE TABLE IF NOT EXISTS memory_entities (
    memory_id INTEGER NOT NULL,
    entity_id INTEGER NOT NULL,
    PRIMARY KEY (memory_id, entity_id),
    FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE,
    FOREIGN KEY (entity_id) REFERENCES entities(id) ON DELETE CASCADE
);

-- v7: Named contexts table
CREATE TABLE IF NOT EXISTS contexts (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    description TEXT,
    created_at INTEGER NOT NULL,
    memory_count INTEGER DEFAULT 0,
    is_default INTEGER DEFAULT 0
);

-- v7: Capture log for auto-capture mode
CREATE TABLE IF NOT EXISTS capture_log (
    id INTEGER PRIMARY KEY,
    event_type TEXT NOT NULL,
    event_data TEXT NOT NULL,
    memory_id INTEGER REFERENCES memories(id),
    captured_at INTEGER NOT NULL,
    processed INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_capture_log_type ON capture_log(event_type);
CREATE INDEX IF NOT EXISTS idx_capture_log_processed ON capture_log(processed);
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

    def migrate(self):
        """Run any pending migrations."""
        current_version = self.get_schema_version()
        conn = self.conn

        if current_version < 2:
            # Migration v1 -> v2: Add session_id column
            try:
                conn.execute("ALTER TABLE memories ADD COLUMN session_id TEXT")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_session ON memories(session_id)")
                conn.commit()
            except sqlite3.OperationalError as e:
                if "duplicate column" not in str(e).lower():
                    raise

        if current_version < 3:
            # Migration v2 -> v3: Add memory_links table
            try:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS memory_links (
                        id INTEGER PRIMARY KEY,
                        source_id INTEGER NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                        target_id INTEGER NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                        link_type TEXT DEFAULT 'related',
                        note TEXT,
                        created_at INTEGER NOT NULL,
                        UNIQUE(source_id, target_id)
                    )
                """)
                conn.execute("CREATE INDEX IF NOT EXISTS idx_links_source ON memory_links(source_id)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_links_target ON memory_links(target_id)")
                conn.commit()
            except sqlite3.OperationalError as e:
                if "already exists" not in str(e).lower():
                    raise

        if current_version < 4:
            # Migration v3 -> v4: Add source, metadata to memories; weight, metadata to links; embedding_versions table
            try:
                # Add source column to memories
                conn.execute("ALTER TABLE memories ADD COLUMN source TEXT DEFAULT 'cli'")
            except sqlite3.OperationalError as e:
                if "duplicate column" not in str(e).lower():
                    raise

            try:
                # Add metadata column to memories
                conn.execute("ALTER TABLE memories ADD COLUMN metadata TEXT")
            except sqlite3.OperationalError as e:
                if "duplicate column" not in str(e).lower():
                    raise

            try:
                # Add weight column to memory_links
                conn.execute("ALTER TABLE memory_links ADD COLUMN weight REAL DEFAULT 1.0")
            except sqlite3.OperationalError as e:
                if "duplicate column" not in str(e).lower():
                    raise

            try:
                # Add metadata column to memory_links
                conn.execute("ALTER TABLE memory_links ADD COLUMN metadata TEXT")
            except sqlite3.OperationalError as e:
                if "duplicate column" not in str(e).lower():
                    raise

            # Create embedding_versions table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS embedding_versions (
                    id INTEGER PRIMARY KEY,
                    memory_id INTEGER NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                    model_version TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    created_at INTEGER NOT NULL,
                    is_current INTEGER DEFAULT 1,
                    UNIQUE(memory_id, model_version)
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_embedding_versions_memory ON embedding_versions(memory_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_embedding_versions_model ON embedding_versions(model_version)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_embedding_versions_current ON embedding_versions(is_current)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_source ON memories(source)")

            conn.commit()

        if current_version < 5:
            # Migration v4 -> v5: Add confidence, parent_id, topic to memories; search_history, topics, memory_conflicts tables
            try:
                conn.execute("ALTER TABLE memories ADD COLUMN confidence REAL DEFAULT 1.0")
            except sqlite3.OperationalError as e:
                if "duplicate column" not in str(e).lower():
                    raise

            try:
                conn.execute("ALTER TABLE memories ADD COLUMN parent_id INTEGER REFERENCES memories(id)")
            except sqlite3.OperationalError as e:
                if "duplicate column" not in str(e).lower():
                    raise

            try:
                conn.execute("ALTER TABLE memories ADD COLUMN topic TEXT")
            except sqlite3.OperationalError as e:
                if "duplicate column" not in str(e).lower():
                    raise

            conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_parent ON memories(parent_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_topic ON memories(topic)")

            # Create search_history table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS search_history (
                    id INTEGER PRIMARY KEY,
                    query TEXT NOT NULL,
                    query_type TEXT DEFAULT 'semantic',
                    result_count INTEGER,
                    top_result_id INTEGER REFERENCES memories(id),
                    session_id TEXT,
                    created_at INTEGER NOT NULL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_search_history_query ON search_history(query)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_search_history_created ON search_history(created_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_search_history_session ON search_history(session_id)")

            # Create topics table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS topics (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL UNIQUE,
                    description TEXT,
                    parent_topic_id INTEGER REFERENCES topics(id),
                    memory_count INTEGER DEFAULT 0,
                    created_at INTEGER NOT NULL,
                    updated_at INTEGER
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_topics_name ON topics(name)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_topics_parent ON topics(parent_topic_id)")

            # Create memory_conflicts table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memory_conflicts (
                    id INTEGER PRIMARY KEY,
                    memory1_id INTEGER NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                    memory2_id INTEGER NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                    conflict_type TEXT DEFAULT 'potential',
                    similarity REAL,
                    note TEXT,
                    resolved_by INTEGER REFERENCES memories(id),
                    created_at INTEGER NOT NULL,
                    resolved_at INTEGER,
                    UNIQUE(memory1_id, memory2_id)
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_conflicts_memory1 ON memory_conflicts(memory1_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_conflicts_memory2 ON memory_conflicts(memory2_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_conflicts_type ON memory_conflicts(conflict_type)")

            conn.commit()

        if current_version < 6:
            # Migration v5 -> v6: Add protected column for decay system
            try:
                conn.execute("ALTER TABLE memories ADD COLUMN protected INTEGER DEFAULT 0")
            except sqlite3.OperationalError as e:
                if "duplicate column" not in str(e).lower():
                    raise

            conn.commit()

        if current_version < 7:
            # Migration v6 -> v7: Add token budgeting, contexts, entities columns
            new_columns = [
                ("token_count", "INTEGER"),
                ("context", "TEXT"),
                ("entities", "TEXT"),
                ("chunk_of", "INTEGER"),
                ("chunk_index", "INTEGER"),
                ("file_path", "TEXT"),
            ]

            for col_name, col_type in new_columns:
                try:
                    conn.execute(f"ALTER TABLE memories ADD COLUMN {col_name} {col_type}")
                except sqlite3.OperationalError as e:
                    if "duplicate column" not in str(e).lower():
                        raise

            # Create new indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_context ON memories(context)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_chunk_of ON memories(chunk_of)")

            # Create entities table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS entities (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    normalized TEXT NOT NULL,
                    type TEXT NOT NULL,
                    first_seen INTEGER NOT NULL,
                    last_seen INTEGER NOT NULL,
                    mention_count INTEGER DEFAULT 1,
                    metadata TEXT,
                    UNIQUE(normalized, type)
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_entities_normalized ON entities(normalized)")

            # Create memory_entities junction table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memory_entities (
                    memory_id INTEGER NOT NULL,
                    entity_id INTEGER NOT NULL,
                    PRIMARY KEY (memory_id, entity_id),
                    FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE,
                    FOREIGN KEY (entity_id) REFERENCES entities(id) ON DELETE CASCADE
                )
            """)

            # Create contexts table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS contexts (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL UNIQUE,
                    description TEXT,
                    created_at INTEGER NOT NULL,
                    memory_count INTEGER DEFAULT 0,
                    is_default INTEGER DEFAULT 0
                )
            """)

            # Create capture_log table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS capture_log (
                    id INTEGER PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    event_data TEXT NOT NULL,
                    memory_id INTEGER,
                    captured_at INTEGER NOT NULL,
                    processed INTEGER DEFAULT 0
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_capture_log_type ON capture_log(event_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_capture_log_processed ON capture_log(processed)")

            # Backfill: Generate summaries and token counts for existing memories
            self._backfill_summaries_and_tokens(conn)

            conn.commit()

        if current_version < 8:
            # Migration v7 -> v8: Add 'session' to valid_type CHECK constraint
            # SQLite doesn't support ALTER CONSTRAINT, so we recreate the table
            self._migrate_v8_session_type(conn)
            conn.commit()

    def _migrate_v8_session_type(self, conn: sqlite3.Connection):
        """Migration v8: Add 'session' to valid memory types."""
        # Disable foreign keys for the migration
        conn.execute("PRAGMA foreign_keys = OFF")

        # Clean up any failed previous attempt
        conn.execute("DROP TABLE IF EXISTS memories_new")

        # Create new table with updated CHECK constraint
        # Note: embedding is stored in vec0 table, not here
        conn.execute("""
            CREATE TABLE memories_new (
                id INTEGER PRIMARY KEY,
                type TEXT NOT NULL DEFAULT 'fact',
                content TEXT NOT NULL,
                summary TEXT,
                project TEXT,
                tags TEXT,
                created_at INTEGER NOT NULL,
                updated_at INTEGER,
                accessed_at INTEGER,
                access_count INTEGER DEFAULT 0,
                importance INTEGER DEFAULT 5,
                retention TEXT DEFAULT 'long-term',
                superseded_by INTEGER REFERENCES memories_new(id),
                archived INTEGER DEFAULT 0,
                embedding_model TEXT,
                session_id TEXT,
                source TEXT DEFAULT 'cli',
                metadata TEXT,
                confidence REAL DEFAULT 1.0,
                parent_id INTEGER,
                topic TEXT,
                protected INTEGER DEFAULT 0,
                token_count INTEGER,
                context TEXT,
                entities TEXT,
                chunk_of INTEGER,
                chunk_index INTEGER,
                file_path TEXT,
                CONSTRAINT valid_type CHECK (type IN ('fact', 'decision', 'event', 'entity', 'context', 'procedure', 'session'))
            )
        """)

        # Copy all data from old table
        conn.execute("""
            INSERT INTO memories_new SELECT * FROM memories
        """)

        # Drop old table
        conn.execute("DROP TABLE memories")

        # Rename new table
        conn.execute("ALTER TABLE memories_new RENAME TO memories")

        # Recreate indexes
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(type)",
            "CREATE INDEX IF NOT EXISTS idx_memories_project ON memories(project)",
            "CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at)",
            "CREATE INDEX IF NOT EXISTS idx_memories_archived ON memories(archived)",
            "CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(importance)",
            "CREATE INDEX IF NOT EXISTS idx_memories_retention ON memories(retention)",
            "CREATE INDEX IF NOT EXISTS idx_memories_superseded ON memories(superseded_by)",
            "CREATE INDEX IF NOT EXISTS idx_memories_session ON memories(session_id)",
            "CREATE INDEX IF NOT EXISTS idx_memories_source ON memories(source)",
            "CREATE INDEX IF NOT EXISTS idx_memories_parent ON memories(parent_id)",
            "CREATE INDEX IF NOT EXISTS idx_memories_topic ON memories(topic)",
            "CREATE INDEX IF NOT EXISTS idx_memories_context ON memories(context)",
            "CREATE INDEX IF NOT EXISTS idx_memories_chunk_of ON memories(chunk_of)",
        ]
        for idx_sql in indexes:
            conn.execute(idx_sql)

        # Recreate FTS table (needs to reference new table)
        conn.execute("DROP TABLE IF EXISTS memories_fts")
        conn.execute("""
            CREATE VIRTUAL TABLE memories_fts USING fts5(
                content,
                content='memories',
                content_rowid='id'
            )
        """)

        # Rebuild FTS index
        conn.execute("INSERT INTO memories_fts(memories_fts) VALUES('rebuild')")

        # Re-enable foreign keys
        conn.execute("PRAGMA foreign_keys = ON")

    def _backfill_summaries_and_tokens(self, conn: sqlite3.Connection):
        """Backfill summaries and token counts for existing memories."""
        from .summarizer import auto_summarize, estimate_tokens

        rows = conn.execute("""
            SELECT id, content, summary FROM memories
            WHERE (summary IS NULL OR summary = '') OR token_count IS NULL
        """).fetchall()

        for row in rows:
            memory_id = row['id']
            content = row['content']
            existing_summary = row['summary']

            # Generate summary if missing
            summary = existing_summary
            if not summary:
                summary = auto_summarize(content, max_words=20)

            # Estimate token count
            token_count = estimate_tokens(content)

            conn.execute("""
                UPDATE memories SET summary = ?, token_count = ? WHERE id = ?
            """, (summary, token_count, memory_id))

    def initialize(self):
        """Initialize database schema."""
        conn = self.conn

        # Check if this is an existing database that needs migration
        needs_migration = False
        try:
            result = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='memories'").fetchone()
            if result:
                needs_migration = True
        except:
            pass

        # Run migrations first for existing databases
        if needs_migration:
            self.migrate()

        # Create main schema (will skip existing tables due to IF NOT EXISTS)
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

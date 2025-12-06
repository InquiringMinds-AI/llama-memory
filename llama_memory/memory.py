"""
Core memory operations for llama-memory.
"""

from __future__ import annotations

import json
import os
import math
from typing import Optional, Literal, Tuple
from datetime import datetime
from dataclasses import dataclass, field

from .config import Config, get_config
from .database import Database, get_database, embedding_to_blob, DECAY_START_DAYS, DECAY_ARCHIVE_DAYS
from .embeddings import get_embedding

MemoryType = Literal['fact', 'decision', 'event', 'entity', 'context', 'procedure']
Retention = Literal['permanent', 'long-term', 'medium', 'short', 'session']
MemorySource = Literal['cli', 'mcp', 'api', 'import', 'auto']


def _row_get(row, key, default=None):
    """Safely get a value from a sqlite3.Row object."""
    if key in row.keys():
        val = row[key]
        return val if val is not None else default
    return default

# Similarity threshold for duplicate detection (lower distance = more similar)
# 0.5 catches near-duplicates, 0.7 catches paraphrases
DUPLICATE_THRESHOLD = 0.5

# Ranking weights for hybrid scoring
RANK_WEIGHT_DISTANCE = 0.5
RANK_WEIGHT_IMPORTANCE = 0.25
RANK_WEIGHT_RECENCY = 0.15
RANK_WEIGHT_ACCESS = 0.10


def get_session_id() -> str:
    """Get current session ID from environment or generate one."""
    return os.environ.get('LLAMA_MEMORY_SESSION', f"session-{int(datetime.now().timestamp())}")


@dataclass
class Memory:
    """A memory record."""
    id: int
    type: MemoryType
    content: str
    summary: Optional[str] = None
    project: Optional[str] = None
    tags: list[str] = None
    importance: int = 5
    retention: Retention = 'long-term'
    created_at: int = 0
    updated_at: Optional[int] = None
    accessed_at: Optional[int] = None
    access_count: int = 0
    superseded_by: Optional[int] = None
    archived: bool = False
    session_id: Optional[str] = None
    source: MemorySource = 'cli'  # v4: where memory came from
    metadata: Optional[dict] = None  # v4: extensible JSON blob
    confidence: float = 1.0  # v5: how certain is this memory (0.0-1.0)
    parent_id: Optional[int] = None  # v5: hierarchical parent
    topic: Optional[str] = None  # v5: auto-assigned topic/cluster
    protected: bool = False  # v6: manually protected from decay
    distance: Optional[float] = None  # Raw vector distance
    score: Optional[float] = None  # Hybrid score (lower = better)

    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'type': self.type,
            'content': self.content,
            'summary': self.summary,
            'project': self.project,
            'tags': self.tags,
            'importance': self.importance,
            'retention': self.retention,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'accessed_at': self.accessed_at,
            'access_count': self.access_count,
            'superseded_by': self.superseded_by,
            'archived': self.archived,
            'session_id': self.session_id,
            'source': self.source,
            'metadata': self.metadata,
            'confidence': self.confidence,
            'parent_id': self.parent_id,
            'topic': self.topic,
            'protected': self.protected,
            'distance': self.distance,
            'score': self.score,
        }


@dataclass
class DuplicateWarning:
    """Warning about potential duplicate memory."""
    existing_id: int
    existing_content: str
    similarity: float  # 0-1, higher = more similar

    def to_dict(self) -> dict:
        return {
            'existing_id': self.existing_id,
            'existing_content': self.existing_content[:200],
            'similarity': round(self.similarity, 3),
        }


class MemoryStore:
    """Memory storage and retrieval operations."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or get_config()
        self.db = get_database(self.config)

    def initialize(self):
        """Initialize the memory store."""
        self.db.initialize()

    def _row_to_memory(self, row, distance: Optional[float] = None, score: Optional[float] = None) -> Memory:
        """Convert a database row to a Memory object."""
        metadata_str = _row_get(row, 'metadata')
        return Memory(
            id=row['id'],
            type=row['type'],
            content=row['content'],
            summary=row['summary'],
            project=row['project'],
            tags=json.loads(row['tags']) if row['tags'] else [],
            importance=row['importance'],
            retention=row['retention'],
            created_at=row['created_at'],
            updated_at=row['updated_at'],
            accessed_at=row['accessed_at'],
            access_count=row['access_count'],
            superseded_by=row['superseded_by'],
            archived=bool(row['archived']),
            session_id=_row_get(row, 'session_id'),
            source=_row_get(row, 'source', 'cli'),
            metadata=json.loads(metadata_str) if metadata_str else {},
            confidence=_row_get(row, 'confidence', 1.0),
            parent_id=_row_get(row, 'parent_id'),
            topic=_row_get(row, 'topic'),
            protected=bool(_row_get(row, 'protected', 0)),
            distance=distance,
            score=score,
        )

    def check_duplicate(
        self,
        content: str,
        embedding: Optional[list[float]] = None,
    ) -> Optional[DuplicateWarning]:
        """Check if similar content already exists."""
        if embedding is None:
            embedding = get_embedding(content, self.config)

        embedding_blob = embedding_to_blob(embedding)
        conn = self.db.conn

        # Find most similar existing memory
        row = conn.execute("""
            SELECT m.id, m.content, e.distance
            FROM memory_embeddings e
            JOIN memories m ON m.id = e.memory_id
            WHERE e.embedding MATCH ?
              AND k = 1
              AND m.archived = 0
              AND m.superseded_by IS NULL
        """, (embedding_blob,)).fetchone()

        if row and row['distance'] < DUPLICATE_THRESHOLD:
            similarity = 1.0 - row['distance']  # Convert distance to similarity
            return DuplicateWarning(
                existing_id=row['id'],
                existing_content=row['content'],
                similarity=similarity
            )
        return None

    def store(
        self,
        content: str,
        type: MemoryType = 'fact',
        summary: Optional[str] = None,
        project: Optional[str] = None,
        tags: Optional[list[str]] = None,
        importance: int = 5,
        retention: Retention = 'long-term',
        session_id: Optional[str] = None,
        source: MemorySource = 'cli',
        metadata: Optional[dict] = None,
        confidence: float = 1.0,
        parent_id: Optional[int] = None,
        topic: Optional[str] = None,
        check_duplicates: bool = True,
        force: bool = False,
    ) -> Tuple[int, Optional[DuplicateWarning]]:
        """Store a new memory with embedding.

        Returns tuple of (memory_id, duplicate_warning).
        If force=False and duplicate found, returns (-1, warning) without storing.
        """
        now = int(datetime.now().timestamp())
        session = session_id or get_session_id()

        # Generate embedding
        embedding = get_embedding(content, self.config)
        embedding_blob = embedding_to_blob(embedding)

        # Check for duplicates
        duplicate = None
        if check_duplicates:
            duplicate = self.check_duplicate(content, embedding)
            if duplicate and not force:
                return (-1, duplicate)

        conn = self.db.conn

        # Insert memory
        cursor = conn.execute("""
            INSERT INTO memories (type, content, summary, project, tags, created_at, importance, retention, embedding_model, session_id, source, metadata, confidence, parent_id, topic)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            type,
            content,
            summary,
            project,
            json.dumps(tags or []),
            now,
            importance,
            retention,
            self.config.embedding_model_version,
            session,
            source,
            json.dumps(metadata) if metadata else None,
            confidence,
            parent_id,
            topic,
        ))

        memory_id = cursor.lastrowid

        # Insert embedding
        conn.execute("""
            INSERT INTO memory_embeddings (memory_id, embedding)
            VALUES (?, ?)
        """, (memory_id, embedding_blob))

        # Also store in embedding_versions for future model migrations
        conn.execute("""
            INSERT INTO embedding_versions (memory_id, model_version, embedding, created_at, is_current)
            VALUES (?, ?, ?, ?, 1)
        """, (memory_id, self.config.embedding_model_version, embedding_blob, now))

        # Log the action
        conn.execute("""
            INSERT INTO memory_log (memory_id, action, timestamp, details)
            VALUES (?, 'create', ?, ?)
        """, (memory_id, now, json.dumps({'type': type, 'project': project, 'session': session, 'source': source})))

        conn.commit()

        return (memory_id, duplicate)

    def _compute_hybrid_score(
        self,
        distance: float,
        importance: int,
        created_at: int,
        access_count: int,
        now: int,
    ) -> float:
        """Compute hybrid ranking score (lower = better).

        Combines:
        - Vector distance (semantic similarity)
        - Importance (user-assigned priority)
        - Recency (newer memories rank higher)
        - Access frequency (frequently accessed = more relevant)
        """
        # Normalize distance (typically 0-2 for cosine, cap at 2)
        norm_distance = min(distance / 2.0, 1.0)

        # Normalize importance (1-10 -> 0-1, inverted so higher = lower score)
        norm_importance = 1.0 - (importance / 10.0)

        # Recency decay: memories older than 30 days start decaying
        age_seconds = now - created_at
        age_days = age_seconds / 86400
        # Logarithmic decay: 0 at 0 days, ~0.5 at 30 days, ~0.7 at 90 days, ~1.0 at 365 days
        if age_days <= 1:
            norm_recency = 0.0
        else:
            norm_recency = min(math.log10(age_days) / 2.5, 1.0)

        # Access frequency boost (inverted so more access = lower score)
        # Logarithmic scaling to prevent runaway
        if access_count <= 1:
            norm_access = 1.0
        else:
            norm_access = max(0.0, 1.0 - (math.log10(access_count) / 2.0))

        # Weighted combination
        score = (
            RANK_WEIGHT_DISTANCE * norm_distance +
            RANK_WEIGHT_IMPORTANCE * norm_importance +
            RANK_WEIGHT_RECENCY * norm_recency +
            RANK_WEIGHT_ACCESS * norm_access
        )

        return score

    def search(
        self,
        query: str,
        limit: int = 10,
        type: Optional[MemoryType] = None,
        project: Optional[str] = None,
        session_id: Optional[str] = None,
        min_importance: Optional[int] = None,
        include_archived: bool = False,
        hybrid_ranking: bool = True,
        after: Optional[int] = None,
        before: Optional[int] = None,
    ) -> list[Memory]:
        """Search memories by semantic similarity with hybrid ranking.

        Args:
            hybrid_ranking: If True, re-ranks results using importance, recency, and access patterns.
        """
        now = int(datetime.now().timestamp())

        # Generate query embedding
        query_embedding = get_embedding(query, self.config)
        query_blob = embedding_to_blob(query_embedding)

        conn = self.db.conn

        # Get more results for hybrid ranking re-sort
        fetch_limit = limit * 5 if hybrid_ranking else limit

        # Build query with filters
        sql = """
            SELECT
                m.*,
                e.distance
            FROM memory_embeddings e
            JOIN memories m ON m.id = e.memory_id
            WHERE e.embedding MATCH ?
              AND k = ?
        """
        params = [query_blob, fetch_limit]

        if not include_archived:
            sql += " AND m.archived = 0"

        if type:
            sql += " AND m.type = ?"
            params.append(type)

        if project:
            sql += " AND (m.project = ? OR m.project IS NULL)"
            params.append(project)

        if session_id:
            sql += " AND m.session_id = ?"
            params.append(session_id)

        if min_importance:
            sql += " AND m.importance >= ?"
            params.append(min_importance)

        if after:
            sql += " AND m.created_at >= ?"
            params.append(after)

        if before:
            sql += " AND m.created_at <= ?"
            params.append(before)

        # Exclude superseded memories
        sql += " AND m.superseded_by IS NULL"

        # Fetch all results first
        rows = conn.execute(sql, params).fetchall()

        results = []
        ids_to_update = []

        for row in rows:
            ids_to_update.append(row['id'])

            # Compute hybrid score
            score = None
            if hybrid_ranking:
                score = self._compute_hybrid_score(
                    distance=row['distance'],
                    importance=row['importance'],
                    created_at=row['created_at'],
                    access_count=row['access_count'],
                    now=now,
                )

            mem = self._row_to_memory(row, distance=row['distance'], score=score)
            # Override with real-time values
            mem.accessed_at = now
            mem.access_count = row['access_count'] + 1
            results.append(mem)

        # Re-sort by hybrid score if enabled
        if hybrid_ranking:
            results.sort(key=lambda m: m.score)

        # Limit results
        results = results[:limit]

        # Update access stats after fetching (only for returned results)
        returned_ids = {m.id for m in results}
        for memory_id in ids_to_update:
            if memory_id in returned_ids:
                conn.execute("""
                    UPDATE memories
                    SET accessed_at = ?, access_count = access_count + 1
                    WHERE id = ?
                """, (now, memory_id))

        conn.commit()

        return results

    def search_text(
        self,
        query: str,
        limit: int = 10,
        type: Optional[MemoryType] = None,
        project: Optional[str] = None,
    ) -> list[Memory]:
        """Search memories using full-text search (faster, less semantic)."""
        conn = self.db.conn

        sql = """
            SELECT m.* FROM memories m
            JOIN memories_fts f ON m.id = f.rowid
            WHERE memories_fts MATCH ?
              AND m.archived = 0
              AND m.superseded_by IS NULL
        """
        params = [query]

        if type:
            sql += " AND m.type = ?"
            params.append(type)

        if project:
            sql += " AND (m.project = ? OR m.project IS NULL)"
            params.append(project)

        sql += " ORDER BY rank LIMIT ?"
        params.append(limit)

        results = []
        for row in conn.execute(sql, params):
            results.append(self._row_to_memory(row))

        return results

    def list(
        self,
        limit: int = 20,
        type: Optional[MemoryType] = None,
        project: Optional[str] = None,
        include_archived: bool = False,
        order_by: str = 'created_at',
        after: Optional[int] = None,
        before: Optional[int] = None,
    ) -> list[Memory]:
        """List memories without search."""
        conn = self.db.conn

        valid_order = ['created_at', 'updated_at', 'accessed_at', 'importance']
        if order_by not in valid_order:
            order_by = 'created_at'

        sql = f"SELECT * FROM memories WHERE superseded_by IS NULL"
        params = []

        if not include_archived:
            sql += " AND archived = 0"

        if type:
            sql += " AND type = ?"
            params.append(type)

        if project:
            sql += " AND (project = ? OR project IS NULL)"
            params.append(project)

        if after:
            sql += " AND created_at >= ?"
            params.append(after)

        if before:
            sql += " AND created_at <= ?"
            params.append(before)

        sql += f" ORDER BY {order_by} DESC LIMIT ?"
        params.append(limit)

        results = []
        for row in conn.execute(sql, params):
            results.append(self._row_to_memory(row))

        return results

    def get(self, id: int) -> Optional[Memory]:
        """Get a specific memory by ID."""
        conn = self.db.conn

        row = conn.execute("SELECT * FROM memories WHERE id = ?", (id,)).fetchone()
        if not row:
            return None

        return self._row_to_memory(row)

    def update(
        self,
        id: int,
        content: Optional[str] = None,
        summary: Optional[str] = None,
        importance: Optional[int] = None,
        tags: Optional[list[str]] = None,
        archived: Optional[bool] = None,
    ) -> bool:
        """Update a memory."""
        conn = self.db.conn
        now = int(datetime.now().timestamp())

        updates = ["updated_at = ?"]
        params = [now]
        log_details = {}

        if content is not None:
            updates.append("content = ?")
            params.append(content)
            log_details['content_changed'] = True

            # Re-embed if content changed
            embedding = get_embedding(content, self.config)
            embedding_blob = embedding_to_blob(embedding)
            conn.execute(
                "UPDATE memory_embeddings SET embedding = ? WHERE memory_id = ?",
                (embedding_blob, id)
            )
            conn.execute(
                "UPDATE memories SET embedding_model = ? WHERE id = ?",
                (self.config.embedding_model_version, id)
            )

        if summary is not None:
            updates.append("summary = ?")
            params.append(summary)
            log_details['summary_changed'] = True

        if importance is not None:
            updates.append("importance = ?")
            params.append(importance)
            log_details['importance'] = importance

        if tags is not None:
            updates.append("tags = ?")
            params.append(json.dumps(tags))
            log_details['tags'] = tags

        if archived is not None:
            updates.append("archived = ?")
            params.append(1 if archived else 0)
            log_details['archived'] = archived

        params.append(id)

        conn.execute(f"UPDATE memories SET {', '.join(updates)} WHERE id = ?", params)

        # Log the update
        conn.execute("""
            INSERT INTO memory_log (memory_id, action, timestamp, details)
            VALUES (?, 'update', ?, ?)
        """, (id, now, json.dumps(log_details)))

        conn.commit()
        return True

    def supersede(
        self,
        old_id: int,
        new_content: str,
        **kwargs
    ) -> int:
        """Create new memory that supersedes an old one."""
        old = self.get(old_id)
        if not old:
            raise ValueError(f"Memory {old_id} not found")

        # Create new memory with same defaults
        new_id = self.store(
            content=new_content,
            type=kwargs.get('type', old.type),
            summary=kwargs.get('summary', old.summary),
            project=kwargs.get('project', old.project),
            tags=kwargs.get('tags', old.tags),
            importance=kwargs.get('importance', old.importance),
            retention=kwargs.get('retention', old.retention),
        )

        # Mark old as superseded
        conn = self.db.conn
        now = int(datetime.now().timestamp())

        conn.execute(
            "UPDATE memories SET superseded_by = ?, updated_at = ? WHERE id = ?",
            (new_id, now, old_id)
        )

        conn.execute("""
            INSERT INTO memory_log (memory_id, action, timestamp, details)
            VALUES (?, 'supersede', ?, ?)
        """, (old_id, now, json.dumps({'superseded_by': new_id})))

        conn.commit()

        return new_id

    def delete(self, id: int, hard: bool = False) -> bool:
        """Delete a memory. Soft delete by default (archive)."""
        conn = self.db.conn
        now = int(datetime.now().timestamp())

        if hard:
            conn.execute("DELETE FROM memory_embeddings WHERE memory_id = ?", (id,))
            conn.execute("DELETE FROM memories WHERE id = ?", (id,))
            action = 'hard_delete'
        else:
            conn.execute(
                "UPDATE memories SET archived = 1, updated_at = ? WHERE id = ?",
                (now, id)
            )
            action = 'archive'

        conn.execute("""
            INSERT INTO memory_log (memory_id, action, timestamp, details)
            VALUES (?, ?, ?, NULL)
        """, (id, action, now))

        conn.commit()
        return True

    def stats(self) -> dict:
        """Get memory statistics."""
        return self.db.get_stats()

    def export_json(self, include_archived: bool = False) -> list[dict]:
        """Export all memories as JSON-serializable list."""
        memories = self.list(limit=100000, include_archived=include_archived)
        return [m.to_dict() for m in memories]

    def unarchive(self, id: int) -> bool:
        """Restore an archived memory."""
        conn = self.db.conn
        now = int(datetime.now().timestamp())

        # Check if memory exists and is archived
        row = conn.execute("SELECT archived FROM memories WHERE id = ?", (id,)).fetchone()
        if not row:
            return False
        if not row['archived']:
            return True  # Already unarchived

        conn.execute(
            "UPDATE memories SET archived = 0, updated_at = ? WHERE id = ?",
            (now, id)
        )

        conn.execute("""
            INSERT INTO memory_log (memory_id, action, timestamp, details)
            VALUES (?, 'unarchive', ?, NULL)
        """, (id, now))

        conn.commit()
        return True

    def reembed(self, id: Optional[int] = None, model_filter: Optional[str] = None) -> int:
        """Regenerate embeddings for memories.

        Args:
            id: If provided, only reembed this specific memory
            model_filter: If provided, only reembed memories with this embedding model

        Returns:
            Number of memories re-embedded
        """
        conn = self.db.conn
        now = int(datetime.now().timestamp())
        count = 0

        if id:
            # Reembed single memory
            row = conn.execute("SELECT content FROM memories WHERE id = ?", (id,)).fetchone()
            if row:
                embedding = get_embedding(row['content'], self.config)
                embedding_blob = embedding_to_blob(embedding)
                conn.execute(
                    "UPDATE memory_embeddings SET embedding = ? WHERE memory_id = ?",
                    (embedding_blob, id)
                )
                conn.execute(
                    "UPDATE memories SET embedding_model = ?, updated_at = ? WHERE id = ?",
                    (self.config.embedding_model_version, now, id)
                )
                count = 1
        else:
            # Reembed all (optionally filtered by model)
            sql = "SELECT id, content FROM memories WHERE archived = 0"
            params = []

            if model_filter:
                sql += " AND embedding_model = ?"
                params.append(model_filter)

            rows = conn.execute(sql, params).fetchall()

            for row in rows:
                embedding = get_embedding(row['content'], self.config)
                embedding_blob = embedding_to_blob(embedding)
                conn.execute(
                    "UPDATE memory_embeddings SET embedding = ? WHERE memory_id = ?",
                    (embedding_blob, row['id'])
                )
                conn.execute(
                    "UPDATE memories SET embedding_model = ?, updated_at = ? WHERE id = ?",
                    (self.config.embedding_model_version, now, row['id'])
                )
                count += 1

        conn.commit()
        return count

    def cleanup_expired(self) -> int:
        """Archive memories past their retention period. Returns count archived."""
        conn = self.db.conn
        now = int(datetime.now().timestamp())

        # Retention periods in seconds
        retention_seconds = {
            'session': 0,  # Already expired
            'short': 30 * 24 * 60 * 60,  # 30 days
            'medium': 365 * 24 * 60 * 60,  # 1 year
            'long-term': 5 * 365 * 24 * 60 * 60,  # 5 years
            'permanent': float('inf'),
        }

        count = 0
        for retention, max_age in retention_seconds.items():
            if max_age == float('inf'):
                continue

            cutoff = now - max_age
            cursor = conn.execute("""
                UPDATE memories
                SET archived = 1, updated_at = ?
                WHERE retention = ?
                  AND created_at < ?
                  AND archived = 0
            """, (now, retention, cutoff))

            count += cursor.rowcount

        conn.commit()
        return count

    def link(
        self,
        source_id: int,
        target_id: int,
        link_type: str = 'related',
        note: Optional[str] = None,
        weight: float = 1.0,
        metadata: Optional[dict] = None,
    ) -> bool:
        """Create a link between two memories.

        Args:
            source_id: The memory creating the link
            target_id: The memory being linked to
            link_type: Type of relationship (related, depends_on, contradicts, etc.)
            note: Optional note explaining the relationship
            weight: Relationship strength from 0.0-1.0 (default 1.0)
            metadata: Optional JSON metadata for extensibility

        Returns:
            True if link created, False if already exists
        """
        if source_id == target_id:
            raise ValueError("Cannot link a memory to itself")

        # Verify both memories exist
        if not self.get(source_id):
            raise ValueError(f"Memory {source_id} not found")
        if not self.get(target_id):
            raise ValueError(f"Memory {target_id} not found")

        conn = self.db.conn
        now = int(datetime.now().timestamp())

        try:
            conn.execute("""
                INSERT INTO memory_links (source_id, target_id, link_type, note, weight, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (source_id, target_id, link_type, note, weight, json.dumps(metadata) if metadata else None, now))
            conn.commit()
            return True
        except Exception as e:
            if "UNIQUE constraint" in str(e):
                return False  # Link already exists
            raise

    def unlink(self, source_id: int, target_id: int) -> bool:
        """Remove a link between two memories.

        Returns:
            True if link removed, False if it didn't exist
        """
        conn = self.db.conn
        cursor = conn.execute("""
            DELETE FROM memory_links
            WHERE source_id = ? AND target_id = ?
        """, (source_id, target_id))
        conn.commit()
        return cursor.rowcount > 0

    def get_links(self, memory_id: int) -> list[dict]:
        """Get all links for a memory (both directions).

        Returns:
            List of dicts with link info and linked memory
        """
        conn = self.db.conn
        links = []

        # Links where this memory is the source
        rows = conn.execute("""
            SELECT l.*, m.content, m.type, m.importance
            FROM memory_links l
            JOIN memories m ON m.id = l.target_id
            WHERE l.source_id = ?
        """, (memory_id,)).fetchall()

        for row in rows:
            links.append({
                'direction': 'outgoing',
                'linked_id': row['target_id'],
                'link_type': row['link_type'],
                'note': row['note'],
                'weight': row['weight'] if 'weight' in row.keys() else 1.0,
                'metadata': json.loads(row['metadata']) if row['metadata'] else {} if 'metadata' in row.keys() else {},
                'created_at': row['created_at'],
                'content': row['content'],
                'type': row['type'],
                'importance': row['importance'],
            })

        # Links where this memory is the target
        rows = conn.execute("""
            SELECT l.*, m.content, m.type, m.importance
            FROM memory_links l
            JOIN memories m ON m.id = l.source_id
            WHERE l.target_id = ?
        """, (memory_id,)).fetchall()

        for row in rows:
            links.append({
                'direction': 'incoming',
                'linked_id': row['source_id'],
                'link_type': row['link_type'],
                'note': row['note'],
                'weight': row['weight'] if 'weight' in row.keys() else 1.0,
                'metadata': json.loads(row['metadata']) if row['metadata'] else {} if 'metadata' in row.keys() else {},
                'created_at': row['created_at'],
                'content': row['content'],
                'type': row['type'],
                'importance': row['importance'],
            })

        return links

    # ========== v5 Features ==========

    def log_search(self, query: str, query_type: str = 'semantic', result_count: int = 0,
                   top_result_id: Optional[int] = None) -> None:
        """Log a search query to history."""
        conn = self.db.conn
        now = int(datetime.now().timestamp())
        session = get_session_id()

        conn.execute("""
            INSERT INTO search_history (query, query_type, result_count, top_result_id, session_id, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (query, query_type, result_count, top_result_id, session, now))
        conn.commit()

    def get_search_history(self, limit: int = 50, session_id: Optional[str] = None) -> list[dict]:
        """Get recent search history."""
        conn = self.db.conn
        sql = "SELECT * FROM search_history"
        params = []

        if session_id:
            sql += " WHERE session_id = ?"
            params.append(session_id)

        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        rows = conn.execute(sql, params).fetchall()
        return [dict(row) for row in rows]

    def get_popular_queries(self, limit: int = 20) -> list[dict]:
        """Get most frequently searched queries."""
        conn = self.db.conn
        rows = conn.execute("""
            SELECT query, COUNT(*) as count, MAX(created_at) as last_searched
            FROM search_history
            GROUP BY query
            ORDER BY count DESC
            LIMIT ?
        """, (limit,)).fetchall()
        return [dict(row) for row in rows]

    def check_conflicts(self, memory_id: int, threshold: float = 0.7) -> list[dict]:
        """Check if a memory potentially conflicts with existing memories.

        Looks for memories that are very similar but might contain contradicting information.
        Returns list of potential conflicts.
        """
        memory = self.get(memory_id)
        if not memory:
            return []

        # Find similar memories
        similar = self.search(query=memory.content, limit=20)

        conflicts = []
        conn = self.db.conn
        now = int(datetime.now().timestamp())

        for other in similar:
            if other.id == memory_id:
                continue

            # High similarity suggests potential conflict
            if other.distance is not None and other.distance < threshold:
                similarity = 1.0 - other.distance

                # Check if conflict already recorded
                existing = conn.execute("""
                    SELECT id FROM memory_conflicts
                    WHERE (memory1_id = ? AND memory2_id = ?) OR (memory1_id = ? AND memory2_id = ?)
                """, (memory_id, other.id, other.id, memory_id)).fetchone()

                if not existing:
                    # Record potential conflict
                    conn.execute("""
                        INSERT INTO memory_conflicts (memory1_id, memory2_id, conflict_type, similarity, created_at)
                        VALUES (?, ?, 'potential', ?, ?)
                    """, (memory_id, other.id, similarity, now))

                conflicts.append({
                    'memory_id': other.id,
                    'content': other.content[:200],
                    'similarity': similarity,
                    'type': other.type,
                })

        conn.commit()
        return conflicts

    def get_conflicts(self, memory_id: Optional[int] = None, conflict_type: Optional[str] = None) -> list[dict]:
        """Get recorded conflicts for a memory or all conflicts."""
        conn = self.db.conn
        sql = """
            SELECT c.*, m1.content as content1, m2.content as content2
            FROM memory_conflicts c
            JOIN memories m1 ON c.memory1_id = m1.id
            JOIN memories m2 ON c.memory2_id = m2.id
            WHERE 1=1
        """
        params = []

        if memory_id:
            sql += " AND (c.memory1_id = ? OR c.memory2_id = ?)"
            params.extend([memory_id, memory_id])

        if conflict_type:
            sql += " AND c.conflict_type = ?"
            params.append(conflict_type)

        sql += " ORDER BY c.created_at DESC"
        rows = conn.execute(sql, params).fetchall()

        return [{
            'id': row['id'],
            'memory1_id': row['memory1_id'],
            'memory2_id': row['memory2_id'],
            'content1': row['content1'][:150],
            'content2': row['content2'][:150],
            'conflict_type': row['conflict_type'],
            'similarity': row['similarity'],
            'note': row['note'],
            'resolved_by': row['resolved_by'],
            'created_at': row['created_at'],
            'resolved_at': row['resolved_at'],
        } for row in rows]

    def resolve_conflict(self, conflict_id: int, resolved_by: int, note: Optional[str] = None) -> bool:
        """Mark a conflict as resolved by choosing a memory as correct."""
        conn = self.db.conn
        now = int(datetime.now().timestamp())

        cursor = conn.execute("""
            UPDATE memory_conflicts
            SET conflict_type = 'resolved', resolved_by = ?, resolved_at = ?, note = COALESCE(?, note)
            WHERE id = ?
        """, (resolved_by, now, note, conflict_id))
        conn.commit()
        return cursor.rowcount > 0

    def get_children(self, parent_id: int) -> list[Memory]:
        """Get all child memories of a parent."""
        conn = self.db.conn
        rows = conn.execute("""
            SELECT * FROM memories WHERE parent_id = ? AND archived = 0
            ORDER BY created_at ASC
        """, (parent_id,)).fetchall()
        return [self._row_to_memory(row) for row in rows]

    def get_ancestors(self, memory_id: int) -> list[Memory]:
        """Get the chain of parent memories (from immediate parent to root)."""
        ancestors = []
        memory = self.get(memory_id)

        while memory and memory.parent_id:
            parent = self.get(memory.parent_id)
            if parent:
                ancestors.append(parent)
                memory = parent
            else:
                break

        return ancestors

    def set_parent(self, child_id: int, parent_id: Optional[int]) -> bool:
        """Set or remove the parent of a memory."""
        # Prevent circular references
        if parent_id:
            ancestors = self.get_ancestors(parent_id)
            if any(a.id == child_id for a in ancestors):
                raise ValueError("Cannot create circular parent reference")

        conn = self.db.conn
        now = int(datetime.now().timestamp())
        cursor = conn.execute("""
            UPDATE memories SET parent_id = ?, updated_at = ? WHERE id = ?
        """, (parent_id, now, child_id))
        conn.commit()
        return cursor.rowcount > 0

    def merge(self, source_ids: list[int], merged_content: str, **kwargs) -> int:
        """Merge multiple memories into one new memory.

        Creates a new memory with the merged content, links it to the sources,
        and optionally archives the source memories.
        """
        if len(source_ids) < 2:
            raise ValueError("Need at least 2 memories to merge")

        # Get source memories
        sources = [self.get(id) for id in source_ids]
        if None in sources:
            raise ValueError("One or more source memories not found")

        # Create merged memory
        # Inherit highest importance
        importance = kwargs.get('importance', max(s.importance for s in sources))
        project = kwargs.get('project', sources[0].project)
        tags = kwargs.get('tags', list(set(t for s in sources for t in s.tags)))

        merged_id, _ = self.store(
            content=merged_content,
            type=kwargs.get('type', sources[0].type),
            summary=kwargs.get('summary'),
            project=project,
            importance=importance,
            tags=tags,
            metadata={'merged_from': source_ids},
            force=True,
        )

        # Link merged memory to sources
        for source_id in source_ids:
            self.link(merged_id, source_id, link_type='merged_from', note='Merged memory')

        # Archive sources if requested
        if kwargs.get('archive_sources', False):
            for source_id in source_ids:
                self.update(source_id, archived=True)

        return merged_id

    def set_topic(self, memory_id: int, topic: str) -> bool:
        """Set the topic for a memory."""
        conn = self.db.conn
        now = int(datetime.now().timestamp())
        cursor = conn.execute("""
            UPDATE memories SET topic = ?, updated_at = ? WHERE id = ?
        """, (topic, now, memory_id))
        conn.commit()
        return cursor.rowcount > 0

    def get_topics(self) -> list[dict]:
        """Get all topics with memory counts."""
        conn = self.db.conn
        rows = conn.execute("""
            SELECT topic, COUNT(*) as count
            FROM memories
            WHERE topic IS NOT NULL AND archived = 0
            GROUP BY topic
            ORDER BY count DESC
        """).fetchall()
        return [{'topic': row['topic'], 'count': row['count']} for row in rows]

    def get_tags(self) -> list[dict]:
        """Get all tags with memory counts."""
        conn = self.db.conn
        rows = conn.execute("""
            SELECT tags FROM memories WHERE archived = 0 AND tags IS NOT NULL AND tags != '[]'
        """).fetchall()

        tag_counts = {}
        for row in rows:
            tags = json.loads(row['tags']) if row['tags'] else []
            for tag in tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1

        return [{'tag': tag, 'count': count} for tag, count in sorted(tag_counts.items(), key=lambda x: -x[1])]

    def get_access_history(self, memory_id: int) -> list[dict]:
        """Get access/modification history for a specific memory from the log."""
        conn = self.db.conn
        rows = conn.execute("""
            SELECT action, timestamp, details FROM memory_log
            WHERE memory_id = ?
            ORDER BY timestamp DESC
        """, (memory_id,)).fetchall()

        return [{
            'action': row['action'],
            'timestamp': row['timestamp'],
            'details': json.loads(row['details']) if row['details'] else None,
        } for row in rows]

    def get_by_topic(self, topic: str, limit: int = 50) -> list[Memory]:
        """Get memories by topic."""
        conn = self.db.conn
        rows = conn.execute("""
            SELECT * FROM memories
            WHERE topic = ? AND archived = 0
            ORDER BY importance DESC, created_at DESC
            LIMIT ?
        """, (topic, limit)).fetchall()
        return [self._row_to_memory(row) for row in rows]

    def set_confidence(self, memory_id: int, confidence: float) -> bool:
        """Set the confidence level for a memory."""
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")

        conn = self.db.conn
        now = int(datetime.now().timestamp())
        cursor = conn.execute("""
            UPDATE memories SET confidence = ?, updated_at = ? WHERE id = ?
        """, (confidence, now, memory_id))
        conn.commit()
        return cursor.rowcount > 0

    # Bulk operations
    def batch_update(self, updates: list[dict]) -> int:
        """Update multiple memories at once.

        Each dict should have 'id' and fields to update.
        Returns count of updated memories.
        """
        count = 0
        for update in updates:
            memory_id = update.pop('id', None)
            if memory_id:
                if self.update(memory_id, **update):
                    count += 1
        return count

    def batch_delete(self, ids: list[int], hard: bool = False) -> int:
        """Delete multiple memories at once. Returns count deleted."""
        count = 0
        for memory_id in ids:
            if self.delete(memory_id, hard=hard):
                count += 1
        return count

    def batch_link(self, links: list[dict]) -> int:
        """Create multiple links at once.

        Each dict should have source_id, target_id, and optionally link_type, note, weight.
        Returns count of links created.
        """
        count = 0
        for link in links:
            try:
                if self.link(
                    source_id=link['source_id'],
                    target_id=link['target_id'],
                    link_type=link.get('link_type', 'related'),
                    note=link.get('note'),
                    weight=link.get('weight', 1.0),
                ):
                    count += 1
            except ValueError:
                pass  # Skip invalid links
        return count

    def batch_set_topic(self, memory_ids: list[int], topic: str) -> int:
        """Set topic for multiple memories at once."""
        count = 0
        for memory_id in memory_ids:
            if self.set_topic(memory_id, topic):
                count += 1
        return count

    # Export formats
    def export_markdown(self, include_archived: bool = False) -> str:
        """Export all memories as Markdown."""
        memories = self.list(limit=100000, include_archived=include_archived)

        lines = ["# Memory Export\n"]
        lines.append(f"Exported: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        lines.append(f"Total memories: {len(memories)}\n\n")

        # Group by project
        by_project = {}
        for m in memories:
            proj = m.project or 'Global'
            if proj not in by_project:
                by_project[proj] = []
            by_project[proj].append(m)

        for project, mems in sorted(by_project.items()):
            lines.append(f"## {project}\n")
            for m in mems:
                lines.append(f"### [{m.id}] {m.type.upper()} (importance: {m.importance})\n")
                if m.topic:
                    lines.append(f"*Topic: {m.topic}*\n")
                if m.tags:
                    lines.append(f"*Tags: {', '.join(m.tags)}*\n")
                lines.append(f"\n{m.content}\n")
                if m.summary:
                    lines.append(f"\n> Summary: {m.summary}\n")
                lines.append(f"\n*Created: {datetime.fromtimestamp(m.created_at).strftime('%Y-%m-%d %H:%M')}*\n")
                if m.confidence < 1.0:
                    lines.append(f"*Confidence: {m.confidence:.0%}*\n")
                lines.append("\n---\n\n")

        return '\n'.join(lines)

    def export_csv(self, include_archived: bool = False) -> str:
        """Export all memories as CSV."""
        import csv
        import io

        memories = self.list(limit=100000, include_archived=include_archived)

        output = io.StringIO()
        writer = csv.writer(output)

        # Header
        writer.writerow([
            'id', 'type', 'content', 'summary', 'project', 'tags', 'importance',
            'confidence', 'topic', 'retention', 'created_at', 'source', 'archived'
        ])

        for m in memories:
            writer.writerow([
                m.id, m.type, m.content, m.summary or '', m.project or '',
                ';'.join(m.tags), m.importance, m.confidence, m.topic or '',
                m.retention, datetime.fromtimestamp(m.created_at).strftime('%Y-%m-%d %H:%M'),
                m.source, 'yes' if m.archived else 'no'
            ])

        return output.getvalue()

    # ========== v6 Decay System ==========

    def is_protected_from_decay(self, memory: Memory) -> bool:
        """Check if a memory is protected from decay based on rules.

        Protected if ANY of:
        - Has project set
        - Has parent_id set (part of hierarchy)
        - Type is entity, decision, or procedure
        - Importance >= 7
        - Retention is 'permanent'
        - Has links to other memories
        - Manually protected flag is set
        """
        # Check manual protection flag
        if memory.protected:
            return True

        # Check project
        if memory.project:
            return True

        # Check hierarchy
        if memory.parent_id:
            return True

        # Check type
        if memory.type in ('entity', 'decision', 'procedure'):
            return True

        # Check importance
        if memory.importance >= 7:
            return True

        # Check retention
        if memory.retention == 'permanent':
            return True

        # Check links
        links = self.get_links(memory.id)
        if links:
            return True

        return False

    def get_decay_candidates(self) -> list[Memory]:
        """Get memories that are candidates for decay (not protected, old enough)."""
        conn = self.db.conn
        now = int(datetime.now().timestamp())
        archive_cutoff = now - (DECAY_ARCHIVE_DAYS * 24 * 60 * 60)

        # Get unarchived memories that haven't been accessed in DECAY_ARCHIVE_DAYS
        # We check protection rules in Python since they're complex
        rows = conn.execute("""
            SELECT * FROM memories
            WHERE archived = 0
              AND COALESCE(accessed_at, created_at) < ?
        """, (archive_cutoff,)).fetchall()

        candidates = []
        for row in rows:
            memory = self._row_to_memory(row)
            if not self.is_protected_from_decay(memory):
                candidates.append(memory)

        return candidates

    def run_decay(self, dry_run: bool = False) -> dict:
        """Run decay process to archive old, unaccessed, unprotected memories.

        Args:
            dry_run: If True, report what would be archived without doing it

        Returns:
            Dict with 'archived' count and 'candidates' list
        """
        candidates = self.get_decay_candidates()

        result = {
            'archived': 0,
            'candidates': [{'id': m.id, 'content': m.content[:100], 'type': m.type,
                           'last_access': m.accessed_at or m.created_at} for m in candidates],
        }

        if not dry_run:
            conn = self.db.conn
            now = int(datetime.now().timestamp())

            for memory in candidates:
                conn.execute("""
                    UPDATE memories SET archived = 1, updated_at = ? WHERE id = ?
                """, (now, memory.id))

                conn.execute("""
                    INSERT INTO memory_log (memory_id, action, timestamp, details)
                    VALUES (?, 'decay_archive', ?, ?)
                """, (memory.id, now, json.dumps({'reason': 'auto_decay', 'days_inactive': DECAY_ARCHIVE_DAYS})))

                result['archived'] += 1

            # Record last decay run
            conn.execute("""
                INSERT OR REPLACE INTO meta (key, value) VALUES ('last_decay_run', ?)
            """, (str(now),))

            conn.commit()

        return result

    def set_protected(self, memory_id: int, protected: bool = True) -> bool:
        """Manually set protection flag for a memory."""
        conn = self.db.conn
        now = int(datetime.now().timestamp())
        cursor = conn.execute("""
            UPDATE memories SET protected = ?, updated_at = ? WHERE id = ?
        """, (1 if protected else 0, now, memory_id))
        conn.commit()
        return cursor.rowcount > 0

    def get_decay_status(self, memory_id: int) -> dict:
        """Get decay status for a specific memory."""
        memory = self.get(memory_id)
        if not memory:
            return {'error': 'Memory not found'}

        now = int(datetime.now().timestamp())
        last_access = memory.accessed_at or memory.created_at
        days_since_access = (now - last_access) // (24 * 60 * 60)

        protected = self.is_protected_from_decay(memory)

        status = {
            'id': memory_id,
            'protected': protected,
            'protection_reasons': [],
            'days_since_access': days_since_access,
            'decay_starts_at': DECAY_START_DAYS,
            'archive_at': DECAY_ARCHIVE_DAYS,
            'will_decay': not protected and days_since_access >= DECAY_ARCHIVE_DAYS,
            'in_decay_window': not protected and DECAY_START_DAYS <= days_since_access < DECAY_ARCHIVE_DAYS,
        }

        # List protection reasons
        if memory.protected:
            status['protection_reasons'].append('manually_protected')
        if memory.project:
            status['protection_reasons'].append(f'has_project:{memory.project}')
        if memory.parent_id:
            status['protection_reasons'].append(f'has_parent:{memory.parent_id}')
        if memory.type in ('entity', 'decision', 'procedure'):
            status['protection_reasons'].append(f'protected_type:{memory.type}')
        if memory.importance >= 7:
            status['protection_reasons'].append(f'high_importance:{memory.importance}')
        if memory.retention == 'permanent':
            status['protection_reasons'].append('retention:permanent')
        if self.get_links(memory.id):
            status['protection_reasons'].append('has_links')

        return status

    def get_last_decay_run(self) -> Optional[int]:
        """Get timestamp of last decay run."""
        conn = self.db.conn
        result = conn.execute("SELECT value FROM meta WHERE key = 'last_decay_run'").fetchone()
        return int(result[0]) if result else None


# Module-level convenience functions
_store: Optional[MemoryStore] = None


def get_store(config: Optional[Config] = None) -> MemoryStore:
    """Get or create memory store."""
    global _store
    if _store is None or (config and config != _store.config):
        _store = MemoryStore(config)
    return _store


def store(content: str, **kwargs) -> int:
    """Store a memory."""
    return get_store().store(content, **kwargs)


def search(query: str, **kwargs) -> list[Memory]:
    """Search memories."""
    return get_store().search(query, **kwargs)


def list_memories(**kwargs) -> list[Memory]:
    """List memories."""
    return get_store().list(**kwargs)


def get_memory(id: int) -> Optional[Memory]:
    """Get a memory by ID."""
    return get_store().get(id)


def update(id: int, **kwargs) -> bool:
    """Update a memory."""
    return get_store().update(id, **kwargs)


def supersede(old_id: int, new_content: str, **kwargs) -> int:
    """Supersede a memory."""
    return get_store().supersede(old_id, new_content, **kwargs)


def stats() -> dict:
    """Get memory statistics."""
    return get_store().stats()

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
from .database import Database, get_database, embedding_to_blob
from .embeddings import get_embedding

MemoryType = Literal['fact', 'decision', 'event', 'entity', 'context', 'procedure']
Retention = Literal['permanent', 'long-term', 'medium', 'short', 'session']

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
    distance: Optional[float] = None  # Raw vector distance
    score: Optional[float] = None  # Hybrid score (lower = better)

    def __post_init__(self):
        if self.tags is None:
            self.tags = []

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
            INSERT INTO memories (type, content, summary, project, tags, created_at, importance, retention, embedding_model, session_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            session
        ))

        memory_id = cursor.lastrowid

        # Insert embedding
        conn.execute("""
            INSERT INTO memory_embeddings (memory_id, embedding)
            VALUES (?, ?)
        """, (memory_id, embedding_blob))

        # Log the action
        conn.execute("""
            INSERT INTO memory_log (memory_id, action, timestamp, details)
            VALUES (?, 'create', ?, ?)
        """, (memory_id, now, json.dumps({'type': type, 'project': project, 'session': session})))

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

            results.append(Memory(
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
                accessed_at=now,
                access_count=row['access_count'] + 1,
                superseded_by=row['superseded_by'],
                archived=bool(row['archived']),
                session_id=row['session_id'] if 'session_id' in row.keys() else None,
                distance=row['distance'],
                score=score,
            ))

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
            results.append(Memory(
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
            ))

        return results

    def list(
        self,
        limit: int = 20,
        type: Optional[MemoryType] = None,
        project: Optional[str] = None,
        include_archived: bool = False,
        order_by: str = 'created_at',
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

        sql += f" ORDER BY {order_by} DESC LIMIT ?"
        params.append(limit)

        results = []
        for row in conn.execute(sql, params):
            results.append(Memory(
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
            ))

        return results

    def get(self, id: int) -> Optional[Memory]:
        """Get a specific memory by ID."""
        conn = self.db.conn

        row = conn.execute("SELECT * FROM memories WHERE id = ?", (id,)).fetchone()
        if not row:
            return None

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
        )

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

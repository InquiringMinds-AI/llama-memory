"""
Session persistence for llama-memory.

Captures and restores Claude Code session state, enabling seamless
resumption after context overflow or between conversations.

Sessions are stored as memories with type="session" and structured metadata
tracking task progress, files touched, decisions made, and next steps.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


@dataclass
class Session:
    """A captured session state."""

    id: Optional[int] = None

    # Core identification
    title: str = ""  # Short title for the session
    summary: str = ""  # What was being worked on
    project: Optional[str] = None

    # Task tracking
    task_description: str = ""  # Detailed task description
    progress: Dict[str, Any] = field(default_factory=dict)  # {"completed": 3, "total": 7, "items": [...]}

    # Context
    files_touched: List[str] = field(default_factory=list)  # Files that were read/modified
    decisions: List[str] = field(default_factory=list)  # Key decisions made
    next_steps: List[str] = field(default_factory=list)  # What to do next
    blockers: List[str] = field(default_factory=list)  # What's blocking progress

    # Metadata
    started_at: Optional[int] = None
    ended_at: Optional[int] = None
    session_id: Optional[str] = None  # Claude Code session ID if available

    # Additional context
    notes: str = ""  # Freeform notes
    tags: List[str] = field(default_factory=list)

    def to_content(self) -> str:
        """Convert session to storable content string."""
        parts = []

        if self.title:
            parts.append(f"# {self.title}")

        if self.summary:
            parts.append(f"\n{self.summary}")

        if self.task_description:
            parts.append(f"\n## Task\n{self.task_description}")

        if self.progress:
            completed = self.progress.get('completed', 0)
            total = self.progress.get('total', 0)
            items = self.progress.get('items', [])
            parts.append(f"\n## Progress: {completed}/{total}")
            for item in items:
                status = "x" if item.get('done') else " "
                parts.append(f"- [{status}] {item.get('task', '')}")

        if self.files_touched:
            parts.append("\n## Files")
            for f in self.files_touched:
                parts.append(f"- {f}")

        if self.decisions:
            parts.append("\n## Decisions")
            for d in self.decisions:
                parts.append(f"- {d}")

        if self.next_steps:
            parts.append("\n## Next Steps")
            for n in self.next_steps:
                parts.append(f"- {n}")

        if self.blockers:
            parts.append("\n## Blockers")
            for b in self.blockers:
                parts.append(f"- {b}")

        if self.notes:
            parts.append(f"\n## Notes\n{self.notes}")

        return "\n".join(parts)

    def to_metadata(self) -> Dict[str, Any]:
        """Convert session fields to metadata dict for storage."""
        return {
            'session': {
                'title': self.title,
                'task_description': self.task_description,
                'progress': self.progress,
                'files_touched': self.files_touched,
                'decisions': self.decisions,
                'next_steps': self.next_steps,
                'blockers': self.blockers,
                'started_at': self.started_at,
                'ended_at': self.ended_at,
                'session_id': self.session_id,
                'notes': self.notes,
            }
        }

    @classmethod
    def from_memory(cls, memory) -> 'Session':
        """Create Session from a Memory object."""
        session = cls(
            id=memory.id,
            summary=memory.summary or "",
            project=memory.project,
            tags=memory.tags or [],
        )

        # Parse metadata if available
        if memory.metadata:
            meta = memory.metadata
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except:
                    meta = {}

            sess_data = meta.get('session', {})
            session.title = sess_data.get('title', '')
            session.task_description = sess_data.get('task_description', '')
            session.progress = sess_data.get('progress', {})
            session.files_touched = sess_data.get('files_touched', [])
            session.decisions = sess_data.get('decisions', [])
            session.next_steps = sess_data.get('next_steps', [])
            session.blockers = sess_data.get('blockers', [])
            session.started_at = sess_data.get('started_at')
            session.ended_at = sess_data.get('ended_at')
            session.session_id = sess_data.get('session_id')
            session.notes = sess_data.get('notes', '')

        return session

    def to_resume_prompt(self) -> str:
        """Generate a context block for resuming this session."""
        parts = [
            "<session-resume>",
            f"Resuming session: {self.title}" if self.title else "Resuming previous session",
        ]

        if self.summary:
            parts.append(f"\n**Summary:** {self.summary}")

        if self.task_description:
            parts.append(f"\n**Task:** {self.task_description}")

        if self.progress:
            completed = self.progress.get('completed', 0)
            total = self.progress.get('total', 0)
            parts.append(f"\n**Progress:** {completed}/{total} complete")
            items = self.progress.get('items', [])
            pending = [i for i in items if not i.get('done')]
            if pending:
                parts.append("Pending items:")
                for item in pending[:5]:  # Limit to avoid context bloat
                    parts.append(f"  - {item.get('task', '')}")

        if self.files_touched:
            parts.append(f"\n**Files involved:** {', '.join(self.files_touched[:10])}")

        if self.decisions:
            parts.append("\n**Key decisions made:**")
            for d in self.decisions[:5]:
                parts.append(f"  - {d}")

        if self.next_steps:
            parts.append("\n**Next steps:**")
            for n in self.next_steps[:5]:
                parts.append(f"  - {n}")

        if self.blockers:
            parts.append("\n**Blockers:**")
            for b in self.blockers:
                parts.append(f"  - {b}")

        parts.append("</session-resume>")

        return "\n".join(parts)


class SessionStore:
    """Manages session persistence using llama-memory as backend."""

    def __init__(self, memory_store=None):
        """Initialize with optional memory store.

        Args:
            memory_store: MemoryStore instance. If None, uses default.
        """
        self._store = memory_store

    @property
    def store(self):
        """Get or create memory store."""
        if self._store is None:
            from .memory import get_store
            self._store = get_store()
        return self._store

    def save(
        self,
        title: str,
        summary: Optional[str] = None,
        task_description: Optional[str] = None,
        progress: Optional[Dict[str, Any]] = None,
        files_touched: Optional[List[str]] = None,
        decisions: Optional[List[str]] = None,
        next_steps: Optional[List[str]] = None,
        blockers: Optional[List[str]] = None,
        project: Optional[str] = None,
        notes: Optional[str] = None,
        tags: Optional[List[str]] = None,
        session_id: Optional[str] = None,
        started_at: Optional[int] = None,
    ) -> Session:
        """Save a session state.

        Args:
            title: Short title for the session
            summary: What was being worked on
            task_description: Detailed task description
            progress: Progress dict {"completed": N, "total": M, "items": [...]}
            files_touched: List of files read/modified
            decisions: Key decisions made during session
            next_steps: What to do next when resuming
            blockers: What's blocking progress
            project: Project name
            notes: Freeform notes
            tags: Tags for categorization
            session_id: Claude Code session ID if available
            started_at: Unix timestamp when session started

        Returns:
            Session object with ID
        """
        now = int(time.time())

        session = Session(
            title=title,
            summary=summary or title,
            task_description=task_description or "",
            progress=progress or {},
            files_touched=files_touched or [],
            decisions=decisions or [],
            next_steps=next_steps or [],
            blockers=blockers or [],
            project=project,
            notes=notes or "",
            tags=tags or [],
            session_id=session_id,
            started_at=started_at or now,
            ended_at=now,
        )

        # Store as memory with type="session"
        content = session.to_content()
        metadata = session.to_metadata()

        # Add session tag
        session_tags = list(session.tags)
        if 'session' not in session_tags:
            session_tags.insert(0, 'session')

        memory_id, _ = self.store.store(
            content=content,
            type='session',
            summary=session.summary,
            project=session.project,
            tags=session_tags,
            importance=7,  # Sessions are important
            metadata=metadata,
            force=True,  # Always store, don't check duplicates
        )

        session.id = memory_id
        return session

    def get(self, session_id: int) -> Optional[Session]:
        """Get a specific session by ID.

        Args:
            session_id: Memory ID of the session

        Returns:
            Session object or None if not found
        """
        memory = self.store.get(session_id)
        if memory and memory.type == 'session':
            return Session.from_memory(memory)
        return None

    def list(
        self,
        project: Optional[str] = None,
        limit: int = 10,
    ) -> List[Session]:
        """List recent sessions.

        Args:
            project: Filter by project
            limit: Maximum number of sessions to return

        Returns:
            List of Session objects, most recent first
        """
        memories = self.store.list(
            type='session',
            project=project,
            limit=limit,
            order_by='created_at',
        )

        return [Session.from_memory(m) for m in memories]

    def resume(
        self,
        session_id: Optional[int] = None,
        project: Optional[str] = None,
    ) -> Optional[str]:
        """Get a resume prompt for a session.

        Args:
            session_id: Specific session ID to resume
            project: If no session_id, get latest for this project

        Returns:
            Resume prompt string or None if no session found
        """
        if session_id:
            session = self.get(session_id)
        else:
            # Get most recent session, optionally filtered by project
            sessions = self.list(project=project, limit=1)
            session = sessions[0] if sessions else None

        if session:
            return session.to_resume_prompt()
        return None

    def search(
        self,
        query: str,
        project: Optional[str] = None,
        limit: int = 5,
    ) -> List[Session]:
        """Search sessions by content.

        Args:
            query: Search query
            project: Filter by project
            limit: Maximum results

        Returns:
            List of matching Session objects
        """
        memories = self.store.search(
            query=query,
            type='session',
            project=project,
            limit=limit,
        )

        return [Session.from_memory(m) for m in memories]


# Module-level convenience functions
_session_store: Optional[SessionStore] = None


def get_session_store() -> SessionStore:
    """Get or create the session store singleton."""
    global _session_store
    if _session_store is None:
        _session_store = SessionStore()
    return _session_store


def save_session(title: str, **kwargs) -> Session:
    """Save a session. See SessionStore.save() for arguments."""
    return get_session_store().save(title, **kwargs)


def resume_session(
    session_id: Optional[int] = None,
    project: Optional[str] = None,
) -> Optional[str]:
    """Get resume prompt for a session."""
    return get_session_store().resume(session_id, project)


def list_sessions(
    project: Optional[str] = None,
    limit: int = 10,
) -> List[Session]:
    """List recent sessions."""
    return get_session_store().list(project, limit)


def get_session(session_id: int) -> Optional[Session]:
    """Get a specific session."""
    return get_session_store().get(session_id)

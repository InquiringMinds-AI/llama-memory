"""
Auto-capture module for llama-memory v2.5.

Provides development modes for automatic memory capture:
- green: Active development - capture everything
- brown: Exploration - capture sessions and discoveries
- quiet: Minimal capture (default)

Note: This module provides configuration and logging infrastructure.
Actual file watching requires external integration (hooks, scripts, etc.).
"""

from __future__ import annotations

import json
import os
from typing import Optional, List, Literal
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path


CaptureMode = Literal['green', 'brown', 'quiet', 'custom']
EventType = Literal['file_change', 'command', 'session_start', 'session_end', 'error', 'discovery']


@dataclass
class CaptureConfig:
    """Configuration for auto-capture behavior."""

    mode: CaptureMode = 'quiet'

    # What to capture
    capture_files: bool = False
    capture_commands: bool = False
    capture_sessions: bool = False
    capture_errors: bool = False

    # Filters
    include_paths: List[str] = field(default_factory=list)
    exclude_paths: List[str] = field(default_factory=lambda: [
        '.git', 'node_modules', '__pycache__', '.venv', 'venv',
        '.cache', '.tmp', '*.pyc', '*.pyo', '*.log'
    ])
    include_commands: List[str] = field(default_factory=list)
    exclude_commands: List[str] = field(default_factory=lambda: [
        'ls', 'cd', 'pwd', 'clear', 'exit', 'cat', 'echo', 'history'
    ])

    # Options
    min_file_change_size: int = 10  # Min chars changed to capture
    session_summary_threshold: int = 5  # Min actions before session summary
    auto_importance: int = 4  # Default importance for auto-captured memories
    auto_context: Optional[str] = None  # Context for auto-captured memories

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> 'CaptureConfig':
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @classmethod
    def for_mode(cls, mode: CaptureMode) -> 'CaptureConfig':
        """Create config for a predefined mode."""
        if mode == 'green':
            return cls(
                mode='green',
                capture_files=True,
                capture_commands=True,
                capture_sessions=True,
                capture_errors=True,
                auto_importance=5,
            )
        elif mode == 'brown':
            return cls(
                mode='brown',
                capture_files=False,
                capture_commands=False,
                capture_sessions=True,
                capture_errors=True,
                auto_importance=4,
            )
        else:  # quiet
            return cls(
                mode='quiet',
                capture_files=False,
                capture_commands=False,
                capture_sessions=False,
                capture_errors=False,
            )


@dataclass
class CaptureEvent:
    """A captured event to potentially store as memory."""
    event_type: EventType
    content: str
    timestamp: int
    metadata: dict = field(default_factory=dict)

    # Optional fields set during processing
    memory_id: Optional[int] = None
    processed: bool = False

    def to_dict(self) -> dict:
        return {
            'event_type': self.event_type,
            'content': self.content,
            'timestamp': self.timestamp,
            'metadata': self.metadata,
            'memory_id': self.memory_id,
            'processed': self.processed,
        }


class CaptureEngine:
    """Engine for managing auto-capture of memories.

    This class manages:
    - Capture configuration and modes
    - Event logging to capture_log table
    - Processing logged events into memories
    """

    CONFIG_KEY = 'capture_config'

    def __init__(self, store=None):
        """Initialize capture engine.

        Args:
            store: MemoryStore instance (will get default if None)
        """
        self._store = store
        self._config: Optional[CaptureConfig] = None

    def _get_store(self):
        """Get or create memory store."""
        if self._store is None:
            from .memory import get_store
            self._store = get_store()
        return self._store

    @property
    def config(self) -> CaptureConfig:
        """Get current capture configuration."""
        if self._config is None:
            self._config = self._load_config()
        return self._config

    def _load_config(self) -> CaptureConfig:
        """Load config from database meta table."""
        store = self._get_store()
        conn = store.db.conn

        row = conn.execute(
            "SELECT value FROM meta WHERE key = ?",
            (self.CONFIG_KEY,)
        ).fetchone()

        if row:
            try:
                data = json.loads(row['value'])
                return CaptureConfig.from_dict(data)
            except (json.JSONDecodeError, TypeError):
                pass

        return CaptureConfig()

    def _save_config(self):
        """Save config to database meta table."""
        store = self._get_store()
        conn = store.db.conn

        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
            (self.CONFIG_KEY, json.dumps(self.config.to_dict()))
        )
        conn.commit()

    def set_mode(self, mode: CaptureMode) -> CaptureConfig:
        """Set capture mode using predefined config.

        Args:
            mode: 'green', 'brown', or 'quiet'

        Returns:
            New configuration
        """
        self._config = CaptureConfig.for_mode(mode)
        self._save_config()
        return self._config

    def update_config(self, **kwargs) -> CaptureConfig:
        """Update specific config values.

        Args:
            **kwargs: Config fields to update

        Returns:
            Updated configuration
        """
        config = self.config
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        config.mode = 'custom'  # Mark as custom when manually updated
        self._config = config
        self._save_config()
        return config

    def get_status(self) -> dict:
        """Get current capture status and stats."""
        store = self._get_store()
        conn = store.db.conn

        config = self.config

        # Count pending events
        pending = conn.execute(
            "SELECT COUNT(*) FROM capture_log WHERE processed = 0"
        ).fetchone()[0]

        # Count total captured events
        total = conn.execute(
            "SELECT COUNT(*) FROM capture_log"
        ).fetchone()[0]

        # Count by event type
        type_rows = conn.execute(
            "SELECT event_type, COUNT(*) as count FROM capture_log GROUP BY event_type"
        ).fetchall()
        by_type = {row['event_type']: row['count'] for row in type_rows}

        return {
            'mode': config.mode,
            'capture_files': config.capture_files,
            'capture_commands': config.capture_commands,
            'capture_sessions': config.capture_sessions,
            'capture_errors': config.capture_errors,
            'pending_events': pending,
            'total_events': total,
            'events_by_type': by_type,
            'auto_context': config.auto_context,
        }

    def should_capture_file(self, path: str) -> bool:
        """Check if a file change should be captured."""
        if not self.config.capture_files:
            return False

        # Check exclude patterns
        path_obj = Path(path)
        for pattern in self.config.exclude_paths:
            if pattern.startswith('*'):
                # Glob pattern for extension
                if path_obj.suffix == pattern[1:] or path_obj.match(pattern):
                    return False
            elif pattern in str(path):
                return False

        # Check include patterns (if any specified, path must match)
        if self.config.include_paths:
            matched = False
            for pattern in self.config.include_paths:
                if pattern in str(path) or path_obj.match(pattern):
                    matched = True
                    break
            if not matched:
                return False

        return True

    def should_capture_command(self, command: str) -> bool:
        """Check if a command should be captured."""
        if not self.config.capture_commands:
            return False

        # Get base command (first word)
        base_cmd = command.split()[0] if command.split() else ''

        # Check excludes
        if base_cmd in self.config.exclude_commands:
            return False

        # Check includes (if any specified)
        if self.config.include_commands:
            if base_cmd not in self.config.include_commands:
                return False

        return True

    def log_event(self, event: CaptureEvent) -> int:
        """Log an event to the capture_log table.

        Args:
            event: Event to log

        Returns:
            Log entry ID
        """
        store = self._get_store()
        conn = store.db.conn

        cursor = conn.execute("""
            INSERT INTO capture_log (event_type, event_data, captured_at, processed)
            VALUES (?, ?, ?, 0)
        """, (event.event_type, json.dumps(event.to_dict()), event.timestamp))

        conn.commit()
        return cursor.lastrowid

    def log_file_change(self, path: str, change_type: str = 'modified',
                        content_preview: Optional[str] = None) -> Optional[int]:
        """Log a file change event.

        Args:
            path: File path
            change_type: 'created', 'modified', 'deleted'
            content_preview: Optional preview of changed content

        Returns:
            Log entry ID if captured, None if filtered out
        """
        if not self.should_capture_file(path):
            return None

        event = CaptureEvent(
            event_type='file_change',
            content=f"File {change_type}: {path}",
            timestamp=int(datetime.now().timestamp()),
            metadata={
                'path': path,
                'change_type': change_type,
                'preview': content_preview[:200] if content_preview else None,
            }
        )
        return self.log_event(event)

    def log_command(self, command: str, exit_code: int = 0,
                    output_preview: Optional[str] = None) -> Optional[int]:
        """Log a command execution event.

        Args:
            command: Command executed
            exit_code: Command exit code
            output_preview: Optional preview of output

        Returns:
            Log entry ID if captured, None if filtered out
        """
        if not self.should_capture_command(command):
            return None

        event = CaptureEvent(
            event_type='command',
            content=command,
            timestamp=int(datetime.now().timestamp()),
            metadata={
                'exit_code': exit_code,
                'output_preview': output_preview[:500] if output_preview else None,
            }
        )
        return self.log_event(event)

    def log_session_start(self, session_id: str) -> int:
        """Log session start."""
        if not self.config.capture_sessions:
            return -1

        event = CaptureEvent(
            event_type='session_start',
            content=f"Session started: {session_id}",
            timestamp=int(datetime.now().timestamp()),
            metadata={'session_id': session_id}
        )
        return self.log_event(event)

    def log_session_end(self, session_id: str, summary: Optional[str] = None) -> int:
        """Log session end with optional summary."""
        if not self.config.capture_sessions:
            return -1

        event = CaptureEvent(
            event_type='session_end',
            content=summary or f"Session ended: {session_id}",
            timestamp=int(datetime.now().timestamp()),
            metadata={'session_id': session_id, 'summary': summary}
        )
        return self.log_event(event)

    def log_error(self, error: str, context: Optional[str] = None) -> int:
        """Log an error event."""
        if not self.config.capture_errors:
            return -1

        event = CaptureEvent(
            event_type='error',
            content=error,
            timestamp=int(datetime.now().timestamp()),
            metadata={'context': context}
        )
        return self.log_event(event)

    def log_discovery(self, content: str, importance: int = 5) -> int:
        """Log a discovery (always captured regardless of mode).

        Use this for important findings that should be remembered.
        """
        event = CaptureEvent(
            event_type='discovery',
            content=content,
            timestamp=int(datetime.now().timestamp()),
            metadata={'importance': importance}
        )
        return self.log_event(event)

    def process_pending(self, limit: int = 100) -> dict:
        """Process pending capture log entries into memories.

        Args:
            limit: Maximum entries to process

        Returns:
            Dict with processing stats
        """
        store = self._get_store()
        conn = store.db.conn
        config = self.config

        stats = {'processed': 0, 'stored': 0, 'skipped': 0, 'errors': 0}

        rows = conn.execute("""
            SELECT id, event_type, event_data, captured_at
            FROM capture_log
            WHERE processed = 0
            ORDER BY captured_at ASC
            LIMIT ?
        """, (limit,)).fetchall()

        for row in rows:
            try:
                data = json.loads(row['event_data'])
                event_type = row['event_type']
                content = data.get('content', '')

                # Determine if this should become a memory
                should_store = event_type in ('discovery', 'session_end', 'error')
                if event_type == 'file_change' and config.capture_files:
                    should_store = True
                if event_type == 'command' and config.capture_commands:
                    should_store = True

                memory_id = None
                if should_store and content:
                    # Determine memory type
                    mem_type = 'event'
                    if event_type == 'discovery':
                        mem_type = 'fact'
                    elif event_type == 'error':
                        mem_type = 'event'

                    # Get importance from metadata or use default
                    importance = data.get('metadata', {}).get('importance', config.auto_importance)

                    mem_id, _ = store.store(
                        content=content,
                        type=mem_type,
                        importance=importance,
                        context=config.auto_context,
                        source='auto',
                        metadata={'capture_event': event_type, 'capture_data': data.get('metadata')},
                        force=True,
                    )
                    memory_id = mem_id
                    stats['stored'] += 1
                else:
                    stats['skipped'] += 1

                # Mark as processed
                conn.execute("""
                    UPDATE capture_log SET processed = 1, memory_id = ?
                    WHERE id = ?
                """, (memory_id, row['id']))

                stats['processed'] += 1

            except Exception as e:
                stats['errors'] += 1

        conn.commit()
        return stats

    def get_recent_events(self, limit: int = 50, event_type: Optional[str] = None) -> List[dict]:
        """Get recent capture log entries.

        Args:
            limit: Maximum entries to return
            event_type: Filter by event type

        Returns:
            List of event dicts
        """
        store = self._get_store()
        conn = store.db.conn

        sql = "SELECT * FROM capture_log"
        params = []

        if event_type:
            sql += " WHERE event_type = ?"
            params.append(event_type)

        sql += " ORDER BY captured_at DESC LIMIT ?"
        params.append(limit)

        rows = conn.execute(sql, params).fetchall()

        return [{
            'id': row['id'],
            'event_type': row['event_type'],
            'event_data': json.loads(row['event_data']) if row['event_data'] else {},
            'captured_at': row['captured_at'],
            'memory_id': row['memory_id'],
            'processed': bool(row['processed']),
        } for row in rows]

    def clear_processed(self, older_than_days: int = 7) -> int:
        """Clear old processed events from capture_log.

        Args:
            older_than_days: Delete entries older than this

        Returns:
            Number of entries deleted
        """
        store = self._get_store()
        conn = store.db.conn

        cutoff = int(datetime.now().timestamp()) - (older_than_days * 24 * 60 * 60)

        cursor = conn.execute("""
            DELETE FROM capture_log
            WHERE processed = 1 AND captured_at < ?
        """, (cutoff,))

        conn.commit()
        return cursor.rowcount


# Module-level convenience
_engine: Optional[CaptureEngine] = None


def get_capture_engine() -> CaptureEngine:
    """Get or create capture engine singleton."""
    global _engine
    if _engine is None:
        _engine = CaptureEngine()
    return _engine


def set_capture_mode(mode: CaptureMode) -> CaptureConfig:
    """Set capture mode."""
    return get_capture_engine().set_mode(mode)


def get_capture_status() -> dict:
    """Get capture status."""
    return get_capture_engine().get_status()

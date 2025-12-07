"""
Transcript parser for Claude Code conversations.

Extracts session state from Claude Code's conversation transcript (JSONL format)
to enable automatic session persistence.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any


@dataclass
class TranscriptState:
    """Extracted state from a Claude Code transcript."""

    # Task tracking
    todos: List[Dict[str, Any]] = field(default_factory=list)  # From TodoWrite
    completed_todos: int = 0
    total_todos: int = 0

    # Files touched
    files_read: List[str] = field(default_factory=list)
    files_written: List[str] = field(default_factory=list)
    files_edited: List[str] = field(default_factory=list)

    # Context
    user_messages: List[str] = field(default_factory=list)
    last_user_message: str = ""
    assistant_summaries: List[str] = field(default_factory=list)

    # Metadata
    session_id: Optional[str] = None
    message_count: int = 0
    tool_calls: int = 0

    @property
    def files_touched(self) -> List[str]:
        """All unique files touched in any way."""
        seen = set()
        result = []
        for f in self.files_written + self.files_edited + self.files_read:
            if f not in seen:
                seen.add(f)
                result.append(f)
        return result

    @property
    def progress(self) -> Dict[str, Any]:
        """Progress dict for session storage."""
        if not self.todos:
            return {}
        return {
            "completed": self.completed_todos,
            "total": self.total_todos,
            "items": [
                {"task": t.get("content", ""), "done": t.get("status") == "completed"}
                for t in self.todos
            ]
        }

    def infer_summary(self, max_length: int = 200) -> str:
        """Infer a summary from the conversation."""
        # Use the first user message as primary context
        if self.user_messages:
            first_msg = self.user_messages[0]
            # Truncate if needed
            if len(first_msg) > max_length:
                return first_msg[:max_length-3] + "..."
            return first_msg
        return "Claude Code session"

    def infer_title(self, max_length: int = 60) -> str:
        """Infer a short title from the conversation."""
        summary = self.infer_summary(max_length)
        # Take first sentence or line
        for sep in ['. ', '\n', '? ', '! ']:
            if sep in summary:
                summary = summary.split(sep)[0]
                break
        if len(summary) > max_length:
            summary = summary[:max_length-3] + "..."
        return summary

    def infer_next_steps(self) -> List[str]:
        """Infer next steps from in-progress and pending todos."""
        # In-progress first, then pending
        in_progress = [
            t.get("content", "")
            for t in self.todos
            if t.get("status") == "in_progress"
        ]
        pending = [
            t.get("content", "")
            for t in self.todos
            if t.get("status") == "pending"
        ]
        return (in_progress + pending)[:5]  # Limit to 5


class TranscriptParser:
    """Parses Claude Code conversation transcripts."""

    def __init__(self):
        self.state = TranscriptState()

    def parse_file(self, transcript_path: str) -> TranscriptState:
        """Parse a transcript file and extract state."""
        path = Path(transcript_path)
        if not path.exists():
            return self.state

        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    self._process_entry(entry)
                except json.JSONDecodeError:
                    continue

        return self.state

    def parse_string(self, content: str) -> TranscriptState:
        """Parse transcript content from a string."""
        for line in content.split('\n'):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                self._process_entry(entry)
            except json.JSONDecodeError:
                continue

        return self.state

    def _process_entry(self, entry: Dict[str, Any]):
        """Process a single transcript entry."""
        self.state.message_count += 1

        # Extract session ID
        if 'session_id' in entry:
            self.state.session_id = entry['session_id']

        # Handle different message types
        msg_type = entry.get('type')
        role = entry.get('role')

        if role == 'user':
            self._process_user_message(entry)
        elif role == 'assistant':
            self._process_assistant_message(entry)

        # Handle tool use
        if 'tool_use' in entry or msg_type == 'tool_use':
            self._process_tool_use(entry)

        # Handle tool results
        if 'tool_result' in entry or msg_type == 'tool_result':
            self._process_tool_result(entry)

    def _process_user_message(self, entry: Dict[str, Any]):
        """Extract user message content."""
        content = entry.get('content', '')
        if isinstance(content, list):
            # Handle content blocks
            text_parts = []
            for block in content:
                if isinstance(block, dict) and block.get('type') == 'text':
                    text_parts.append(block.get('text', ''))
                elif isinstance(block, str):
                    text_parts.append(block)
            content = '\n'.join(text_parts)

        if content and isinstance(content, str):
            # Filter out system messages and very short inputs
            if not content.startswith('<') and len(content) > 5:
                self.state.user_messages.append(content)
                self.state.last_user_message = content

    def _process_assistant_message(self, entry: Dict[str, Any]):
        """Extract assistant message summaries."""
        content = entry.get('content', '')
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    if block.get('type') == 'text':
                        text = block.get('text', '')
                        # Extract first paragraph as potential summary
                        if text and '\n' in text:
                            first_para = text.split('\n\n')[0]
                            if 20 < len(first_para) < 500:
                                self.state.assistant_summaries.append(first_para)

    def _process_tool_use(self, entry: Dict[str, Any]):
        """Process tool use entries."""
        self.state.tool_calls += 1

        # Handle nested tool_use structure
        tool_use = entry.get('tool_use', entry)
        if isinstance(tool_use, dict):
            name = tool_use.get('name', '')
            input_data = tool_use.get('input', {})
        else:
            return

        # Also check content blocks for tool_use
        content = entry.get('content', [])
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get('type') == 'tool_use':
                    name = block.get('name', name)
                    input_data = block.get('input', input_data)

        self._handle_tool(name, input_data)

    def _process_tool_result(self, entry: Dict[str, Any]):
        """Process tool result entries (might contain tool info)."""
        # Sometimes tool name is in the result
        tool_result = entry.get('tool_result', entry)
        if isinstance(tool_result, dict):
            name = tool_result.get('tool_name', '')
            if name:
                # We don't have input here, but note it was used
                pass

    def _handle_tool(self, name: str, input_data: Dict[str, Any]):
        """Handle specific tool types."""
        if not name:
            return

        # File operations
        if name == 'Read':
            file_path = input_data.get('file_path', '')
            if file_path and file_path not in self.state.files_read:
                self.state.files_read.append(file_path)

        elif name == 'Write':
            file_path = input_data.get('file_path', '')
            if file_path and file_path not in self.state.files_written:
                self.state.files_written.append(file_path)

        elif name == 'Edit':
            file_path = input_data.get('file_path', '')
            if file_path and file_path not in self.state.files_edited:
                self.state.files_edited.append(file_path)

        # Todo tracking
        elif name == 'TodoWrite':
            todos = input_data.get('todos', [])
            if todos:
                self.state.todos = todos
                self.state.completed_todos = sum(
                    1 for t in todos if t.get('status') == 'completed'
                )
                self.state.total_todos = len(todos)

    def to_session_kwargs(self) -> Dict[str, Any]:
        """Convert state to kwargs for session.save()."""
        return {
            'title': self.state.infer_title(),
            'summary': self.state.infer_summary(),
            'task_description': self.state.last_user_message[:500] if self.state.last_user_message else '',
            'progress': self.state.progress,
            'files_touched': self.state.files_touched[:20],  # Limit
            'next_steps': self.state.infer_next_steps(),
            'session_id': self.state.session_id,
        }


def parse_transcript(transcript_path: str) -> TranscriptState:
    """Parse a transcript file and return extracted state."""
    parser = TranscriptParser()
    return parser.parse_file(transcript_path)


def transcript_to_session_kwargs(transcript_path: str) -> Dict[str, Any]:
    """Parse transcript and return kwargs for session.save()."""
    parser = TranscriptParser()
    parser.parse_file(transcript_path)
    return parser.to_session_kwargs()

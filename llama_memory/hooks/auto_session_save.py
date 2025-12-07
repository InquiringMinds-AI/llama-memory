#!/usr/bin/env python3
"""
Auto-save session hook for Claude Code.

Triggered by: PreCompact, SessionEnd
Parses the conversation transcript and saves session state to llama-memory.

Install: mem hooks install
Configure in ~/.claude/settings.json
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Add llama_memory package to path for imports when run as script
package_dir = Path(__file__).parent.parent.parent  # Go up to the directory containing llama_memory/
if str(package_dir) not in sys.path:
    sys.path.insert(0, str(package_dir))


def main():
    """Main hook entry point."""
    try:
        # Read hook input from stdin
        input_data = json.load(sys.stdin)
    except (json.JSONDecodeError, EOFError):
        # No input or invalid JSON - exit silently
        sys.exit(0)

    hook_event = input_data.get('hook_event_name', '')
    transcript_path = input_data.get('transcript_path', '')
    session_id = input_data.get('session_id', '')
    cwd = input_data.get('cwd', os.getcwd())

    # Only process relevant hooks
    if hook_event not in ('PreCompact', 'SessionEnd'):
        sys.exit(0)

    # Need transcript to extract state
    if not transcript_path or not Path(transcript_path).exists():
        sys.exit(0)

    # For PreCompact, check if it's auto-triggered (context pressure)
    if hook_event == 'PreCompact':
        trigger = input_data.get('trigger', '')
        if trigger != 'auto':
            # Manual compaction, skip
            sys.exit(0)

    try:
        # Import here to avoid startup cost if hook exits early
        from llama_memory.transcript import TranscriptParser
        from llama_memory.session import save_session

        # Parse transcript
        parser = TranscriptParser()
        state = parser.parse_file(transcript_path)

        # Skip if no meaningful content
        if state.message_count < 2:
            sys.exit(0)

        # Determine project from cwd
        project = None
        cwd_path = Path(cwd)
        if cwd_path.name and cwd_path.name != 'home':
            project = cwd_path.name

        # Build session data
        kwargs = parser.to_session_kwargs()
        kwargs['project'] = project
        kwargs['session_id'] = session_id

        # Add note about how it was saved
        if hook_event == 'PreCompact':
            kwargs['notes'] = f"Auto-saved before context compaction at {datetime.now().isoformat()}"
        else:
            kwargs['notes'] = f"Auto-saved on session end at {datetime.now().isoformat()}"

        # Tag appropriately
        kwargs['tags'] = ['auto-saved', hook_event.lower()]

        # Save the session
        session = save_session(**kwargs)

        # Print visual indicator to stderr (visible to user)
        title = kwargs.get('title', 'Session')
        sys.stderr.write(f"\n\033[1;32m>>> Session saved: {title} (ID: {session.id})\033[0m\n")
        sys.stderr.flush()

        # Output success message (shown in verbose mode)
        result = {
            "continue": True,
            "systemMessage": f"Session auto-saved (ID: {session.id})"
        }
        print(json.dumps(result))

    except ImportError as e:
        # llama-memory not installed properly
        sys.stderr.write(f"llama-memory import error: {e}\n")
        sys.exit(0)  # Non-blocking error
    except Exception as e:
        # Log error but don't block
        sys.stderr.write(f"Auto-save error: {e}\n")
        sys.exit(0)  # Non-blocking error


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Auto-resume session hook for Claude Code.

Triggered by: SessionStart
Injects relevant session context when starting/resuming a session.

Install: mem hooks install
Configure in ~/.claude/settings.json
"""

import json
import os
import sys
from pathlib import Path

# Add llama_memory package to path for imports when run as script
package_dir = Path(__file__).parent.parent.parent
if str(package_dir) not in sys.path:
    sys.path.insert(0, str(package_dir))


def main():
    """Main hook entry point."""
    try:
        # Read hook input from stdin
        input_data = json.load(sys.stdin)
    except (json.JSONDecodeError, EOFError):
        sys.exit(0)

    hook_event = input_data.get('hook_event_name', '')
    source = input_data.get('source', '')  # 'startup', 'resume', 'compact', 'clear'
    cwd = input_data.get('cwd', os.getcwd())

    # Only process SessionStart
    if hook_event != 'SessionStart':
        sys.exit(0)

    # Determine when to inject context
    # - 'compact': Context was just compacted, inject session state
    # - 'resume': Resuming a previous session
    # - 'startup': Fresh start, optionally inject recent session
    inject_on = ('compact', 'resume')

    # For fresh startup, check if there's a very recent session (within 1 hour)
    inject_recent_on_startup = True

    if source not in inject_on and not (source == 'startup' and inject_recent_on_startup):
        sys.exit(0)

    try:
        from llama_memory.session import get_session_store, list_sessions
        import time

        # Determine project from cwd
        project = None
        cwd_path = Path(cwd)
        if cwd_path.name and cwd_path.name != 'home':
            project = cwd_path.name

        # Get recent sessions
        sessions = list_sessions(project=project, limit=1)

        if not sessions:
            # Try global sessions
            sessions = list_sessions(limit=1)

        if not sessions:
            sys.exit(0)

        session = sessions[0]

        # For startup, only inject if session is recent (within 1 hour)
        if source == 'startup':
            now = int(time.time())
            if session.ended_at and (now - session.ended_at) > 3600:
                # Session is old, don't auto-inject
                sys.exit(0)

        # Generate resume prompt
        resume_prompt = session.to_resume_prompt()

        # Print visual indicator to stderr (visible to user)
        title = session.title or "Previous session"
        sys.stderr.write(f"\n\033[1;36m>>> Session restored: {title}\033[0m\n")
        sys.stderr.flush()

        # Build context message
        context_parts = []

        if source == 'compact':
            context_parts.append("**Note: Context was just compacted. Here's your session state:**\n")
        elif source == 'resume':
            context_parts.append("**Resuming previous session:**\n")
        else:
            context_parts.append("**Recent session detected:**\n")

        context_parts.append(resume_prompt)

        # Also inject relevant memories if available
        try:
            from llama_memory.memory import get_store
            store = get_store()

            # Search for relevant memories based on session summary
            if session.summary:
                memories = store.search(
                    query=session.summary,
                    limit=3,
                    project=project
                )
                if memories:
                    context_parts.append("\n**Relevant memories:**")
                    for mem in memories:
                        context_parts.append(f"- [{mem.type}] {mem.summary or mem.content[:100]}")
        except Exception:
            pass  # Memory search is optional

        # Output context (printed to stdout becomes injected context)
        print('\n'.join(context_parts))

    except ImportError as e:
        sys.stderr.write(f"llama-memory import error: {e}\n")
        sys.exit(0)
    except Exception as e:
        sys.stderr.write(f"Auto-resume error: {e}\n")
        sys.exit(0)


if __name__ == '__main__':
    main()

"""
Claude Code hooks for automatic session persistence.

These hooks integrate with Claude Code's hook system to automatically
save and restore session state.

Hooks:
- auto_session_save: Saves session state on PreCompact and SessionEnd
- auto_session_resume: Restores session context on SessionStart

Install with: llama-memory hooks install
"""

from pathlib import Path

HOOKS_DIR = Path(__file__).parent

AUTO_SESSION_SAVE = HOOKS_DIR / "auto_session_save.py"
AUTO_SESSION_RESUME = HOOKS_DIR / "auto_session_resume.py"

__all__ = [
    "HOOKS_DIR",
    "AUTO_SESSION_SAVE",
    "AUTO_SESSION_RESUME",
]

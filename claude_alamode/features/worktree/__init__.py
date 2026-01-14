"""Git worktree management feature.

Provides isolated feature development via git worktrees.
"""

# Public API - only what app.py needs
from claude_alamode.features.worktree.git import FinishInfo, list_worktrees
from claude_alamode.features.worktree.commands import handle_worktree_command

__all__ = [
    "FinishInfo",
    "list_worktrees",
    "handle_worktree_command",
]

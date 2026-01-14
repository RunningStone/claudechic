"""Worktree selection prompt."""

from textual.app import ComposeResult
from textual.widgets import Static

from claude_alamode.widgets.prompts import BasePrompt


class WorktreePrompt(BasePrompt):
    """Prompt for selecting or creating worktrees."""

    def __init__(self, worktrees: list[tuple[str, str]]) -> None:
        """Create worktree prompt.

        Args:
            worktrees: List of (path, branch) tuples for existing worktrees
        """
        super().__init__()
        self.worktrees = worktrees

    def compose(self) -> ComposeResult:
        yield Static("Worktrees", classes="prompt-title")
        for i, (path, branch) in enumerate(self.worktrees):
            classes = "prompt-option selected" if i == 0 else "prompt-option"
            yield Static(f"{i + 1}. {branch}", classes=classes, id=f"opt-{i}")
        # "New" option at the end
        new_idx = len(self.worktrees)
        classes = "prompt-option prompt-placeholder"
        if new_idx == 0:
            classes += " selected"
        yield Static(f"{new_idx + 1}. {self._text_option_placeholder()}", classes=classes, id=f"opt-{new_idx}")

    def _total_options(self) -> int:
        return len(self.worktrees) + 1  # +1 for "New"

    def _text_option_idx(self) -> int:
        return len(self.worktrees)

    def _text_option_placeholder(self) -> str:
        return "Enter name..."

    def _select_option(self, idx: int) -> None:
        if idx < len(self.worktrees):
            path, branch = self.worktrees[idx]
            self._resolve(("switch", path))
        else:
            self._text_buffer = ""
            self._enter_text_mode()
            self._update_text_display()

    def _submit_text(self, text: str) -> None:
        self._resolve(("new", text))

    async def wait(self) -> tuple[str, str] | None:
        """Wait for selection. Returns (action, value) or None if cancelled."""
        await super().wait()
        return self._result_value

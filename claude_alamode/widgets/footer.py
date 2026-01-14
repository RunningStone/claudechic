"""Custom footer widget."""

import subprocess

from textual.app import ComposeResult
from textual.reactive import reactive
from textual.widgets import Static
from textual.containers import Horizontal


def get_git_branch(cwd: str | None = None) -> str:
    """Get current git branch name."""
    try:
        result = subprocess.run(
            ["git", "branch", "--show-current"],
            capture_output=True,
            text=True,
            timeout=1,
            cwd=cwd,
        )
        return result.stdout.strip() or "detached"
    except Exception:
        return ""


class StatusFooter(Static):
    """Footer showing git branch, model, and auto-edit status."""

    can_focus = False
    auto_edit = reactive(False)
    model = reactive("")
    branch = reactive("")

    def on_mount(self) -> None:
        self.branch = get_git_branch()

    def refresh_branch(self, cwd: str | None = None) -> None:
        """Update branch from given directory."""
        self.branch = get_git_branch(cwd)

    def compose(self) -> ComposeResult:
        with Horizontal(id="footer-content"):
            yield Static("", id="model-label", classes="footer-label")
            yield Static("·", classes="footer-sep")
            yield Static("Auto-edit: off", id="auto-edit-label", classes="footer-label")
            yield Static("", id="footer-spacer")
            yield Static("", id="branch-label", classes="footer-label")

    def watch_branch(self, value: str) -> None:
        try:
            label = self.query_one("#branch-label", Static)
            label.update(f"⎇ {value}" if value else "")
        except Exception:
            pass

    def watch_model(self, value: str) -> None:
        try:
            label = self.query_one("#model-label", Static)
            label.update(value if value else "")
        except Exception:
            pass

    def watch_auto_edit(self, value: bool) -> None:
        try:
            label = self.query_one("#auto-edit-label", Static)
            label.update("Auto-edit: on" if value else "Auto-edit: off")
            label.set_class(value, "active")
        except Exception:
            pass

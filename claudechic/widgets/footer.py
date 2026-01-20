"""Custom footer widget."""

import asyncio

from textual.app import ComposeResult
from textual.message import Message
from textual.reactive import reactive
from textual.containers import Horizontal
from textual.widgets import Static

from claudechic.widgets.button import Button
from claudechic.widgets.indicators import CPUBar, ContextBar


class AutoEditLabel(Button):
    """Clickable auto-edit status label."""

    class Toggled(Message):
        """Emitted when auto-edit is toggled."""

    def on_click(self, event) -> None:
        self.post_message(self.Toggled())


class ModelLabel(Button):
    """Clickable model label."""

    class Clicked(Message):
        """Emitted when model label is clicked."""

    def on_click(self, event) -> None:
        self.post_message(self.Clicked())


async def get_git_branch(cwd: str | None = None) -> str:
    """Get current git branch name (async)."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "git",
            "branch",
            "--show-current",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=1)
        return stdout.decode().strip() or "detached"
    except Exception:
        return ""


class StatusFooter(Static):
    """Footer showing git branch, model, auto-edit status, and resource indicators."""

    can_focus = False
    auto_edit = reactive(False)
    model = reactive("")
    branch = reactive("")

    async def on_mount(self) -> None:
        self.branch = await get_git_branch()

    async def refresh_branch(self, cwd: str | None = None) -> None:
        """Update branch from given directory (async)."""
        self.branch = await get_git_branch(cwd)

    def compose(self) -> ComposeResult:
        with Horizontal(id="footer-content"):
            yield ModelLabel("", id="model-label", classes="footer-label")
            yield Static("·", classes="footer-sep")
            yield AutoEditLabel(
                "Auto-edit: off", id="auto-edit-label", classes="footer-label"
            )
            yield Static("", id="footer-spacer")
            yield ContextBar(id="context-bar")
            yield CPUBar(id="cpu-bar")
            yield Static("", id="branch-label", classes="footer-label")

    def watch_branch(self, value: str) -> None:
        """Update branch label when branch changes."""
        try:
            label = self.query_one("#branch-label", Static)
            label.update(f"⎇ {value}" if value else "")
        except Exception:
            pass

    def watch_model(self, value: str) -> None:
        """Update model label when model changes."""
        try:
            label = self.query_one("#model-label", ModelLabel)
            label.update(value if value else "")
        except Exception:
            pass

    def watch_auto_edit(self, value: bool) -> None:
        """Update auto-edit label when setting changes."""
        try:
            label = self.query_one("#auto-edit-label", AutoEditLabel)
            label.update("Auto-edit: on" if value else "Auto-edit: off")
            label.set_class(value, "active")
        except Exception:
            pass

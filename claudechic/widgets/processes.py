"""Process panel widget for displaying background processes."""

from dataclasses import dataclass
from datetime import datetime

from rich.text import Text
from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import Static


@dataclass
class BackgroundProcess:
    """A background process being tracked."""

    pid: int
    command: str  # Short description of the command
    start_time: datetime


class ProcessItem(Static):
    """Single process item with PID and command."""

    can_focus = False

    def __init__(self, process: BackgroundProcess) -> None:
        super().__init__()
        self.process = process

    def render(self) -> Text:
        # Show running indicator and truncated command
        cmd = self.process.command
        if len(cmd) > 20:
            cmd = cmd[:19] + "…"
        return Text.assemble(("● ", "yellow"), (cmd, ""))


class ProcessPanel(Widget):
    """Sidebar panel for background processes."""

    DEFAULT_CSS = """
    ProcessPanel {
        width: 100%;
        height: auto;
        max-height: 30%;
        border-top: solid $panel;
        padding: 1;
    }
    ProcessPanel.hidden {
        display: none;
    }
    ProcessPanel .process-title {
        color: $text-muted;
        text-style: bold;
        padding: 0 0 1 0;
    }
    ProcessItem {
        height: 1;
    }
    """

    can_focus = False

    def compose(self) -> ComposeResult:
        yield Static("Processes", classes="process-title")

    def update_processes(self, processes: list[BackgroundProcess]) -> None:
        """Replace processes with new list."""
        # Remove old items
        for item in self.query(ProcessItem):
            item.remove()

        # Add new items
        for proc in processes:
            self.mount(ProcessItem(proc))

        # Show/hide based on whether we have processes
        self.set_class(len(processes) == 0, "hidden")

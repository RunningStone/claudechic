"""Profile statistics modal."""

import pyperclip
from rich.console import Console

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, Horizontal, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Static, Button

from claudechic.profiling import get_stats_table, get_stats_text, _stats


def _get_table_width() -> int:
    """Calculate the rendered width of the stats table."""
    if not _stats:
        return 40
    console = Console(width=500, record=True)
    console.print(get_stats_table())
    text = console.export_text()
    max_width = max(len(line) for line in text.split("\n")) if text else 40
    return max_width + 8  # padding for container


class ProfileModal(ModalScreen):
    """Modal showing profiling statistics."""

    BINDINGS = [
        Binding("escape", "dismiss", "Close"),
    ]

    DEFAULT_CSS = """
    ProfileModal {
        align: center middle;
    }

    ProfileModal #profile-container {
        height: auto;
        max-height: 80%;
        background: $surface;
        border: solid $panel;
        padding: 1 2;
    }

    ProfileModal #profile-header {
        height: 1;
        margin-bottom: 1;
    }

    ProfileModal #profile-title {
        width: 1fr;
    }

    ProfileModal #copy-btn {
        width: 3;
        min-width: 3;
        height: 1;
        padding: 0;
        background: transparent;
        border: none;
        color: $text-muted;
    }

    ProfileModal #copy-btn:hover {
        color: $primary;
        background: transparent;
    }

    ProfileModal #profile-scroll {
        height: auto;
        max-height: 30;
    }

    ProfileModal #profile-content {
        height: auto;
    }

    ProfileModal #profile-footer {
        height: 1;
        margin-top: 1;
        align: center middle;
    }

    ProfileModal #close-btn {
        min-width: 10;
    }
    """

    def compose(self) -> ComposeResult:
        width = _get_table_width()
        with Vertical(id="profile-container"):
            with Horizontal(id="profile-header"):
                yield Static("[bold]Profiling Statistics[/]", id="profile-title", markup=True)
                yield Button("\u29c9", id="copy-btn")
            with VerticalScroll(id="profile-scroll"):
                if _stats:
                    yield Static(get_stats_table(), id="profile-content")
                else:
                    yield Static("No profiling data collected.", id="profile-content")
            with Horizontal(id="profile-footer"):
                yield Button("Close", id="close-btn")
        # Set container width after compose
        self.call_later(lambda: self._set_width(width))

    def _set_width(self, width: int) -> None:
        try:
            container = self.query_one("#profile-container")
            container.styles.width = width
        except Exception:
            pass

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "copy-btn":
            try:
                pyperclip.copy(get_stats_text())
                self.notify("Copied to clipboard")
            except Exception as e:
                self.notify(f"Copy failed: {e}", severity="error")
        elif event.button.id == "close-btn":
            self.dismiss()

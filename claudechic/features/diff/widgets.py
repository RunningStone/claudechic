"""Diff view widgets - sidebar, main view, and file panels."""

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.message import Message
from textual.widgets import Label, Static, TextArea

from claudechic.widgets.content.diff import DiffWidget

from .git import FileChange, Hunk, HunkComment


class DiffFileItem(Static):
    """A file entry in the diff sidebar. Click to scroll to that file's panel."""

    DEFAULT_CSS = """
    DiffFileItem {
        padding: 0 1;
        height: 1;
    }
    """

    class Selected(Message):
        """Posted when file should be highlighted (programmatic, no scroll)."""

        def __init__(self, path: str) -> None:
            super().__init__()
            self.path = path

    class Clicked(Message):
        """Posted when file is clicked by user (should scroll)."""

        def __init__(self, path: str) -> None:
            super().__init__()
            self.path = path

    def __init__(self, path: str, status: str, hunk_count: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.path = path
        self.status = status
        self.hunk_count = hunk_count

    def compose(self) -> ComposeResult:
        # Status indicator - use primary color, red for deleted
        indicator = {"modified": "M", "added": "A", "deleted": "D", "renamed": "R"}.get(
            self.status, "?"
        )
        color = "$primary" if self.status != "deleted" else "$error"
        # Show hunk count if > 1
        count_str = f" ({self.hunk_count})" if self.hunk_count > 1 else ""
        # Truncate path from front if too long (leave room for indicator + count)
        max_path_len = 24
        display_path = self.path
        if len(display_path) > max_path_len:
            display_path = "…" + display_path[-(max_path_len - 1) :]
        yield Label(f"[{color}]{indicator}[/] {display_path}[dim]{count_str}[/]")

    def on_click(self) -> None:
        self.post_message(self.Clicked(self.path))


class DiffSidebar(Vertical):
    """Left sidebar listing all changed files."""

    DEFAULT_CSS = """
    DiffSidebar {
        width: 30;
        border-right: solid $surface-darken-1;
        padding: 1 0;
    }
    DiffSidebar .section-header {
        padding: 0 1;
        text-style: bold;
        margin-bottom: 1;
    }
    """

    def __init__(self, changes: list[FileChange], **kwargs) -> None:
        super().__init__(**kwargs)
        self.changes = changes
        self._active_path: str | None = None

    def compose(self) -> ComposeResult:
        yield Label("Changed Files", classes="section-header")
        for change in self.changes:
            yield DiffFileItem(
                change.path,
                change.status,
                len(change.hunks),
                id=f"sidebar-{_sanitize_id(change.path)}",
            )

    def set_active(self, path: str) -> None:
        """Highlight the active file in the sidebar."""
        if self._active_path:
            try:
                old_item = self.query_one(
                    f"#sidebar-{_sanitize_id(self._active_path)}", DiffFileItem
                )
                old_item.remove_class("active")
            except Exception:
                pass
        self._active_path = path
        try:
            new_item = self.query_one(f"#sidebar-{_sanitize_id(path)}", DiffFileItem)
            new_item.add_class("active")
        except Exception:
            pass


class CommentInput(TextArea):
    """Multi-line input for adding comments to hunks. Enter submits, Ctrl+J for newline."""

    BINDINGS = [
        Binding("enter", "submit", "Submit", priority=True, show=False),
        Binding("ctrl+j", "newline", "Newline", priority=True, show=False),
        Binding("escape", "cancel", "Cancel", priority=True, show=False),
    ]

    DEFAULT_CSS = """
    CommentInput {
        height: auto;
        max-height: 10;
        min-height: 3;
        max-width: 80;
        margin-top: 1;
        background: $surface;
        border: solid $primary;
    }
    """

    class CommentSubmitted(Message):
        """Posted when comment is submitted (Enter pressed)."""

        def __init__(self, comment: str) -> None:
            super().__init__()
            self.comment = comment

    class CommentCancelled(Message):
        """Posted when comment is cancelled (Escape pressed)."""

    def __init__(self, value: str = "", **kwargs) -> None:
        kwargs.setdefault("soft_wrap", True)
        kwargs.setdefault("show_line_numbers", False)
        super().__init__(**kwargs)
        if value:
            self.text = value

    def action_submit(self) -> None:
        self.post_message(self.CommentSubmitted(self.text))

    def action_newline(self) -> None:
        self.insert("\n")

    def action_cancel(self) -> None:
        self.post_message(self.CommentCancelled())


class CommentLabel(Static):
    """Label showing a saved comment on a hunk."""

    DEFAULT_CSS = """
    CommentLabel {
        margin-top: 1;
        padding: 0 1;
        color: $warning;
    }
    """


class HunkSeparator(Static):
    """Full-width horizontal rule between hunks."""

    DEFAULT_CSS = """
    HunkSeparator {
        height: 1;
        width: 100%;
        margin: 1 0;
        color: #444444;
    }
    """

    def render(self):
        return "─" * (self.size.width or 80)


class HunkWidget(Static, can_focus=True):
    """Widget displaying a single hunk with syntax-highlighted diff."""

    DEFAULT_CSS = """
    HunkWidget {
        height: auto;
        border-left: tall $panel;
        padding-left: 1;
    }
    HunkWidget.has-comment {
        border-left: tall $warning;
    }
    HunkWidget:focus {
        border-left: tall $primary;
    }
    """

    def __init__(self, hunk: Hunk, path: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self.hunk = hunk
        self.path = path
        self.comment: str | None = None
        self._input: CommentInput | None = None
        self._label: CommentLabel | None = None

    def on_mouse_down(self) -> None:
        """Focus this hunk on click."""
        self.focus()

    def compose(self) -> ComposeResult:
        old_content = "\n".join(self.hunk.old_lines)
        new_content = "\n".join(self.hunk.new_lines)
        yield DiffWidget(
            old=old_content,
            new=new_content,
            path=self.path,
            old_start=self.hunk.old_start,
            new_start=self.hunk.new_start,
        )

    @property
    def editing(self) -> bool:
        """Return True if comment input is active."""
        return self._input is not None

    def start_editing(self) -> None:
        """Show comment input below the diff."""
        if self._input is not None:
            return
        # Hide label while editing
        if self._label:
            self._label.remove()
            self._label = None
        self._input = CommentInput(
            placeholder="Add comment...",
            value=self.comment or "",
        )
        self.mount(self._input)
        self._input.focus()

    def stop_editing(self, save: bool = True) -> None:
        """Hide comment input and optionally save the comment."""
        if self._input is None:
            return
        if save:
            self.comment = self._input.text.strip() or None
            if self.comment:
                self.add_class("has-comment")
            else:
                self.remove_class("has-comment")
        self._input.remove()
        self._input = None
        # Show label with comment text
        self._update_label()
        self.focus()

    def _update_label(self) -> None:
        """Show or hide the comment label based on current comment."""
        if self._label:
            self._label.remove()
            self._label = None
        if self.comment:
            self._label = CommentLabel(f"💬 {self.comment}")
            self.mount(self._label)

    def on_comment_input_comment_submitted(
        self, event: CommentInput.CommentSubmitted
    ) -> None:
        """Handle comment submission."""
        event.stop()
        self.stop_editing(save=True)

    def on_comment_input_comment_cancelled(
        self, event: CommentInput.CommentCancelled
    ) -> None:
        """Handle comment cancellation."""
        event.stop()
        self.stop_editing(save=False)


class FileDiffPanel(Vertical):
    """Panel showing all hunks for a single file."""

    DEFAULT_CSS = """
    FileDiffPanel {
        margin-bottom: 2;
        height: auto;
    }
    FileDiffPanel .file-header {
        background: $surface;
        padding: 0 1;
        text-style: bold;
        margin-bottom: 1;
    }
    """

    def __init__(self, change: FileChange, **kwargs) -> None:
        super().__init__(**kwargs)
        self.change = change

    def compose(self) -> ComposeResult:
        # Status color for header - primary color, red for deleted
        color = "$primary" if self.change.status != "deleted" else "$error"
        yield Label(f"[{color}]{self.change.path}[/]", classes="file-header")

        # Show each hunk as a separate widget with separators between
        if self.change.hunks:
            for i, hunk in enumerate(self.change.hunks):
                if i > 0:
                    yield HunkSeparator()
                yield HunkWidget(
                    hunk,
                    self.change.path,
                    id=f"hunk-{_sanitize_id(self.change.path)}-{i}",
                )
        else:
            yield Label("[dim]Binary file or no diff available[/]")


class DiffView(VerticalScroll):
    """Main scrollable container of file diff panels with hunk navigation."""

    DEFAULT_CSS = """
    DiffView {
        padding: 1;
    }
    """

    def __init__(self, changes: list[FileChange], **kwargs) -> None:
        super().__init__(**kwargs)
        self.changes = changes
        # Build flat list of (file_idx, hunk_idx) for navigation
        self._hunk_list: list[tuple[int, int]] = []
        for file_idx, change in enumerate(changes):
            if change.hunks:
                for hunk_idx in range(len(change.hunks)):
                    self._hunk_list.append((file_idx, hunk_idx))
            else:
                # File with no hunks (binary) - still navigable as a unit
                self._hunk_list.append((file_idx, -1))
        self._current_idx = 0

    def compose(self) -> ComposeResult:
        if not self.changes:
            yield Label("[dim]No changes to display[/]")
            return

        for change in self.changes:
            yield FileDiffPanel(change, id=f"panel-{_sanitize_id(change.path)}")

    def on_mount(self) -> None:
        """Focus the first hunk on mount."""
        self._focus_hunk(0)

    def on_descendant_focus(self, event) -> None:
        """Sync _current_idx when a hunk is focused (by click or programmatically)."""
        widget = event.widget
        if isinstance(widget, HunkWidget) and widget.id:
            for i, (file_idx, hunk_idx) in enumerate(self._hunk_list):
                if hunk_idx >= 0:
                    path = self.changes[file_idx].path
                    if f"hunk-{_sanitize_id(path)}-{hunk_idx}" == widget.id:
                        self._current_idx = i
                        self.post_message(DiffFileItem.Selected(path))
                        return

    def scroll_to_file(self, path: str) -> None:
        """Scroll to bring the specified file's panel into view."""
        for i, (file_idx, hunk_idx) in enumerate(self._hunk_list):
            if self.changes[file_idx].path == path and hunk_idx >= 0:
                self._focus_hunk(i)
                return

    def action_next_file(self) -> None:
        """Navigate to next hunk (j/down key)."""
        if self._hunk_list:
            self._focus_hunk(min(self._current_idx + 1, len(self._hunk_list) - 1))

    def action_prev_file(self) -> None:
        """Navigate to previous hunk (k/up key)."""
        if self._hunk_list:
            self._focus_hunk(max(self._current_idx - 1, 0))

    def _focus_hunk(self, idx: int) -> None:
        """Focus hunk at given index in _hunk_list."""
        if not self._hunk_list or idx < 0 or idx >= len(self._hunk_list):
            return
        self._current_idx = idx
        file_idx, hunk_idx = self._hunk_list[idx]
        path = self.changes[file_idx].path
        if hunk_idx >= 0:
            hunk_id = f"hunk-{_sanitize_id(path)}-{hunk_idx}"
            try:
                self.query_one(f"#{hunk_id}", HunkWidget).focus()
            except Exception:
                pass
        else:
            # Binary file with no hunks
            self.post_message(DiffFileItem.Selected(path))

    def get_current_hunk_widget(self) -> HunkWidget | None:
        """Get the currently focused hunk widget."""
        if not self._hunk_list:
            return None
        file_idx, hunk_idx = self._hunk_list[self._current_idx]
        if hunk_idx < 0:
            return None
        path = self.changes[file_idx].path
        hunk_id = f"hunk-{_sanitize_id(path)}-{hunk_idx}"
        try:
            return self.query_one(f"#{hunk_id}", HunkWidget)
        except Exception:
            return None

    def get_comments(self) -> list[HunkComment]:
        """Collect all non-empty comments from hunks."""
        comments = []
        for hunk_widget in self.query(HunkWidget):
            if hunk_widget.comment:
                comments.append(
                    HunkComment(
                        path=hunk_widget.path,
                        hunk=hunk_widget.hunk,
                        comment=hunk_widget.comment,
                    )
                )
        return comments

    def is_editing(self) -> bool:
        """Return True if any hunk is in editing mode."""
        for hunk_widget in self.query(HunkWidget):
            if hunk_widget.editing:
                return True
        return False


def _sanitize_id(path: str) -> str:
    """Convert a file path to a valid CSS ID."""
    return path.replace("/", "-").replace(".", "-").replace(" ", "-")

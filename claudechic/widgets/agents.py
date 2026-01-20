"""Agent sidebar widget for multi-agent management."""

from pathlib import Path

from textual.app import ComposeResult
from textual.events import Click
from textual.message import Message
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Static
from rich.text import Text

from claudechic.enums import AgentStatus
from claudechic.cursor import ClickableMixin
from claudechic.widgets.button import Button


class HamburgerButton(Button):
    """Floating hamburger button for narrow screens."""

    class Clicked(Message):
        """Posted when hamburger is clicked."""

        pass

    DEFAULT_CSS = """
    HamburgerButton {
        layer: above;
        width: 10;
        height: 3;
        content-align: center middle;
        background: $surface;
        color: $text-muted;
        display: none;
        /* Position top-right */
        offset: -1 1;
        dock: right;
        border: round $panel;
    }
    HamburgerButton:hover {
        color: $text;
    }
    HamburgerButton.visible {
        display: block;
    }
    HamburgerButton.needs-attention {
        color: $primary;
        border: round $primary;
    }
    """

    def __init__(self, id: str | None = None) -> None:
        super().__init__(id=id)

    def render(self) -> str:
        return "Agents"

    def on_click(self, event) -> None:
        self.post_message(self.Clicked())


class PlanButton(Button):
    """Button to open the current session's plan file."""

    class Clicked(Message):
        """Posted when plan button is clicked."""

        def __init__(self, plan_path: Path) -> None:
            self.plan_path = plan_path
            super().__init__()

    DEFAULT_CSS = """
    PlanButton {
        height: 3;
        min-height: 3;
        padding: 1 1 1 2;
        dock: bottom;
    }
    PlanButton:hover {
        background: $surface-lighten-1;
    }
    """

    def __init__(self, plan_path: Path) -> None:
        super().__init__()
        self.plan_path = plan_path

    def render(self) -> Text:
        return Text.assemble(("📋", ""), " ", ("Plan", "dim"))

    def on_click(self, event) -> None:
        self.post_message(self.Clicked(self.plan_path))


class WorktreeItem(Widget, ClickableMixin):
    """A ghost worktree in the sidebar (not yet an agent)."""

    class Selected(Message):
        """Posted when worktree is clicked."""

        def __init__(self, branch: str, path: Path) -> None:
            self.branch = branch
            self.path = path
            super().__init__()

    DEFAULT_CSS = """
    WorktreeItem {
        height: 3;
        min-height: 3;
        padding: 1 1 1 2;
    }
    WorktreeItem:hover {
        background: $surface-lighten-1;
    }
    """

    def __init__(self, branch: str, path: Path) -> None:
        super().__init__()
        self.branch = branch
        self.path = path

    def render(self) -> Text:
        name = self.branch
        if len(name) > 16:
            name = name[:15] + "…"
        return Text.assemble(("◌", ""), " ", (name, "dim"))

    def on_click(self, event) -> None:
        self.post_message(self.Selected(self.branch, self.path))


class AgentItem(Widget, ClickableMixin):
    """A single agent in the sidebar."""

    class Selected(Message):
        """Posted when agent is clicked."""

        def __init__(self, agent_id: str) -> None:
            self.agent_id = agent_id
            super().__init__()

    class CloseRequested(Message):
        """Posted when close button is clicked."""

        def __init__(self, agent_id: str) -> None:
            self.agent_id = agent_id
            super().__init__()

    DEFAULT_CSS = """
    AgentItem {
        height: 3;
        padding: 1 1 1 2;
        layout: horizontal;
    }
    AgentItem:hover {
        background: $surface-lighten-1;
    }
    AgentItem.active {
        padding: 1 1 1 1;
        border-left: wide $primary;
        background: $surface;
    }
    AgentItem .agent-label {
        width: 1fr;
        height: 1;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    AgentItem .agent-close {
        width: 3;
        min-width: 3;
        height: 1;
        padding: 0;
        background: $panel;
        color: $text-muted;
        text-align: center;
    }
    AgentItem .agent-close:hover {
        color: $error;
        background: $panel-lighten-1;
    }
    """

    status: reactive[AgentStatus] = reactive(AgentStatus.IDLE)

    def __init__(
        self, agent_id: str, display_name: str, status: AgentStatus = AgentStatus.IDLE
    ) -> None:
        super().__init__()
        self.agent_id = agent_id
        self.display_name = display_name
        self.status = status

    def compose(self) -> ComposeResult:
        yield Static(self._render_label(), classes="agent-label")
        yield Static(Text("X"), classes="agent-close")

    def _render_label(self) -> Text:
        if self.status == AgentStatus.BUSY:
            indicator = "\u25cf"
            style = ""  # default text color
        elif self.status == AgentStatus.NEEDS_INPUT:
            indicator = "\u25cf"
            style = self.app.current_theme.primary if self.app else "bold"
        else:
            indicator = "\u25cb"
            style = "dim"
        name = self.display_name
        if len(name) > 14:
            name = name[:13] + "…"
        return Text.assemble((indicator, style), " ", (name, ""))

    def watch_status(self, _status: str) -> None:
        """Update label when status changes."""
        try:
            label = self.query_one(".agent-label", Static)
            label.update(self._render_label())
        except Exception:
            pass  # Widget may not be mounted yet

    def on_click(self, event: Click) -> None:
        """Handle clicks - check if on close button."""
        if event.widget and event.widget.has_class("agent-close"):
            event.stop()
            self.post_message(self.CloseRequested(self.agent_id))
        else:
            self.post_message(self.Selected(self.agent_id))


class AgentSidebar(Widget):
    """Sidebar showing all agents with status indicators."""

    DEFAULT_CSS = """
    AgentSidebar {
        width: 24;
        height: 100%;
        padding: 0;
        overflow-y: auto;
    }
    AgentSidebar .sidebar-title {
        color: $text-muted;
        text-style: bold;
        padding: 1 1 1 1;
    }
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._agents: dict[str, AgentItem] = {}
        self._worktrees: dict[str, WorktreeItem] = {}  # branch -> item
        self._plan_button: PlanButton | None = None

    def compose(self) -> ComposeResult:
        yield Static("Agents", classes="sidebar-title")

    def add_agent(
        self, agent_id: str, name: str, status: AgentStatus = AgentStatus.IDLE
    ) -> None:
        """Add an agent to the sidebar."""
        if agent_id in self._agents:
            return
        # Remove ghost worktree if there's one for this name
        if name in self._worktrees:
            self._worktrees[name].remove()
            del self._worktrees[name]
        item = AgentItem(agent_id, name, status)
        item.id = f"agent-{agent_id}"
        self._agents[agent_id] = item
        self.mount(item)

    def remove_agent(self, agent_id: str) -> None:
        """Remove an agent from the sidebar."""
        if agent_id in self._agents:
            self._agents[agent_id].remove()
            del self._agents[agent_id]

    def set_active(self, agent_id: str) -> None:
        """Mark an agent as active (selected)."""
        for aid, item in self._agents.items():
            if aid == agent_id:
                item.add_class("active")
            else:
                item.remove_class("active")

    def update_status(self, agent_id: str, status: AgentStatus) -> None:
        """Update an agent's status."""
        if agent_id in self._agents:
            self._agents[agent_id].status = status

    def add_worktree(self, branch: str, path: Path) -> None:
        """Add a ghost worktree to the sidebar."""
        # Skip if already have an agent with this name
        for agent_item in self._agents.values():
            if agent_item.display_name == branch:
                return
        if branch in self._worktrees:
            return
        item = WorktreeItem(branch, path)
        item.id = f"worktree-{branch}"
        self._worktrees[branch] = item
        self.mount(item)

    def remove_worktree(self, branch: str) -> None:
        """Remove a ghost worktree from the sidebar."""
        if branch in self._worktrees:
            self._worktrees[branch].remove()
            del self._worktrees[branch]

    def set_plan(self, plan_path: Path | None) -> None:
        """Show or hide the plan button."""
        if plan_path:
            if self._plan_button is None:
                self._plan_button = PlanButton(plan_path)
                self._plan_button.id = "plan-button"
                self.mount(self._plan_button)
            else:
                self._plan_button.plan_path = plan_path
        else:
            if self._plan_button is not None:
                self._plan_button.remove()
                self._plan_button = None

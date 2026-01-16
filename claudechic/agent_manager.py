"""AgentManager: coordinates multiple concurrent agents."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Awaitable, Callable, Iterator

from claude_agent_sdk import ClaudeAgentOptions

from claudechic.agent import Agent

if TYPE_CHECKING:
    from claude_agent_sdk import ResultMessage
    from claudechic.permissions import PermissionRequest

log = logging.getLogger(__name__)


class AgentManager:
    """Coordinates multiple concurrent Claude agents.

    Responsibilities:
    - Create and connect agents
    - Track active agent
    - Switch between agents
    - Close agents cleanly
    - Wire agent callbacks for UI integration

    This class has no UI dependencies - it's pure async coordination.
    """

    def __init__(
        self,
        options_factory: Callable[..., ClaudeAgentOptions],
    ):
        """Initialize the agent manager.

        Args:
            options_factory: Function to create SDK options. Called with
                keyword args: cwd, resume. Should NOT set can_use_tool
                (Agent sets its own permission handler).
        """
        self.agents: dict[str, Agent] = {}
        self.active_id: str | None = None
        self._options_factory = options_factory

        # Callbacks for UI integration (set by ChatApp)
        self.on_created: Callable[[Agent], None] | None = None
        self.on_switched: Callable[[Agent, Agent | None], None] | None = None
        self.on_closed: Callable[[str], None] | None = None

        # Callback factory for permission UI (set by ChatApp)
        # Returns the callback to set on each agent
        self.permission_ui_callback: (
            Callable[[Agent, PermissionRequest], Awaitable[str]] | None
        ) = None

        # Agent event callbacks (set by ChatApp, applied to all agents)
        self.on_agent_status_changed: Callable[[Agent], None] | None = None
        self.on_agent_error: Callable[[Agent, str, Exception | None], None] | None = None
        self.on_agent_complete: Callable[[Agent, ResultMessage | None], None] | None = None
        self.on_agent_todos_updated: Callable[[Agent], None] | None = None

        # Fine-grained streaming callbacks
        from claudechic.agent import ToolUse
        self.on_agent_text_chunk: Callable[[Agent, str, bool, str | None], None] | None = None
        self.on_agent_tool_use: Callable[[Agent, ToolUse], None] | None = None
        self.on_agent_tool_result: Callable[[Agent, ToolUse], None] | None = None

    @property
    def active(self) -> Agent | None:
        """Get the currently active agent."""
        if self.active_id and self.active_id in self.agents:
            return self.agents[self.active_id]
        return None

    def get(self, agent_id: str | None = None) -> Agent | None:
        """Get agent by ID, or active agent if None."""
        if agent_id:
            return self.agents.get(agent_id)
        return self.active

    def find_by_name(self, name: str) -> Agent | None:
        """Find agent by name."""
        for agent in self.agents.values():
            if agent.name == name:
                return agent
        return None

    def create_unconnected(
        self,
        name: str,
        cwd: Path,
        *,
        worktree: str | None = None,
        switch_to: bool = True,
    ) -> Agent:
        """Create a new agent without connecting to SDK.

        Use this when you need to populate UI immediately and connect later.
        Call agent.connect() separately to establish SDK connection.

        Args:
            name: Display name for the agent
            cwd: Working directory
            worktree: Git worktree branch name if applicable
            switch_to: Whether to make this the active agent

        Returns:
            The created agent (not yet connected)
        """
        agent = Agent(name=name, cwd=cwd, worktree=worktree)

        # Wire callbacks
        self._wire_agent_callbacks(agent)

        # Register agent
        self.agents[agent.id] = agent
        log.info(f"Created agent '{name}' (id={agent.id}, cwd={cwd})")

        if self.on_created:
            self.on_created(agent)

        # Switch to new agent if requested or if it's the first agent
        if switch_to or self.active_id is None:
            self.switch(agent.id)

        return agent

    async def create(
        self,
        name: str,
        cwd: Path,
        *,
        worktree: str | None = None,
        resume: str | None = None,
        switch_to: bool = True,
    ) -> Agent:
        """Create and connect a new agent.

        Args:
            name: Display name for the agent
            cwd: Working directory
            worktree: Git worktree branch name if applicable
            resume: Session ID to resume
            switch_to: Whether to make this the active agent

        Returns:
            The created agent (connected and ready)
        """
        agent = Agent(name=name, cwd=cwd, worktree=worktree)

        # Wire callbacks
        self._wire_agent_callbacks(agent)

        # Create options and connect
        options = self._options_factory(cwd=cwd, resume=resume)
        await agent.connect(options, resume=resume)

        # Register agent
        self.agents[agent.id] = agent
        log.info(f"Created agent '{name}' (id={agent.id}, cwd={cwd})")

        if self.on_created:
            self.on_created(agent)

        # Switch to new agent if requested or if it's the first agent
        if switch_to or self.active_id is None:
            self.switch(agent.id)

        return agent

    def _wire_agent_callbacks(self, agent: Agent) -> None:
        """Wire up agent callbacks to manager callbacks."""
        # Permission UI
        if self.permission_ui_callback:
            agent.permission_ui_callback = self.permission_ui_callback

        # Event callbacks
        if self.on_agent_status_changed:
            agent.on_status_changed = self.on_agent_status_changed
        if self.on_agent_error:
            agent.on_error = self.on_agent_error
        if self.on_agent_complete:
            agent.on_complete = self.on_agent_complete
        if self.on_agent_todos_updated:
            agent.on_todos_updated = self.on_agent_todos_updated

        # Fine-grained streaming callbacks
        if self.on_agent_text_chunk:
            agent.on_text_chunk = self.on_agent_text_chunk
        if self.on_agent_tool_use:
            agent.on_tool_use = self.on_agent_tool_use
        if self.on_agent_tool_result:
            agent.on_tool_result = self.on_agent_tool_result

    def switch(self, agent_id: str) -> bool:
        """Switch to a different agent.

        Args:
            agent_id: ID of agent to switch to

        Returns:
            True if switch succeeded, False if agent not found
        """
        if agent_id not in self.agents:
            log.warning(f"Cannot switch to unknown agent: {agent_id}")
            return False

        old_agent = self.active
        self.active_id = agent_id
        new_agent = self.agents[agent_id]

        log.info(f"Switched to agent '{new_agent.name}' (id={agent_id})")

        if self.on_switched:
            self.on_switched(new_agent, old_agent)

        return True

    async def close(self, agent_id: str) -> None:
        """Close an agent and clean up.

        Args:
            agent_id: ID of agent to close
        """
        agent = self.agents.pop(agent_id, None)
        if not agent:
            log.warning(f"Cannot close unknown agent: {agent_id}")
            return

        name = agent.name
        was_active = agent_id == self.active_id

        # Disconnect
        await agent.disconnect()
        log.info(f"Closed agent '{name}' (id={agent_id})")

        if self.on_closed:
            self.on_closed(agent_id)

        # Switch to another agent if we closed the active one
        if was_active and self.agents:
            next_id = next(iter(self.agents))
            self.switch(next_id)
        elif was_active:
            self.active_id = None

    async def close_all(self) -> None:
        """Close all agents."""
        for agent_id in list(self.agents.keys()):
            await self.close(agent_id)

    def __len__(self) -> int:
        """Number of agents."""
        return len(self.agents)

    def __iter__(self) -> Iterator[Agent]:
        """Iterate over agents."""
        return iter(self.agents.values())

    def __contains__(self, agent_id: str) -> bool:
        """Check if agent exists."""
        return agent_id in self.agents

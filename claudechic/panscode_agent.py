"""panscode_agent.py — PansCode backend adapter for claudechic.

Translates PansCodeApp EventBus events into claudechic AgentObserver
callbacks, so the TUI layer is fully unaware of backend differences.

Communication path:
    TUI (Textual) → PansCodeBackend → PansCodeApp.run() → AgentLoop → LLM
         ↑                              │
         └── EventBus subscription ←────┘
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from claudechic.compat import (
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    SystemMessage,
    PermissionResult,
    PermissionResultAllow,
    PermissionResultDeny,
    ToolPermissionContext,
)
from claudechic.enums import AgentStatus, PermissionChoice

if TYPE_CHECKING:
    from claudechic.agent import Agent, ToolUse
    from claudechic.protocols import AgentObserver
    from panscode.kernel.bus import EventBus

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Event constants (must match panscode.kernel.bus)
# ---------------------------------------------------------------------------

_AGENT_TEXT_DELTA = "agent.text_delta"
_AGENT_FINISHED = "agent.finished"
_TOOL_STARTED = "tool.started"
_TOOL_COMPLETED = "tool.completed"
_PERMISSION_ASKED = "permission.asked"
_SESSION_CREATED = "session.created"


class PansCodeBackend:
    """Wraps PansCodeApp to provide the backend interface that Agent needs.

    Lifecycle:
        backend = PansCodeBackend(cwd)
        await backend.initialize()        # creates PansCodeApp
        session_id = await backend.connect(agent)
        await backend.send(agent, prompt)  # drives observer callbacks
        await backend.disconnect()
    """

    def __init__(
        self,
        cwd: Path,
        config_path: Path | None = None,
    ) -> None:
        self.cwd = cwd
        self._config_path = config_path
        self._app: Any = None  # PansCodeApp (lazy import)
        self._unsubs: list[Any] = []
        self._initialized = False

    async def initialize(self) -> None:
        """Create PansCodeApp instance (lazy import to avoid circular deps)."""
        if self._initialized:
            return
        from panscode.app import PansCodeApp

        self._app = PansCodeApp(config_path=self._config_path)
        self._app.ensure_project_dir()
        self._initialized = True

    @property
    def bus(self) -> EventBus:
        """Access the PansCode EventBus."""
        assert self._app is not None, "Call initialize() first"
        return self._app.bus

    async def connect(self, agent: Agent) -> str:
        """Connect backend and return a session_id."""
        await self.initialize()
        session_id = str(uuid.uuid4())
        agent.session_id = session_id

        # Emit init system message
        if agent.observer:
            init_msg = SystemMessage(
                subtype="init",
                data={"session_id": session_id},
            )
            agent.observer.on_system_message(agent, init_msg)

        return session_id

    async def send(self, agent: Agent, prompt: str) -> None:
        """Send prompt via PansCodeApp.run() and drive observer callbacks.

        Subscribes to EventBus events for the duration of the call,
        translating them into AgentObserver callbacks.
        """
        assert self._app is not None, "Call initialize() first"
        observer = agent.observer
        if not observer:
            log.warning("Agent has no observer, skipping send")
            return

        bus = self._app.bus
        unsubs: list[Any] = []

        # Map PansCode permission to claudechic permission
        permission_mode = agent.permission_mode
        mode = _map_permission_mode(permission_mode)

        # --- EventBus → Observer translation ---

        def on_text_delta(_event_type: str, props: dict) -> None:
            text = props.get("delta", "")
            if text:
                agent._handle_text_chunk(text, agent._needs_new_message, None)
                agent._needs_new_message = False

        def on_tool_started(_event_type: str, props: dict) -> None:
            from claudechic.agent import ToolUse as ChicToolUse

            tool = ChicToolUse(
                id=props.get("call_id", ""),
                name=props.get("tool_id", ""),
                input=props.get("args", {}),
            )
            agent.pending_tools[tool.id] = tool
            agent.response_had_tools = True
            agent._needs_new_message = True

            # Add to current assistant content
            from claudechic.agent import AssistantContent, ChatItem

            if agent._current_assistant is None:
                agent._current_assistant = AssistantContent()
                agent.messages.append(
                    ChatItem(role="assistant", content=agent._current_assistant)
                )
            agent._current_assistant.blocks.append(tool)
            if observer:
                observer.on_message_updated(agent)
                observer.on_tool_use(agent, tool)

        def on_tool_completed(_event_type: str, props: dict) -> None:
            call_id = props.get("call_id", "")
            tool = agent.pending_tools.pop(call_id, None)
            if tool:
                tool.result = props.get("output") or props.get("error") or ""
                tool.is_error = bool(props.get("error"))
                if observer:
                    observer.on_message_updated(agent)
                    observer.on_tool_result(agent, tool)

        unsubs.append(bus.subscribe(_AGENT_TEXT_DELTA, on_text_delta))
        unsubs.append(bus.subscribe(_TOOL_STARTED, on_tool_started))
        unsubs.append(bus.subscribe(_TOOL_COMPLETED, on_tool_completed))

        try:
            agent._set_status(AgentStatus.BUSY)
            agent._needs_new_message = True

            # Set up permission callback: PansCode → claudechic
            self._app.set_permission_callback(
                _make_permission_callback(agent)
            )

            # Execute the agent loop
            result_text = await self._app.run(
                prompt,
                mode=mode,
                session_id=agent.session_id,
            )

            # Flush any remaining text
            agent._flush_current_text()

            # Notify completion
            result = ResultMessage(session_id=agent.session_id or "")
            if observer:
                observer.on_complete(agent, result)

        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.exception("PansCode response processing failed")
            if observer:
                observer.on_error(agent, f"Response failed: {e}", e)
                observer.on_complete(agent, None)
        finally:
            for unsub in unsubs:
                unsub()
            agent._set_status(AgentStatus.IDLE)

    async def disconnect(self) -> None:
        """Shutdown PansCodeApp."""
        if self._app:
            self._app.shutdown()

    async def interrupt(self) -> None:
        """Interrupt current execution (placeholder)."""
        # TODO: implement cancellation via AgentLoop
        pass


# ---------------------------------------------------------------------------
# Permission callback bridge
# ---------------------------------------------------------------------------


def _make_permission_callback(agent: Agent):
    """Create an async callback that bridges PansCode permission requests
    to claudechic's permission UI flow.

    PansCode calls: reply = await callback(PermissionRequest)
    claudechic uses: PermissionRequest -> observer.on_prompt_added -> UI -> result
    """

    async def _permission_callback(panscode_request) -> str:
        """Handle a PansCode PermissionRequest, return ReplyAction string."""
        from claudechic.permissions import (
            PermissionRequest as ChicPermRequest,
            PermissionResponse,
        )

        # Create claudechic-style permission request
        tool_name = panscode_request.permission
        # Build tool_input from patterns
        tool_input = {"patterns": panscode_request.patterns}

        chic_request = ChicPermRequest(tool_name, tool_input)
        agent.pending_prompts.append(chic_request)

        if agent.observer:
            agent.observer.on_prompt_added(agent, chic_request)

        agent._set_status(AgentStatus.NEEDS_INPUT)

        # Wait for UI response
        if agent.permission_handler:
            result = await agent.permission_handler(agent, chic_request)
        else:
            result = await chic_request.wait()

        # Remove from queue
        if chic_request in agent.pending_prompts:
            agent.pending_prompts.remove(chic_request)

        agent._set_status(AgentStatus.BUSY)

        # Map claudechic PermissionChoice → PansCode ReplyAction
        if result.choice == PermissionChoice.ALLOW:
            return "once"
        elif result.choice == PermissionChoice.ALLOW_ALL:
            return "always"
        elif result.choice == PermissionChoice.ALLOW_SESSION:
            return "always"
        else:
            return "reject"

    return _permission_callback


# ---------------------------------------------------------------------------
# Mode mapping
# ---------------------------------------------------------------------------


def _map_permission_mode(chic_mode: str) -> str:
    """Map claudechic permission mode to PansCode AgentMode."""
    mapping = {
        "default": "build",
        "acceptEdits": "build",
        "plan": "plan",
        "planSwarm": "plan",
    }
    return mapping.get(chic_mode, "build")

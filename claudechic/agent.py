"""Agent: autonomous Claude agent with PansCode backend and message history."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import mimetypes
import re
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from claudechic.compat import (
    AssistantMessage,
    CLIJSONDecodeError,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    PermissionResult,
    PermissionResultAllow,
    PermissionResultDeny,
    ResultMessage,
    StreamEvent,
    SystemMessage,
    ToolPermissionContext,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)

from claudechic.enums import AgentStatus, PermissionChoice, ToolName
from claudechic.file_index import FileIndex
from claudechic.permissions import PermissionRequest
from claudechic.sessions import get_plan_path_for_session
from claudechic.tasks import create_safe_task

if TYPE_CHECKING:
    from claudechic.features.worktree.git import FinishState
    from claudechic.protocols import AgentObserver, PermissionHandler

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Message types for chat history
# ---------------------------------------------------------------------------


@dataclass
class ImageAttachment:
    """An image attached to a message."""

    path: str
    filename: str
    media_type: str
    base64_data: str


@dataclass
class UserContent:
    """A user message in chat history."""

    text: str
    images: list[ImageAttachment] = field(default_factory=list)


@dataclass
class ToolUse:
    """A tool use within an assistant turn."""

    id: str
    name: str
    input: dict[str, Any]
    parent_tool_use_id: str | None = None
    result: str | None = None
    is_error: bool = False


@dataclass
class TextBlock:
    """A text block within an assistant turn."""

    text: str


@dataclass
class AssistantContent:
    """An assistant message in chat history.

    Contains an ordered list of blocks (TextBlock or ToolUse) to preserve
    the original interleaving of text and tool uses.
    """

    blocks: list[TextBlock | ToolUse] = field(default_factory=list)


@dataclass
class ChatItem:
    """A single item in chat history."""

    role: Literal["user", "assistant"]
    content: UserContent | AssistantContent


# ---------------------------------------------------------------------------
# Agent class
# ---------------------------------------------------------------------------


class Agent:
    """Autonomous Claude agent with PansCode backend and state.

    The Agent owns:
    - PansCode backend connection lifecycle
    - Message history (list of ChatItem)
    - Permission request queue
    - Per-agent state (images, todos, file index, etc.)

    Events are emitted via the observer protocol for UI integration.
    """

    # Tools to auto-approve when auto_approve_edits is True
    AUTO_EDIT_TOOLS = {ToolName.EDIT, ToolName.WRITE}

    # Tools blocked in plan mode (read-only enforcement)
    PLAN_MODE_BLOCKED_TOOLS = {
        ToolName.EDIT,
        ToolName.WRITE,
        ToolName.BASH,
        ToolName.NOTEBOOK_EDIT,
    }

    def __init__(
        self,
        name: str,
        cwd: Path,
        *,
        id: str | None = None,
        worktree: str | None = None,
    ):
        # Identity
        self.id = id or str(uuid.uuid4())[:8]
        self.name = name
        self.cwd = cwd
        self.worktree = worktree

        # PansCode backend (replaces SDK client)
        self.backend: Any = None  # PansCodeBackend, lazily imported
        # Keep client as a property alias for backward compatibility
        self.client: ClaudeSDKClient | None = None
        self.session_id: str | None = None
        self._response_task: asyncio.Task | None = None

        # Status
        self.status: AgentStatus = AgentStatus.IDLE
        self._thinking: bool = False  # Whether this agent is currently thinking
        self._interrupted: bool = False  # Suppress errors after intentional interrupt

        # Chat history
        self.messages: list[ChatItem] = []
        self._current_assistant: AssistantContent | None = None
        self._current_text_buffer: str = ""

        # Permission queue
        self.pending_prompts: deque[PermissionRequest] = deque()

        # Tool tracking (within current response)
        self.pending_tools: dict[str, ToolUse] = {}
        self.active_tasks: dict[str, str] = {}  # task_id -> accumulated text
        self.response_had_tools: bool = False
        self._needs_new_message: bool = True  # Start new ChatMessage on next text
        self._thinking_hidden: bool = (
            False  # Track if thinking indicator was hidden this response
        )

        # Per-agent state
        self.pending_images: list[ImageAttachment] = []
        self.file_index: FileIndex | None = None
        self.todos: list[dict] = []
        self.permission_mode: str = "default"  # default, acceptEdits, plan
        self.session_allowed_tools: set[str] = set()  # Tools allowed for this session
        self._pending_followup: str | None = None  # Auto-send after current response
        self.model: str | None = None  # Model override (None = SDK default)

        # Worktree finish state (for /worktree finish flow)
        self.finish_state: Any = None  # FinishState | None

        # Plan file path (cached after first lookup)
        self.plan_path: Path | None = None

        # UI state (managed by ChatApp, not widget references)
        self.pending_input: str = ""  # Saved input text when switching away

        # Observer for UI integration (set by AgentManager)
        self.observer: AgentObserver | None = None
        self.permission_handler: PermissionHandler | None = None

        # Background process tracking (not used in PansCode mode)
        self._claude_pid: int | None = None
        # Background task output files: command -> output_file path
        self._background_outputs: dict[str, str] = {}

        # Pending plan execution (set when "clear context + auto-approve" chosen)
        self.pending_plan_execution: dict | None = None  # {"plan": str, "mode": str}

        # Checkpoint tracking for /rewind command (UUIDs of user messages)
        self.checkpoint_uuids: list[str] = []

    @property
    def analytics_id(self) -> str:
        """ID for analytics events (session_id if connected, else internal id)."""
        return self.session_id or self.id

    # -----------------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------------

    async def connect(
        self,
        options: ClaudeAgentOptions,
        resume: str | None = None,
    ) -> None:
        """Connect to PansCode backend.

        Args:
            options: Agent options (kept for API compatibility with AgentManager)
            resume: Optional session ID to resume
        """
        from claudechic.panscode_agent import PansCodeBackend

        # Create and initialize PansCode backend
        self.backend = PansCodeBackend(cwd=self.cwd)
        await self.backend.initialize()

        # Connect and get session_id
        session_id = await self.backend.connect(self)

        if resume:
            self.session_id = resume
        elif not self.session_id:
            self.session_id = session_id

        # Initialize file index
        self.file_index = FileIndex(root=self.cwd)
        await self.file_index.refresh()

    async def disconnect(self) -> None:
        """Disconnect and cleanup."""
        if self._response_task and not self._response_task.done():
            self._response_task.cancel()
            try:
                await self._response_task
            except asyncio.CancelledError:
                pass

        if self.backend:
            try:
                await self.backend.disconnect()
            except Exception:
                pass
            self.backend = None

        await asyncio.sleep(0)

    async def load_history(self, cwd: Path | None = None) -> None:
        """Load message history from session file into self.messages.

        This populates Agent.messages from the persisted session,
        making Agent.messages the single source of truth for history.
        Call ChatView._render_full() after this to update UI.

        Args:
            cwd: Working directory for session lookup (defaults to self.cwd)
        """
        from claudechic.sessions import load_session_messages

        if not self.session_id:
            return

        self.messages.clear()
        self.checkpoint_uuids.clear()  # Clear stale UUIDs; SDK will repopulate via replay
        raw_messages = await load_session_messages(self.session_id, cwd=cwd or self.cwd)

        current_assistant: AssistantContent | None = None

        for m in raw_messages:
            if m["type"] == "user":
                # Flush any pending assistant content
                if current_assistant is not None:
                    self.messages.append(
                        ChatItem(role="assistant", content=current_assistant)
                    )
                    current_assistant = None
                # Add user message
                self.messages.append(
                    ChatItem(role="user", content=UserContent(text=m["content"]))
                )
            elif m["type"] == "assistant":
                # Add text block to current assistant content (preserving order)
                if current_assistant is None:
                    current_assistant = AssistantContent()
                current_assistant.blocks.append(TextBlock(text=m["content"]))
            elif m["type"] == "tool_use":
                # Add tool use to current assistant content (preserving order)
                if current_assistant is None:
                    current_assistant = AssistantContent()
                current_assistant.blocks.append(
                    ToolUse(
                        id=m.get("id", ""),
                        name=m["name"],
                        input=m.get("input", {}),
                    )
                )

        # Flush final assistant content
        if current_assistant is not None:
            self.messages.append(ChatItem(role="assistant", content=current_assistant))

        log.info(f"Loaded {len(self.messages)} messages from session {self.session_id}")

    # -----------------------------------------------------------------------
    # Sending messages
    # -----------------------------------------------------------------------

    def attach_image(self, path: Path) -> ImageAttachment | None:
        """Attach an image to the next message.

        Returns ImageAttachment on success, None on failure.
        """
        try:
            data = base64.b64encode(path.read_bytes()).decode()
            media_type = mimetypes.guess_type(str(path))[0] or "image/png"
            img = ImageAttachment(str(path), path.name, media_type, data)
            self.pending_images.append(img)
            return img
        except Exception:
            return None

    def clear_images(self) -> None:
        """Clear pending images."""
        self.pending_images.clear()

    async def send(self, prompt: str, *, display_as: str | None = None) -> None:
        """Send a message and start processing response.

        The response is processed concurrently - this method returns immediately.

        Args:
            prompt: The prompt to send to Claude
            display_as: Optional shorter text to show in UI instead of full prompt
        """
        if not self.backend:
            raise RuntimeError("Agent not connected")

        # Add user message to history (store display text if provided)
        display_text = display_as or prompt
        self.messages.append(
            ChatItem(
                role="user",
                content=UserContent(
                    text=display_text, images=list(self.pending_images)
                ),
            )
        )

        # Notify UI to display user message (pass full image info before clearing)
        if self.observer:
            self.observer.on_prompt_sent(self, display_text, list(self.pending_images))

        self.response_had_tools = False
        self._current_assistant = None
        self._current_text_buffer = ""
        self._needs_new_message = True
        self._thinking_hidden = False  # Reset for new response
        self._interrupted = False  # Clear interrupt flag for new query

        # Clear pending images after sending
        self.pending_images.clear()

        # Start response processing via PansCode backend
        self._response_task = asyncio.create_task(
            self.backend.send(self, prompt),
            name=f"agent-{self.id}-response",
        )

    async def interrupt(self) -> None:
        """Interrupt current response."""
        self._interrupted = True
        if self._response_task and not self._response_task.done():
            self._response_task.cancel()
            try:
                await self._response_task
            except asyncio.CancelledError:
                pass

        if self.backend:
            try:
                await self.backend.interrupt()
            except Exception:
                pass

        self._set_status(AgentStatus.IDLE)

    async def _send_followup(self, message: str) -> None:
        """Send a follow-up message after brief delay (for 'do something else' flow)."""
        await asyncio.sleep(0.1)  # Let UI update
        await self.send(message)

    # -----------------------------------------------------------------------
    # Response processing (handled by PansCodeBackend.send())
    # -----------------------------------------------------------------------

    def _handle_text_chunk(
        self, text: str, new_message: bool, parent_tool_use_id: str | None
    ) -> None:
        """Handle incoming text chunk."""
        # If this belongs to a Task, accumulate there
        if parent_tool_use_id and parent_tool_use_id in self.active_tasks:
            self.active_tasks[parent_tool_use_id] += text
            return

        if new_message:
            self._flush_current_text()

        # Ensure we have an assistant content to accumulate into
        if self._current_assistant is None:
            self._current_assistant = AssistantContent()
            self.messages.append(
                ChatItem(role="assistant", content=self._current_assistant)
            )

        self._current_text_buffer += text
        # Update the current TextBlock in-place for live streaming display
        self._update_current_text_block()
        if self.observer:
            self.observer.on_message_updated(self)
            self.observer.on_text_chunk(self, text, new_message, parent_tool_use_id)

    def _update_current_text_block(self) -> None:
        """Update the current TextBlock with accumulated text (for streaming)."""
        if not self._current_assistant or not self._current_text_buffer:
            return
        # Find or create the trailing TextBlock
        if self._current_assistant.blocks and isinstance(
            self._current_assistant.blocks[-1], TextBlock
        ):
            self._current_assistant.blocks[-1].text = self._current_text_buffer
        else:
            self._current_assistant.blocks.append(
                TextBlock(text=self._current_text_buffer)
            )

    def _flush_current_text(self) -> None:
        """Flush accumulated text to current assistant message and reset buffer."""
        if self._current_assistant and self._current_text_buffer:
            self._update_current_text_block()
            self._current_text_buffer = ""
            if self.observer:
                self.observer.on_message_updated(self)

    def _handle_command_output(self, content: str) -> None:
        """Handle command output from UserMessage (e.g., /context)."""
        import re

        # Extract content from <local-command-stdout>...</local-command-stdout>
        match = re.search(
            r"<local-command-stdout>(.*?)</local-command-stdout>", content, re.DOTALL
        )
        if match and self.observer:
            self.observer.on_command_output(self, match.group(1).strip())

    def _handle_tool_use(
        self, block: ToolUseBlock, parent_tool_use_id: str | None
    ) -> None:
        """Handle tool use start."""
        self._flush_current_text()
        self.response_had_tools = True
        self._needs_new_message = True  # Next text chunk starts a new ChatMessage

        # TodoWrite updates todos
        if block.name == ToolName.TODO_WRITE:
            self.todos = block.input.get("todos", [])
            if self.observer:
                self.observer.on_todos_updated(self)
            return

        tool = ToolUse(
            id=block.id,
            name=block.name,
            input=block.input,
            parent_tool_use_id=parent_tool_use_id,
        )

        # Track Task tools specially
        if block.name == ToolName.TASK:
            self.active_tasks[block.id] = ""

        self.pending_tools[block.id] = tool

        # Add to current assistant content
        if self._current_assistant is None:
            self._current_assistant = AssistantContent()
            self.messages.append(
                ChatItem(role="assistant", content=self._current_assistant)
            )
        self._current_assistant.blocks.append(tool)
        if self.observer:
            self.observer.on_message_updated(self)
            self.observer.on_tool_use(self, tool)

    def _handle_tool_result(self, block: ToolResultBlock) -> None:
        """Handle tool result."""
        from claudechic.processes import parse_background_task_output

        tool = self.pending_tools.pop(block.tool_use_id, None)
        if tool:
            tool.result = (
                block.content if isinstance(block.content, str) else str(block.content)
            )
            tool.is_error = block.is_error or False

            # Track background task output files
            if tool.name == ToolName.BASH and tool.result:
                output_file = parse_background_task_output(tool.result)
                if output_file:
                    command = tool.input.get("command", "")
                    self._background_outputs[command] = output_file

            # Update permission mode based on plan mode tools
            if tool.name == ToolName.EXIT_PLAN_MODE and not tool.is_error:
                self._set_permission_mode_local("default")
            elif tool.name == ToolName.ENTER_PLAN_MODE and not tool.is_error:
                self._set_permission_mode_local("plan")
                # Fetch plan path asynchronously (needed for ExitPlanMode later)
                create_safe_task(self.ensure_plan_path(), name="fetch-plan-path")

            if self.observer:
                self.observer.on_message_updated(self)
                self.observer.on_tool_result(self, tool)

        # Clean up active tasks
        self.active_tasks.pop(block.tool_use_id, None)

    # -----------------------------------------------------------------------
    # Permissions
    # -----------------------------------------------------------------------

    async def _handle_permission(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        context: ToolPermissionContext,  # noqa: ARG002  # pyright: ignore[reportUnusedParameter]
    ) -> PermissionResult:
        """Handle permission request from SDK."""
        log.info(f"Permission requested for {tool_name}: {str(tool_input)[:100]}")

        # AskUserQuestion needs special handling
        if tool_name == ToolName.ASK_USER_QUESTION:
            return await self._handle_ask_user_question(tool_input)

        # Auto-allow EnterPlanMode; ExitPlanMode falls through to normal permission flow
        if tool_name == ToolName.ENTER_PLAN_MODE:
            return PermissionResultAllow()
        if tool_name.startswith("mcp__chic__"):
            return PermissionResultAllow()

        # Block mutating tools in plan mode (except writes to plan file)
        # Note: PreToolUse hook in app.py also blocks these; this is a fallback
        if self.permission_mode == "plan" and tool_name in self.PLAN_MODE_BLOCKED_TOOLS:
            # Allow Write/Edit to files in ~/.claude/plans/
            if tool_name in (ToolName.WRITE, ToolName.EDIT):
                file_path = tool_input.get("file_path", "")
                if file_path:
                    plans_dir = Path.home() / ".claude" / "plans"
                    resolved = Path(file_path).expanduser().resolve()
                    if str(resolved).startswith(str(plans_dir)):
                        self.plan_path = resolved  # Capture for ExitPlanMode display
                        log.info(f"Auto-approved {tool_name} to plan file (plan mode)")
                        return PermissionResultAllow()
            log.info(f"Denied {tool_name} (plan mode)")
            return PermissionResultDeny(
                message=f"{tool_name} is not available in plan mode. Write your plan to the plan file and use ExitPlanMode when ready.",
                interrupt=False,
            )

        # Auto-approve edits if in acceptEdits mode
        if self.permission_mode == "acceptEdits" and tool_name in self.AUTO_EDIT_TOOLS:
            log.info(f"Auto-approved {tool_name} (acceptEdits mode)")
            return PermissionResultAllow()

        # Auto-approve if tool was allowed for session
        if tool_name in self.session_allowed_tools:
            log.info(f"Auto-approved {tool_name} (session allowed)")
            return PermissionResultAllow()

        # Auto-approve git commands during worktree finish
        if self.finish_state and tool_name == ToolName.BASH:
            command = tool_input.get("command", "")
            if command.startswith("git "):
                log.info(f"Auto-approved git during finish: {command[:50]}")
                return PermissionResultAllow()

        # Create permission request and queue it
        request = PermissionRequest(tool_name, tool_input)
        self.pending_prompts.append(request)
        if self.observer:
            self.observer.on_prompt_added(self, request)

        self._set_status(AgentStatus.NEEDS_INPUT)

        # Wait for UI to respond
        if self.permission_handler:
            result = await self.permission_handler(self, request)
        else:
            # No UI callback - wait for programmatic response
            result = await request.wait()

        # Remove from queue
        if request in self.pending_prompts:
            self.pending_prompts.remove(request)

        self._set_status(AgentStatus.BUSY)

        log.info(f"Permission result: {result.choice}")
        if result.choice == PermissionChoice.ALLOW_ALL:
            self._set_permission_mode_local("acceptEdits")
            return PermissionResultAllow()
        elif result.choice == PermissionChoice.ALLOW_SESSION:
            self.session_allowed_tools.add(tool_name)
            return PermissionResultAllow()
        elif result.choice == PermissionChoice.ALLOW:
            return PermissionResultAllow()
        elif result.choice == PermissionChoice.DENY and result.alternative_message:
            # User provided alternative instructions - don't interrupt so model continues
            return PermissionResultDeny(
                message=result.alternative_message, interrupt=False
            )
        else:
            return PermissionResultDeny(message="User denied permission")

    async def _handle_ask_user_question(
        self, tool_input: dict[str, Any]
    ) -> PermissionResult:
        """Handle AskUserQuestion tool - needs UI to collect answers."""
        questions = tool_input.get("questions", [])
        if not questions:
            return PermissionResultAllow(updated_input=tool_input)

        # Create a special request for question prompts
        request = PermissionRequest(ToolName.ASK_USER_QUESTION, tool_input)
        self.pending_prompts.append(request)
        if self.observer:
            self.observer.on_prompt_added(self, request)

        self._set_status(AgentStatus.NEEDS_INPUT)

        # The UI callback should handle question collection
        if self.permission_handler:
            result = await self.permission_handler(self, request)
        else:
            result = await request.wait()

        if request in self.pending_prompts:
            self.pending_prompts.remove(request)

        self._set_status(AgentStatus.BUSY)

        if result == PermissionChoice.DENY:
            return PermissionResultDeny(message="User cancelled questions")

        # Result should be the answers dict (stored in request._result by UI)
        answers = getattr(request, "_answers", {})
        return PermissionResultAllow(
            updated_input={"questions": questions, "answers": answers}
        )

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _set_status(self, status: AgentStatus) -> None:
        """Update status and emit event."""
        if self.status != status:
            self.status = status
            if self.observer:
                self.observer.on_status_changed(self)

    # Valid permission modes
    PERMISSION_MODES = {"default", "acceptEdits", "plan", "planSwarm"}

    def _set_permission_mode_local(self, mode: str) -> None:
        """Update permission mode locally without calling SDK.

        Used when SDK already knows (e.g., EnterPlanMode/ExitPlanMode tools).
        """
        assert mode in self.PERMISSION_MODES, f"Invalid permission mode: {mode}"
        if self.permission_mode != mode:
            self.permission_mode = mode
            if self.observer:
                self.observer.on_permission_mode_changed(self)

    async def ensure_plan_path(self) -> None:
        """Fetch and cache the plan path for this session (if not already set)."""
        if self.session_id and not self.plan_path:
            self.plan_path = await get_plan_path_for_session(
                self.session_id, cwd=self.cwd, must_exist=False
            )

    async def set_permission_mode(self, mode: str) -> None:
        """Update permission mode and emit event.

        Args:
            mode: One of 'default', 'acceptEdits', 'plan'
        """
        assert mode in self.PERMISSION_MODES, f"Invalid permission mode: {mode}"
        if self.permission_mode != mode:
            self.permission_mode = mode
            # Fetch plan path when entering plan mode
            if mode == "plan":
                await self.ensure_plan_path()
            if self.observer:
                self.observer.on_permission_mode_changed(self)

    def get_background_processes(self) -> list:
        """Get list of background processes for this agent.

        Returns:
            Empty list in PansCode mode (no subprocess tracking needed).
        """
        return []

"""Agent session management for multi-agent support."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal
import uuid

if TYPE_CHECKING:
    from claude_agent_sdk import ClaudeSDKClient
    from textual.containers import VerticalScroll
    from claude_alamode.widgets import ChatMessage, ToolUseWidget, TaskWidget
    from claude_alamode.widgets.prompts import SelectionPrompt, QuestionPrompt
    from claude_alamode.features.worktree.git import FinishState


@dataclass
class AgentSession:
    """State for a single Claude agent."""

    id: str
    name: str
    cwd: Path
    worktree: str | None = None
    client: "ClaudeSDKClient | None" = None
    session_id: str | None = None  # SDK session ID for resume

    # Status: dim=idle, gray=busy, orange=needs_input
    status: Literal["idle", "busy", "needs_input"] = "idle"

    # UI state
    chat_view: "VerticalScroll | None" = None
    current_response: "ChatMessage | None" = None
    pending_tools: dict[str, "ToolUseWidget | TaskWidget"] = field(default_factory=dict)
    active_tasks: dict[str, "TaskWidget"] = field(default_factory=dict)
    recent_tools: list["ToolUseWidget | TaskWidget"] = field(default_factory=list)
    todos: list[dict] = field(default_factory=list)
    active_prompt: "SelectionPrompt | QuestionPrompt | None" = None

    # Track if current response used tools (for summary styling)
    response_had_tools: bool = False

    # Auto-approve Edit/Write tools for this agent
    auto_approve_edits: bool = False

    # Worktree finish state (scoped to this agent)
    finish_state: "FinishState | None" = None


def create_agent_session(
    name: str,
    cwd: Path,
    worktree: str | None = None,
) -> AgentSession:
    """Create a new agent session."""
    return AgentSession(
        id=str(uuid.uuid4())[:8],
        name=name,
        cwd=cwd,
        worktree=worktree,
    )

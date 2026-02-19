"""compat.py — SDK type compatibility layer.

Replaces all types from claude_agent_sdk with lightweight dataclasses
that have the same attribute interface. This lets claudechic widgets and
agent code work without the actual SDK installed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Message / Block types (used by agent.py, app.py, messages.py, widgets)
# ---------------------------------------------------------------------------


@dataclass
class ToolUseBlock:
    """Represents a tool use request from the assistant."""

    id: str
    name: str
    input: dict[str, Any]


@dataclass
class ToolResultBlock:
    """Represents the result of a tool execution."""

    tool_use_id: str
    content: str | list = ""
    is_error: bool = False


@dataclass
class ResultMessage:
    """Sent when a response completes."""

    session_id: str = ""


@dataclass
class SystemMessage:
    """SDK system message (init, etc.)."""

    subtype: str = ""
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class StreamEvent:
    """Streaming event from SDK."""

    event: dict[str, Any] = field(default_factory=dict)
    parent_tool_use_id: str | None = None


@dataclass
class AssistantMessage:
    """An assistant message containing content blocks."""

    content: list[ToolUseBlock | ToolResultBlock] = field(default_factory=list)
    parent_tool_use_id: str | None = None


@dataclass
class UserMessage:
    """A user message from SDK."""

    uuid: str | None = None
    content: str | list = ""


# ---------------------------------------------------------------------------
# Permission types (used by agent.py)
# ---------------------------------------------------------------------------


@dataclass
class PermissionResultAllow:
    """Permission granted, optionally with updated input."""

    updated_input: dict[str, Any] | None = None


@dataclass
class PermissionResultDeny:
    """Permission denied with optional message."""

    message: str = ""
    interrupt: bool = False


PermissionResult = PermissionResultAllow | PermissionResultDeny


@dataclass
class ToolPermissionContext:
    """Context passed to permission handler (unused in PansCode adapter)."""

    pass


# ---------------------------------------------------------------------------
# Connection errors (used by app.py)
# ---------------------------------------------------------------------------


class CLIConnectionError(Exception):
    """Raised when SDK connection fails."""

    pass


class CLIJSONDecodeError(Exception):
    """Raised when SDK returns invalid JSON."""

    pass


# ---------------------------------------------------------------------------
# Hook types (used by app.py _plan_mode_hooks)
# ---------------------------------------------------------------------------


@dataclass
class HookMatcher:
    """Matches hooks for SDK hook events."""

    matcher: Any | None = None
    hooks: list = field(default_factory=list)


# HookEvent is only used in TYPE_CHECKING, so a string alias suffices
HookEvent = str


# ---------------------------------------------------------------------------
# SDK Client stubs (used by agent.py, app.py, agent_manager.py)
# ---------------------------------------------------------------------------


@dataclass
class ClaudeAgentOptions:
    """Options for creating an SDK client (stub for type compatibility).

    In PansCode mode, these are not used for actual SDK connection.
    They exist to keep AgentManager's type signatures compatible.
    """

    permission_mode: str = "default"
    env: dict[str, str] = field(default_factory=dict)
    setting_sources: list[str] = field(default_factory=list)
    cwd: Any = None
    resume: str | None = None
    model: str | None = None
    mcp_servers: dict[str, Any] = field(default_factory=dict)
    include_partial_messages: bool = False
    stderr: Any = None
    hooks: dict[str, Any] = field(default_factory=dict)
    enable_file_checkpointing: bool = False
    extra_args: dict[str, Any] = field(default_factory=dict)
    can_use_tool: Any = None


class ClaudeSDKClient:
    """Stub SDK client — not used in PansCode mode.

    Exists only so that type annotations don't break.
    All actual communication goes through PansCodeBackend.
    """

    def __init__(self, options: ClaudeAgentOptions | None = None) -> None:
        self._options = options
        self._transport = None
        self.session_id: str | None = None

    async def connect(self) -> None:
        raise NotImplementedError("Use PansCodeBackend instead of ClaudeSDKClient")

    async def disconnect(self) -> None:
        pass

    async def query(self, prompt: str) -> None:
        raise NotImplementedError("Use PansCodeBackend instead of ClaudeSDKClient")

    async def receive_response(self):
        raise NotImplementedError("Use PansCodeBackend instead of ClaudeSDKClient")
        yield  # make it an async generator

    async def interrupt(self) -> None:
        pass

    async def set_permission_mode(self, mode: str) -> None:
        pass

    async def get_server_info(self) -> dict[str, Any] | None:
        return None


# ---------------------------------------------------------------------------
# MCP stubs (used by mcp.py)
# ---------------------------------------------------------------------------


def tool(name=None, description=None, input_schema=None):
    """No-op @tool decorator (MCP stub).

    Preserves the decorated function and attaches MCP metadata.
    """
    def decorator(func):
        func._mcp_tool = {
            "name": name,
            "description": description,
            "input_schema": input_schema,
        }
        return func
    return decorator


def create_sdk_mcp_server(name: str, version: str, tools: list | None = None):
    """Stub MCP server factory — returns None in PansCode mode."""
    return None

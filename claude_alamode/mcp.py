"""In-process MCP server for alamode agent control.

Exposes tools for Claude to manage agents within alamode:
- spawn_agent: Create new agent, optionally with initial prompt
- spawn_worktree: Create git worktree + agent
- ask_agent: Send prompt to existing agent, wait for response
- list_agents: List current agents and their status
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Any

from claude_agent_sdk import tool, create_sdk_mcp_server

from claude_alamode.features.worktree.git import start_worktree

if TYPE_CHECKING:
    from claude_alamode.app import ChatApp

# Global app reference, set by ChatApp.on_mount()
_app: ChatApp | None = None


def set_app(app: ChatApp) -> None:
    """Register the app instance for MCP tools to use."""
    global _app
    _app = app


def _text_response(text: str) -> dict[str, Any]:
    """Format a text response for MCP."""
    return {"content": [{"type": "text", "text": text}]}


def _find_agent_by_name(name: str):
    """Find an agent by name. Returns (agent, error_message)."""
    if _app is None:
        return None, "App not initialized"
    for agent in _app.agents.values():
        if agent.name == name:
            return agent, None
    return None, f"Agent '{name}' not found. Use list_agents to see available agents."


async def _wait_for_agent_ready(name: str):
    """Poll until agent is ready. Returns (agent, error_message)."""
    for _ in range(50):  # 5 second timeout
        await asyncio.sleep(0.1)
        agent, _ = _find_agent_by_name(name)
        if agent and agent.client:
            return agent, None
    return None, f"Agent '{name}' creation timed out"


def _send_prompt_without_switch(agent, prompt: str) -> None:
    """Send prompt to agent, switching back to original agent after."""
    original_agent_id = _app.active_agent_id
    _app._switch_to_agent(agent.id)
    _app._handle_prompt(prompt)
    if original_agent_id:
        _app._switch_to_agent(original_agent_id)


@tool(
    "spawn_agent",
    "Create a new Claude agent in alamode. The agent gets its own chat view and can work independently.",
    {"name": str, "path": str, "prompt": str},
)
async def spawn_agent(args: dict[str, Any]) -> dict[str, Any]:
    """Spawn a new agent, optionally with an initial prompt."""
    if _app is None:
        return _text_response("Error: App not initialized")

    name = args["name"]
    path = Path(args.get("path", ".")).resolve()
    prompt = args.get("prompt")

    if not path.exists():
        return _text_response(f"Error: Path '{path}' does not exist")

    # Check if agent with this name already exists
    for agent in _app.agents.values():
        if agent.name == name:
            return _text_response(f"Error: Agent '{name}' already exists")

    # Create the agent (async worker) without switching to it
    _app._create_new_agent(name, path, switch_to=False)

    agent, error = await _wait_for_agent_ready(name)
    if agent is None:
        return _text_response(f"Error: {error}")

    result = f"Created agent '{name}' in {path}"

    if prompt:
        _send_prompt_without_switch(agent, prompt)
        result += f"\nSent initial prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}"

    return _text_response(result)


@tool(
    "spawn_worktree",
    "Create a git worktree (feature branch) with a new agent. Useful for isolated feature development.",
    {"name": str, "base_branch": str, "prompt": str},
)
async def spawn_worktree(args: dict[str, Any]) -> dict[str, Any]:
    """Create a git worktree and spawn an agent in it."""
    if _app is None:
        return _text_response("Error: App not initialized")

    name = args["name"]
    prompt = args.get("prompt")

    # Create the worktree
    success, message, wt_path = start_worktree(name)
    if not success:
        return _text_response(f"Error creating worktree: {message}")

    # Create agent in the worktree without switching to it
    _app._create_new_agent(name, wt_path, worktree=name, switch_to=False)

    agent, error = await _wait_for_agent_ready(name)
    if agent is None:
        return _text_response(f"Worktree created at {wt_path}, but {error}")

    result = f"Created worktree '{name}' at {wt_path} with new agent"

    if prompt:
        _send_prompt_without_switch(agent, prompt)
        result += f"\nSent initial prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}"

    return _text_response(result)


@tool(
    "ask_agent",
    "Send a prompt to an existing agent and wait for its response. Returns the agent's full response text.",
    {"name": str, "prompt": str},
)
async def ask_agent(args: dict[str, Any]) -> dict[str, Any]:
    """Send prompt to an agent and wait for response."""
    if _app is None:
        return _text_response("Error: App not initialized")

    name = args["name"]
    prompt = args["prompt"]

    agent, error = _find_agent_by_name(name)
    if agent is None:
        return _text_response(f"Error: {error}")

    # Clear the completion event before sending prompt
    agent._completion_event.clear()

    _send_prompt_without_switch(agent, prompt)

    # Wait for THIS agent's response with timeout (5 minutes)
    try:
        await asyncio.wait_for(agent._completion_event.wait(), timeout=300)
    except asyncio.TimeoutError:
        return _text_response(f"Error: Agent '{name}' response timed out after 5 minutes")

    # Get the response text from the agent
    response_text = agent._last_response or ""

    # Truncate if too long
    max_len = 4000
    if len(response_text) > max_len:
        response_text = response_text[:max_len] + f"\n\n[Truncated - full response was {len(response_text)} chars]"

    return _text_response(f"Response from '{name}':\n\n{response_text}")


@tool(
    "list_agents",
    "List all agents currently running in alamode with their status and working directory.",
    {},
)
async def list_agents(args: dict[str, Any]) -> dict[str, Any]:  # noqa: ARG001
    """List all agents and their status."""
    if _app is None:
        return _text_response("Error: App not initialized")

    if not _app.agents:
        return _text_response("No agents running")

    lines = ["Agents:"]
    for i, agent in enumerate(_app.agents.values(), 1):
        active = "*" if agent.id == _app.active_agent_id else " "
        wt = f" (worktree)" if agent.worktree else ""
        lines.append(f"{active}{i}. {agent.name} [{agent.status}] - {agent.cwd}{wt}")

    return _text_response("\n".join(lines))


def create_alamode_server():
    """Create the alamode MCP server with all tools."""
    return create_sdk_mcp_server(
        name="alamode",
        version="1.0.0",
        tools=[spawn_agent, spawn_worktree, ask_agent, list_agents],
    )

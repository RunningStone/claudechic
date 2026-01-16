# Multi-Agent Architecture

## Overview

Claude Chic supports multiple concurrent Claude agents, each with its own SDK connection, chat history, and working directory. The architecture uses a single source of truth pattern with `AgentManager` coordinating agents and `Agent` owning per-agent state.

## Core Classes

### Agent (`agent.py`)

The `Agent` class owns everything for a single Claude agent:

```python
class Agent:
    # Identity
    id: str                    # UUID (first 8 chars)
    name: str                  # Display name
    cwd: Path                  # Working directory
    worktree: str | None       # Git worktree branch if applicable

    # SDK connection
    client: ClaudeSDKClient | None
    session_id: str | None     # For resume

    # Status: "idle" | "busy" | "needs_input"
    status: str

    # Chat history (data model)
    messages: list[ChatItem]   # User/assistant messages with tool uses

    # Tool tracking
    pending_tools: dict[str, ToolUse]      # Tools awaiting results
    active_tasks: dict[str, str]           # Task tool accumulated text

    # UI widgets (stored for convenience)
    chat_view: VerticalScroll | None       # This agent's chat container
    current_response: ChatMessage | None   # Message being streamed
    pending_tool_widgets: dict[str, Widget]  # Tool widgets awaiting results
    active_task_widgets: dict[str, TaskWidget]

    # Per-agent state
    todos: list[dict]
    auto_approve_edits: bool
    file_index: FileIndex | None
    pending_images: list[tuple]
```

**Key methods:**
- `connect(options, resume)` - Connect to SDK
- `send(prompt)` - Send message (async, starts background task)
- `wait_for_completion()` - Wait for response to finish
- `interrupt()` - Cancel current response

**Callbacks for UI integration:**
```python
on_status_changed: Callable[[Agent], None]
on_error: Callable[[Agent, str, Exception], None]
on_complete: Callable[[Agent, ResultMessage], None]
on_todos_updated: Callable[[Agent], None]

# Fine-grained streaming (for real-time UI)
on_text_chunk: Callable[[Agent, str, bool, str | None], None]
on_tool_use: Callable[[Agent, ToolUse], None]
on_tool_result: Callable[[Agent, ToolUse], None]

# Permission UI
permission_ui_callback: Callable[[Agent, PermissionRequest], Awaitable[str]]
```

### AgentManager (`agent_manager.py`)

Coordinates multiple agents. Single source of truth for agent state.

```python
class AgentManager:
    agents: dict[str, Agent]   # All agents by ID
    active_id: str | None      # Currently active agent

    # Lifecycle callbacks
    on_created: Callable[[Agent], None]
    on_switched: Callable[[Agent, Agent | None], None]
    on_closed: Callable[[str], None]
```

**Key methods:**
- `create(name, cwd, ...)` - Create and connect new agent
- `create_unconnected(name, cwd)` - Create without connecting (for initial agent)
- `switch(agent_id)` - Switch active agent
- `close(agent_id)` - Disconnect and remove agent
- `get(agent_id)` - Get agent by ID (or active if None)

**Callback wiring:**
When agents are created, AgentManager wires up callbacks from app to agent:
```python
def _wire_agent_callbacks(self, agent):
    agent.on_status_changed = self.on_agent_status_changed
    agent.on_text_chunk = self.on_agent_text_chunk
    # etc.
```

### ChatApp (`app.py`)

Thin UI layer. Delegates to AgentManager.

```python
class ChatApp:
    agent_mgr: AgentManager

    # Properties delegate to AgentManager
    @property
    def agents(self) -> dict[str, Agent]:
        return self.agent_mgr.agents

    @property
    def active_agent_id(self) -> str | None:
        return self.agent_mgr.active_id

    @property
    def _agent(self) -> Agent | None:
        return self.agent_mgr.active
```

## Message Flow

### Sending a message

```
User types → ChatInput.Submitted
    → _handle_prompt(text)
        → Mount user message to chat_view
        → _send_to_active_agent(text)
            → agent = agent_mgr.active
            → asyncio.create_task(agent.send(prompt))
```

### Receiving responses

```
Agent._process_response() processes SDK stream
    → For each TextBlock: agent.on_text_chunk(agent, text, ...)
        → AgentManager forwards to ChatApp._on_agent_text_chunk
            → app.post_message(StreamChunk(..., agent_id=agent.id))
                → on_stream_chunk() handler
                    → agent = _get_agent(event.agent_id)
                    → Mount/update ChatMessage on agent.chat_view
```

### Agent switching

```
User clicks agent in sidebar → AgentItem.Selected
    → _switch_to_agent(agent_id)
        → self.active_agent_id = agent_id  # Property setter syncs to AgentManager
        → Hide old agent's chat_view
        → Show new agent's chat_view
        → Update sidebar, footer, todo panel
```

Or via AgentManager (e.g., when creating new agent with switch_to=True):
```
AgentManager.switch(agent_id)
    → Sets active_id
    → Calls on_switched callback
        → _on_agent_switched() in ChatApp
            → Updates UI
```

## Creating Agents

### Initial agent (on_mount)

```python
# Create synchronously so UI is ready immediately
agent = agent_mgr.create_unconnected(name=cwd.name, cwd=cwd)

# Connect in background
run_worker(_connect_initial_client)
```

### New agent via /agent command

```python
agent = await agent_mgr.create(name=name, cwd=cwd, switch_to=True)
# AgentManager handles:
#   1. Create Agent instance
#   2. Wire callbacks
#   3. Connect to SDK
#   4. Add to agents dict
#   5. Switch to new agent (triggers on_switched)
```

### MCP spawn_agent

```python
agent = await app.agent_mgr.create(name=name, cwd=path, switch_to=False)
if prompt:
    await agent.send(prompt)
```

## UI State on Agent

Each agent stores its own UI widgets:

- `chat_view` - The VerticalScroll containing this agent's messages
- `current_response` - ChatMessage widget being streamed to
- `pending_tool_widgets` - Tool widgets awaiting results
- `active_task_widgets` - Task widgets with nested content
- `recent_tools` - For collapsing older tools
- `active_prompt` - SelectionPrompt/QuestionPrompt if showing

This allows switching agents to show/hide the correct widgets.

## Permissions

Each agent handles its own permissions via `permission_ui_callback`:

```python
async def _handle_agent_permission_ui(self, agent, request):
    # Show SelectionPrompt in UI
    # Wait for user choice
    # Return "allow" | "deny" | "allow_all"
```

The callback is set by AgentManager when wiring up the agent.

## Key Design Decisions

1. **Single source of truth**: `AgentManager.agents` is THE dict of agents. `ChatApp.agents` is a property that returns it.

2. **Agent owns SDK lifecycle**: Agent.connect(), Agent.send(), Agent.disconnect() - not ChatApp.

3. **Callbacks for UI**: Agent emits events, ChatApp subscribes. No direct widget manipulation in Agent.

4. **UI widgets on Agent**: For convenience, Agent stores references to its widgets. This avoids lookups.

5. **Concurrent async**: Agents run via `asyncio.create_task()`, not Textual's `@work`. This allows true concurrency.

## Commands

- `/agent` - List all agents
- `/agent <name>` - Create new agent in current directory
- `/agent <name> <path>` - Create new agent in specified directory
- `/agent close` - Close current agent
- `/agent close <name>` - Close agent by name
- `Ctrl+1-9` - Switch to agent by position

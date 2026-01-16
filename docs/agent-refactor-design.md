# Agent Refactor Design

## Status

**In Progress** - Agent, AgentManager, and ChatView created. ChatApp integration pending.

## Overview

Refactor multi-agent state management from "ChatApp with dict of AgentSession" to "Agent as autonomous unit, AgentManager as coordinator, ChatApp as thin UI layer".

## Motivation

The codebase started as a single-agent application. Multi-agent support was added via a dict of `AgentSession` dataclasses with property delegators. This led to:

1. **ChatApp is too fat** - 1300+ lines mixing agent lifecycle, SDK communication, UI events, commands
2. **9 property delegators** - Boilerplate to access `self._agent.X` for backward compatibility
3. **Scattered agent logic** - Create/switch/close spread across methods
4. **Global state bleeding** - `pending_images`, `file_index` are global but should be per-agent
5. **Permission handling split** - `auto_approve_edits` on AgentSession but prompt logic in ChatApp
6. **No concurrent agents** - Only active agent can have in-flight query

## Key Design Decisions

### Agent Owns Its World

Each Agent is fully autonomous:
- SDK client and connection lifecycle
- Message history (as data, not widgets)
- Permission request queue
- Response processing loop (concurrent async task)
- Per-agent state: images, todos, file index, auto-edit mode

### Data-Driven UI

Agent buffers everything as data. UI renders the active agent's state.

- `agent.messages: list[ChatItem]` - full chat history
- `agent.pending_prompts: deque[PermissionRequest]` - queued permission requests
- `agent.todos: list[dict]` - todo items

On agent switch: UI re-renders from new agent's data.

On agent event while inactive: data buffers naturally, UI updates when switched to.

### Concurrent Agents

Agents run concurrently via `asyncio.create_task()`. No Textual `@work` decorator - just standard async.

Multiple agents can process responses simultaneously. User can chat with one agent while another works in background.

### Callbacks for UI Integration

Agent emits events via callbacks. ChatApp subscribes and updates UI:

```python
agent.on_message_updated = lambda a: self._render_messages(a)
agent.on_status_changed = lambda a: self._update_sidebar(a)
agent.on_prompt_added = lambda a, req: self._show_prompt(a, req)
```

### Permission Flow

1. SDK calls `agent._handle_permission()`
2. Agent checks auto-approve rules
3. If needs user input: creates `PermissionRequest`, adds to queue, calls `permission_ui_callback`
4. ChatApp shows prompt UI, collects response, returns result
5. Agent removes request from queue, returns SDK result

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        ChatApp                               │
│  - Global UI (header, footer, sidebar, input)               │
│  - Keybindings                                               │
│  - Routes input to active agent                             │
│  - Subscribes to agent events, updates UI                   │
│  - Handles permission UI (prompts), sends result to agent   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      AgentManager                            │
│  - Creates/closes agents (including SDK connection)         │
│  - Tracks active agent                                       │
│  - Provides lookup (by id, by name)                         │
│  - Wires agent callbacks                                    │
│  - No UI dependencies, pure async                           │
└─────────────────────────────────────────────────────────────┘
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
              ┌─────────┐         ┌─────────┐
              │  Agent  │         │  Agent  │  (concurrent)
              └─────────┘         └─────────┘
```

## Message Types

```python
@dataclass
class UserContent:
    text: str
    images: list[tuple[str, str]]  # (filename, media_type)

@dataclass
class ToolUse:
    id: str
    name: str
    input: dict[str, Any]
    result: str | None = None
    is_error: bool = False

@dataclass
class AssistantContent:
    text: str = ""
    tool_uses: list[ToolUse] = field(default_factory=list)

@dataclass
class ChatItem:
    role: Literal["user", "assistant"]
    content: UserContent | AssistantContent
```

## Agent Class

See `claudechic/agent.py` for implementation.

Key attributes:
- **Identity**: `id`, `name`, `cwd`, `worktree`
- **SDK**: `client`, `session_id`, `_response_task`
- **Status**: `status` (idle/busy/needs_input)
- **History**: `messages: list[ChatItem]`
- **Permissions**: `pending_prompts: deque[PermissionRequest]`, `auto_approve_edits`
- **State**: `pending_images`, `file_index`, `todos`

Key methods:
- `connect(options, resume)` - connect to SDK
- `disconnect()` - cleanup
- `send(prompt)` - send message, start concurrent response task
- `interrupt()` - cancel current response
- `wait_for_completion()` - for MCP ask_agent

Callbacks:
- `on_status_changed(agent)`
- `on_message_updated(agent)`
- `on_prompt_added(agent, request)`
- `on_error(agent, message, exception)`
- `on_complete(agent, result)`
- `on_todos_updated(agent)`
- `permission_ui_callback(agent, request) -> str` - async, returns "allow"/"deny"/"allow_all"

## AgentManager Class

See `claudechic/agent_manager.py` for implementation.

```python
class AgentManager:
    agents: dict[str, Agent]
    active_id: str | None

    # Callbacks for ChatApp
    on_created: Callable[[Agent], None] | None
    on_switched: Callable[[Agent, Agent | None], None] | None
    on_closed: Callable[[str], None] | None

    @property
    def active(self) -> Agent | None

    def get(self, agent_id: str | None = None) -> Agent | None
    def find_by_name(self, name: str) -> Agent | None

    async def create(
        self,
        name: str,
        cwd: Path,
        options: ClaudeAgentOptions,
        worktree: str | None = None,
        resume: str | None = None,
        switch_to: bool = True,
    ) -> Agent

    def switch(self, agent_id: str) -> bool

    async def close(self, agent_id: str) -> None
    async def close_all(self) -> None
```

## ChatApp Changes

### Removed
- `self.agents` dict → `self.agent_mgr.agents`
- `self.active_agent_id` → `self.agent_mgr.active_id`
- 9 property delegators
- `run_claude()` → moved to `Agent._process_response()`
- `pending_images` → moved to `Agent`
- `file_index` → moved to `Agent`

### Added
- `self.agent_mgr: AgentManager`
- Chat view renderer (renders `agent.messages` to widgets)
- Agent callback wiring

### Simplified
- `on_mount()` - creates AgentManager, first agent
- `_handle_prompt()` - calls `agent_mgr.active.send(prompt)`
- `_switch_to_agent()` - calls `agent_mgr.switch()`, re-renders UI

## MCP Integration

```python
@tool("spawn_agent", ...)
async def spawn_agent(args):
    agent = await _app.agent_mgr.create(
        name=args["name"],
        cwd=Path(args["path"]),
        switch_to=False,
    )
    if args.get("prompt"):
        await agent.send(args["prompt"])
    return _text_response(f"Created agent '{agent.name}'")

@tool("ask_agent", ...)
async def ask_agent(args):
    agent = _app.agent_mgr.find_by_name(args["name"])
    await agent.send(args["prompt"])
    response = await agent.wait_for_completion()
    return _text_response(response)

@tool("list_agents", ...)
async def list_agents(args):
    lines = [f"{a.name} [{a.status}]" for a in _app.agent_mgr]
    return _text_response("\n".join(lines))
```

## Migration Steps

1. ✅ Create Agent class in `agent.py`
2. ✅ Create AgentManager class in `agent_manager.py`
3. ✅ Create ChatView widget in `widgets/chat_view.py`
4. ✅ Add backward compatibility (AgentSession, create_agent_session)
5. ✅ Add imports to app.py (Agent, AgentManager, ChatView)
6. ✅ Initialize AgentManager in ChatApp.__init__ and on_mount()
7. ✅ Wire agent callbacks in ChatApp (_wire_agent_manager_callbacks)
8. ✅ Migrate run_claude() to use Agent.send()
   - Added `_send_to_active_agent()` method that calls `agent.send()`
   - Agent callbacks emit Textual Messages via `_on_agent_*` handlers
   - Fine-grained callbacks: `on_text_chunk`, `on_tool_use`, `on_tool_result`
   - Legacy `run_claude()` still exists but no longer used for main message flow
9. ✅ Update MCP tools to use agent_mgr
   - `spawn_agent` uses `agent_mgr.create()` and `agent.send()`
   - `spawn_worktree` uses `agent_mgr.create()` with worktree param
   - `ask_agent` uses `agent.send()` and `agent.wait_for_completion()`
   - `list_agents` iterates over `agent_mgr`
   - No more UI switching needed - agents run concurrently
10. ⬜ Remove old code (delegators, AgentSession, etc.)
11. ⬜ Test concurrent agents

**Current state**: All 50 tests pass. MCP tools now use AgentManager API.
Agents can be created and prompted without UI switching.

**Next step**: Clean up old code (run_claude, _create_new_agent, etc.) once confident in new path.

## File Changes

```
claudechic/
├── agent.py              # AgentSession dataclass → Agent class ✅
├── agent_manager.py      # NEW ✅
├── widgets/
│   └── chat_view.py      # NEW - ChatView widget ✅
├── app.py                # Significantly smaller (TODO)
├── mcp.py                # Simplified (TODO)
└── ...
```

## Testing

- Unit test Agent with mock SDK client
- Unit test AgentManager without UI
- Integration test concurrent agents
- E2E test permission flow

## Open Questions

1. **History loading**: When resuming a session, should we populate `agent.messages` from session file? (Probably yes for consistency)

2. **Widget recycling**: Re-rendering full history on switch could be slow. May need to optimize later (virtual list, widget pooling). Start simple.

3. **Task tool nesting**: Current design tracks `active_tasks` but doesn't fully model nested agent output in history. May need `TaskContent` type.

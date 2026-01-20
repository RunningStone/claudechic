# Claude Chic

A stylish terminal UI for Claude Code, built with Textual and wrapping the `claude-agent-sdk`.

## Run

```bash
uv run claudechic
uv run claudechic --resume     # Resume most recent session
uv run claudechic -s <uuid>    # Resume specific session
```

Requires Claude Code to be logged in with a Max/Pro subscription (`claude /login`).

## File Map

```
claudechic/
├── __init__.py        # Package entry, exports ChatApp
├── __main__.py        # CLI entry point
├── agent.py           # Agent class - SDK connection, history, permissions, state
├── agent_manager.py   # AgentManager - coordinates multiple concurrent agents
├── app.py             # ChatApp - main application, event handlers
├── commands.py        # Slash command routing (/agent, /shell, /clear, etc.)
├── compact.py         # Session compaction - shrink old tool uses to save context
├── errors.py          # Logging infrastructure, error handling
├── file_index.py      # Fuzzy file search using git ls-files
├── formatting.py      # Tool formatting, diff rendering (pure functions)
├── history.py         # Global history loading from ~/.claude/history.jsonl
├── mcp.py             # In-process MCP server for agent control tools
├── messages.py        # Custom Textual Message types for SDK events
├── permissions.py     # PermissionRequest dataclass for tool approval
├── profiling.py       # Lightweight profiling utilities (@profile decorator)
├── sampling.py        # CPU-conditional sampling profiler for high-CPU investigation
├── protocols.py       # Observer protocols (AgentObserver, AgentManagerObserver)
├── sessions.py        # Session file loading and listing (pure functions)
├── styles.tcss        # Textual CSS - visual styling
├── theme.py           # Textual theme definition
├── usage.py           # OAuth usage API fetching (rate limits)
├── features/
│   ├── __init__.py    # Feature module exports
│   └── worktree/
│       ├── __init__.py   # Public API (list_worktrees, handle_worktree_command)
│       ├── commands.py   # /worktree command handlers
│       ├── git.py        # Git worktree operations
│       └── prompts.py    # WorktreePrompt widget
└── widgets/
    ├── __init__.py    # Re-exports all widgets
    ├── agents.py      # AgentSidebar, AgentItem for multi-agent UI
    ├── autocomplete.py # Autocomplete for slash commands and file paths
    ├── chat.py        # ChatMessage, ChatInput, ThinkingIndicator
    ├── chat_view.py   # ChatView - renders agent messages, handles streaming
    ├── context_report.py # ContextReport - visual 2D grid for /context
    ├── diff.py        # Syntax-highlighted diff widget
    ├── footer.py      # Custom footer with git branch, CPU/context bars
    ├── history_search.py # Reverse history search widget (Ctrl+R)
    ├── indicators.py  # CPUBar (clickable), ContextBar resource monitors
    ├── profile_modal.py # ProfileModal - shows profiling stats on CPU click
    ├── prompts.py     # SelectionPrompt, QuestionPrompt, SessionItem
    ├── scroll.py      # AutoHideScroll - auto-hiding scrollbar container
    ├── todo.py        # TodoPanel for TodoWrite tool display
    ├── tools.py       # ToolUseWidget, TaskWidget, AgentToolWidget
    └── usage.py       # UsageReport, UsageBar for /usage command

tests/
├── __init__.py        # Package marker
├── conftest.py        # Shared fixtures (wait_for)
├── test_app.py        # E2E tests with real SDK
├── test_app_ui.py     # App UI tests without SDK
├── test_autocomplete.py # Autocomplete widget tests
├── test_file_index.py # Fuzzy file search tests
└── test_widgets.py    # Pure widget tests
```

## Architecture

### Module Boundaries

**Pure functions (no UI dependencies):**
- `formatting.py` - Tool header formatting, diff rendering, language detection
- `sessions.py` - Session file I/O, listing, filtering
- `file_index.py` - Fuzzy file search, git ls-files integration
- `compact.py` - Session compaction to reduce context window usage
- `usage.py` - OAuth API for rate limit info

**Agent layer (no UI dependencies):**
- `agent.py` - `Agent` class owns SDK client, message history, permissions, state
- `agent_manager.py` - Coordinates multiple agents, switching, lifecycle
- `protocols.py` - Observer protocols (`AgentObserver`, `AgentManagerObserver`, `PermissionHandler`)

**Internal protocol:**
- `messages.py` - Custom `Message` subclasses for async event communication
- `permissions.py` - `PermissionRequest` dataclass bridging SDK callbacks to UI
- `mcp.py` - MCP server exposing agent control tools to Claude

**Features:**
- `features/worktree/` - Git worktree management for isolated development

**UI components:**
- `widgets/` - Textual widgets with associated styles
- `widgets/chat_view.py` - `ChatView` renders agent messages, handles streaming
- `app.py` - Main app orchestrating widgets and agents via observer pattern

### Widget Hierarchy

```
ChatApp
├── Horizontal #main
│   ├── ListView #session-picker (hidden by default)
│   ├── ChatView (one per agent, only active visible)
│   │   ├── ChatMessage (user/assistant)
│   │   ├── ToolUseWidget (collapsible tool display)
│   │   ├── TaskWidget (for Task tool - nested content)
│   │   ├── AgentToolWidget (for MCP chic tools)
│   │   └── ThinkingIndicator (animated spinner)
│   └── Vertical #right-sidebar (hidden when narrow or single agent)
│       ├── AgentSidebar (list of agents with status)
│       └── TodoPanel (todos for active agent)
├── Horizontal #input-wrapper
│   └── Vertical #input-container
│       ├── ImageAttachments (hidden by default)
│       ├── ChatInput (or SelectionPrompt/QuestionPrompt when prompting)
│       └── TextAreaAutoComplete (slash commands, file paths)
└── StatusFooter (git branch, CPU/context bars)
```

### Observer Pattern

Agent and AgentManager emit events via protocol-based observers:

```
Agent events (AgentObserver)         ChatApp handlers
────────────────────────────         ────────────────
on_text_chunk()                  ->  ChatView.append_text()
on_tool_use()                    ->  ChatView.append_tool_use()
on_tool_result()                 ->  ChatView.update_tool_result()
on_complete()                    ->  end response, update UI
on_status_changed()              ->  update AgentSidebar indicator
on_prompt_added()                ->  show SelectionPrompt/QuestionPrompt

AgentManager events                  ChatApp handlers
───────────────────                  ────────────────
on_agent_created()               ->  create ChatView, update sidebar
on_agent_switched()              ->  show/hide ChatViews
on_agent_closed()                ->  remove ChatView, update sidebar
```

This decouples Agent (pure async) from UI (Textual widgets).

### Permission Flow

When SDK needs tool approval:
1. `can_use_tool` callback creates `PermissionRequest`
2. Request queued to `app.interactions` (for testing)
3. `SelectionPrompt` mounted, replacing input
4. User selects allow/deny/allow-all
5. Callback returns `PermissionResultAllow` or `PermissionResultDeny`

For `AskUserQuestion` tool: `QuestionPrompt` handles multi-question flow.

### Styling

Visual language uses left border bars to indicate content type:
- **Orange** (`#cc7700`) - User messages
- **Blue** (`#334455`) - Assistant messages
- **Gray** (`#333333`) - Tool uses (brightens on hover)
- **Blue-gray** (`#445566`) - Task widgets

Context/CPU bars color-code by threshold (dim → yellow → red).

Copy buttons appear on hover. Collapsibles auto-collapse older tool uses.

## Key SDK Usage

```python
from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions

client = ClaudeSDKClient(ClaudeAgentOptions(
    permission_mode="default",
    env={"ANTHROPIC_API_KEY": ""},
    can_use_tool=permission_callback,
    resume=session_id,
))
await client.connect()
await client.query("prompt")
async for message in client.receive_response():
    # Handle AssistantMessage, TextBlock, ToolUseBlock, ToolResultBlock, ResultMessage
```

## Keybindings

- Enter: Send message
- Ctrl+C (x2): Quit
- Ctrl+L: Clear chat (UI only)
- Ctrl+R: Reverse history search
- Shift+Tab: Toggle auto-edit mode
- Ctrl+N: New agent (hint)
- Ctrl+1-9: Switch to agent by position

## Commands

### Multi-Agent
- `/agent` - List all agents
- `/agent <name>` - Create new agent in current directory
- `/agent <name> <path>` - Create new agent in specified directory
- `/agent close` - Close current agent
- `/agent close <name>` - Close agent by name

Agent status indicators: ○ (idle), ● gray (busy), ● orange (needs input)

### Session Management
- `/resume` - Show session picker
- `/resume <id>` - Resume specific session
- `/compactish` - Compact session to reduce context (dry run with `-n`)
- `/usage` - Show API rate limit usage
- `/clear` - Clear chat UI
- `/shell <cmd>` - Suspend TUI and run shell command
- `/exit` - Quit

## Testing

```bash
uv run pytest tests/ -v
```

Tests use `app.interactions` queue to programmatically respond to permission prompts, and `app.completions` queue to wait for response completion. Real SDK required.

## Pre-commit Hooks

```bash
uv run pre-commit install  # Install hooks (one-time)
uv run pre-commit run --all-files  # Run manually
```

Hooks: ruff (lint + fix), ruff-format, pyright. Run automatically on commit.

# Claude Code Textual UI

Terminal UI for Claude Code built with Textual, wrapping the `claude-agent-sdk`.

## Run

```bash
uv run python app.py
uv run python app.py --resume     # Resume most recent session
uv run python app.py -s <uuid>    # Resume specific session
```

Requires Claude Code to be logged in with a Max/Pro subscription (`claude /login`).

## Architecture

- `app.py` - Single-file Textual app (~860 lines)
- `styles.tcss` - Textual CSS styling

### Key Components

- `ChatApp` - Main app, manages SDK client and UI state
- `ChatMessage` - User/assistant message with copy button
- `ToolUseWidget` - Collapsible tool use display with colored diffs for edits
- `ContextHeader` / `ContextBar` - Header showing context window usage as progress bar
- `SessionItem` - Sidebar item for session selection

### Message Flow

1. User types in `ChatInput`, Enter submits
2. `run_claude()` worker calls SDK in background
3. SDK responses post custom messages: `StreamChunk`, `ToolUseMessage`, `ToolResultMessage`, `ResponseComplete`
4. Main thread handlers update UI widgets

### Session Management

Sessions stored in `~/.claude/projects/-path-to-project/*.jsonl`. The app can:
- List recent sessions in sidebar (Ctrl+B)
- Resume sessions by loading history + reconnecting SDK with `resume=session_id`
- Filter out `/context` and other internal commands from history display

## Key SDK Usage

```python
from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions

client = ClaudeSDKClient(ClaudeAgentOptions(
    allowed_tools=["Read", "Edit", "Write", "Bash", "Glob", "Grep"],
    permission_mode="acceptEdits",
    env={"ANTHROPIC_API_KEY": ""},  # Force Max subscription
    resume=session_id,  # Optional: resume existing session
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
- Ctrl+B: Toggle session sidebar

## Features

- **Context bar**: Shows token usage in header (green/yellow/red by %)
- **Tool use display**: Collapsible widgets, last 2 expanded, older auto-collapsed
- **Edit diffs**: Word-level highlighting with red/green backgrounds
- **Copy buttons**: On messages and tool uses
- **Session resume**: Via CLI flags or sidebar selection

## Future Work

- Permission prompts in UI
- Thinking blocks display
- Tool result content display improvements

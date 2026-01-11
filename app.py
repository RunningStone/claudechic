"""Claude Code Textual UI - A terminal interface for Claude Code."""

import difflib
import json
import logging
import re
from pathlib import Path

# Set up file logging
logging.basicConfig(
    filename="cc-textual.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
log = logging.getLogger(__name__)

from textual.app import App, ComposeResult, RenderResult
from textual.widgets import Markdown, TextArea, Header, Footer, Static, ListView, ListItem, Label, Collapsible, Button
from textual.widgets._header import HeaderIcon, HeaderTitle
from textual.containers import VerticalScroll, Horizontal
from textual.message import Message
from textual.binding import Binding
from textual.reactive import reactive
from textual import work, on
from textual.events import Key
from textual.widget import Widget
from rich.text import Text

from claude_agent_sdk import (
    ClaudeSDKClient,
    ClaudeAgentOptions,
    AssistantMessage,
    SystemMessage,
    UserMessage,
    TextBlock,
    ToolUseBlock,
    ToolResultBlock,
    ResultMessage,
)


def is_valid_uuid(s: str) -> bool:
    """Check if string is a valid UUID (not agent-* internal sessions)."""
    return bool(re.match(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', s, re.I))


def get_project_sessions_dir() -> Path | None:
    """Get the sessions directory for the current project."""
    cwd = Path.cwd().absolute()
    # Claude stores sessions in ~/.claude/projects/-path-to-project (with dashes instead of slashes)
    project_key = str(cwd).replace("/", "-")
    sessions_dir = Path.home() / ".claude/projects" / project_key
    return sessions_dir if sessions_dir.exists() else None


def get_recent_sessions(limit: int = 20) -> list[tuple[str, str, float, int]]:
    """Get recent sessions from current project only. Returns [(session_id, preview, mtime, msg_count)]."""
    sessions = []
    sessions_dir = get_project_sessions_dir()
    if not sessions_dir:
        return sessions

    for f in sessions_dir.glob("*.jsonl"):
        # Skip non-UUID sessions (agent-* are internal)
        if not is_valid_uuid(f.stem):
            continue
        if f.stat().st_size == 0:
            continue
        try:
            preview = ""
            msg_count = 0
            with open(f) as fh:
                for line in fh:
                    d = json.loads(line)
                    if d.get("type") == "user":
                        msg_count += 1
                        if not preview:
                            content = d.get("message", {}).get("content", "")
                            preview = (content[:50] if isinstance(content, str) else str(content)[:50]).replace("\n", " ")
            if preview:
                sessions.append((f.stem, preview, f.stat().st_mtime, msg_count))
        except (json.JSONDecodeError, IOError):
            continue

    sessions.sort(key=lambda x: x[2], reverse=True)
    return sessions[:limit]


def load_session_messages(session_id: str, limit: int = 10) -> list[dict]:
    """Load recent messages from a session file. Returns list of message dicts.

    Each dict has 'type' key: 'user', 'assistant', or 'tool_use'.
    - user: {'type': 'user', 'content': str}
    - assistant: {'type': 'assistant', 'content': str}
    - tool_use: {'type': 'tool_use', 'name': str, 'input': dict}
    """
    sessions_dir = get_project_sessions_dir()
    if not sessions_dir:
        return []

    session_file = sessions_dir / f"{session_id}.jsonl"
    if not session_file.exists():
        return []

    messages = []
    try:
        with open(session_file) as f:
            for line in f:
                d = json.loads(line)
                if d.get("type") == "user":
                    content = d.get("message", {}).get("content", "")
                    if isinstance(content, str) and content.strip():
                        # Skip slash commands and their output (XML-wrapped format)
                        if content.strip().startswith("/"):
                            continue
                        if "<command-name>/" in content:
                            continue
                        if "<local-command-stdout>" in content:
                            continue
                        if "<local-command-caveat>" in content:
                            continue
                        messages.append({"type": "user", "content": content})
                elif d.get("type") == "assistant":
                    msg = d.get("message", {})
                    content_blocks = msg.get("content", [])
                    for block in content_blocks:
                        if isinstance(block, dict):
                            if block.get("type") == "text":
                                text = block.get("text", "")
                                if text.strip():
                                    messages.append({"type": "assistant", "content": text})
                            elif block.get("type") == "tool_use":
                                messages.append({
                                    "type": "tool_use",
                                    "name": block.get("name", "?"),
                                    "input": block.get("input", {}),
                                    "id": block.get("id", ""),
                                })
    except (json.JSONDecodeError, IOError):
        pass

    # Return last N messages
    return messages[-limit:]


MAX_CONTEXT_TOKENS = 200_000  # Claude's context window


def parse_context_tokens(content: str) -> int | None:
    """Parse token count from /context output. Returns tokens used or None."""
    # Look for "**Tokens:** 17.7k / 200.0k" pattern
    match = re.search(r'\*\*Tokens:\*\*\s*([\d.]+)(k)?\s*/\s*[\d.]+k', content)
    if match:
        used = float(match.group(1))
        if match.group(2):  # has 'k' suffix
            used *= 1000
        return int(used)
    return None


class ContextBar(Widget):
    """Display context usage as a progress bar in the header."""

    DEFAULT_CSS = """
    ContextBar {
        dock: right;
        width: 20;
        padding: 0 1;
        content-align: right middle;
    }
    """

    tokens = reactive(0)
    max_tokens = reactive(MAX_CONTEXT_TOKENS)

    def render(self) -> RenderResult:
        pct = min(self.tokens / self.max_tokens, 1.0) if self.max_tokens else 0
        bar_width = 10
        filled = int(pct * bar_width)
        bar = "█" * filled + "░" * (bar_width - filled)
        # Color based on usage: green < 50%, yellow < 80%, red >= 80%
        if pct < 0.5:
            color = "green"
        elif pct < 0.8:
            color = "yellow"
        else:
            color = "red"
        return Text.assemble((bar, color), f" {pct*100:.0f}%")


class ContextHeader(Header):
    """Header with context bar instead of clock."""

    def compose(self) -> ComposeResult:
        yield HeaderIcon().data_bind(Header.icon)
        yield HeaderTitle()
        yield ContextBar(id="context-bar")


class SessionItem(ListItem):
    """A session in the sidebar."""
    def __init__(self, session_id: str, preview: str, msg_count: int = 0) -> None:
        super().__init__()
        self.session_id = session_id
        self.preview = preview
        self.msg_count = msg_count

    def compose(self) -> ComposeResult:
        yield Label(f"{self.session_id[:8]}… ({self.msg_count})\n{self.preview[:40]}")


class ChatInput(TextArea):
    """Text input that submits on Enter, newline on Shift+Enter."""

    BINDINGS = [
        Binding("enter", "submit", "Send", priority=True),
        Binding("ctrl+j", "newline", "Newline", priority=True),
        Binding("ctrl+c", "quit_app", "Quit", priority=True),
    ]

    def action_quit_app(self) -> None:
        self.app.action_quit()

    class Submitted(Message):
        """Posted when user presses Enter."""
        def __init__(self, text: str) -> None:
            self.text = text
            super().__init__()

    def action_submit(self) -> None:
        self.post_message(self.Submitted(self.text))

    def action_newline(self) -> None:
        self.insert("\n")


class StreamChunk(Message):
    """Message sent when a chunk of text is received."""
    def __init__(self, text: str, new_message: bool = False) -> None:
        self.text = text
        self.new_message = new_message  # Start a new ChatMessage widget
        super().__init__()


class ResponseComplete(Message):
    """Message sent when response is complete."""
    def __init__(self, result: ResultMessage | None = None) -> None:
        self.result = result
        super().__init__()


class ToolUseMessage(Message):
    """Message sent when a tool use starts."""
    def __init__(self, block: ToolUseBlock) -> None:
        self.block = block
        super().__init__()


class ToolResultMessage(Message):
    """Message sent when a tool result arrives."""
    def __init__(self, block: ToolResultBlock) -> None:
        self.block = block
        super().__init__()


class ContextUpdate(Message):
    """Message sent when context usage is known."""
    def __init__(self, tokens: int) -> None:
        self.tokens = tokens
        super().__init__()


def format_tool_header(name: str, input: dict) -> str:
    """Format a one-line header for a tool use."""
    if name == "Edit":
        return f"Edit: {input.get('file_path', '?')}"
    elif name == "Write":
        return f"Write: {input.get('file_path', '?')}"
    elif name == "Read":
        return f"Read: {input.get('file_path', '?')}"
    elif name == "Bash":
        cmd = input.get('command', '?')
        desc = input.get('description', '')
        if desc:
            return f"Bash: {desc}"
        return f"Bash: {cmd[:50]}{'...' if len(cmd) > 50 else ''}"
    elif name == "Glob":
        return f"Glob: {input.get('pattern', '?')}"
    elif name == "Grep":
        return f"Grep: {input.get('pattern', '?')}"
    else:
        return f"{name}"


def get_lang_from_path(path: str) -> str:
    """Guess language from file extension for syntax highlighting."""
    ext = Path(path).suffix.lower()
    return {
        '.py': 'python', '.js': 'javascript', '.ts': 'typescript',
        '.jsx': 'jsx', '.tsx': 'tsx', '.rs': 'rust', '.go': 'go',
        '.rb': 'ruby', '.java': 'java', '.c': 'c', '.cpp': 'cpp',
        '.h': 'c', '.hpp': 'cpp', '.css': 'css', '.html': 'html',
        '.json': 'json', '.yaml': 'yaml', '.yml': 'yaml', '.toml': 'toml',
        '.md': 'markdown', '.sh': 'bash', '.bash': 'bash',
    }.get(ext, '')


def _tokenize(s: str) -> list[str]:
    """Split string into words and punctuation for word-level diff."""
    return re.findall(r'\w+|[^\w\s]|\s+', s)


def _render_word_diff(old_line: str, new_line: str, result: Text) -> None:
    """Render a single line pair with word-level highlighting."""
    old_tokens = _tokenize(old_line)
    new_tokens = _tokenize(new_line)
    sm = difflib.SequenceMatcher(None, old_tokens, new_tokens)

    # Build old line with highlights
    result.append("- ", style="on #3c1f1f")
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        chunk = ''.join(old_tokens[i1:i2])
        if tag == 'equal':
            result.append(chunk, style="on #3c1f1f")
        elif tag in ('delete', 'replace'):
            result.append(chunk, style="bold red on #5c2f2f")
    result.append("\n")

    # Build new line with highlights
    result.append("+ ", style="on #1f3c1f")
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        chunk = ''.join(new_tokens[j1:j2])
        if tag == 'equal':
            result.append(chunk, style="on #1f3c1f")
        elif tag in ('insert', 'replace'):
            result.append(chunk, style="bold green on #2f5c2f")
    result.append("\n")


def format_diff_text(old: str, new: str, max_len: int = 300) -> Text:
    """Format a diff with red/green backgrounds and word-level highlighted changes."""
    result = Text()
    old_preview = old[:max_len] + ('...' if len(old) > max_len else '')
    new_preview = new[:max_len] + ('...' if len(new) > max_len else '')
    old_lines = old_preview.split('\n') if old else []
    new_lines = new_preview.split('\n') if new else []

    # Use difflib to match lines
    sm = difflib.SequenceMatcher(None, old_lines, new_lines)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == 'equal':
            for line in old_lines[i1:i2]:
                result.append(f"  {line}\n", style="dim")
        elif tag == 'delete':
            for line in old_lines[i1:i2]:
                result.append(f"- {line}\n", style="on #3c1f1f")
        elif tag == 'insert':
            for line in new_lines[j1:j2]:
                result.append(f"+ {line}\n", style="on #1f3c1f")
        elif tag == 'replace':
            # For replaced lines, highlight word-level changes
            for old_line, new_line in zip(old_lines[i1:i2], new_lines[j1:j2]):
                _render_word_diff(old_line, new_line, result)
            # Handle unequal line counts
            for line in old_lines[i1+len(new_lines[j1:j2]):i2]:
                result.append(f"- {line}\n", style="on #3c1f1f")
            for line in new_lines[j1+len(old_lines[i1:i2]):j2]:
                result.append(f"+ {line}\n", style="on #1f3c1f")
    return result


def format_tool_details(name: str, input: dict) -> str:
    """Format expanded details for a tool use."""
    if name == "Edit":
        path = input.get('file_path', '?')
        old = input.get('old_string', '')
        new = input.get('new_string', '')
        lang = get_lang_from_path(path)
        diff_lines = []
        if old:
            preview = old[:300] + ('...' if len(old) > 300 else '')
            for line in preview.split('\n'):
                diff_lines.append(f"- {line}")
        if new:
            preview = new[:300] + ('...' if len(new) > 300 else '')
            for line in preview.split('\n'):
                diff_lines.append(f"+ {line}")
        if diff_lines:
            return f"```{lang}\n" + "\n".join(diff_lines) + "\n```"
        return f"`{path}`"
    elif name == "Write":
        path = input.get('file_path', '?')
        content = input.get('content', '')
        lang = get_lang_from_path(path)
        preview = content[:400] + ('...' if len(content) > 400 else '')
        return f"```{lang}\n{preview}\n```"
    elif name == "Read":
        path = input.get('file_path', '?')
        offset = input.get('offset')
        limit = input.get('limit')
        details = f"**File:** `{path}`"
        if offset or limit:
            details += f"\nLines: {offset or 0} - {(offset or 0) + (limit or 'end')}"
        return details
    elif name == "Bash":
        cmd = input.get('command', '?')
        return f"```bash\n{cmd}\n```"
    elif name == "Glob":
        pattern = input.get('pattern', '?')
        path = input.get('path', '.')
        return f"**Pattern:** `{pattern}`\n**Path:** `{path}`"
    elif name == "Grep":
        pattern = input.get('pattern', '?')
        path = input.get('path', '.')
        return f"**Pattern:** `{pattern}`\n**Path:** `{path}`"
    else:
        return f"```\n{json.dumps(input, indent=2)}\n```"


class ToolUseWidget(Static):
    """A collapsible widget showing a tool use."""

    def __init__(self, block: ToolUseBlock, collapsed: bool = False) -> None:
        super().__init__()
        self.block = block
        self.tool_use_id = block.id
        self.result: ToolResultBlock | None = None
        self._initial_collapsed = collapsed

    def compose(self) -> ComposeResult:
        yield Button("⎘", id="tool-copy-btn", classes="tool-copy-btn")
        header = format_tool_header(self.block.name, self.block.input)
        with Collapsible(title=header, collapsed=self._initial_collapsed):
            if self.block.name == "Edit":
                # Use colored diff display
                diff = format_diff_text(
                    self.block.input.get('old_string', ''),
                    self.block.input.get('new_string', '')
                )
                yield Static(diff, id="diff-content")
            else:
                details = format_tool_details(self.block.name, self.block.input)
                yield Markdown(details, id="md-content")

    def collapse(self) -> None:
        """Collapse this widget."""
        try:
            self.query_one(Collapsible).collapsed = True
        except Exception:
            pass

    def get_copyable_content(self) -> str:
        """Get content suitable for copying - preserves exact content."""
        inp = self.block.input
        parts = []
        if self.block.name == "Edit":
            parts.append(f"File: {inp.get('file_path', '?')}")
            if inp.get('old_string'):
                parts.append(f"Old:\n```\n{inp['old_string']}\n```")
            if inp.get('new_string'):
                parts.append(f"New:\n```\n{inp['new_string']}\n```")
        elif self.block.name == "Bash":
            parts.append(f"Command:\n```\n{inp.get('command', '?')}\n```")
        elif self.block.name == "Write":
            parts.append(f"File: {inp.get('file_path', '?')}")
            if inp.get('content'):
                parts.append(f"Content:\n```\n{inp['content']}\n```")
        elif self.block.name == "Read":
            parts.append(f"File: {inp.get('file_path', '?')}")
        else:
            parts.append(json.dumps(inp, indent=2))
        if self.result and self.result.content:
            content = self.result.content if isinstance(self.result.content, str) else str(self.result.content)
            parts.append(f"Result:\n```\n{content}\n```")
        return "\n\n".join(parts)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "tool-copy-btn":
            event.stop()
            import pyperclip
            try:
                pyperclip.copy(self.get_copyable_content())
                self.app.notify("Copied tool output")
            except Exception as e:
                self.app.notify(f"Copy failed: {e}", severity="error")

    def set_result(self, result: ToolResultBlock) -> None:
        """Update with tool result."""
        self.result = result
        log.info(f"Tool result for {self.block.name}: {type(result.content)} - {str(result.content)[:200]}")
        try:
            collapsible = self.query_one(Collapsible)
            if result.is_error:
                collapsible.add_class("error")
            # Edit uses Static for diff, others use Markdown
            if self.block.name == "Edit":
                # For edits, result is usually just success/error - no content to add
                return
            md = collapsible.query_one("#md-content", Markdown)
            details = format_tool_details(self.block.name, self.block.input)
            if result.content:
                content = result.content if isinstance(result.content, str) else str(result.content)
                preview = content[:500] + ('...' if len(content) > 500 else '')
                if result.is_error:
                    details += f"\n\n**Error:**\n```\n{preview}\n```"
                elif self.block.name == "Read":
                    lang = get_lang_from_path(self.block.input.get('file_path', ''))
                    details += f"\n\n```{lang}\n{preview}\n```"
                elif self.block.name in ("Bash", "Grep", "Glob"):
                    details += f"\n\n```\n{preview}\n```"
                else:
                    details += f"\n\n{preview}"
            md.update(details)
        except Exception:
            pass  # Widget may not be mounted yet


class ChatMessage(Static):
    """A single chat message."""

    def __init__(self, role: str, content: str = "") -> None:
        super().__init__()
        self.role = role
        self._content = content

    def compose(self) -> ComposeResult:
        yield Button("⎘", id="copy-btn", classes="copy-btn")
        yield Markdown(self._content, id="content")

    def append_content(self, text: str) -> None:
        self._content += text
        md = self.query_one("#content", Markdown)
        md.update(self._content)

    def get_raw_content(self) -> str:
        """Get content without the role prefix for copying."""
        # Strip the **Claude:**\n\n or **You:** prefix
        content = self._content
        if content.startswith("**Claude:**\n\n"):
            return content[13:]
        if content.startswith("**You:** "):
            return content[9:]
        return content

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "copy-btn":
            import pyperclip
            try:
                pyperclip.copy(self.get_raw_content())
                self.app.notify("Copied to clipboard")
            except Exception as e:
                self.app.notify(f"Copy failed: {e}", severity="error")


class ChatApp(App):
    """Main chat application."""

    CSS_PATH = "styles.tcss"
    BINDINGS = [
        ("ctrl+c", "quit", "Quit"),
        ("ctrl+l", "clear", "Clear"),
        ("ctrl+b", "toggle_sidebar", "Sessions"),
    ]

    RECENT_TOOLS_EXPANDED = 2  # Keep last N tool uses expanded

    def __init__(self, resume_session_id: str | None = None) -> None:
        super().__init__()
        self.options = ClaudeAgentOptions(
            allowed_tools=["Read", "Edit", "Write", "Bash", "Glob", "Grep"],
            permission_mode="acceptEdits",
            env={"ANTHROPIC_API_KEY": ""},  # Use Max subscription, not API key
            setting_sources=["user", "project", "local"],  # Respect normal CC settings/hooks
        )
        self.client: ClaudeSDKClient | None = None
        self.current_response: ChatMessage | None = None
        self.session_id: str | None = None
        self.pending_tools: dict[str, ToolUseWidget] = {}  # tool_use_id -> widget
        self.recent_tools: list[ToolUseWidget] = []  # Track recent for auto-collapse
        self._resume_on_start = resume_session_id  # Session to resume on startup

    def compose(self) -> ComposeResult:
        yield ContextHeader()
        with Horizontal(id="main"):
            yield ListView(id="sidebar", classes="hidden")
            yield VerticalScroll(id="chat-view")
        yield ChatInput(id="input")
        yield Footer()

    async def on_mount(self) -> None:
        self.client = ClaudeSDKClient(self.options)
        await self.client.connect()
        self.query_one("#input", ChatInput).focus()
        if self._resume_on_start:
            self._load_and_display_history(self._resume_on_start)
            self.notify(f"Resuming {self._resume_on_start[:8]}...")
            self.resume_session(self._resume_on_start)
        else:
            self.refresh_context()

    def _load_and_display_history(self, session_id: str) -> None:
        """Load session history and display in chat view."""
        chat_view = self.query_one("#chat-view", VerticalScroll)
        chat_view.remove_children()
        for m in load_session_messages(session_id, limit=50):
            if m["type"] == "user":
                msg = ChatMessage("user", f"**You:** {m['content'][:500]}")
                msg.add_class("user-message")
                chat_view.mount(msg)
            elif m["type"] == "assistant":
                msg = ChatMessage("assistant", f"**Claude:** {m['content'][:1000]}")
                msg.add_class("assistant-message")
                chat_view.mount(msg)
            elif m["type"] == "tool_use":
                block = ToolUseBlock(id=m.get("id", ""), name=m["name"], input=m["input"])
                widget = ToolUseWidget(block, collapsed=True)
                widget.add_class("tool-use")
                chat_view.mount(widget)
        self.call_later(lambda: chat_view.scroll_end(animate=False))

    async def on_unmount(self) -> None:
        # Don't disconnect - causes task boundary errors on Ctrl+C exit
        # Process exit will clean up resources
        pass

    @work(group="context", exclusive=True, exit_on_error=False)
    async def refresh_context(self) -> None:
        """Silently run /context to get current usage."""
        if not self.client:
            return
        await self.client.query("/context")
        async for message in self.client.receive_response():
            if isinstance(message, UserMessage):
                content = getattr(message, 'content', '')
                tokens = parse_context_tokens(content)
                if tokens is not None:
                    self.post_message(ContextUpdate(tokens))

    def on_chat_input_submitted(self, event: ChatInput.Submitted) -> None:
        if not event.text.strip():
            return

        prompt = event.text
        self.query_one("#input", ChatInput).clear()
        chat_view = self.query_one("#chat-view", VerticalScroll)

        # Handle /clear specially - also clear UI
        if prompt.strip() == "/clear":
            chat_view.remove_children()
            self.notify("Conversation cleared")
            self.run_claude(prompt)
            return

        # Handle /resume - reconnect with session ID
        if prompt.strip().startswith("/resume"):
            parts = prompt.strip().split(maxsplit=1)
            session_id = parts[1] if len(parts) > 1 else self.session_id
            if session_id:
                self.resume_session(session_id)
            else:
                self.notify("No session ID to resume", severity="error")
            return

        # Add user message
        user_msg = ChatMessage("user", f"**You:** {prompt}")
        user_msg.add_class("user-message")
        chat_view.mount(user_msg)
        chat_view.scroll_end(animate=False)

        # Reset current response - will be created when first text arrives
        self.current_response = None

        # Start the query
        self.run_claude(prompt)

    @work(group="claude", exclusive=True, exit_on_error=False)
    async def run_claude(self, prompt: str) -> None:
        if not self.client:
            return

        await self.client.query(prompt)
        had_tool_use = False
        async for message in self.client.receive_response():
            log.info(f"Message type: {type(message).__name__}")
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        # After tool use, need fresh message widget
                        if had_tool_use:
                            self.post_message(StreamChunk(block.text, new_message=True))
                            had_tool_use = False
                        else:
                            self.post_message(StreamChunk(block.text))
                    elif isinstance(block, ToolUseBlock):
                        self.post_message(ToolUseMessage(block))
                        had_tool_use = True
                    elif isinstance(block, ToolResultBlock):
                        self.post_message(ToolResultMessage(block))
            elif isinstance(message, UserMessage):
                # Check for /context response
                content = getattr(message, 'content', '')
                if '<local-command-stdout>' in content:
                    tokens = parse_context_tokens(content)
                    if tokens is not None:
                        self.post_message(ContextUpdate(tokens))
            elif isinstance(message, SystemMessage):
                # Handle system messages (compact, etc.)
                subtype = getattr(message, 'subtype', '')
                if subtype == 'compact_boundary':
                    meta = getattr(message, 'compact_metadata', None)
                    if meta:
                        self.call_from_thread(
                            self.notify,
                            f"Compacted: {getattr(meta, 'pre_tokens', '?')} tokens"
                        )
            elif isinstance(message, ResultMessage):
                self.post_message(ResponseComplete(message))

    def on_stream_chunk(self, event: StreamChunk) -> None:
        chat_view = self.query_one("#chat-view", VerticalScroll)
        # Create new message widget if needed (after tool use)
        if event.new_message or not self.current_response:
            self.current_response = ChatMessage("assistant", "**Claude:**\n\n")
            self.current_response.add_class("assistant-message")
            chat_view.mount(self.current_response)
        self.current_response.append_content(event.text)
        chat_view.scroll_end(animate=False)

    def on_tool_use_message(self, event: ToolUseMessage) -> None:
        """Handle a tool use starting."""
        chat_view = self.query_one("#chat-view", VerticalScroll)
        # Collapse older tools beyond the threshold
        while len(self.recent_tools) >= self.RECENT_TOOLS_EXPANDED:
            old = self.recent_tools.pop(0)
            old.collapse()
        widget = ToolUseWidget(event.block, collapsed=False)
        widget.add_class("tool-use")
        self.pending_tools[event.block.id] = widget
        self.recent_tools.append(widget)
        chat_view.mount(widget)
        chat_view.scroll_end(animate=False)

    def on_tool_result_message(self, event: ToolResultMessage) -> None:
        """Handle a tool result arriving."""
        widget = self.pending_tools.get(event.block.tool_use_id)
        if widget:
            widget.set_result(event.block)
            del self.pending_tools[event.block.tool_use_id]

    def on_context_update(self, event: ContextUpdate) -> None:
        """Update context bar from /context command."""
        self.query_one("#context-bar", ContextBar).tokens = event.tokens

    def on_response_complete(self, event: ResponseComplete) -> None:
        if event.result:
            self.session_id = event.result.session_id
            cost = event.result.total_cost_usd or 0
            if cost > 0:
                self.notify(f"Cost: ${cost:.4f}")
            # Update context bar with usage (input tokens = context window usage)
            if event.result.usage:
                u = event.result.usage
                # Context = input + cached input (output doesn't count toward limit)
                total = (u.get("input_tokens", 0) +
                        u.get("cache_creation_input_tokens", 0) +
                        u.get("cache_read_input_tokens", 0))
                self.query_one("#context-bar", ContextBar).tokens = total
        self.current_response = None
        self.query_one("#input", ChatInput).focus()

    @work(group="resume", exclusive=True, exit_on_error=False)
    async def resume_session(self, session_id: str) -> None:
        """Resume a session by creating a new client (abandoning the old one)."""
        log.info(f"resume_session started: {session_id}")
        try:
            # Don't disconnect - just abandon old client to avoid task boundary issues
            self.client = None

            options = ClaudeAgentOptions(
                allowed_tools=["Read", "Edit", "Write", "Bash", "Glob", "Grep"],
                permission_mode="acceptEdits",
                env={"ANTHROPIC_API_KEY": ""},
                setting_sources=["user", "project", "local"],
                resume=session_id,
            )
            log.info("Creating new client")
            client = ClaudeSDKClient(options)
            log.info("Connecting new client")
            await client.connect()
            log.info("Connected!")
            self.client = client
            self.session_id = session_id
            self.post_message(ResponseComplete(None))  # Trigger focus back to input
            self.refresh_context()  # Update context bar
            log.info(f"Resume complete for {session_id}")
        except Exception as e:
            log.exception(f"Resume failed: {e}")
            self.post_message(ResponseComplete(None))

    def action_clear(self) -> None:
        chat_view = self.query_one("#chat-view", VerticalScroll)
        chat_view.remove_children()

    def action_quit(self) -> None:
        import time
        now = time.time()
        if hasattr(self, '_last_quit_time') and now - self._last_quit_time < 1.0:
            self.exit()
        else:
            self._last_quit_time = now
            self.notify("Press Ctrl+C again to quit")

    def action_toggle_sidebar(self) -> None:
        sidebar = self.query_one("#sidebar", ListView)
        if sidebar.has_class("hidden"):
            sidebar.remove_class("hidden")
            sidebar.clear()
            for session_id, preview, _, msg_count in get_recent_sessions():
                sidebar.append(SessionItem(session_id, preview, msg_count))
            sidebar.focus()
        else:
            sidebar.add_class("hidden")
            self.query_one("#input", ChatInput).focus()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        if isinstance(event.item, SessionItem):
            session_id = event.item.session_id
            log.info(f"Resuming session: {session_id}")
            self.query_one("#sidebar", ListView).add_class("hidden")
            self._load_and_display_history(session_id)
            self.notify(f"Resuming {session_id[:8]}...")
            self.resume_session(session_id)

    def on_app_focus(self) -> None:
        self.query_one("#input", ChatInput).focus()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Claude Code Textual UI")
    parser.add_argument("--resume", "-r", action="store_true",
                        help="Resume the most recent session")
    parser.add_argument("--session", "-s", type=str,
                        help="Resume a specific session ID")
    args = parser.parse_args()

    resume_id = None
    if args.session:
        resume_id = args.session
    elif args.resume:
        sessions = get_recent_sessions(limit=1)
        if sessions:
            resume_id = sessions[0][0]

    try:
        app = ChatApp(resume_session_id=resume_id)
        app.run()
    except (KeyboardInterrupt, SystemExit):
        pass
    except Exception as e:
        import traceback
        with open("/tmp/cc-textual-crash.log", "w") as f:
            traceback.print_exc(file=f)
        raise

"""Command handlers for slash commands.

This module extracts command routing from app.py. Commands receive an app
reference and access only what they need.
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from claudechic.app import ChatApp


def handle_command(app: "ChatApp", prompt: str) -> bool:
    """Route slash commands. Returns True if handled, False to send to Claude."""
    cmd = prompt.strip()

    # Handle ! prefix for inline shell commands
    if cmd.startswith("!"):
        return _handle_bang(app, cmd[1:].strip())

    if cmd == "/clear":
        chat_view = app._chat_view
        if chat_view:
            chat_view.clear()
            app.notify("Conversation cleared")
            app._send_to_active_agent(cmd)
        return True

    if cmd.startswith("/resume"):
        return _handle_resume(app, cmd)

    if cmd.startswith("/worktree"):
        from claudechic.features.worktree import handle_worktree_command
        handle_worktree_command(app, cmd)
        return True

    if cmd.startswith("/agent"):
        return _handle_agent(app, cmd)

    if cmd.startswith("/shell"):
        return _handle_shell(app, cmd)

    if cmd == "/theme":
        app.search_themes()
        return True

    if cmd.startswith("/compactish"):
        return _handle_compactish(app, cmd)

    if cmd == "/usage":
        app._handle_usage_command()
        return True

    if cmd == "/exit":
        app.exit()
        return True

    if cmd == "/welcome":
        return _handle_welcome(app)

    return False


def _handle_resume(app: "ChatApp", command: str) -> bool:
    """Handle /resume [session_id] command."""
    parts = command.split(maxsplit=1)
    if len(parts) > 1:
        session_id = parts[1]
        app.run_worker(app._load_and_display_history(session_id))
        app.notify(f"Resuming {session_id[:8]}...")
        app.resume_session(session_id)
    else:
        app._show_session_picker()
    return True


def _handle_agent(app: "ChatApp", command: str) -> bool:
    """Handle /agent commands: list, create, close."""
    from claudechic.widgets import ChatMessage

    parts = command.split(maxsplit=2)

    if len(parts) == 1:
        # In narrow mode, open the sidebar overlay instead of listing
        width = app.size.width
        has_content = len(app.agents) > 1 or app.agent_sidebar._worktrees or app.todo_panel.todos
        if width < app.SIDEBAR_MIN_WIDTH and has_content:
            app._sidebar_overlay_open = True
            app._position_right_sidebar()
            return True

        # List agents as markdown table
        lines = ["| # | Agent | Status | Directory |", "|---|-------|--------|-----------|"]
        for i, (aid, agent) in enumerate(app.agents.items(), 1):
            marker = "▸" if aid == app.active_agent_id else " "
            # Shorten home directory
            path = str(agent.cwd).replace(str(Path.home()), "~")
            lines.append(f"| {marker}{i} | {agent.name} | {agent.status} | {path} |")

        chat_view = app._chat_view
        if chat_view:
            msg = ChatMessage("\n".join(lines))
            msg.add_class("system-message")
            chat_view.mount(msg)
            chat_view.scroll_if_tailing()
        return True

    subcommand = parts[1]
    if subcommand == "close":
        target = parts[2] if len(parts) > 2 else None
        app._close_agent(target)
        return True

    # Create new agent
    name = subcommand
    path = Path(parts[2]) if len(parts) > 2 else Path.cwd()
    app._create_new_agent(name, path)
    return True


def _handle_shell(app: "ChatApp", command: str) -> bool:
    """Run shell command inline, or interactive shell if no command or -i flag."""
    parts = command.split(maxsplit=1)
    cmd = parts[1] if len(parts) > 1 else None

    # Check for -i flag (interactive mode)
    interactive = False
    if cmd and cmd.startswith("-i "):
        interactive = True
        cmd = cmd[3:].lstrip()

    agent = app._agent
    cwd = str(agent.cwd) if agent else None
    env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
    # Force color output, disable pagers for captured output
    env.update({"FORCE_COLOR": "1", "CLICOLOR_FORCE": "1", "TERM": "xterm-256color", "BAT_PAGER": "", "PAGER": ""})
    shell = os.environ.get("SHELL", "/bin/sh")

    if cmd and not interactive:
        # Async execution with captured output
        app.run_shell_command(cmd, shell, cwd, env)
    else:
        # Interactive: suspend TUI and run in real terminal
        with app.suspend():
            args = [shell, "-lc", cmd] if cmd else [shell, "-l"]
            start = time.monotonic()
            subprocess.run(args, cwd=cwd, env=env)
            # If command was fast, wait for keypress so user can see output
            if cmd and time.monotonic() - start < 1.0:
                import sys
                import termios
                import tty
                print("\nPress any key to continue...", end="", flush=True)
                fd = sys.stdin.fileno()
                old = termios.tcgetattr(fd)
                try:
                    tty.setraw(fd)
                    sys.stdin.read(1)
                finally:
                    termios.tcsetattr(fd, termios.TCSADRAIN, old)

    return True


def _handle_bang(app: "ChatApp", command: str) -> bool:
    """Alias for /shell <command>."""
    if not command:
        app.notify("Usage: !<command>")
        return True
    return _handle_shell(app, f"/shell {command}")


def _handle_welcome(app: "ChatApp") -> bool:
    """Send welcome message to Claude to present to user."""
    welcome_prompt = """\
Welcome the user to Claude Chic, the Claude Code app that they're using. Present this information to them exactly:

---

# Welcome to Claude Chic 👋

Claude Chic is a drop-in replacement for the `claude` terminal UI that adds the following:

## Visual Design

Information is organized to be easy to scan and explore.

-  Tool outputs are collapsible
-  Border colors indicates user (orange), assistant (blue), and tools (gray) messages
-  Context and CPU indicators are in the footer
-  Maximum width makes scanning easier on the eyes

**Example**: Resume (`/resume`) a long Claude session to see the differnce side-by-side.

## Multi-Agent

Run multiple Claude agents concurrently in the same UI.

-  Create agents with `/agent <name>` (or ask me to make them).
-  Enable parallel workstreams without lots of terminals
-  Agents able to communicate with each other

**Example:** Ask me to set up a game of chess between two agents.

## Git Worktrees

Create isolated git worktrees for parallel feature development.

-  Launch multiple improvements simultaneously without branch conflicts
-  Use `/worktree branch-nanme` to create a worktree
-  Use `/worktree finish` to merge it back into main with linear history

**Example:** Run `claudechic` in your repo, and ask me to do a thorough review, launching different workstreams for each issue.

## Hackable

Claude Chic is open source and written in Python with Textual.  It's easy to extend with fun tools.

**Example:** Use simple quality of life features like shell support with `!ls`. or `!git diff`.

Enjoy!

---

Repeat this message verbatim.  Help them if they have further questions.
"""

    app._send_to_active_agent(welcome_prompt, display_as="/welcome")
    return True


def _handle_compactish(app: "ChatApp", command: str) -> bool:
    """Handle /compactish command - compact the current session.

    Flags:
        -n, --dry: Show stats without modifying
        -a, --aggressive: Use lower size thresholds
        --no-reconnect: Don't reconnect after compaction
    """
    from claudechic.compact import compact_session, format_compact_summary
    from claudechic.widgets import ChatMessage

    agent = app._agent
    if not agent or not agent.session_id:
        app.notify("No active session to compact", severity="warning")
        return True

    session_id = agent.session_id
    parts = command.split()

    # Parse flags
    dry_run = "--dry" in parts or "-n" in parts
    aggressive = "--aggressive" in parts or "-a" in parts
    reconnect = "--no-reconnect" not in parts

    result = compact_session(session_id, cwd=agent.cwd, aggressive=aggressive, dry_run=dry_run)
    if "error" in result:
        app.notify(f"Error: {result['error']}", severity="error")
        return True

    # Display summary table
    summary_md = format_compact_summary(result, dry_run=dry_run)
    chat_view = app._chat_view
    if chat_view:
        summary_msg = ChatMessage(summary_md)
        summary_msg.add_class("system-message")
        chat_view.mount(summary_msg)
        chat_view.scroll_if_tailing()

    if dry_run:
        app.notify("Dry run - no changes made", timeout=3)
    elif reconnect:
        app.run_worker(app._reconnect_agent(agent, session_id))
        app.notify("Session compacted, reconnecting...", timeout=3)
    else:
        app.notify("Session compacted", timeout=3)

    return True

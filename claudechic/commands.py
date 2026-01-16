"""Command handlers for slash commands.

This module extracts command routing from app.py. Commands receive an app
reference and access only what they need.
"""

from __future__ import annotations

import os
import subprocess
import sys
import termios
import time
import tty
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from claudechic.app import ChatApp


def handle_command(app: "ChatApp", prompt: str) -> bool:
    """Route slash commands. Returns True if handled, False to send to Claude."""
    cmd = prompt.strip()

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
    parts = command.split(maxsplit=2)

    if len(parts) == 1:
        # List agents
        for i, (aid, agent) in enumerate(app.agents.items(), 1):
            marker = "*" if aid == app.active_agent_id else " "
            app.notify(f"{marker}{i}. {agent.name} ({agent.status})")
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
    """Suspend TUI and run shell command."""
    parts = command.split(maxsplit=1)
    if len(parts) < 2:
        app.notify("Usage: /shell <command>")
        return True

    cmd = parts[1]
    agent = app._agent
    cwd = str(agent.cwd) if agent else None

    with app.suspend():
        env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
        shell = os.environ.get("SHELL", "/bin/sh")
        start = time.monotonic()
        subprocess.run([shell, "-lc", cmd], cwd=cwd, env=env)

        # Only prompt if command completed quickly (likely non-interactive)
        if time.monotonic() - start < 1.0:
            print("\nPress any key to continue...", end="", flush=True)
            fd = sys.stdin.fileno()
            old = termios.tcgetattr(fd)
            try:
                tty.setraw(fd)
                sys.stdin.read(1)
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old)
                print()

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
    from claudechic.app import _scroll_if_at_bottom

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
        _scroll_if_at_bottom(chat_view)

    if dry_run:
        app.notify("Dry run - no changes made", timeout=3)
    elif reconnect:
        app.run_worker(app._reconnect_agent(agent, session_id))
        app.notify("Session compacted, reconnecting...", timeout=3)
    else:
        app.notify("Session compacted", timeout=3)

    return True

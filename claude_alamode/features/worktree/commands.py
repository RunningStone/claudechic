"""Worktree command handlers extracted from app.py."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from textual.containers import Center
from textual import work

from claude_alamode.features.worktree.git import (
    FinishInfo,
    get_finish_info,
    get_finish_prompt,
    get_cleanup_fix_prompt,
    finish_cleanup,
    list_worktrees,
    cleanup_worktrees,
    remove_worktree,
    start_worktree,
    needs_rebase,
    fast_forward_merge,
)
from claude_alamode.features.worktree.prompts import WorktreePrompt

if TYPE_CHECKING:
    from claude_alamode.app import ChatApp
    from claude_alamode.agent import AgentSession

# Max retries for worktree cleanup before giving up
MAX_CLEANUP_ATTEMPTS = 3


def handle_worktree_command(app: "ChatApp", command: str) -> None:
    """Handle /worktree commands.

    Args:
        app: The ChatApp instance
        command: Full command string (e.g. "/worktree finish")
    """
    parts = command.split(maxsplit=2)

    if len(parts) == 1:
        _show_worktree_modal(app)
        return

    subcommand = parts[1]
    if subcommand == "finish":
        _handle_finish(app)
    elif subcommand == "cleanup":
        branches = parts[2].split() if len(parts) > 2 else None
        _handle_cleanup(app, branches)
    else:
        _switch_or_create_worktree(app, subcommand)


def _handle_finish(app: "ChatApp") -> None:
    """Handle /worktree finish command."""
    from claude_alamode.widgets import ChatMessage

    agent = app._agent
    if not agent:
        app.notify("No active agent", severity="error")
        return

    success, message, info = get_finish_info(app.sdk_cwd)
    if not success:
        app.notify(message, severity="error")
        return

    # Store pending state on the agent, not the app
    agent.pending_worktree_finish = info
    agent.worktree_cleanup_attempts = 0

    chat_view = app._chat_view
    if chat_view:
        user_msg = ChatMessage("/worktree finish")
        user_msg.add_class("user-message")
        chat_view.mount(user_msg)

    # Check if we can skip Claude entirely (fast-forward possible)
    if not needs_rebase(info):
        app.notify("Fast-forward merge possible, skipping rebase...")
        success, error = fast_forward_merge(info)
        if success:
            # Go directly to cleanup
            attempt_worktree_cleanup(app, agent)
        else:
            # Need Claude to handle the issue
            app._show_thinking()
            app.run_claude(f"Fast-forward merge failed: {error}\n\n" + get_finish_prompt(info))
    else:
        # Need Claude to rebase
        app._show_thinking()
        app.run_claude(get_finish_prompt(info))


def _switch_or_create_worktree(app: "ChatApp", feature_name: str) -> None:
    """Switch to existing worktree agent or create new one."""
    # Check if we already have an agent for this worktree
    for agent in app.agents.values():
        if agent.worktree == feature_name:
            app._switch_to_agent(agent.id)
            app.notify(f"Switched to {feature_name}")
            return

    # Check if worktree exists on disk
    existing = [wt for wt in list_worktrees() if wt.branch == feature_name]
    if existing:
        wt = existing[0]
        app._create_new_agent(feature_name, wt.path, worktree=feature_name, auto_resume=True)
    else:
        # Create new worktree
        success, message, new_cwd = start_worktree(feature_name)
        if success and new_cwd:
            app._create_new_agent(feature_name, new_cwd, worktree=feature_name, auto_resume=True)
        else:
            app.notify(message, severity="error")


def _handle_cleanup(app: "ChatApp", branches: list[str] | None) -> None:
    """Handle /worktree cleanup command."""
    results = cleanup_worktrees(branches)

    if not results:
        app.notify("No worktrees to clean up")
        return

    # Check if any need confirmation
    needs_confirm = [(b, msg) for b, success, msg, confirm in results if confirm]
    removed = [(b, msg) for b, success, msg, confirm in results if success]
    failed = [(b, msg) for b, success, msg, confirm in results if not success and not confirm]

    # Report results
    for branch, msg in removed:
        app.notify(f"Removed: {branch}")
    for branch, msg in failed:
        app.notify(f"Failed: {branch} - {msg}", severity="error")

    # Prompt for confirmation on dirty/unmerged
    if needs_confirm:
        _run_cleanup_prompt(app, needs_confirm)


@work(group="cleanup_prompt", exclusive=True, exit_on_error=False)
async def _run_cleanup_prompt(app: "ChatApp", needs_confirm: list[tuple[str, str]]) -> None:
    """Show prompt for confirming worktree removal."""
    from claude_alamode.widgets import SelectionPrompt, ChatInput

    branches_to_confirm = [b for b, _ in needs_confirm]
    options = [("all", f"Remove all ({len(needs_confirm)})")]
    options.extend((b, f"Remove {b} ({msg})") for b, msg in needs_confirm)
    options.append(("cancel", "Cancel"))

    async with app._show_prompt(SelectionPrompt("Worktrees with changes or unmerged:", options)) as prompt:
        prompt.focus()
        selected = await prompt.wait()

    if selected and selected != "cancel":
        to_remove = branches_to_confirm if selected == "all" else [selected]
        worktrees = list_worktrees()
        for branch in to_remove:
            wt = next((w for w in worktrees if w.branch == branch), None)
            if wt:
                success, msg = remove_worktree(wt, force=True)
                app.notify(f"Removed: {branch}" if success else msg, severity="error" if not success else "information")
    else:
        app.notify("Cleanup cancelled")

    app.query_one("#input", ChatInput).focus()


def attempt_worktree_cleanup(app: "ChatApp", agent: "AgentSession") -> None:
    """Attempt to clean up worktree, asking Claude for help if it fails.

    Called from app.on_response_complete when agent.pending_worktree_finish is set.
    Also called directly from _handle_finish for fast-forward merges.
    """
    info = agent.pending_worktree_finish
    if not info:
        return

    success, message = finish_cleanup(info)
    if not success:
        _handle_cleanup_failure(app, agent, message, info)
        return

    agent.pending_worktree_finish = None
    agent.worktree_cleanup_attempts = 0

    if message:  # Branch deletion warning
        app.notify(f"Cleaned up {info.branch_name}{message}", severity="warning")
    else:
        app.notify(f"Cleaned up {info.branch_name}")

    # Close the worktree agent if it exists
    worktree_agent = next(
        (a for a in app.agents.values() if a.worktree == info.branch_name),
        None
    )
    if worktree_agent and len(app.agents) > 1:
        # Find or create a main repo agent to switch to
        main_agent = next(
            (a for a in app.agents.values() if a.worktree is None),
            None
        )
        if main_agent:
            app._switch_to_agent(main_agent.id)
        app._do_close_agent(worktree_agent.id)


def _handle_cleanup_failure(app: "ChatApp", agent: "AgentSession", error: str, info: FinishInfo) -> None:
    """Handle cleanup failure by asking Claude to fix it or giving up."""
    from claude_alamode.widgets import ChatMessage

    # If the worktree directory no longer exists but we got here, that's strange
    # but we should work from main_dir to avoid cwd issues
    if not info.worktree_dir.exists():
        # Worktree is gone - switch to main and consider it mostly done
        agent.pending_worktree_finish = None
        agent.worktree_cleanup_attempts = 0
        app.notify(f"Worktree removed, but: {error}", severity="warning")
        # Switch agent to main dir if needed
        if agent.cwd == info.worktree_dir:
            agent.cwd = info.main_dir
            app._reconnect_sdk(info.main_dir)
        return

    agent.worktree_cleanup_attempts += 1

    if agent.worktree_cleanup_attempts >= MAX_CLEANUP_ATTEMPTS:
        agent.pending_worktree_finish = None
        agent.worktree_cleanup_attempts = 0
        app.notify(f"Cleanup failed after {MAX_CLEANUP_ATTEMPTS} attempts: {error}", severity="error")
        return

    # Use the agent's chat view, not the app's current one
    chat_view = agent.chat_view
    if chat_view:
        user_msg = ChatMessage(f"[Cleanup attempt {agent.worktree_cleanup_attempts}/{MAX_CLEANUP_ATTEMPTS} failed]\n\n{error}")
        user_msg.add_class("user-message")
        chat_view.mount(user_msg)

    # Ensure we're on the right agent before running Claude
    if app.active_agent_id != agent.id:
        app._switch_to_agent(agent.id)

    app._show_thinking()
    app.run_claude(get_cleanup_fix_prompt(error, info.worktree_dir))


def _show_worktree_modal(app: "ChatApp") -> None:
    """Show worktree selection modal."""
    worktrees = [(str(wt.path), wt.branch) for wt in list_worktrees() if not wt.is_main]
    prompt = WorktreePrompt(worktrees)
    container = Center(prompt, id="worktree-modal")
    app.mount(container)
    _wait_for_worktree_selection(app, prompt, container)


@work(group="worktree", exclusive=True, exit_on_error=False)
async def _wait_for_worktree_selection(app: "ChatApp", prompt: WorktreePrompt, container: Center) -> None:
    """Wait for worktree modal selection and act on it."""
    try:
        result = await prompt.wait()
        container.remove()
        if result is None:
            return  # Cancelled

        action, value = result
        if action == "switch":
            # value is the path; find the branch name from worktrees
            worktrees = {str(wt.path): wt.branch for wt in list_worktrees()}
            branch = worktrees.get(value, Path(value).name)
            _switch_or_create_worktree(app, branch)
        elif action == "new":
            _switch_or_create_worktree(app, value)
    except Exception as e:
        app.show_error("Worktree selection failed", e)

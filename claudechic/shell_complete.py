"""Shell completion utilities - command and path completion."""

from __future__ import annotations

import os
from pathlib import Path

# Cached executables from PATH
_executable_cache: list[str] | None = None


def get_executables() -> list[str]:
    """Get all executable commands from PATH (cached)."""
    global _executable_cache
    if _executable_cache is not None:
        return _executable_cache

    executables = set()
    path_dirs = os.environ.get("PATH", "").split(os.pathsep)

    for dir_path in path_dirs:
        p = Path(dir_path)
        if not p.is_dir():
            continue
        try:
            for entry in p.iterdir():
                if entry.is_file() and os.access(entry, os.X_OK):
                    executables.add(entry.name)
        except (PermissionError, OSError):
            continue

    _executable_cache = sorted(executables)
    return _executable_cache


def complete_command(prefix: str, limit: int = 20) -> list[str]:
    """Complete a command name prefix."""
    prefix_lower = prefix.lower()
    executables = get_executables()

    # Prioritize exact prefix matches, then contains
    exact = [e for e in executables if e.lower().startswith(prefix_lower)]
    return exact[:limit]


def complete_path(partial: str, cwd: Path | None = None, limit: int = 20) -> list[str]:
    """Complete a partial file/directory path."""
    cwd = cwd or Path.cwd()

    if not partial:
        base = cwd
        prefix = ""
        match_part = ""
    elif partial.startswith("/"):
        # Absolute path
        p = Path(partial)
        if p.is_dir() and partial.endswith("/"):
            base = p
            prefix = partial
            match_part = ""
        else:
            base = p.parent if p.parent.exists() else Path("/")
            prefix = str(base).rstrip("/") + "/"
            match_part = p.name
    elif partial.startswith("~"):
        # Home directory
        expanded = Path(partial).expanduser()
        if expanded.is_dir() and partial.endswith("/"):
            base = expanded
            prefix = partial
            match_part = ""
        else:
            base = expanded.parent if expanded.parent.exists() else Path.home()
            # Keep the ~ prefix in output
            prefix = partial.rsplit("/", 1)[0] + "/" if "/" in partial else "~/"
            match_part = expanded.name
    else:
        # Relative path
        p = cwd / partial
        if p.is_dir() and partial.endswith("/"):
            base = p
            prefix = partial
            match_part = ""
        else:
            base = p.parent if p.parent.exists() else cwd
            if "/" in partial:
                prefix = partial.rsplit("/", 1)[0] + "/"
            else:
                prefix = ""
            match_part = p.name

    results = []
    match_lower = match_part.lower()

    try:
        for entry in base.iterdir():
            name = entry.name
            # Skip hidden unless explicitly requested
            if name.startswith(".") and not match_part.startswith("."):
                continue
            if match_part and not name.lower().startswith(match_lower):
                continue
            suffix = "/" if entry.is_dir() else ""
            results.append(prefix + name + suffix)
    except (PermissionError, OSError):
        pass

    return sorted(results)[:limit]


def parse_shell_input(text: str) -> tuple[str, str]:
    """Parse shell input into (command, current_arg).

    Returns:
        (command, current_arg) where current_arg is what's being typed
    """
    # Strip leading ! or "/shell "
    if text.startswith("!"):
        text = text[1:]
    elif text.startswith("/shell "):
        text = text[7:]
    else:
        return "", text

    # Split by whitespace, keeping track of what's being typed
    parts = text.split()

    if not parts:
        return "", ""

    if text.endswith(" "):
        # Space after last word - completing new argument
        return parts[0], ""
    elif len(parts) == 1:
        # Still typing the command
        return "", parts[0]
    else:
        # Typing an argument
        return parts[0], parts[-1]

"""Git worktree management for isolated feature work."""

import subprocess
from dataclasses import dataclass
from pathlib import Path


def get_repo_name() -> str:
    """Get the current repository name."""
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True, text=True, check=True
    )
    return Path(result.stdout.strip()).name


def get_current_branch() -> str:
    """Get the current git branch."""
    result = subprocess.run(
        ["git", "branch", "--show-current"],
        capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def start_worktree(feature_name: str) -> tuple[bool, str, Path | None]:
    """
    Create a worktree for the given feature.

    Returns (success, message, worktree_path).
    """
    try:
        repo_name = get_repo_name()
        base_branch = get_current_branch()

        # Find main worktree to put new worktree next to it
        main_wt = get_main_worktree()
        if main_wt:
            parent_dir = main_wt[0].parent
        else:
            parent_dir = Path.cwd().parent

        worktree_dir = parent_dir / f"{repo_name}-{feature_name}"

        if worktree_dir.exists():
            return False, f"Directory {worktree_dir} already exists", None

        # Create the worktree with a new branch
        subprocess.run(
            ["git", "worktree", "add", "-b", feature_name, str(worktree_dir), "HEAD"],
            check=True, capture_output=True, text=True
        )

        return True, f"Created worktree at {worktree_dir}", worktree_dir

    except subprocess.CalledProcessError as e:
        return False, f"Git error: {e.stderr}", None
    except Exception as e:
        return False, f"Error: {e}", None


def get_main_worktree() -> tuple[Path, str] | None:
    """Find the main worktree (non-feature) path and its branch."""
    worktrees = list_worktrees()
    for wt in worktrees:
        if wt.is_main:
            return wt.path, wt.branch
    return None


def get_finish_worktree_info(cwd: Path | None = None) -> tuple[bool, str, dict | None]:
    """
    Get info needed to finish a worktree (for Claude to execute).

    Args:
        cwd: Current working directory (SDK's cwd). If None, uses Path.cwd().

    Returns (success, message, info_dict or None).
    info_dict contains: branch_name, base_branch, worktree_dir, main_dir
    """
    # Detect current worktree from git
    if cwd is None:
        cwd = Path.cwd()
    cwd = cwd.resolve()  # Normalize for comparison with git output
    worktrees = list_worktrees()
    current_wt = next((wt for wt in worktrees if wt.path.resolve() == cwd), None)

    if current_wt is None or current_wt.is_main:
        return False, "Not in a feature worktree. Switch to a worktree first.", None

    main_wt = get_main_worktree()
    if main_wt is None:
        return False, "Cannot find main worktree.", None

    original_dir, base_branch = main_wt

    return True, "Ready to finish worktree", {
        "branch_name": current_wt.branch,
        "base_branch": base_branch,
        "worktree_dir": str(current_wt.path),
        "main_dir": str(original_dir),
    }


@dataclass
class WorktreeInfo:
    """Info about an existing worktree."""
    path: Path
    branch: str
    is_main: bool


def list_worktrees() -> list[WorktreeInfo]:
    """List all git worktrees for this repo."""
    result = subprocess.run(
        ["git", "worktree", "list", "--porcelain"],
        capture_output=True, text=True, check=True
    )

    worktrees = []
    current_path = None
    current_branch = None

    for line in result.stdout.strip().split("\n"):
        if line.startswith("worktree "):
            current_path = Path(line[9:])
        elif line.startswith("branch refs/heads/"):
            current_branch = line[18:]
        elif line == "":
            if current_path and current_branch:
                # Main worktree is the one without a hyphenated name pattern
                main_repo = get_repo_name()
                is_main = current_path.name == main_repo
                worktrees.append(WorktreeInfo(current_path, current_branch, is_main))
            current_path = None
            current_branch = None

    # Handle last entry if no trailing newline
    if current_path and current_branch:
        main_repo = get_repo_name()
        is_main = current_path.name == main_repo
        worktrees.append(WorktreeInfo(current_path, current_branch, is_main))

    return worktrees

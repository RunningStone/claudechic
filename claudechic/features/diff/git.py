"""Git diff parsing - pure functions for extracting file changes."""

import asyncio
import difflib
import re
from dataclasses import dataclass, field


@dataclass
class Hunk:
    """A single hunk (@@-delimited section) from a diff."""

    old_start: int
    old_count: int
    new_start: int
    new_count: int
    old_lines: list[str]  # Lines from old file (context + removed)
    new_lines: list[str]  # Lines from new file (context + added)


@dataclass
class HunkComment:
    """A comment on a specific hunk."""

    path: str
    hunk: Hunk
    comment: str


def format_hunk_comments(comments: list[HunkComment]) -> str:
    """Format hunk comments as markdown for chat input."""
    parts = []
    for c in comments:
        # Reconstruct unified diff from old/new lines
        diff_lines = list(
            difflib.unified_diff(
                c.hunk.old_lines,
                c.hunk.new_lines,
                lineterm="",
                n=3,
            )
        )[2:]  # Skip --- and +++ headers

        diff_text = "\n".join(diff_lines) if diff_lines else "\n".join(c.hunk.new_lines)
        line_range = f"L{c.hunk.new_start}-{c.hunk.new_start + c.hunk.new_count - 1}"

        parts.append(
            f"## {c.path} ({line_range})\n```diff\n{diff_text}\n```\n{c.comment}"
        )

    return "\n\n".join(parts)


@dataclass
class FileChange:
    """A single file's changes from git diff."""

    path: str
    status: str  # "modified", "added", "deleted", "renamed"
    hunks: list[Hunk] = field(default_factory=list)


async def get_changes(cwd: str) -> list[FileChange]:
    """Get all uncommitted changes (staged + unstaged) via git diff HEAD."""
    # Get list of changed files with status
    proc = await asyncio.create_subprocess_exec(
        "git",
        "diff",
        "HEAD",
        "--name-status",
        cwd=cwd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, _ = await proc.communicate()
    if proc.returncode != 0:
        return []

    files = _parse_name_status(stdout.decode())
    if not files:
        return []

    # Get full diff content for parsing
    # Use --no-ext-diff to ensure we get unified diff format (not difft, delta, etc.)
    proc = await asyncio.create_subprocess_exec(
        "git",
        "diff",
        "HEAD",
        "--no-ext-diff",
        cwd=cwd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, _ = await proc.communicate()
    if proc.returncode != 0:
        return files  # Return files without hunks

    diff_content = stdout.decode()
    return _merge_diff_content(files, diff_content)


def _parse_name_status(output: str) -> list[FileChange]:
    """Parse git diff --name-status output."""
    changes = []
    for line in output.strip().split("\n"):
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue

        status_code = parts[0][0]  # First char (M, A, D, R, etc.)
        path = parts[-1]  # Last part is the (new) path

        status = {
            "M": "modified",
            "A": "added",
            "D": "deleted",
            "R": "renamed",
            "C": "copied",
        }.get(status_code, "modified")

        changes.append(FileChange(path=path, status=status, hunks=[]))

    return changes


def _merge_diff_content(files: list[FileChange], diff_text: str) -> list[FileChange]:
    """Parse unified diff and merge hunks into FileChange objects."""
    # Split by file headers
    file_diffs = re.split(r"^diff --git ", diff_text, flags=re.MULTILINE)

    path_to_diff: dict[str, str] = {}
    for file_diff in file_diffs[1:]:  # Skip empty first split
        # Extract path from "a/path b/path" line
        first_line = file_diff.split("\n")[0]
        match = re.search(r"b/(.+)$", first_line)
        if match:
            path = match.group(1)
            path_to_diff[path] = file_diff

    # Merge hunks into FileChange objects
    result = []
    for fc in files:
        if fc.path in path_to_diff:
            hunks = _parse_hunks(path_to_diff[fc.path])
            result.append(FileChange(path=fc.path, status=fc.status, hunks=hunks))
        else:
            result.append(fc)

    return result


def _parse_hunks(diff_section: str) -> list[Hunk]:
    """Parse a file's diff section into individual hunks."""
    hunks = []
    lines = diff_section.split("\n")

    # Find all hunk headers
    hunk_starts = []
    for i, line in enumerate(lines):
        if line.startswith("@@"):
            hunk_starts.append(i)

    # Parse each hunk
    for idx, start in enumerate(hunk_starts):
        # Determine end of this hunk
        end = hunk_starts[idx + 1] if idx + 1 < len(hunk_starts) else len(lines)

        header = lines[start]
        # Parse @@ -old_start,old_count +new_start,new_count @@
        match = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", header)
        if not match:
            continue

        old_start = int(match.group(1))
        old_count = int(match.group(2) or 1)
        new_start = int(match.group(3))
        new_count = int(match.group(4) or 1)

        old_lines = []
        new_lines = []

        for line in lines[start + 1 : end]:
            if line.startswith("-"):
                old_lines.append(line[1:])
            elif line.startswith("+"):
                new_lines.append(line[1:])
            elif line.startswith(" "):
                # Context line - present in both
                old_lines.append(line[1:])
                new_lines.append(line[1:])
            elif line.startswith("\\"):
                # "\ No newline at end of file" - skip
                continue

        hunks.append(
            Hunk(
                old_start=old_start,
                old_count=old_count,
                new_start=new_start,
                new_count=new_count,
                old_lines=old_lines,
                new_lines=new_lines,
            )
        )

    return hunks

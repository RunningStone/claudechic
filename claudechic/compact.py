"""Session compaction - reduce context by removing old tool uses.

This module modifies session JSONL files in-place. Unlike Claude's built-in
microcompact (which truncates content), we remove entire tool_use/tool_result
pairs that are both old AND large.

The SDK reads the JSONL file on resume, so modifying it directly affects
what Claude sees.
"""

import json
import shutil
from collections import defaultdict
from pathlib import Path

from claudechic.sessions import get_project_sessions_dir


def compact_session(
    session_id: str,
    cwd: Path | None = None,
    keep_last_n: int = 5,  # Keep last N tool results regardless of size
    min_result_size: int = 1000,  # Only truncate results larger than this (bytes)
    min_input_size: int = 2000,  # Only truncate inputs larger than this (bytes)
    aggressive: bool = False,  # If True, use lower thresholds (500/1000)
    dry_run: bool = False,
) -> dict:
    """Compact a session by removing old, large tool_use/tool_result pairs.

    Strategy: Only remove things that are BOTH old AND large.
    - Small tool results (<1KB) kept regardless of age
    - Small tool inputs (<2KB) kept regardless of age
    - Recent items (last N per tool type) kept regardless of size
    """
    # Aggressive mode uses lower thresholds
    if aggressive:
        min_result_size = min(min_result_size, 500)
        min_input_size = min(min_input_size, 1000)

    sessions_dir = get_project_sessions_dir(cwd)
    if not sessions_dir:
        return {"error": "No sessions directory found"}

    session_file = sessions_dir / f"{session_id}.jsonl"
    if not session_file.exists():
        return {"error": f"Session file not found: {session_file}"}

    # Load all messages
    messages = []
    with open(session_file) as f:
        for line in f:
            if line.strip():
                messages.append(json.loads(line))

    # First pass: collect tool_use info and track order
    # tool_id -> (name, input, input_size, msg_idx, block_idx)
    tool_uses: dict = {}
    tool_order: list = []  # [(tool_id, msg_idx), ...] in order

    for msg_idx, m in enumerate(messages):
        if m.get("type") == "assistant":
            for block_idx, block in enumerate(m.get("message", {}).get("content", [])):
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    tool_id = block["id"]
                    inp = block.get("input", {})
                    input_size = len(json.dumps(inp))
                    tool_uses[tool_id] = {
                        "name": block.get("name"),
                        "input": inp,
                        "input_size": input_size,
                        "msg_idx": msg_idx,
                        "block_idx": block_idx,
                    }
                    tool_order.append(tool_id)

    # Second pass: collect tool_result info
    # tool_id -> result_size
    tool_results: dict = {}
    for m in messages:
        if m.get("type") == "user":
            content = m.get("message", {}).get("content", [])
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        tool_id = block.get("tool_use_id")
                        result_content = block.get("content", "")
                        tool_results[tool_id] = len(str(result_content))

    # Decide what to truncate based on size AND recency
    # Keep last N of each tool type
    tool_counts: dict = defaultdict(int)
    recent_tools: set = set()

    for tool_id in reversed(tool_order):
        info = tool_uses.get(tool_id, {})
        name = info.get("name", "unknown")
        if tool_counts[name] < keep_last_n:
            recent_tools.add(tool_id)
            tool_counts[name] += 1

    # Truncate results: old AND large
    truncate_result_ids = set()
    for tool_id, result_size in tool_results.items():
        if tool_id in recent_tools:
            continue  # Keep recent
        if result_size < min_result_size:
            continue  # Keep small
        truncate_result_ids.add(tool_id)

    # Truncate inputs: old AND large
    truncate_input_ids = set()
    for tool_id, info in tool_uses.items():
        if tool_id in recent_tools:
            continue  # Keep recent
        if info["input_size"] < min_input_size:
            continue  # Keep small
        truncate_input_ids.add(tool_id)

    # IDs to remove entirely (both tool_use and tool_result)
    remove_ids = truncate_result_ids | truncate_input_ids

    # Create compacted messages by removing tool_use/tool_result blocks
    compacted_messages = []

    for m in messages:
        msg_type = m.get("type")

        # Handle assistant messages (remove tool_use blocks)
        if msg_type == "assistant":
            content = m.get("message", {}).get("content", [])
            if not isinstance(content, list):
                compacted_messages.append(m)
                continue

            new_content = [
                block for block in content
                if not (isinstance(block, dict) and block.get("type") == "tool_use" and block.get("id") in remove_ids)
            ]

            if len(new_content) != len(content):
                if new_content:  # Only keep message if there's content left
                    new_msg = {**m, "message": {**m["message"], "content": new_content}}
                    compacted_messages.append(new_msg)
                # else: drop the entire message
            else:
                compacted_messages.append(m)

        # Handle user messages (remove tool_result blocks)
        elif msg_type == "user":
            content = m.get("message", {}).get("content", [])
            if not isinstance(content, list):
                compacted_messages.append(m)
                continue

            new_content = [
                block for block in content
                if not (isinstance(block, dict) and block.get("type") == "tool_result" and block.get("tool_use_id") in remove_ids)
            ]

            if len(new_content) != len(content):
                if new_content:  # Only keep message if there's content left
                    new_msg = {**m, "message": {**m["message"], "content": new_content}}
                    # Remove toolUseResult if we removed the tool_result
                    if "toolUseResult" in new_msg:
                        del new_msg["toolUseResult"]
                    compacted_messages.append(new_msg)
                # else: drop the entire message
            else:
                compacted_messages.append(m)

        else:
            compacted_messages.append(m)

    # Calculate before/after token breakdown by category
    def calc_tokens(msgs: list) -> dict[str, int]:
        """Calculate token breakdown for a message list."""
        breakdown: dict[str, float] = defaultdict(float)
        for m in msgs:
            t = m.get("type")
            if t == "assistant":
                for block in m.get("message", {}).get("content", []):
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            breakdown["assistant_text"] += len(block.get("text", "")) / 4
                        elif block.get("type") == "tool_use":
                            breakdown["tool_inputs"] += len(json.dumps(block.get("input", {}))) / 4
            elif t == "user":
                content = m.get("message", {}).get("content", [])
                if isinstance(content, str):
                    breakdown["user_text"] += len(content) / 4
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict):
                            if block.get("type") == "tool_result":
                                breakdown["tool_results"] += len(str(block.get("content", ""))) / 4
                            elif block.get("type") == "text":
                                breakdown["user_text"] += len(block.get("text", "")) / 4
        return {k: int(v) for k, v in breakdown.items()}

    before_breakdown = calc_tokens(messages)
    after_breakdown = calc_tokens(compacted_messages)

    before_total = sum(before_breakdown.values())
    after_total = sum(after_breakdown.values())
    tokens_saved = before_total - after_total

    stats = {
        "truncated_results": len(truncate_result_ids),
        "truncated_inputs": len(truncate_input_ids),
        "tokens_saved": tokens_saved,
        "before_total": before_total,
        "after_total": after_total,
        "before_breakdown": before_breakdown,
        "after_breakdown": after_breakdown,
        "file": str(session_file),
    }

    if dry_run:
        stats["dry_run"] = True
        return stats

    # Write compacted file
    backup_file = session_file.with_suffix(".jsonl.bak")
    shutil.copy(session_file, backup_file)

    with open(session_file, "w") as f:
        for m in compacted_messages:
            f.write(json.dumps(m) + "\n")

    stats["backup"] = str(backup_file)
    return stats


def format_compact_summary(stats: dict, dry_run: bool = False) -> str:
    """Format compaction stats as a markdown summary for display."""
    before = stats.get("before_total", 0)
    after = stats.get("after_total", 0)

    before_bd = stats.get("before_breakdown", {})
    after_bd = stats.get("after_breakdown", {})

    def pct(val: int, total: int) -> str:
        if total == 0:
            return f"{val:,} (0%)"
        return f"{val:,} ({val * 100 // total}%)"

    # Build markdown table
    header = "## Compaction Preview (dry run)" if dry_run else "## Session Compacted"
    lines = [
        header,
        "",
        "| Category | Before | After |",
        "|----------|-------:|------:|",
    ]

    categories = ["tool_results", "tool_inputs", "assistant_text", "user_text"]
    for cat in categories:
        b = before_bd.get(cat, 0)
        a = after_bd.get(cat, 0)
        if b > 0 or a > 0:
            cat_display = cat.replace("_", " ").title()
            lines.append(f"| {cat_display} | {pct(b, before)} | {pct(a, after)} |")

    lines.append(f"| **Total** | **{before:,}** | **{after:,}** |")

    return "\n".join(lines)

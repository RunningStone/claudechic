# Claude Chic

A stylish terminal UI for [Claude Code](https://docs.anthropic.com/en/docs/claude-code), built with [Textual](https://textual.textualize.io/).

## Install

```bash
uv tool install claudechic
```

Requires Claude Code to be logged in (`claude /login`).

## Usage

```bash
claudechic                     # Start new session
claudechic --resume            # Resume most recent session
claudechic -s <session-id>     # Resume specific session
claudechic "your prompt here"  # Start with initial prompt
```

## Development

```bash
git clone https://github.com/mrocklin/claudechic
cd claudechic
uv sync
uv run claudechic
```

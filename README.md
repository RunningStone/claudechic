# Claude à la Mode

A stylish terminal UI for [Claude Code](https://docs.anthropic.com/en/docs/claude-code), built with [Textual](https://textual.textualize.io/).

## Install

```bash
uv tool install claude-claude-alamode
```

Requires Claude Code to be logged in (`claude /login`).

## Usage

```bash
claude-alamode                     # Start new session
claude-alamode --resume            # Resume most recent session
claude-alamode -s <session-id>     # Resume specific session
claude-alamode "your prompt here"  # Start with initial prompt
```

## Development

```bash
git clone https://github.com/mrocklin/claude-claude-alamode
cd claude-claude-alamode
uv sync
uv run claude-alamode
```

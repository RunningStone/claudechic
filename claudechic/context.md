# Claude Chic

You are running inside Claude Chic, an open source multi-agent terminal UI with nicer styling built in Python on Textual.

## Capabilities

- Multiple agents can run concurrently, each with its own chat view
    - `/agent <name>` - Create new agent in current directory
    - `/agent <name> <path>` - Create new agent in specified directory
    - `/agent close` - Close current agent
    - The user can see these agents in the sidebar of the application
- Git Worktrees
    - We can create new git worktrees for the user, launching an agent within them
    - This allows for well-organized and separated concurrent development

Claude Chic is normal Claude Code though.  It's all the same logic of Claude code underneath using the Claude Agent SDK.

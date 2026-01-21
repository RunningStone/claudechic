# Analytics Implementation

## Files

- `analytics.py` - `capture(event, **properties)` async function, direct HTTP to PostHog
- `config.py` - `~/.claude/claudechic.yaml` management (user ID, opt-out flag)

## Adding New Events

1. Call `capture("event_name", prop1=value1, prop2=value2)` from app.py
2. Use `self.run_worker(capture(...))` for fire-and-forget from sync context
3. For shutdown events, use `await capture(...)` to ensure delivery

## Current Events

- `app_started` - in `on_mount()`, includes env context (version, terminal, os)
- `app_closed` - in `_cleanup_and_exit()`, includes duration
- `agent_created` - in `on_agent_created()` observer
- `agent_closed` - in `on_agent_closed()` observer, includes message_count

## Design Decisions

- **No PostHog SDK** - direct HTTP keeps dependencies minimal
- **Context only on app_started** - other events are minimal (just `$session_id` + event-specific props)
- **Opt-out not opt-in** - check `get_analytics_enabled()` before sending
- **Silent failures** - analytics must never crash or slow the app

## Testing

Add debug logging temporarily:
```python
# In analytics.py before the try block
import json
with open("/tmp/posthog_debug.log", "a") as f:
    f.write(json.dumps(payload, indent=2) + "\n---\n")
```

Then restart app via remote: `curl -s -X POST localhost:9999/exit`

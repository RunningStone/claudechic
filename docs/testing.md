# Testing

cc-textual uses Textual's testing framework with event-driven architecture for reliable e2e tests.

## Running Tests

```bash
uv run pytest test_app.py -v
```

## Architecture

The app exposes two async queues for testing:

```python
app.interactions  # PermissionRequest objects (permission prompts)
app.completions   # ResponseComplete events (Claude finished responding)
```

### PermissionRequest

When a permission prompt would appear, a `PermissionRequest` is added to the queue:

```python
request = await asyncio.wait_for(app.interactions.get(), timeout=30)
assert request.tool_name == "Write"
request.respond("allow")  # or "allow_all" or "deny"
```

### ResponseComplete

When Claude finishes a response, a `ResponseComplete` event is added:

```python
await asyncio.wait_for(app.completions.get(), timeout=30)
# Claude has finished responding
```

## Writing Tests

```python
@pytest.mark.asyncio
async def test_example(tmp_path: Path):
    app = ChatApp()
    async with app.run_test(size=(120, 40)) as pilot:
        # Wait for SDK connection
        await wait_for(lambda: app.client is not None, timeout=10)

        # Send a message
        input_widget = app.query_one(ChatInput)
        input_widget.text = "Your prompt here"
        await pilot.press("enter")

        # Wait for permission request
        request = await asyncio.wait_for(app.interactions.get(), timeout=30)
        request.respond("allow")

        # Wait for completion
        await asyncio.wait_for(app.completions.get(), timeout=30)

        # Assert results
        assert some_condition
```

## Utilities

### wait_for

Fast-polling condition waiter for state changes:

```python
async def wait_for(condition, timeout=5, poll=0.01):
    """Wait for a condition to be true."""
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        if condition():
            return True
        await asyncio.sleep(poll)
    raise TimeoutError(f"Condition not met within {timeout}s")

# Usage
await wait_for(lambda: app.auto_approve_edits is True, timeout=2)
```

## Key Principles

1. **No polling loops** - Use queues and `wait_for` instead of `sleep`
2. **Event-driven** - Wait on specific events, not arbitrary delays
3. **Real SDK** - Tests use the actual Claude SDK (requires auth)
4. **Temp files** - Use `tmp_path` fixture for file operations

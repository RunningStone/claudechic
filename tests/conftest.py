"""Shared test fixtures and utilities."""

import asyncio
import time


async def wait_for(condition, timeout=5, poll=0.01):
    """Wait for a condition to be true, with fast polling.

    Args:
        condition: Callable that returns True when ready
        timeout: Maximum seconds to wait
        poll: Poll interval in seconds

    Returns:
        True if condition was met

    Raises:
        TimeoutError: If condition not met within timeout
    """
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        if condition():
            return True
        await asyncio.sleep(poll)
    raise TimeoutError(f"Condition not met within {timeout}s")

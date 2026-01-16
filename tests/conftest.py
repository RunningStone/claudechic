"""Shared test fixtures."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


async def empty_async_gen():
    """Empty async generator for mocking receive_response."""
    return
    yield  # Makes this an async generator


async def wait_for_workers(app):
    """Wait for all workers to complete."""
    await app.workers.wait_for_complete()


@pytest.fixture
def mock_sdk():
    """Patch SDK to not actually connect."""
    mock_client = MagicMock()
    mock_client.connect = AsyncMock()
    mock_client.query = AsyncMock()
    mock_client.interrupt = AsyncMock()
    mock_client.get_server_info = AsyncMock(return_value={"commands": [], "models": []})
    mock_client.receive_response = lambda: empty_async_gen()

    with patch("claudechic.app.ClaudeSDKClient", return_value=mock_client):
        yield mock_client

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


async def submit_command(app, pilot, command: str):
    """Submit a command, handling autocomplete properly.

    When setting input text directly, autocomplete may activate.
    This helper hides it before submitting to ensure the command goes through.
    """
    from claudechic.widgets import ChatInput

    input_widget = app.query_one("#input", ChatInput)
    input_widget.text = command
    await pilot.pause()

    # Hide autocomplete if it's showing (triggered by / or @)
    if input_widget._autocomplete and input_widget._autocomplete.display:
        input_widget._autocomplete.action_hide()
        await pilot.pause()

    input_widget.action_submit()
    await pilot.pause()


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

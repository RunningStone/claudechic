"""Permission request handling for tool approvals."""

import asyncio
from dataclasses import dataclass, field
from typing import Any

from claudechic.formatting import format_tool_header


@dataclass
class PermissionRequest:
    """Represents a pending permission request.

    Used for both UI display and programmatic testing.
    """

    tool_name: str
    tool_input: dict[str, Any]
    _event: asyncio.Event = field(default_factory=asyncio.Event)
    _result: str = "deny"

    @property
    def title(self) -> str:
        """Format permission prompt title."""
        return f"Allow {format_tool_header(self.tool_name, self.tool_input)}?"

    def respond(self, result: str) -> None:
        """Respond to this permission request.

        Args:
            result: One of "allow", "allow_all", or "deny"
        """
        self._result = result
        self._event.set()

    async def wait(self) -> str:
        """Wait for response (from UI or programmatic).

        Returns:
            The response string ("allow", "allow_all", or "deny")
        """
        await self._event.wait()
        return self._result

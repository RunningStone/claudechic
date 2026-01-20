"""Button widget - simple clickable label."""

from textual.message import Message
from textual.widgets import Static

from claudechic.cursor import ClickableMixin


class Button(Static, ClickableMixin):
    """Simple clickable label with hand cursor and hover state.

    For widgets that render text directly. For containers that
    compose children, use ClickableMixin directly on your Widget subclass.

    Emits Button.Pressed on click for parent handlers.
    """

    class Pressed(Message):
        """Posted when button is clicked."""

        def __init__(self, button: "Button") -> None:
            self.button = button
            super().__init__()

    def on_click(self, event) -> None:
        self.post_message(self.Pressed(self))

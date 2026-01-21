"""Copyable mixin for widgets with copy-to-clipboard functionality."""

from __future__ import annotations

from typing import TYPE_CHECKING

from claudechic.widgets.button import Button

if TYPE_CHECKING:
    from textual.app import App


class CopyButton(Button):
    """Copy button with hand cursor on hover."""

    pass


class CopyableMixin:
    """Mixin for widgets that support copying content to clipboard.

    Usage:
        class MyWidget(Static, CopyableMixin):
            def compose(self):
                yield CopyButton("⧉", classes="copy-btn")
                ...

            def get_copyable_content(self) -> str:
                return "content to copy"

            def on_button_pressed(self, event):
                if self.handle_copy_button(event):
                    return
    """

    def get_copyable_content(self) -> str:
        """Return content to copy. Override in subclass."""
        raise NotImplementedError("Subclass must implement get_copyable_content()")

    # Type hint for app - mixin expects to be used with Widget
    app: App

    def handle_copy_button(self, event: Button.Pressed) -> bool:
        """Handle copy button press. Returns True if handled."""
        if "copy-btn" in event.button.classes:
            event.stop()
            try:
                import pyperclip

                pyperclip.copy(self.get_copyable_content())
                self.app.notify("Copied to clipboard")
            except Exception as e:
                self.app.notify(f"Copy failed: {e}", severity="error")
            return True
        return False

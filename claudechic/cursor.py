"""Mouse cursor and hover mixins for Textual widgets.

Provides three mixins for different use cases:
- ClickableMixin: Hand cursor + .hovered class (for buttons)
- PointerMixin: Configurable cursor style (for text areas)
- HoverableMixin: Just .hovered class (for hover effects without cursor)

Uses OSC 22 escape sequences supported by modern terminals
(Ghostty, Kitty, WezTerm, foot). Unsupported terminals ignore the sequence.
"""

import os
from typing import Literal

# CSS cursor names supported by OSC 22
# Ghostty on macOS only supports: default, pointer, text
PointerStyle = Literal[
    "default",
    "pointer",  # hand/finger for clickable
    "text",  # I-beam for selectable text
    "crosshair",
    "move",
    "wait",
    "progress",
    "not-allowed",
    "grab",
    "grabbing",
]

# Cache the tty file descriptor for direct terminal writes
_tty_fd: int | None = None


def _get_tty() -> int | None:
    """Get file descriptor for the controlling terminal."""
    global _tty_fd
    if _tty_fd is None:
        try:
            _tty_fd = os.open("/dev/tty", os.O_WRONLY)
        except OSError:
            _tty_fd = -1  # Mark as unavailable
    return _tty_fd if _tty_fd >= 0 else None


def set_pointer(style: PointerStyle = "default") -> None:
    """Emit OSC 22 to change mouse pointer shape."""
    tty = _get_tty()
    if tty is None:
        return
    try:
        os.write(tty, f"\033]22;{style}\033\\".encode())
    except OSError:
        pass


class PointerMixin:
    """Mixin for widgets that change mouse pointer on hover.

    Default shows hand cursor. Set pointer_style = "text" for I-beam.
    """

    pointer_style: PointerStyle = "pointer"

    def on_enter(self) -> None:
        set_pointer(self.pointer_style)
        if hasattr(super(), "on_enter"):
            super().on_enter()  # type: ignore[misc]

    def on_leave(self) -> None:
        set_pointer("default")
        if hasattr(super(), "on_leave"):
            super().on_leave()  # type: ignore[misc]


class HoverableMixin:
    """Mixin for widgets that need a .hovered class for CSS styling.

    Adds/removes 'hovered' class on mouse enter/leave. Useful when
    CSS :hover doesn't propagate properly through child widgets.
    """

    def on_enter(self) -> None:
        if not self.has_class("hovered"):  # type: ignore[attr-defined]
            self.add_class("hovered")  # type: ignore[attr-defined]

    def on_leave(self) -> None:
        if self.has_class("hovered"):  # type: ignore[attr-defined]
            self.remove_class("hovered")  # type: ignore[attr-defined]
        set_pointer("default")


class ClickableMixin:
    """Mixin for clickable widgets with hand cursor and hover state.

    Combines pointer cursor with .hovered class. Use for buttons and
    clickable containers.

    Example:
        class MyButton(Static, ClickableMixin):
            pass

        class MyContainer(Widget, ClickableMixin):
            def compose(self):
                yield Child()
    """

    def on_enter(self) -> None:
        set_pointer("pointer")
        if not self.has_class("hovered"):  # type: ignore[attr-defined]
            self.add_class("hovered")  # type: ignore[attr-defined]

    def on_leave(self) -> None:
        set_pointer("default")
        if self.has_class("hovered"):  # type: ignore[attr-defined]
            self.remove_class("hovered")  # type: ignore[attr-defined]

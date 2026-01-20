"""OSC 22 mouse pointer cursor control.

Modern terminals (Ghostty, Kitty, WezTerm, foot) support OSC 22 to change
the mouse pointer shape. This provides visual feedback for clickable elements.

Unsupported terminals simply ignore the escape sequence.

Usage:
    from claudechic.cursor import PointerMixin

    class MyButton(Widget, PointerMixin):
        pass  # shows hand cursor on hover

    class MyText(Widget, PointerMixin):
        pointer_style = "text"  # shows I-beam cursor
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


class ContainerPointerMixin(PointerMixin):
    """PointerMixin for widgets with children that also change cursor.

    Uses on_mouse_move to maintain cursor when moving between children.
    """

    def on_mouse_move(self) -> None:
        set_pointer(self.pointer_style)


class HoverableMixin:
    """Mixin for widgets that need a .hovered class for CSS styling.

    Adds/removes 'hovered' class on mouse enter/leave/move. Useful when
    CSS :hover doesn't propagate properly through child widgets.
    """

    def on_enter(self) -> None:
        self.add_class("hovered")  # type: ignore[attr-defined]

    def on_leave(self) -> None:
        self.remove_class("hovered")  # type: ignore[attr-defined]
        set_pointer("default")

    def on_mouse_move(self) -> None:
        self.add_class("hovered")  # type: ignore[attr-defined]

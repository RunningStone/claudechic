"""Base classes and mixins for widgets."""

from claudechic.widgets.base.cursor import (
    ClickableMixin,
    HoverableMixin,
    PointerMixin,
    set_pointer,
)
from claudechic.widgets.base.copyable import CopyButton, CopyableMixin

__all__ = [
    "ClickableMixin",
    "HoverableMixin",
    "PointerMixin",
    "set_pointer",
    "CopyButton",
    "CopyableMixin",
]

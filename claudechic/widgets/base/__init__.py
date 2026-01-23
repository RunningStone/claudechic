"""Base classes and mixins for widgets."""

from claudechic.widgets.base.cursor import (
    ClickableMixin,
    PointerMixin,
    set_pointer,
)
from claudechic.widgets.base.clickable import ClickableLabel
from claudechic.widgets.base.tool_protocol import ToolWidget
from claudechic.widgets.base.tool_base import BaseToolWidget

__all__ = [
    "ClickableMixin",
    "PointerMixin",
    "set_pointer",
    "ClickableLabel",
    "ToolWidget",
    "BaseToolWidget",
]

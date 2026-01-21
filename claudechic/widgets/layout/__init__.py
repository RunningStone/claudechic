"""Layout widgets - chat view, sidebar, footer."""

from claudechic.widgets.layout.chat_view import ChatView
from claudechic.widgets.layout.sidebar import (
    AgentItem,
    AgentSidebar,
    WorktreeItem,
    PlanButton,
    HamburgerButton,
)
from claudechic.widgets.base.clickable import ClickableLabel
from claudechic.widgets.layout.footer import (
    AutoEditLabel,
    ModelLabel,
    StatusFooter,
)
from claudechic.widgets.layout.indicators import (
    IndicatorWidget,
    CPUBar,
    ContextBar,
    ProcessIndicator,
)
from claudechic.widgets.layout.processes import (
    ProcessPanel,
    ProcessItem,
)
from claudechic.processes import BackgroundProcess

__all__ = [
    "ChatView",
    "AgentItem",
    "AgentSidebar",
    "WorktreeItem",
    "PlanButton",
    "HamburgerButton",
    "ClickableLabel",
    "AutoEditLabel",
    "ModelLabel",
    "StatusFooter",
    "IndicatorWidget",
    "CPUBar",
    "ContextBar",
    "ProcessIndicator",
    "ProcessPanel",
    "ProcessItem",
    "BackgroundProcess",
]

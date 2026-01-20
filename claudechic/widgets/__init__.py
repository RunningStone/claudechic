"""Textual widgets for Claude Code UI."""

from claudechic.cursor import ClickableMixin
from claudechic.widgets.button import Button
from claudechic.widgets.indicators import CPUBar, ContextBar
from claudechic.widgets.chat import (
    ChatMessage,
    ChatInput,
    ThinkingIndicator,
    ImageAttachments,
    ErrorMessage,
    SystemInfo,
    ChatAttachment,
    Spinner,
)
from claudechic.widgets.tools import (
    ToolUseWidget,
    TaskWidget,
    AgentToolWidget,
    AgentListWidget,
    ShellOutputWidget,
    EditPlanRequested,
)
from claudechic.widgets.diff import DiffWidget
from claudechic.widgets.todo import TodoWidget, TodoPanel
from claudechic.widgets.prompts import (
    BasePrompt,
    SelectionPrompt,
    QuestionPrompt,
    SessionItem,
)
from claudechic.widgets.model_prompt import ModelPrompt
from claudechic.widgets.autocomplete import TextAreaAutoComplete
from claudechic.widgets.agents import (
    AgentItem,
    AgentSidebar,
    WorktreeItem,
    PlanButton,
    HamburgerButton,
)
from claudechic.widgets.scroll import AutoHideScroll
from claudechic.widgets.chat_view import ChatView
from claudechic.widgets.collapsible import QuietCollapsible
from claudechic.widgets.history_search import HistorySearch
from claudechic.widgets.usage import UsageReport
from claudechic.widgets.profile_modal import ProfileModal

__all__ = [
    "Button",
    "ClickableMixin",
    "CPUBar",
    "ContextBar",
    "ChatMessage",
    "ChatInput",
    "ChatAttachment",
    "Spinner",
    "ThinkingIndicator",
    "ImageAttachments",
    "ErrorMessage",
    "SystemInfo",
    "ToolUseWidget",
    "TaskWidget",
    "AgentToolWidget",
    "AgentListWidget",
    "ShellOutputWidget",
    "EditPlanRequested",
    "DiffWidget",
    "TodoWidget",
    "TodoPanel",
    "BasePrompt",
    "SelectionPrompt",
    "QuestionPrompt",
    "SessionItem",
    "TextAreaAutoComplete",
    "AgentItem",
    "AgentSidebar",
    "WorktreeItem",
    "PlanButton",
    "HamburgerButton",
    "AutoHideScroll",
    "ChatView",
    "QuietCollapsible",
    "HistorySearch",
    "UsageReport",
    "ProfileModal",
    "ModelPrompt",
]

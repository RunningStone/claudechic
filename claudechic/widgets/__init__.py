"""Textual widgets for Claude Code UI.

Re-exports all widgets from submodules for backward compatibility.
"""

# Base classes and mixins
from claudechic.widgets.base import (
    ClickableMixin,
    HoverableMixin,
    PointerMixin,
    CopyButton,
    CopyableMixin,
)

# Primitives
from claudechic.widgets.primitives import Button, QuietCollapsible, AutoHideScroll

# Content widgets
from claudechic.widgets.content import (
    ChatMessage,
    ChatInput,
    ThinkingIndicator,
    ImageAttachments,
    ErrorMessage,
    SystemInfo,
    ChatAttachment,
    Spinner,
    ToolUseWidget,
    TaskWidget,
    AgentToolWidget,
    AgentListWidget,
    ShellOutputWidget,
    EditPlanRequested,
    DiffWidget,
    TodoWidget,
    TodoPanel,
)

# Input widgets
from claudechic.widgets.input import TextAreaAutoComplete, HistorySearch

# Layout widgets
from claudechic.widgets.layout import (
    ChatView,
    AgentItem,
    AgentSidebar,
    WorktreeItem,
    PlanButton,
    HamburgerButton,
    ClickableLabel,
    AutoEditLabel,
    ModelLabel,
    StatusFooter,
    IndicatorWidget,
    CPUBar,
    ContextBar,
    ProcessIndicator,
    ProcessPanel,
    ProcessItem,
    BackgroundProcess,
)

# Report widgets
from claudechic.widgets.reports import UsageReport, ProfileModal

# Prompts (still in top-level prompts.py)
from claudechic.widgets.prompts import (
    BasePrompt,
    SelectionPrompt,
    QuestionPrompt,
    SessionItem,
    ModelPrompt,
    WorktreePrompt,
    UncommittedChangesPrompt,
)

__all__ = [
    # Base
    "ClickableMixin",
    "HoverableMixin",
    "PointerMixin",
    "CopyButton",
    "CopyableMixin",
    # Primitives
    "Button",
    "QuietCollapsible",
    "AutoHideScroll",
    # Content
    "ChatMessage",
    "ChatInput",
    "ThinkingIndicator",
    "ImageAttachments",
    "ErrorMessage",
    "SystemInfo",
    "ChatAttachment",
    "Spinner",
    "ToolUseWidget",
    "TaskWidget",
    "AgentToolWidget",
    "AgentListWidget",
    "ShellOutputWidget",
    "EditPlanRequested",
    "DiffWidget",
    "TodoWidget",
    "TodoPanel",
    # Input
    "TextAreaAutoComplete",
    "HistorySearch",
    # Layout
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
    # Reports
    "UsageReport",
    "ProfileModal",
    # Prompts
    "BasePrompt",
    "SelectionPrompt",
    "QuestionPrompt",
    "SessionItem",
    "ModelPrompt",
    "WorktreePrompt",
    "UncommittedChangesPrompt",
]

"""Textual widgets for Claude Code UI."""

from claudechic.widgets.indicators import CPUBar, ContextBar
from claudechic.widgets.chat import ChatMessage, ChatInput, ThinkingIndicator, ImageAttachments, ErrorMessage, SystemInfo, ChatAttachment, Spinner
from claudechic.widgets.tools import ToolUseWidget, TaskWidget, AgentToolWidget
from claudechic.widgets.diff import DiffWidget
from claudechic.widgets.todo import TodoWidget, TodoPanel
from claudechic.widgets.prompts import BasePrompt, SelectionPrompt, QuestionPrompt, SessionItem
from claudechic.widgets.autocomplete import TextAreaAutoComplete
from claudechic.widgets.agents import AgentItem, AgentSidebar, WorktreeItem
from claudechic.widgets.scroll import AutoHideScroll
from claudechic.widgets.chat_view import ChatView
from claudechic.widgets.history_search import HistorySearch
from claudechic.widgets.usage import UsageReport

__all__ = [
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
    "AutoHideScroll",
    "ChatView",
    "HistorySearch",
    "UsageReport",
]

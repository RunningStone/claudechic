"""Auto-hiding scrollbar container."""

from textual.containers import VerticalScroll


class AutoHideScroll(VerticalScroll):
    """VerticalScroll with always-visible scrollbar.

    Previously auto-hid after inactivity, but layout shifts caused rendering issues.
    """

    DEFAULT_CSS = """
    AutoHideScroll {
        scrollbar-size-vertical: 1;
    }
    """

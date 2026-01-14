"""Auto-hiding scrollbar container."""

from textual.containers import VerticalScroll


class AutoHideScroll(VerticalScroll):
    """VerticalScroll that hides scrollbar after inactivity."""

    HIDE_DELAY = 2.0  # seconds

    DEFAULT_CSS = """
    AutoHideScroll {
        scrollbar-size-vertical: 0;
    }
    AutoHideScroll.scrollbar-visible {
        scrollbar-size-vertical: 1;
    }
    """

    def _on_scroll(self) -> None:
        """Show scrollbar briefly on scroll."""
        self.add_class("scrollbar-visible")
        self.set_timer(self.HIDE_DELAY, lambda: self.remove_class("scrollbar-visible"), name="hide_scrollbar")

    on_mouse_scroll_down = on_mouse_scroll_up = lambda self, event: self._on_scroll()

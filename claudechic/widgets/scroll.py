"""Auto-hiding scrollbar container."""

from textual.containers import VerticalScroll
from textual.scrollbar import ScrollTo


class AutoHideScroll(VerticalScroll):
    """VerticalScroll with always-visible scrollbar and smart tailing.

    Tracks whether user is at bottom to enable/disable auto-scroll on new content.

    Tailing mode is disabled when user scrolls up, and re-enabled when scroll
    position reaches the bottom. Uses watch_scroll_y to detect position changes
    after deferred scrolls complete.
    """

    DEFAULT_CSS = """
    AutoHideScroll {
        scrollbar-size-vertical: 1;
    }
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._tailing = True  # Whether to auto-scroll on new content
        self._user_scrolling_up = False  # Track if user initiated upward scroll

    def _is_near_bottom(self) -> bool:
        """Check if scroll position is near the bottom."""
        return self.scroll_y >= self.max_scroll_y - 50

    def watch_scroll_y(self, old_value: float, new_value: float) -> None:
        """React to scroll position changes after they complete."""
        super().watch_scroll_y(old_value, new_value)
        if self._user_scrolling_up:
            # User scrolled up - disable tailing
            self._tailing = False
            self._user_scrolling_up = False
        elif not self._tailing and self._is_near_bottom():
            # User scrolled to bottom - re-enable tailing
            self._tailing = True

    def action_scroll_up(self) -> None:
        """User scrolled up via keyboard."""
        self._user_scrolling_up = True
        super().action_scroll_up()

    def action_page_up(self) -> None:
        """User paged up via keyboard."""
        self._user_scrolling_up = True
        super().action_page_up()

    def _on_mouse_scroll_up(self, event) -> None:
        """User scrolled up via mouse wheel."""
        self._user_scrolling_up = True
        super()._on_mouse_scroll_up(event)

    def _on_scroll_to(self, message: ScrollTo) -> None:
        """User dragged scrollbar."""
        if message.y is not None and message.y < self.scroll_y:
            self._user_scrolling_up = True
        super()._on_scroll_to(message)

    def scroll_if_tailing(self) -> None:
        """Scroll to end if in tailing mode."""
        if self._tailing:
            self.scroll_end(animate=False)

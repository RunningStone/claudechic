"""Animated spinner widget."""

from textual.widgets import Static

from claudechic.profiling import profile


class Spinner(Static):
    """Animated spinner - all instances share a single timer for efficiency."""

    FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    DEFAULT_CSS = """
    Spinner {
        width: 1;
        height: 1;
        color: $text-muted;
    }
    """

    # Class-level shared state
    _instances: set["Spinner"] = set()
    _frame: int = 0
    _timer = None

    def __init__(self, text: str = "") -> None:
        self._text = f" {text}" if text else ""
        super().__init__()

    def render(self) -> str:
        """Return current frame from shared counter."""
        return f"{self.FRAMES[Spinner._frame]}{self._text}"

    def on_mount(self) -> None:
        Spinner._instances.add(self)
        # Start shared timer if this is the first spinner
        # Use app.set_interval so timer survives widget unmount
        if Spinner._timer is None:
            Spinner._timer = self.app.set_interval(1 / 10, Spinner._tick_all)  # 10 FPS

    def on_unmount(self) -> None:
        Spinner._instances.discard(self)
        # Stop timer if no spinners left
        if not Spinner._instances and Spinner._timer is not None:
            Spinner._timer.stop()
            Spinner._timer = None

    @staticmethod
    @profile
    def _tick_all() -> None:
        """Advance frame and refresh all spinners.

        Note: We don't check visibility - refresh() on hidden widgets is cheap,
        and the DOM-walking visibility check was more expensive than the savings.
        """
        Spinner._frame = (Spinner._frame + 1) % len(Spinner.FRAMES)
        for spinner in list(Spinner._instances):
            spinner.refresh(layout=False)

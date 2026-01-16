"""Resource indicator widgets - context bar and CPU monitor."""

import psutil

from textual.app import RenderResult
from textual.reactive import reactive
from textual.widget import Widget
from rich.text import Text

from claudechic.formatting import MAX_CONTEXT_TOKENS
from claudechic.profiling import profile


class CPUBar(Widget):
    """Display CPU usage."""

    cpu_pct = reactive(0.0)

    def on_mount(self) -> None:
        self._process = psutil.Process()
        self._process.cpu_percent()  # Prime the measurement
        self.set_interval(2.0, self._update_cpu)

    @profile
    def _update_cpu(self) -> None:
        try:
            self.cpu_pct = self._process.cpu_percent()
        except Exception:
            pass  # Process may have exited

    def render(self) -> RenderResult:
        pct = min(self.cpu_pct / 100.0, 1.0)
        if pct < 0.3:
            color = "dim"
        elif pct < 0.7:
            color = "yellow"
        else:
            color = "red"
        return Text.assemble(("CPU ", "dim"), (f"{self.cpu_pct:3.0f}%", color))


class ContextBar(Widget):
    """Display context usage as a progress bar."""

    tokens = reactive(0)
    max_tokens = reactive(MAX_CONTEXT_TOKENS)

    def render(self) -> RenderResult:
        pct = min(self.tokens / self.max_tokens, 1.0) if self.max_tokens else 0
        bar_width = 10
        filled = int(pct * bar_width)
        bar = "\u2588" * filled + "\u2591" * (bar_width - filled)
        # Dim when low, yellow when moderate, red when high
        if pct < 0.5:
            color = "dim"
        elif pct < 0.8:
            color = "yellow"
        else:
            color = "red"
        return Text.assemble((bar, color), (f" {pct*100:.0f}%", color))

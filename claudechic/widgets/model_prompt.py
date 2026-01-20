"""Model selection prompt."""

from textual.app import ComposeResult
from textual.widgets import Static

from claudechic.widgets.prompts import BasePrompt


class ModelPrompt(BasePrompt):
    """Prompt for selecting a model from SDK-provided list."""

    def __init__(self, models: list[dict], current_value: str | None = None) -> None:
        """Create model prompt.

        Args:
            models: List of model dicts from SDK with 'value', 'displayName', 'description'
            current_value: Currently selected model value (e.g., 'opus', 'sonnet')
        """
        super().__init__()
        self.models = models
        self.current_value = current_value
        # Find current model index for initial selection
        self.selected_idx = 0
        for i, m in enumerate(models):
            if m.get("value") == current_value:
                self.selected_idx = i
                break

    def compose(self) -> ComposeResult:
        yield Static("Select Model", classes="prompt-title")
        for i, m in enumerate(self.models):
            value = m.get("value", "")
            # Extract short name from description like "Opus 4.5 · ..."
            desc = m.get("description", "")
            name = (
                desc.split("·")[0].strip()
                if "·" in desc
                else m.get("displayName", value)
            )
            current = " *" if value == self.current_value else ""
            classes = "prompt-option"
            if i == self.selected_idx:
                classes += " selected"
            yield Static(f"{i + 1}. {name}{current}", classes=classes, id=f"opt-{i}")

    def _total_options(self) -> int:
        return len(self.models)

    def _select_option(self, idx: int) -> None:
        value = self.models[idx].get("value", "")
        self._resolve(value)

    async def wait(self) -> str | None:
        """Wait for selection. Returns model value or None if cancelled."""
        await super().wait()
        return self._result_value

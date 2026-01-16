"""Theme definition for Claude Chic."""

from textual.theme import Theme

# Custom theme for Claude Chic
CHIC_THEME = Theme(
    name="chic",
    primary="#cc7700",
    secondary="#5599dd",  # Sky blue for syntax highlighting
    accent="#445566",
    background="black",
    surface="#111111",
    panel="#555555",  # Used for borders and subtle UI elements
    success="#5599dd",  # Same as secondary - strings in code
    warning="#ffaa33",  # Bright orange - numbers in code
    error="#ff6666",  # Red - errors
    dark=True,
)


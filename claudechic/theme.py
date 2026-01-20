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
    warning="#aaaa00",  # Yellow - moderate usage/caution
    error="#cc3333",  # Red - high usage/errors
    dark=True,
)

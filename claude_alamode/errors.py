"""Error handling and logging infrastructure.

Provides centralized exception handling to make errors visible instead of
silently swallowed. All errors are logged to file; optionally displayed in UI.
"""

import logging
import traceback
from pathlib import Path

# Log to file in user's home directory
LOG_FILE = Path.home() / "claude-alamode.log"

# Configure module logger
log = logging.getLogger("claude_alamode")


def setup_logging(level: int = logging.DEBUG) -> None:
    """Initialize logging to file. Call once at app startup."""
    handler = logging.FileHandler(LOG_FILE, mode="a")
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    log.addHandler(handler)
    log.setLevel(level)


def log_exception(e: Exception, context: str = "") -> str:
    """Log an exception with context. Returns formatted message for display."""
    tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
    if context:
        log.error(f"{context}: {e}\n{tb}")
        return f"{context}: {e}"
    else:
        log.error(f"{e}\n{tb}")
        return str(e)

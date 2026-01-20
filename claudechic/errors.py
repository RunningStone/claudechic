"""Error handling and logging infrastructure.

Provides centralized exception handling to make errors visible instead of
silently swallowed. All errors are logged to file; optionally displayed in UI.
"""

import logging
import os
import traceback
from pathlib import Path

# Only log to file in development mode (set CLAUDECHIC_DEBUG=1)
DEBUG_MODE = os.environ.get("CLAUDECHIC_DEBUG", "").lower() in ("1", "true", "yes")
LOG_FILE = Path.home() / "claudechic.log"

# Configure module logger
log = logging.getLogger("claudechic")


def setup_logging(level: int = logging.DEBUG) -> None:
    """Initialize logging to file. Call once at app startup.

    Configures the root 'claudechic' logger so all child loggers
    (claudechic.app, claudechic.widgets.*, etc.) inherit the handler.

    Only logs to file if CLAUDECHIC_DEBUG env var is set.
    """
    if not DEBUG_MODE:
        return

    handler = logging.FileHandler(LOG_FILE, mode="a")
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    log.addHandler(handler)
    log.setLevel(level)
    log.propagate = False  # Avoid duplicates if root logger is configured
    log.info("Logging initialized")


def log_exception(e: Exception, context: str = "") -> str:
    """Log an exception with context. Returns formatted message for display."""
    tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
    if context:
        log.error(f"{context}: {e}\n{tb}")
        return f"{context}: {e}"
    else:
        log.error(f"{e}\n{tb}")
        return str(e)

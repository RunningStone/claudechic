"""Configuration management for claudechic via ~/.claude/claudechic.yaml."""

import uuid
from pathlib import Path

import yaml

CONFIG_PATH = Path.home() / ".claude" / "claudechic.yaml"

_config: dict = {}
_loaded: bool = False


def _load_config() -> dict:
    """Load config from disk, creating with defaults if missing."""
    global _config, _loaded
    if _loaded:
        return _config

    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            _config = yaml.safe_load(f) or {}
    else:
        _config = {}

    # Ensure analytics section with defaults
    if "analytics" not in _config:
        _config["analytics"] = {}
    if "enabled" not in _config["analytics"]:
        _config["analytics"]["enabled"] = True
    if "id" not in _config["analytics"]:
        _config["analytics"]["id"] = str(uuid.uuid4())
        _save_config()

    _loaded = True
    return _config


def _save_config() -> None:
    """Write config to disk."""
    if not _config:
        return
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(_config, f, default_flow_style=False)


def get_analytics_enabled() -> bool:
    """Check if analytics collection is enabled."""
    return _load_config()["analytics"]["enabled"]


def get_analytics_id() -> str:
    """Get the anonymous analytics ID, generating if needed."""
    return _load_config()["analytics"]["id"]

"""Claude Chic - A stylish terminal UI for PansCode."""


def __getattr__(name: str):
    """Lazy imports to avoid requiring textual at module level."""
    if name == "ChatApp":
        from claudechic.app import ChatApp
        return ChatApp
    if name == "CHIC_THEME":
        from claudechic.theme import CHIC_THEME
        return CHIC_THEME
    if name in ("AgentManagerObserver", "AgentObserver", "PermissionHandler"):
        from claudechic import protocols
        return getattr(protocols, name)
    if name == "__version__":
        try:
            from importlib.metadata import version
            return version("claudechic")
        except Exception:
            return "0.0.0"
    raise AttributeError(f"module 'claudechic' has no attribute {name!r}")


__all__ = [
    "ChatApp",
    "CHIC_THEME",
    "AgentManagerObserver",
    "AgentObserver",
    "PermissionHandler",
]

"""Single import point for the MagicMind bindings."""

from __future__ import annotations

from typing import Any

__all__ = ["magicmind"]

_MODULE: Any = None


def magicmind() -> Any:
    """Return ``magicmind.python.runtime``, importing it on first use."""

    global _MODULE
    if _MODULE is None:
        import magicmind.python.runtime as runtime

        _MODULE = runtime
    return _MODULE

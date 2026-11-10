"""Cambricon MagicMind graph backend for TensorPlay.

Registering through the ``tensorplay_compiler_backends`` entry-point group
makes the backend discoverable by name: after installing this package and
the MagicMind SDK, ``tensorplay.compile(model, backend="magicmind")`` lowers
the captured region through MagicMind and runs it on the MLU.
"""

from .backend import backend
from .settings import CompilationSettings

__all__ = ["backend", "CompilationSettings"]

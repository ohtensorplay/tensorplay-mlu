"""The ``magicmind`` backend, selected by ``tensorplay.compile(backend="magicmind")``.

The captured region is lowered through Cambricon MagicMind into an MLU
model: the graph is exported to ONNX, the MagicMind front end parses it into
a network, and the builder produces a serialized model that runs with MLU
buffers.  A region that fails to build stays on its interpreter instead of
failing the whole compile, unless ``pass_through_build_failures`` is set -- a
build failure is a coverage gap, not a user error.  Only inference regions
lower here; training regions are wrapped ahead-of-time by the compiler
frontend (forward through this backend, backward as traced).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from .runtime import Runner, build_plan
from .settings import CompilationSettings

__all__ = ["backend"]

log = logging.getLogger(__name__)


def _backend(
    graph_module: Any,
    example_inputs: list[Any],
    **kwargs: Any,
) -> Callable[..., Any]:
    try:
        import magicmind.python.runtime  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "The magicmind backend requires the Cambricon MagicMind SDK. "
            "Install the toolkit that ships the `magicmind` package, then retry."
        ) from exc

    settings = CompilationSettings.from_kwargs(kwargs)

    try:
        plan = build_plan(graph_module, example_inputs, settings)
    except Exception:
        if settings.pass_through_build_failures:
            raise
        log.warning(
            "magicmind model build failed on this region; "
            "running the region uncompiled",
            exc_info=True,
        )
        return graph_module

    return Runner(plan, settings)


try:
    from tensorplay.compiler import BackendCapabilities, declares_capabilities

    # Training regions never reach a backend directly: the frontend routes
    # them through ahead-of-time autograd before lookup, so no training
    # capability needs declaring.
    backend = declares_capabilities(
        BackendCapabilities(
            optional_deps=("magicmind",),
        )
    )(_backend)
except ImportError:
    # A core without the capability contract cannot read the declaration;
    # the backend stays importable and the contract gap surfaces when the
    # compiler resolves backends by name.
    backend = _backend

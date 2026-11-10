"""Compilation settings for the MagicMind backend.

Only what an MLU model build actually consumes: arithmetic precision, the
MLU architecture the binary targets, and where serialized models live.
Anything left unset is decided by the builder.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Mapping

__all__ = ["CompilationSettings", "default_cache_dir"]

#: Precision names the builder understands, keyed by the short spelling this
#: package accepts.  The values are the strings a build configuration carries.
_PRECISION_MODES = {
    "fp32": "force_float32",
    "force_float32": "force_float32",
    "fp16": "force_float16",
    "force_float16": "force_float16",
}


def default_cache_dir() -> str:
    return os.environ.get(
        "TENSORPLAY_MLU_MAGICMIND_CACHE",
        os.path.join(os.path.expanduser("~"), ".cache", "tensorplay-magicmind"),
    )


@dataclass(frozen=True)
class CompilationSettings:
    """Immutable knobs for one model build."""

    #: ``"fp32"``/``"force_float32"`` or ``"fp16"``/``"force_float16"``.
    #: Half precision lets the builder pick half-precision tactics; regions
    #: whose accuracy would suffer keep running uncompiled instead.
    precision: str = "fp32"
    #: MLU architectures the built binary targets, for example
    #: ``("mtp_372",)``.  ``None`` leaves the choice to the builder.
    mlu_archs: tuple[str, ...] | None = None
    #: Directory for serialized models.  ``None`` disables the disk cache.
    cache_dir: str | None = field(default_factory=default_cache_dir)
    #: Raise on build failure instead of running the region uncompiled.
    pass_through_build_failures: bool = False

    def __post_init__(self) -> None:
        mode = _PRECISION_MODES.get(str(self.precision))
        if mode is None:
            raise ValueError(
                f"unsupported precision {self.precision!r}; expected one of "
                f"{sorted(_PRECISION_MODES)}"
            )
        if self.mlu_archs is not None:
            archs = tuple(str(a) for a in self.mlu_archs)
            if not archs:
                raise ValueError("mlu_archs must be a non-empty sequence or None")
            object.__setattr__(self, "mlu_archs", archs)

    @property
    def precision_mode(self) -> str:
        """The build-configuration spelling of :attr:`precision`."""

        return _PRECISION_MODES[str(self.precision)]

    @classmethod
    def from_kwargs(cls, kwargs: Mapping[str, Any]) -> "CompilationSettings":
        """Build settings from ``compile(..., backend="magicmind", options=...)``.

        Accepts an ``options`` mapping or flat keyword spelling; the two may
        not carry the same key, and an unknown option is rejected so a typo
        never silently changes nothing.  Keywords the compiler frontend adds
        on its own (``mode``, ``name``, ``dynamic``, ``strict_native``) are
        accepted and ignored here.
        """

        option_keys = {
            "precision",
            "mlu_archs",
            "cache_dir",
            "pass_through_build_failures",
        }
        frontend_keys = {"options", "mode", "name", "dynamic", "strict_native"}
        unknown = (set(kwargs) - option_keys - frontend_keys) | (
            set(kwargs.get("options") or ()) - option_keys
        )
        if unknown:
            raise TypeError(
                f"magicmind backend received unsupported option(s): {sorted(unknown)}"
            )
        options = dict(kwargs.get("options") or {})
        payload: dict[str, Any] = {}
        for name in option_keys:
            in_options = name in options
            in_kwargs = name in kwargs
            if in_options and in_kwargs:
                raise TypeError(f"magicmind backend option {name!r} given twice")
            if in_options:
                payload[name] = options.pop(name)
            elif in_kwargs:
                payload[name] = kwargs[name]
        return cls(**payload)


#: Historical spelling kept as an alias.
Settings = CompilationSettings

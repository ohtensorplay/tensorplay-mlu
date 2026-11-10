"""Cambricon MLU device backend for TensorPlay.

The backend fills TensorPlay's out-of-tree device slot: memory comes from
CNRT through a caching allocator, kernels run on CNNL, and every operator
without an MLU kernel runs on the host through the generic fallback.  The
device is named ``"mlu"``::

    import tensorplay as tp
    import tensorplay_mlu  # or let the tensorplay.backends entry point load it

    x = tp.randn(64, 64, device="mlu")
    y = (x @ x).relu().sum()

The native part is compiled against the installed Neuware toolkit the first
time it is loaded (``$NEUWARE_HOME``, default ``/usr/local/neuware``) and
cached under ``~/.cache/tensorplay_extensions``.
"""

from __future__ import annotations

import ctypes
import glob
import os
import threading

__all__ = ["is_loaded", "load", "neuware_home"]

_HERE = os.path.dirname(os.path.abspath(__file__))
_CSRC = os.path.join(_HERE, "csrc")
_LOCK = threading.Lock()
_LIB: ctypes.CDLL | None = None


def neuware_home() -> str:
    """Root of the Neuware toolkit the backend builds against."""
    return os.environ.get("NEUWARE_HOME") or "/usr/local/neuware"


def is_loaded() -> bool:
    return _LIB is not None


def _library() -> ctypes.CDLL:
    if _LIB is None:
        raise RuntimeError("the MLU backend is not loaded; call tensorplay_mlu.load()")
    return _LIB


def _build(verbose: bool) -> ctypes.CDLL:
    from tensorplay.utils import cpp_extension

    home = neuware_home()
    if not os.path.isfile(os.path.join(home, "include", "cnnl.h")):
        raise RuntimeError(f"Neuware toolkit not found under {home} (set NEUWARE_HOME)")
    libdir = os.path.join(home, "lib64")
    return cpp_extension.load(
        "tensorplay_mlu",
        sorted(glob.glob(os.path.join(_CSRC, "*.cpp"))),
        extra_cflags=["-O3", "-DNDEBUG", "-Wno-deprecated-declarations"],
        extra_ldflags=[f"-L{libdir}", "-lcnnl", "-lcnrt", f"-Wl,-rpath,{libdir}"],
        extra_include_paths=[_CSRC, os.path.join(home, "include")],
        verbose=verbose,
    )


def load(verbose: bool = False) -> None:
    """Build (once) and load the native backend, then name the device."""
    global _LIB
    with _LOCK:
        if _LIB is not None:
            return
        import tensorplay as tp

        lib = _build(verbose)
        lib.tp_mlu_init.restype = ctypes.c_int
        if lib.tp_mlu_init() != 0:
            raise RuntimeError("registering the MLU host fallbacks failed")
        lib.tp_mlu_device_count.restype = ctypes.c_int
        lib.tp_mlu_memory_stats.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_int64)]
        lib.tp_mlu_reset_peak_memory_stats.argtypes = [ctypes.c_int]
        _LIB = lib

        tp.utils.rename_privateuse1_backend("mlu")
        tp.utils.generate_methods_for_privateuse1_backend()
        from . import runtime

        tp._register_device_module("mlu", runtime)


def _autoload() -> None:
    """``tensorplay.backends`` entry point: load when the toolkit is present."""
    if os.path.isfile(os.path.join(neuware_home(), "include", "cnnl.h")):
        load()

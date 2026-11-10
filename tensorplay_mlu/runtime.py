"""``tensorplay.mlu``: device management and memory statistics."""

from __future__ import annotations

import contextlib
import ctypes

import tensorplay as tp

from . import _library

__all__ = [
    "current_device", "device", "device_count", "empty_cache", "is_available",
    "is_initialized", "max_memory_allocated", "max_memory_reserved", "mem_get_info",
    "memory_allocated", "memory_reserved", "memory_stats", "reset_peak_memory_stats",
    "set_device", "synchronize",
]


def is_available() -> bool:
    return tp._C._privateuse1_is_available()


def is_initialized() -> bool:
    return True


def device_count() -> int:
    return tp._C._privateuse1_device_count()


def current_device() -> int:
    return tp._C._privateuse1_current_device()


def _index(device=None) -> int:
    if device is None:
        return current_device()
    if isinstance(device, int):
        return device
    index = tp.device(device).index
    return current_device() if index is None or index < 0 else index


def set_device(device) -> None:
    tp._C._privateuse1_set_device(_index(device))


@contextlib.contextmanager
def device(index):
    """Make ``index`` the current MLU device inside the block."""
    previous = current_device()
    set_device(index)
    try:
        yield
    finally:
        set_device(previous)


def synchronize(device=None) -> None:
    tp._C._privateuse1_synchronize(_index(device))


def memory_stats(device=None) -> dict[str, int]:
    out = (ctypes.c_int64 * 4)()
    _library().tp_mlu_memory_stats(_index(device), out)
    return {
        "allocated_bytes": out[0],
        "reserved_bytes": out[1],
        "peak_allocated_bytes": out[2],
        "peak_reserved_bytes": out[3],
    }


def memory_allocated(device=None) -> int:
    return memory_stats(device)["allocated_bytes"]


def memory_reserved(device=None) -> int:
    return memory_stats(device)["reserved_bytes"]


def max_memory_allocated(device=None) -> int:
    return memory_stats(device)["peak_allocated_bytes"]


def max_memory_reserved(device=None) -> int:
    return memory_stats(device)["peak_reserved_bytes"]


def reset_peak_memory_stats(device=None) -> None:
    _library().tp_mlu_reset_peak_memory_stats(_index(device))


def empty_cache() -> None:
    """Return cached, unused device memory to the driver."""
    _library().tp_mlu_empty_cache()


def mem_get_info(device=None) -> tuple[int, int]:
    """(free, total) bytes on the device, as the driver reports them."""
    return tuple(tp._C._accelerator_getMemoryInfo(_index(device)))

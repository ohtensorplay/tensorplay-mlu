"""Recording a fixed sequence of device work and replaying it in one call.

A decode step issues the same operators on the same shapes every token, so
the cost of issuing them is overhead that a recording can remove: the device
records the work once and runs the whole sequence again with a single
invocation.

    graph = Graph()
    logits = graph.capture(lambda: model(token, position_ids=positions, past_key_values=cache))
    token.copy_(next_token)
    graph.replay()
    ... logits now holds the result of the replayed step ...

The values a capture returns are held by the graph, because the recorded work
reads and writes those buffers on every replay.  A body may only issue device
work: anything the host does while recording is not recorded, and anything
that reads a value back would replay the value it read rather than the value
the buffers hold.
"""

from __future__ import annotations

import ctypes

from . import _library, runtime

__all__ = ["Graph"]


def _check(status: int) -> None:
    if status == 0:
        return
    message = _library().tp_mlu_last_error()
    text = message.decode("utf-8", "replace") if message else ""
    raise RuntimeError(text or "recording the sequence failed")


class Graph:
    """A recorded sequence of device work that replays in one call."""

    def __init__(self) -> None:
        self._outputs = None
        self._ready = False

    @property
    def ready(self) -> bool:
        return bool(_library().tp_mlu_capture_ready())

    def capture(self, body):
        """Record what one call to ``body`` queues, and return its value."""
        lib = _library()
        index = runtime.current_device()
        runtime.synchronize()
        _check(lib.tp_mlu_capture_begin(index))
        try:
            outputs = body()
        except BaseException:
            self._abandon(index)
            raise
        try:
            _check(lib.tp_mlu_capture_end(index))
        except BaseException:
            self._abandon(index)
            raise
        self._outputs = outputs
        self._ready = True
        return outputs

    def replay(self) -> None:
        """Queue another run of the recording on the current device."""
        if not self._ready:
            raise RuntimeError("nothing has been recorded yet")
        _check(_library().tp_mlu_capture_invoke(runtime.current_device()))

    def reset(self) -> None:
        """Drop the recording and give back the blocks it held."""
        _check(_library().tp_mlu_capture_reset())
        self._outputs = None
        self._ready = False

    def stats(self) -> tuple[int, int]:
        """Blocks handed out while recording, and blocks the recording holds."""
        out = (ctypes.c_int64 * 2)()
        _library().tp_mlu_capture_stats(out)
        return int(out[0]), int(out[1])

    def _abandon(self, index: int) -> None:
        try:
            _library().tp_mlu_capture_end(index)
        except Exception:  # noqa: BLE001 - the original failure is the one to raise
            pass
        _check(_library().tp_mlu_capture_reset())

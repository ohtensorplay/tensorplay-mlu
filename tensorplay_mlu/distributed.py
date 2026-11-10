"""CNCL process group for MLU tensors.

The ``cncl`` distributed backend is implemented here: a process group that
marshals MLU tensors as device pointers into the native entry points of
``tensorplay_mlu`` and submits them to the CNCL library.  It plugs into
``tensorplay.distributed`` through the device-backend registry, so from the
user's point of view it is one more backend name::

    import tensorplay.distributed as dist
    import tensorplay_mlu  # registers the "cncl" backend

    dist.init_process_group("cncl")
    dist.all_reduce(x)

Only the entry points the collectives above need are wrapped; a group whose
process count is larger than the number of ranks that can reach the same
clique is the user's responsibility, exactly as with the CUDA backend.
"""

from __future__ import annotations

import ctypes
import functools
from typing import Any, List, Optional

from . import _library

__all__ = ["is_available"]


# ---------------------------------------------------------------------------
# Native entry points
# ---------------------------------------------------------------------------
# Element-type codes and reduction codes share the convention of
# ``csrc/Cncl.cpp``; keeping them here avoids spelling the CNCL enums out in
# two places.
_DTYPE_CODES = {
    "uint8": 0,
    "int8": 1,
    "int16": 2,
    "int32": 3,
    "int64": 4,
    "float16": 5,
    "float32": 6,
    "bfloat16": 7,
}

# tensorplay.distributed.ReduceOp value -> (CNCL reduction code, post-scale).
# CNCL has no mean reduction, so AVG is a SUM whose result is scaled by the
# group size afterwards (``None`` marks that).
_SUM_CODE, _PROD_CODE, _MIN_CODE, _MAX_CODE = 0, 1, 3, 2
_REDUCTIONS = {
    0: (_SUM_CODE, 1.0),    # SUM
    1: (_SUM_CODE, None),   # AVG
    2: (_PROD_CODE, 1.0),   # PRODUCT
    3: (_MIN_CODE, 1.0),    # MIN
    4: (_MAX_CODE, 1.0),    # MAX
}


@functools.lru_cache(maxsize=1)
def _bindings():
    lib = _library()
    ptr = ctypes.c_void_p
    size = ctypes.c_size_t

    def declare(name, argtypes, restype=ctypes.c_int):
        fn = getattr(lib, name)
        fn.argtypes = argtypes
        fn.restype = restype
        return fn

    declare("tp_mlu_cncl_clique_id_bytes", [])
    declare("tp_mlu_cncl_get_clique_id", [ptr])
    declare("tp_mlu_cncl_comm_init", [ctypes.c_int, ctypes.c_int, ctypes.c_int, ptr,
                                      ctypes.POINTER(ptr)])
    declare("tp_mlu_cncl_comm_destroy", [ptr])
    declare("tp_mlu_cncl_comm_abort", [ptr])
    declare("tp_mlu_cncl_comm_rank", [ptr, ctypes.POINTER(ctypes.c_int)])
    declare("tp_mlu_cncl_comm_count", [ptr, ctypes.POINTER(ctypes.c_int)])
    declare("tp_mlu_cncl_all_reduce", [ptr, ptr, size, ctypes.c_int, ctypes.c_int,
                                       ctypes.c_int, ptr])
    declare("tp_mlu_cncl_broadcast", [ptr, ptr, size, ctypes.c_int, ctypes.c_int,
                                      ctypes.c_int, ptr])
    declare("tp_mlu_cncl_reduce", [ptr, ptr, size, ctypes.c_int, ctypes.c_int,
                                   ctypes.c_int, ctypes.c_int, ptr])
    declare("tp_mlu_cncl_all_gather", [ptr, ptr, size, ctypes.c_int, ctypes.c_int, ptr])
    declare("tp_mlu_cncl_reduce_scatter", [ptr, ptr, size, ctypes.c_int, ctypes.c_int,
                                           ctypes.c_int, ptr])
    declare("tp_mlu_cncl_all_to_all", [ptr, ptr, size, ctypes.c_int, ctypes.c_int, ptr])
    declare("tp_mlu_cncl_send", [ptr, size, ctypes.c_int, ctypes.c_int, ctypes.c_int, ptr])
    declare("tp_mlu_cncl_recv", [ptr, size, ctypes.c_int, ctypes.c_int, ctypes.c_int, ptr])
    declare("tp_mlu_cncl_group_start", [])
    declare("tp_mlu_cncl_group_end", [])
    declare("tp_mlu_cncl_synchronize", [ctypes.c_int])
    declare("tp_mlu_cncl_queue_idle", [ctypes.c_int])
    declare("tp_mlu_cncl_last_error", [], ctypes.c_char_p)
    return lib


def _check(code: int) -> None:
    if code >= 0:
        return
    message = _bindings().tp_mlu_cncl_last_error()
    raise RuntimeError(message.decode() if message else "CNCL call failed")


def _handle(tensor) -> ctypes.c_void_p:
    return ctypes.c_void_p(int(tensor.data_ptr()))


def _dtype_code(tensor) -> int:
    name = str(tensor.dtype).rsplit(".", 1)[-1]
    try:
        return _DTYPE_CODES[name]
    except KeyError as error:
        raise TypeError(f"CNCL does not support {name} tensors") from error


def _flat(tensor):
    """A contiguous 1-D view of ``tensor``."""
    if tensor.dim() == 1 and tensor.is_contiguous():
        return tensor
    return tensor.contiguous().reshape(-1)


class _CnclWork:
    """Handle for a collective that the device queue has not drained yet."""

    def __init__(self, device: int, done=None) -> None:
        self._device = device
        self._done = done
        self._finished = False

    def _finish(self) -> None:
        if not self._finished:
            self._finished = True
            if self._done is not None:
                self._done()

    def is_completed(self) -> bool:
        if self._finished:
            return True
        completed = _bindings().tp_mlu_cncl_queue_idle(self._device) == 1
        if completed:
            self._finish()
        return completed

    def wait(self, timeout=None) -> bool:
        _check(_bindings().tp_mlu_cncl_synchronize(self._device))
        self._finish()
        return True

    def abort(self) -> None:
        raise RuntimeError("abort is unavailable for CNCL work")


class _CnclProcessGroup:
    """Collective provider for one ``cncl`` process group."""

    def __init__(self, pg) -> None:
        self._pg = pg
        self._comm = None
        self._device = None
        self._timeout_s = pg._timeout_s
        self._barrier_flag = None

    # -- lifecycle ---------------------------------------------------------
    def ensure(self) -> None:
        """Create the communicator, rendezvousing through the process store."""
        if self._comm is not None:
            return
        from tensorplay.distributed import distributed_core as dc

        from .runtime import current_device

        pg = self._pg
        lib = _bindings()
        rank = pg.group_rank(dc._global_rank())
        size = pg.size()
        device = current_device()
        nbytes = lib.tp_mlu_cncl_clique_id_bytes()
        store = dc._current_store()
        key = f"tensorplay_distributed/cncl_clique_id/{pg.group_name}"
        if rank == 0:
            buffer = ctypes.create_string_buffer(nbytes)
            _check(lib.tp_mlu_cncl_get_clique_id(buffer))
            store.set(key, buffer.raw.hex())
        else:
            raw = bytes.fromhex(store.get(key, timeout=self._timeout_s).decode())
            buffer = ctypes.create_string_buffer(raw, nbytes)

        comm = ctypes.c_void_p()
        _check(lib.tp_mlu_cncl_comm_init(rank, size, device, buffer,
                                         ctypes.byref(comm)))
        self._comm = comm
        self._device = device

    def destroy(self) -> None:
        if self._comm is None:
            return
        _check(_bindings().tp_mlu_cncl_synchronize(self._device))
        _check(_bindings().tp_mlu_cncl_comm_destroy(self._comm))
        self._comm = None

    def set_timeout(self, timeout_ms: int) -> None:
        self._timeout_s = timeout_ms / 1000.0

    def _work(self) -> _CnclWork:
        return _CnclWork(self._device)

    def _barrier_tensor(self):
        import tensorplay as tp
        if self._barrier_flag is None:
            self._barrier_flag = tp.zeros(1, dtype=tp.float32,
                                          device=f"mlu:{self._device}")
        return self._barrier_flag

    # -- collective primitives --------------------------------------------
    def _reduction(self, op: int):
        """(CNCL reduction code, post-scale) for a ReduceOp value."""
        try:
            code, scale = _REDUCTIONS[int(op)]
        except KeyError:
            from tensorplay.distributed import distributed_core as dc
            raise RuntimeError(f"CNCL does not support reduction {dc.ReduceOp(op)}")
        if scale is None:
            scale = 1.0 / self._pg.size()
        return code, scale

    def allreduce(self, tensors: List[Any], op: int, timeout_ms: int):
        self.ensure()
        lib = _bindings()
        tensor = tensors[0]
        code, scale = self._reduction(int(op))
        count = int(tensor.numel())
        if count:
            _check(lib.tp_mlu_cncl_all_reduce(_handle(tensor), _handle(tensor),
                                              count, _dtype_code(tensor), code,
                                              self._device, self._comm))
            if scale != 1.0:
                tensor.mul_(scale)
        return self._work()

    def allreduce_coalesced(self, tensors: List[Any], op: int, timeout_ms: int):
        self.ensure()
        lib = _bindings()
        code, scale = self._reduction(int(op))
        _check(lib.tp_mlu_cncl_group_start())
        for tensor in tensors:
            count = int(tensor.numel())
            if not count:
                continue
            _check(lib.tp_mlu_cncl_all_reduce(_handle(tensor), _handle(tensor),
                                              count, _dtype_code(tensor), code,
                                              self._device, self._comm))
        _check(lib.tp_mlu_cncl_group_end())
        if scale != 1.0:
            for tensor in tensors:
                tensor.mul_(scale)
        return self._work()

    def broadcast(self, tensors: List[Any], root: int, root_tensor: int,
                  timeout_ms: int):
        self.ensure()
        tensor = tensors[0]
        count = int(tensor.numel())
        if count:
            _check(_bindings().tp_mlu_cncl_broadcast(
                _handle(tensor), _handle(tensor), count, _dtype_code(tensor),
                root, self._device, self._comm))
        return self._work()

    def reduce(self, tensors: List[Any], root: int, op: int, root_tensor: int,
               timeout_ms: int):
        self.ensure()
        tensor = tensors[0]
        code, scale = self._reduction(int(op))
        count = int(tensor.numel())
        if count:
            _check(_bindings().tp_mlu_cncl_reduce(
                _handle(tensor), _handle(tensor), count, _dtype_code(tensor),
                code, root, self._device, self._comm))
            if scale != 1.0 and self._pg.rank() == root:
                tensor.mul_(scale)
        return self._work()

    def _into_buffer(self, chunks):
        """A dense buffer holding ``chunks`` end to end."""
        import tensorplay as tp
        total = sum(int(chunk.numel()) for chunk in chunks)
        buffer = tp.empty((total,), dtype=chunks[0].dtype, device=chunks[0].device)
        offset = 0
        for chunk in chunks:
            n = int(chunk.numel())
            if n:
                buffer.narrow(0, offset, n).copy_(_flat(chunk))
            offset += n
        return buffer

    def allgather(self, output_lists, input_tensors: List[Any], timeout_ms: int):
        self.ensure()
        outputs = output_lists[0]
        tensor = input_tensors[0]
        count = int(tensor.numel())
        if count:
            flat = self._into_buffer(outputs)
            _check(_bindings().tp_mlu_cncl_all_gather(
                _handle(tensor), _handle(flat), count, _dtype_code(tensor),
                self._device, self._comm))
            offset = 0
            for output in outputs:
                n = int(output.numel())
                if n:
                    output.copy_(flat.narrow(0, offset, n).reshape(output.shape))
                offset += n
        return self._work()

    def all_gather_single(self, output, input_tensor, timeout_ms: int):
        self.ensure()
        count = int(input_tensor.numel())
        if count:
            _check(_bindings().tp_mlu_cncl_all_gather(
                _handle(input_tensor), _handle(output), count,
                _dtype_code(input_tensor), self._device, self._comm))
        return self._work()

    def all_gather_single_coalesced(self, outputs: List[Any],
                                    inputs: List[Any], timeout_ms: int):
        self.ensure()
        _check(_bindings().tp_mlu_cncl_group_start())
        for output, input_tensor in zip(outputs, inputs):
            count = int(input_tensor.numel())
            if not count:
                continue
            _check(_bindings().tp_mlu_cncl_all_gather(
                _handle(input_tensor), _handle(output), count,
                _dtype_code(input_tensor), self._device, self._comm))
        _check(_bindings().tp_mlu_cncl_group_end())
        return self._work()

    def allgather_coalesced(self, output_lists, input_tensors: List[Any],
                            timeout_ms: int):
        self.ensure()
        _check(_bindings().tp_mlu_cncl_group_start())
        for index, tensor in enumerate(input_tensors):
            count = int(tensor.numel())
            if not count:
                continue
            chunks = [rank_outputs[index] for rank_outputs in output_lists]
            flat = self._into_buffer(chunks)
            _check(_bindings().tp_mlu_cncl_all_gather(
                _handle(tensor), _handle(flat), count, _dtype_code(tensor),
                self._device, self._comm))
            offset = 0
            for chunk in chunks:
                n = int(chunk.numel())
                if n:
                    chunk.copy_(flat.narrow(0, offset, n).reshape(chunk.shape))
                offset += n
        _check(_bindings().tp_mlu_cncl_group_end())
        return self._work()

    def reduce_scatter(self, recv_tensors: List[Any], input_lists, op: int,
                       timeout_ms: int):
        self.ensure()
        output = recv_tensors[0]
        code, scale = self._reduction(int(op))
        count = int(output.numel())
        if count:
            send = self._into_buffer(input_lists[0])
            _check(_bindings().tp_mlu_cncl_reduce_scatter(
                _handle(send), _handle(output), count, _dtype_code(output),
                code, self._device, self._comm))
            if scale != 1.0:
                output.mul_(scale)
        return self._work()

    def reduce_scatter_single(self, output, input_tensor, op: int, timeout_ms: int):
        self.ensure()
        code, scale = self._reduction(int(op))
        count = int(output.numel())
        if count:
            _check(_bindings().tp_mlu_cncl_reduce_scatter(
                _handle(input_tensor), _handle(output), count,
                _dtype_code(output), code, self._device, self._comm))
            if scale != 1.0:
                output.mul_(scale)
        return self._work()

    def reduce_scatter_single_coalesced(self, outputs: List[Any],
                                        inputs: List[Any], op: int, timeout_ms: int):
        self.ensure()
        code, scale = self._reduction(int(op))
        _check(_bindings().tp_mlu_cncl_group_start())
        for output, input_tensor in zip(outputs, inputs):
            count = int(output.numel())
            if not count:
                continue
            _check(_bindings().tp_mlu_cncl_reduce_scatter(
                _handle(input_tensor), _handle(output), count,
                _dtype_code(output), code, self._device, self._comm))
        _check(_bindings().tp_mlu_cncl_group_end())
        if scale != 1.0:
            for output in outputs:
                output.mul_(scale)
        return self._work()

    # -- gather / scatter built on the point-to-point primitives -----------
    def gather(self, output_lists, input_tensors: List[Any], root: int,
               timeout_ms: int):
        self.ensure()
        lib = _bindings()
        rank = self._pg.rank()
        tensor = input_tensors[0]
        count = int(tensor.numel())
        dtype = _dtype_code(tensor)
        _check(lib.tp_mlu_cncl_group_start())
        if rank == root:
            for peer in range(self._pg.size()):
                if peer == root:
                    output_lists[0][peer].copy_(_flat(tensor))
                elif count:
                    target = _flat(output_lists[0][peer])
                    _check(lib.tp_mlu_cncl_recv(_handle(target), count, dtype, peer,
                                                self._device, self._comm))
        elif count:
            _check(lib.tp_mlu_cncl_send(_handle(tensor), count, dtype, root,
                                       self._device, self._comm))
        _check(lib.tp_mlu_cncl_group_end())
        return self._work()

    def gather_single(self, output, input_tensor, root: int, timeout_ms: int):
        self.ensure()
        lib = _bindings()
        rank = self._pg.rank()
        count = int(input_tensor.numel())
        dtype = _dtype_code(input_tensor)
        _check(lib.tp_mlu_cncl_group_start())
        if rank == root:
            for peer in range(self._pg.size()):
                if peer == root:
                    output.narrow(0, peer * count, count).copy_(_flat(input_tensor))
                elif count:
                    _check(lib.tp_mlu_cncl_recv(
                        _handle(output.narrow(0, peer * count, count)), count,
                        dtype, peer, self._device, self._comm))
        elif count:
            _check(lib.tp_mlu_cncl_send(_handle(input_tensor), count, dtype, root,
                                       self._device, self._comm))
        _check(lib.tp_mlu_cncl_group_end())
        return self._work()

    def scatter(self, recv_tensors: List[Any], input_lists, root: int,
                timeout_ms: int):
        self.ensure()
        lib = _bindings()
        rank = self._pg.rank()
        output = recv_tensors[0]
        count = int(output.numel())
        dtype = _dtype_code(output)
        _check(lib.tp_mlu_cncl_group_start())
        if rank == root:
            for peer in range(self._pg.size()):
                if peer == root:
                    output.copy_(_flat(input_lists[0][peer]))
                elif count:
                    _check(lib.tp_mlu_cncl_send(_handle(input_lists[0][peer]), count,
                                               dtype, peer, self._device, self._comm))
        elif count:
            _check(lib.tp_mlu_cncl_recv(_handle(output), count, dtype, root,
                                       self._device, self._comm))
        _check(lib.tp_mlu_cncl_group_end())
        return self._work()

    # -- exchange ----------------------------------------------------------
    def all_to_all_single(self, output, input_tensor, output_split_sizes,
                          input_split_sizes, timeout_ms: int):
        self.ensure()
        lib = _bindings()
        dtype = _dtype_code(input_tensor)
        rank = self._pg.rank()
        world = self._pg.size()
        flat_in = _flat(input_tensor)
        flat_out = _flat(output)

        if not output_split_sizes and not input_split_sizes:
            if world == 1:
                flat_out.copy_(flat_in)
            elif flat_in.numel():
                _check(lib.tp_mlu_cncl_all_to_all(
                    _handle(flat_in), _handle(flat_out),
                    int(flat_in.numel()) // world, dtype, self._device,
                    self._comm))
            self._write_back(output, flat_out)
            return self._work()

        # The split sizes count leading-dimension rows; convert to elements
        # so the exchange works for tensors of any rank.
        in_counts = _element_counts(input_tensor, input_split_sizes, world)
        out_counts = _element_counts(output, output_split_sizes, world)
        in_offsets = _prefix_offsets(in_counts)
        out_offsets = _prefix_offsets(out_counts)

        if world > 1:
            _check(lib.tp_mlu_cncl_group_start())
            for peer in range(world):
                if peer != rank and in_counts[peer]:
                    _check(lib.tp_mlu_cncl_send(
                        _handle(flat_in.narrow(0, in_offsets[peer], in_counts[peer])),
                        in_counts[peer], dtype, peer, self._device, self._comm))
            for peer in range(world):
                if peer != rank and out_counts[peer]:
                    _check(lib.tp_mlu_cncl_recv(
                        _handle(flat_out.narrow(0, out_offsets[peer], out_counts[peer])),
                        out_counts[peer], dtype, peer, self._device, self._comm))
            _check(lib.tp_mlu_cncl_group_end())
        # The self exchange never leaves the device.
        if in_counts[rank] and out_counts[rank]:
            flat_out.narrow(0, out_offsets[rank], out_counts[rank]).copy_(
                flat_in.narrow(0, in_offsets[rank], in_counts[rank]))
        self._write_back(output, flat_out)
        return self._work()

    def alltoall(self, outputs: List[Any], inputs: List[Any], timeout_ms: int):
        self.ensure()
        lib = _bindings()
        rank = self._pg.rank()
        world = self._pg.size()
        if world == 1:
            if inputs[0].numel():
                outputs[0].copy_(_flat(inputs[0]))
            return self._work()
        _check(lib.tp_mlu_cncl_group_start())
        for peer, tensor in enumerate(inputs):
            count = int(tensor.numel())
            if count and peer != rank:
                _check(lib.tp_mlu_cncl_send(_handle(tensor), count,
                                            _dtype_code(tensor), peer,
                                            self._device, self._comm))
        for peer, tensor in enumerate(outputs):
            count = int(tensor.numel())
            if count and peer != rank:
                _check(lib.tp_mlu_cncl_recv(_handle(tensor), count,
                                            _dtype_code(tensor), peer,
                                            self._device, self._comm))
        _check(lib.tp_mlu_cncl_group_end())
        if inputs[rank].numel() and outputs[rank].numel():
            outputs[rank].copy_(_flat(inputs[rank]))
        return self._work()

    def _write_back(self, output, flat_out) -> None:
        """Undo a flattening copy when ``_flat`` had to allocate one."""
        if flat_out.data_ptr() != output.data_ptr() or not output.is_contiguous():
            output.copy_(flat_out.reshape(output.shape))

    # -- point to point ----------------------------------------------------
    def send(self, tensors: List[Any], dst: int, tag: int):
        self.ensure()
        tensor = tensors[0]
        count = int(tensor.numel())
        if count:
            _check(_bindings().tp_mlu_cncl_send(_handle(tensor), count,
                                                _dtype_code(tensor), dst,
                                                self._device, self._comm))
        return self._work()

    def recv(self, tensors: List[Any], src: int, tag: int):
        self.ensure()
        tensor = tensors[0]
        count = int(tensor.numel())
        if count:
            _check(_bindings().tp_mlu_cncl_recv(_handle(tensor), count,
                                                _dtype_code(tensor), src,
                                                self._device, self._comm))
        return self._work()

    def recv_anysource(self, tensors: List[Any], tag: int):
        raise RuntimeError("receiving from any source is unavailable for CNCL")

    # -- synchronisation ---------------------------------------------------
    def barrier(self, timeout_ms: int):
        self.ensure()
        flag = self._barrier_tensor()
        _check(_bindings().tp_mlu_cncl_all_reduce(
            _handle(flag), _handle(flag), 1, _DTYPE_CODES["float32"], _SUM_CODE,
            self._device, self._comm))
        return self._work()

    def monitored_barrier(self, timeout_ms: int, wait_all_ranks: bool):
        raise RuntimeError("monitored_barrier is unavailable for CNCL")


def _prefix_offsets(sizes):
    """Exclusive prefix sums of ``sizes`` for ``narrow``-style offsets."""
    offsets = []
    total = 0
    for size in sizes:
        offsets.append(total)
        total += int(size)
    return offsets


def _element_counts(tensor, split_sizes, world: int):
    """Per-peer element counts for ``all_to_all`` split sizes.

    ``split_sizes`` counts leading-dimension rows, as the frontend passes
    them; the exchange itself works in elements, so scale by the row size.
    An absent list means the buffer is divided evenly between the peers.
    """
    dim0 = int(tensor.shape[0]) if tensor.dim() > 0 else 1
    row = int(tensor.numel()) // dim0 if dim0 else 0
    if split_sizes:
        return [int(size) * row for size in split_sizes]
    return [(dim0 // world) * row] * world


def is_available() -> bool:
    """Whether the CNCL backend can be used on this host."""
    try:
        from . import is_loaded
        from . import runtime
        return is_loaded() and runtime.device_count() > 0
    except Exception:
        return False


def create_process_group(pg):
    """Device-backend factory: the collective provider for ``pg``."""
    impl = _CnclProcessGroup(pg)
    impl.ensure()
    return impl


def register() -> None:
    """Register the ``cncl`` backend with ``tensorplay.distributed``.

    Idempotent, and safe to call before ``tensorplay`` is imported: it
    imports the distributed core lazily and only when it is available.
    """
    try:
        from tensorplay.distributed import distributed_core as dc
    except Exception:
        return
    if "cncl" not in dc.Backend.backend_list:
        dc.Backend.register_backend("cncl", create_process_group, devices=["mlu"],
                                    _backend_type="cncl")
    dc.register_device_backend("cncl", create_process_group)

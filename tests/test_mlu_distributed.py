"""CNCL collective backend on a single MLU rank.

The two-container path (``world_size=2``) is exercised by the remote smoke
script; here every collective is checked against a hand-computed result with
one rank, which already covers the native entry points, the process-group
object and the backend registration.

Needs the Neuware toolkit and a Cambricon MLU; skipped otherwise.
"""

import contextlib
import os
import socket

import pytest

tp = pytest.importorskip("tensorplay")
tensorplay_mlu = pytest.importorskip("tensorplay_mlu")

if not os.path.isfile(os.path.join(tensorplay_mlu.neuware_home(), "include", "cnnl.h")):
    pytest.skip("Neuware toolkit not installed", allow_module_level=True)

tensorplay_mlu.load()
if not tp.mlu.is_available():
    pytest.skip("no MLU device", allow_module_level=True)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@contextlib.contextmanager
def _single_rank():
    import tensorplay_mlu

    tensorplay_mlu.load()
    import tensorplay.distributed as dist

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(_free_port())
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    dist.init_process_group("cncl", init_method="env://")
    try:
        yield dist
    finally:
        dist.destroy_process_group()


def _mlu_tensor(values, dtype=None):
    dtype = tp.float32 if dtype is None else dtype
    return tp.tensor(values, dtype=dtype, device="mlu")


def test_cncl_backend_is_registered():
    import tensorplay_mlu

    tensorplay_mlu.load()
    import tensorplay.distributed as dist

    assert dist.is_backend_available("cncl")
    assert dist.get_default_backend_for_device("mlu") == "cncl"


def test_all_reduce_sum():
    with _single_rank() as dist:
        t = _mlu_tensor([1.0, 2.0, 3.0])
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        assert t.tolist() == [1.0, 2.0, 3.0]


def test_all_reduce_avg():
    with _single_rank() as dist:
        t = _mlu_tensor([2.0, 4.0])
        dist.all_reduce(t, op=dist.ReduceOp.AVG)
        assert t.tolist() == [2.0, 4.0]


def test_all_reduce_bf16():
    with _single_rank() as dist:
        t = _mlu_tensor([1.5, 2.5], dtype=tp.bfloat16)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        assert t.tolist() == [1.5, 2.5]


def test_all_reduce_coalesced():
    with _single_rank() as dist:
        a = _mlu_tensor([1.0, 2.0])
        b = _mlu_tensor([3.0])
        dist.all_reduce_coalesced([a, b], op=dist.ReduceOp.SUM)
        assert a.tolist() == [1.0, 2.0]
        assert b.tolist() == [3.0]


def test_broadcast():
    with _single_rank() as dist:
        t = _mlu_tensor([5.0, 6.0])
        dist.broadcast(t, src=0)
        assert t.tolist() == [5.0, 6.0]


def test_reduce():
    with _single_rank() as dist:
        t = _mlu_tensor([7.0, 8.0])
        dist.reduce(t, dst=0, op=dist.ReduceOp.SUM)
        assert t.tolist() == [7.0, 8.0]


def test_all_gather():
    with _single_rank() as dist:
        send = _mlu_tensor([1.0, 2.0])
        outputs = [_mlu_tensor([0.0, 0.0])]
        dist.all_gather(outputs, send)
        assert outputs[0].tolist() == [1.0, 2.0]


def test_reduce_scatter():
    with _single_rank() as dist:
        inputs = [_mlu_tensor([1.0, 2.0])]
        output = _mlu_tensor([0.0, 0.0])
        dist.reduce_scatter(output, inputs, op=dist.ReduceOp.SUM)
        assert output.tolist() == [1.0, 2.0]


def test_all_to_all_single():
    with _single_rank() as dist:
        send = _mlu_tensor([1.0, 2.0, 3.0, 4.0])
        recv = _mlu_tensor([0.0, 0.0, 0.0, 0.0])
        dist.all_to_all_single(recv, send)
        assert recv.tolist() == [1.0, 2.0, 3.0, 4.0]


def test_all_to_all_single_with_splits():
    with _single_rank() as dist:
        send = _mlu_tensor([[1.0, 2.0], [3.0, 4.0]])
        recv = _mlu_tensor([[0.0, 0.0], [0.0, 0.0]])
        dist.all_to_all_single(recv, send, output_split_sizes=[2],
                               input_split_sizes=[2])
        assert recv.tolist() == [[1.0, 2.0], [3.0, 4.0]]


def test_all_to_all_list():
    with _single_rank() as dist:
        sends = [_mlu_tensor([1.0, 2.0])]
        recvs = [_mlu_tensor([0.0, 0.0])]
        dist.all_to_all(recvs, sends)
        assert recvs[0].tolist() == [1.0, 2.0]


def test_barrier_and_send_recv_self():
    with _single_rank() as dist:
        t = _mlu_tensor([1.0])
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        dist.barrier()
        assert t.tolist() == [1.0]

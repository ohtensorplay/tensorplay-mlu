"""The MLU backend against the host, kernel by kernel and end to end.

Needs a Cambricon MLU and the Neuware toolkit; skipped otherwise.  Every
check compares the device result with the same computation on the host.
"""

import os

import pytest

tp = pytest.importorskip("tensorplay")
tensorplay_mlu = pytest.importorskip("tensorplay_mlu")

if not os.path.isfile(os.path.join(tensorplay_mlu.neuware_home(), "include", "cnnl.h")):
    pytest.skip("Neuware toolkit not installed", allow_module_level=True)

tensorplay_mlu.load()
if not tp.mlu.is_available():
    pytest.skip("no MLU device", allow_module_level=True)

DTYPES = [tp.float32, tp.float16, tp.bfloat16]
TOL = {tp.float32: 1e-5, tp.float16: 2e-3, tp.bfloat16: 2e-2}


def close(actual, expected, dtype=tp.float32, scale=1.0):
    assert actual.device.type == "mlu", actual.device
    got = actual.cpu().to(tp.float64)
    want = expected.to(tp.float64)
    assert tuple(got.shape) == tuple(want.shape), (tuple(got.shape), tuple(want.shape))
    if got.numel() == 0:
        return
    err = (got - want).abs().max().item()
    ref = max(want.abs().max().item(), 1.0) * scale
    assert err <= TOL[dtype] * ref * 10, f"max error {err} (tolerance {TOL[dtype] * ref * 10})"


def rand(*shape, dtype=tp.float32, seed=0):
    tp.manual_seed(seed)
    return tp.randn(*shape).to(dtype)


# -- placement, copies, casts ------------------------------------------------

def test_device_is_named_and_counted():
    assert tp.device("mlu:0").type == "mlu"
    assert tp.mlu.device_count() >= 1
    assert tp.accelerator.current_accelerator().type == "mlu"


def test_kernel_library_sees_a_product_name():
    # CNNL picks its kernels by the product name; a generic one ("mlu")
    # selects kernels that fault or return wrong data on ordinary shapes.
    import ctypes
    driver = ctypes.CDLL(None)
    dev = ctypes.c_uint64()
    assert driver.cnDeviceGet(ctypes.byref(dev), 0) == 0
    name = ctypes.create_string_buffer(64)
    get_name = driver.cnDeviceGetName
    get_name.argtypes = [ctypes.c_char_p, ctypes.c_uint, ctypes.c_uint64]
    assert get_name(name, 64, dev) == 0
    assert name.value.startswith(b"MLU") and name.value[3:4].isdigit(), name.value


@pytest.mark.parametrize("dtype", DTYPES + [tp.float64, tp.int64, tp.int32, tp.bool])
def test_round_trip(dtype):
    x = (rand(5, 7) * 10).to(dtype)
    y = x.to("mlu")
    assert y.device.type == "mlu" and y.dtype == dtype
    assert tp.equal(y.cpu(), x)


def test_strided_copies_both_ways():
    x = rand(6, 8)
    assert tp.equal(x.t().to("mlu").cpu(), x.t())
    d = x.to("mlu")
    assert tp.equal(d.t().cpu(), x.t())
    assert tp.equal(d[:, 1::3].cpu(), x[:, 1::3])
    out = tp.zeros(8, 6, device="mlu")
    out.copy_(d.t())
    assert tp.equal(out.cpu(), x.t())


def test_broadcasting_copy_on_device():
    row = rand(1, 5).to("mlu")
    out = tp.empty(4, 5, device="mlu")
    out.copy_(row)
    assert tp.equal(out.cpu(), row.cpu().expand(4, 5))


@pytest.mark.parametrize("src,dst", [(tp.float32, tp.float16), (tp.float16, tp.float32),
                                     (tp.float32, tp.bfloat16), (tp.float16, tp.bfloat16),
                                     (tp.float32, tp.int32), (tp.int64, tp.float32),
                                     (tp.float64, tp.float32), (tp.float32, tp.float64)])
def test_casts_on_device(src, dst):
    x = (rand(3, 9) * 4).to(src)
    y = x.to("mlu").to(dst)
    assert y.dtype == dst
    assert tp.equal(y.cpu(), x.to(dst))


@pytest.mark.parametrize("dtype", DTYPES + [tp.int32, tp.int64, tp.float64])
def test_fill_and_factories(dtype):
    assert tp.equal(tp.full((3, 4), 2.5 if dtype.is_floating_point else 3,
                            dtype=dtype, device="mlu").cpu(),
                    tp.full((3, 4), 2.5 if dtype.is_floating_point else 3, dtype=dtype))
    assert tp.equal(tp.zeros(5, dtype=dtype, device="mlu").cpu(), tp.zeros(5, dtype=dtype))
    assert tp.equal(tp.ones(2, 3, dtype=dtype, device="mlu").cpu(), tp.ones(2, 3, dtype=dtype))
    d = tp.zeros(2, 3, dtype=dtype, device="mlu")
    for like, value in ((tp.ones_like(d), 1), (tp.zeros_like(d), 0), (tp.full_like(d, 7), 7)):
        assert like.device.type == "mlu" and like.dtype == dtype
        assert tp.equal(like.cpu(), tp.full((2, 3), value, dtype=dtype))
    assert tp.ones_like(d, dtype=tp.float32).dtype == tp.float32
    assert tp.equal(tp.ones_like(d, device="cpu"), tp.ones(2, 3, dtype=dtype))


def test_views_share_device_memory():
    x = rand(4, 6)
    d = x.to("mlu")
    v = d.view(6, 4)
    v.mul_(2.0)
    assert tp.equal(d.cpu(), x * 2)
    assert tp.equal(d.permute(1, 0).reshape(-1).cpu(), (x * 2).t().reshape(-1))
    assert tp.equal(d.unsqueeze(0).squeeze(0).cpu(), x * 2)
    parts = d.split(2, dim=1)
    assert len(parts) == 3 and tp.equal(parts[1].cpu(), (x * 2)[:, 2:4])
    assert tp.equal(d.narrow(0, 1, 2).cpu(), (x * 2)[1:3])


def test_resize_keeps_contents():
    d = tp.arange(6, dtype=tp.float32).to("mlu")
    d.resize_(12)
    assert tp.equal(d[:6].cpu(), tp.arange(6, dtype=tp.float32))


# -- arithmetic --------------------------------------------------------------

@pytest.mark.parametrize("dtype", DTYPES)
def test_binary_with_broadcasting(dtype):
    a, b = rand(4, 1, 5, dtype=dtype), rand(3, 1, dtype=dtype, seed=1)
    da, db = a.to("mlu"), b.to("mlu")
    close(da + db, a.float() + b.float(), dtype)
    close(da - 0.5 * db, a.float() - 0.5 * b.float(), dtype)
    close(tp.sub(da, db, alpha=2), a.float() - 2 * b.float(), dtype)
    close(da * db, a.float() * b.float(), dtype)
    close(da / (db.abs() + 1), a.float() / (b.float().abs() + 1), dtype)


@pytest.mark.parametrize("dtype", DTYPES)
def test_scalar_forms_and_host_scalars(dtype):
    a = rand(8, 3, dtype=dtype)
    d = a.to("mlu")
    af = a.float()
    close(d + 2, af + 2, dtype)
    close(d * 3, af * 3, dtype)
    close(d / 4, af / 4, dtype)
    close(1 - d, 1 - af, dtype)
    close(d * tp.tensor(2.0), af * 2, dtype)
    close(tp.tensor(2.0) + d, af + 2, dtype)


@pytest.mark.parametrize("dtype", DTYPES)
def test_in_place(dtype):
    a, b = rand(6, 4, dtype=dtype), rand(4, dtype=dtype, seed=2)
    d = a.to("mlu")
    d.add_(b.to("mlu"), alpha=0.5)
    d.mul_(2)
    d.sub_(b.to("mlu"))
    d.div_(3)
    close(d, ((a.float() + 0.5 * b.float()) * 2 - b.float()) / 3, dtype)


@pytest.mark.parametrize("dtype", DTYPES)
def test_scalar_forms_of_strided_operands(dtype):
    # An expanded operand (the gradient of a sum, for instance) and a strided
    # in-place target.
    base = rand(1, 5, dtype=dtype)
    expanded = base.to("mlu").expand(4, 5)
    close(expanded * 3, base.float().expand(4, 5) * 3, dtype)
    close(expanded + 2, base.float().expand(4, 5) + 2, dtype)
    a = rand(6, 8, dtype=dtype)
    d = a.to("mlu")
    d.t().mul_(0.5)
    d[:, ::2].add_(1.0)
    want = a.float() * 0.5
    want[:, ::2] += 1.0
    close(d, want, dtype)


def test_strided_operands():
    a = rand(5, 7)
    d = a.to("mlu")
    close(d.t() + d.t(), a.t() * 2)
    close(d[:, ::2][:, :3] * d[:, 1::2], a[:, ::2][:, :3] * a[:, 1::2])


def test_integer_math_runs_on_host_fallback():
    x = tp.arange(10).to("mlu")
    assert tp.equal((x * 3 + 1).cpu(), tp.arange(10) * 3 + 1)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("name", ["exp", "log", "sqrt", "rsqrt", "neg", "abs", "relu",
                                  "sigmoid", "tanh"])
def test_unary(dtype, name):
    a = rand(33, 17, dtype=dtype)
    if name in ("log", "sqrt", "rsqrt"):
        a = a.abs() + 0.1
    got = getattr(tp, name)(a.to("mlu"))
    close(got, getattr(tp, name)(a.float()), dtype)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("exponent", [2, 1, 0, 0.5, -0.5, 3, 1.7])
def test_pow_scalar(dtype, exponent):
    a = (rand(6, 9, dtype=dtype).float().abs() + 0.5).to(dtype)
    close(a.to("mlu") ** exponent, a.float() ** exponent, dtype, scale=4)


def test_squared_error_backward():
    x, t = rand(8, 5), rand(8, 5, seed=1)
    xd = x.to("mlu").requires_grad_()
    ((xd - t.to("mlu")) ** 2).mean().backward()
    close(xd.grad, 2 * (x - t) / x.numel())


@pytest.mark.parametrize("dtype", DTYPES)
def test_activations(dtype):
    a = rand(16, 32, dtype=dtype)
    d, af = a.to("mlu"), a.float()
    F = tp.nn.functional
    close(F.gelu(d), F.gelu(af), dtype)
    close(F.gelu(d, approximate="tanh"), F.gelu(af, approximate="tanh"), dtype)
    close(F.silu(d), F.silu(af), dtype)


# -- products, reductions, softmax -------------------------------------------

@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("ta,tb", [(False, False), (True, False), (False, True), (True, True)])
def test_mm_transposes(dtype, ta, tb):
    a, b = rand(37, 19, dtype=dtype), rand(19, 23, dtype=dtype, seed=3)
    da = a.t().contiguous().to("mlu").t() if ta else a.to("mlu")
    db = b.t().contiguous().to("mlu").t() if tb else b.to("mlu")
    close(da @ db, a.float() @ b.float(), dtype, scale=5)


@pytest.mark.parametrize("dtype", DTYPES)
def test_addmm_bmm_matmul(dtype):
    bias, a, b = rand(23, dtype=dtype), rand(37, 19, dtype=dtype, seed=1), rand(19, 23, dtype=dtype, seed=2)
    close(tp.addmm(bias.to("mlu"), a.to("mlu"), b.to("mlu"), beta=0.5, alpha=2.0),
          0.5 * bias.float() + 2.0 * (a.float() @ b.float()), dtype, scale=10)
    x, y = rand(4, 9, 6, dtype=dtype), rand(4, 6, 5, dtype=dtype, seed=4)
    close(tp.bmm(x.to("mlu"), y.to("mlu")), tp.bmm(x.float(), y.float()), dtype, scale=5)
    w = rand(6, 5, dtype=dtype, seed=5)
    close(x.to("mlu") @ w.to("mlu"), x.float() @ w.float(), dtype, scale=5)
    v = rand(2, 1, 6, 3, dtype=dtype, seed=6)
    close(x.to("mlu") @ v.to("mlu"), x.float() @ v.float(), dtype, scale=5)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("m,k,n", [(2, 1024, 1024), (128, 1024, 256), (256, 4096, 1024),
                                   (1024, 4096, 128), (4096, 4096, 4096)])
def test_mm_shapes(dtype, m, k, n):
    a, b = rand(m, k, dtype=dtype, seed=1), rand(k, n, dtype=dtype, seed=2)
    close(a.to("mlu") @ b.to("mlu"), a.float() @ b.float(), dtype, scale=k ** 0.5)


@pytest.mark.parametrize("dtype", DTYPES)
def test_linear(dtype):
    F = tp.nn.functional
    x, w, b = rand(3, 5, 64, dtype=dtype), rand(48, 64, dtype=dtype, seed=1), rand(48, dtype=dtype, seed=2)
    xd, wd, bd = x.to("mlu"), w.to("mlu"), b.to("mlu")
    want = x.float() @ w.float().t()
    close(F.linear(xd, wd, bd), want + b.float(), dtype, scale=8)
    close(F.linear(xd, wd), want, dtype, scale=8)
    close(F.linear(xd[0, 0], wd, bd), want[0, 0] + b.float(), dtype, scale=8)


@pytest.mark.parametrize("dtype", DTYPES)
def test_bmm_shapes(dtype):
    for (bt, m, k, n) in ((32, 512, 64, 512), (1, 256, 4096, 1024), (3, 1, 7, 5)):
        a, b = rand(bt, m, k, dtype=dtype, seed=1), rand(bt, k, n, dtype=dtype, seed=2)
        close(tp.bmm(a.to("mlu"), b.to("mlu")), tp.bmm(a.float(), b.float()), dtype, scale=k ** 0.5)
    a, b = rand(4, 9, 6, dtype=dtype), rand(4, 5, 6, dtype=dtype, seed=3)
    close(tp.bmm(a.to("mlu"), b.to("mlu").transpose(1, 2)), tp.bmm(a.float(), b.float().transpose(1, 2)),
          dtype, scale=5)


def test_float32_matmul_precision_selects_tf32():
    a, b = rand(512, 512, seed=1), rand(512, 512, seed=2)
    want = a.double() @ b.double()
    exact = ((a.to("mlu") @ b.to("mlu")).cpu().double() - want).abs().max().item()
    previous = tp.get_float32_matmul_precision()
    tp.set_float32_matmul_precision("high")
    try:
        fast = ((a.to("mlu") @ b.to("mlu")).cpu().double() - want).abs().max().item()
    finally:
        tp.set_float32_matmul_precision(previous)
    assert exact < 1e-3, exact
    assert fast < 0.5, fast  # TF32 keeps 10 mantissa bits


@pytest.mark.parametrize("dtype", DTYPES)
def test_reductions(dtype):
    a = rand(6, 7, 8, dtype=dtype)
    d, af = a.to("mlu"), a.float()
    close(d.sum(), af.sum(), dtype, scale=20)
    close(d.sum(dim=1), af.sum(dim=1), dtype, scale=5)
    close(d.sum(dim=(0, 2), keepdim=True), af.sum(dim=(0, 2), keepdim=True), dtype, scale=20)
    close(d.mean(), af.mean(), dtype)
    close(d.mean(dim=-1), af.mean(dim=-1), dtype)


@pytest.mark.parametrize("dtype", DTYPES)
def test_long_reductions(dtype):
    # Long contiguous reductions and whole-tensor reductions over more than
    # 65536 elements.
    for shape in ((2, 4096), (2, 65536), (16, 8192), (3, 32768), (4096, 2)):
        a = rand(*shape, dtype=dtype)
        d, af = a.to("mlu"), a.float()
        close(d.sum(), af.sum(), dtype, scale=20)
        close(d.mean(), af.mean(), dtype)
        close(d.sum(dim=1), af.sum(dim=1), dtype, scale=20)
        close(d.mean(dim=0), af.mean(dim=0), dtype)
    a = rand(65537, dtype=dtype)
    close(a.to("mlu").sum(), a.float().sum(), dtype, scale=20)
    close(a.to("mlu").mean(), a.float().mean(), dtype)
    a = rand(2, 4096, 2, dtype=dtype)
    close(a.to("mlu").sum(dim=1), a.float().sum(dim=1), dtype, scale=20)
    close(a.to("mlu").mean(dim=1), a.float().mean(dim=1), dtype)
    a = rand(16, 8192, dtype=dtype)
    close(a.to("mlu").amax(dim=-1), a.float().amax(dim=-1), dtype)
    close(a.to("mlu").amin(dim=-1), a.float().amin(dim=-1), dtype)


@pytest.mark.parametrize("dtype", [tp.float16, tp.bfloat16])
def test_large_narrow_float_elementwise(dtype):
    a = rand(4096, 8192, dtype=dtype)  # 64 MiB
    d = a.to("mlu")
    close(d + d, a.float() * 2, dtype)
    close(tp.sigmoid(d), tp.sigmoid(a.float()), dtype)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("dim", [0, 1, -1])
def test_softmax_and_log_softmax(dtype, dim):
    a = rand(5, 6, 7, dtype=dtype)
    d, af = a.to("mlu"), a.float()
    close(tp.softmax(d, dim=dim), tp.softmax(af, dim=dim), dtype)
    close(tp.log_softmax(d, dim=dim), tp.log_softmax(af, dim=dim), dtype)


# -- whole programs ----------------------------------------------------------

def test_operator_without_mlu_kernel_runs_on_host():
    a = rand(4, 5)
    close(tp.cumsum(a.to("mlu"), dim=1), tp.cumsum(a, dim=1))
    vals, idx = tp.sort(a.to("mlu"), dim=1)
    ref_vals, ref_idx = tp.sort(a, dim=1)
    assert vals.device.type == "mlu" and tp.equal(idx.cpu(), ref_idx)


def test_training_step_matches_host():
    tp.manual_seed(0)
    model = tp.nn.Sequential(tp.nn.Linear(12, 32), tp.nn.GELU(), tp.nn.Linear(32, 3))
    x, target = tp.randn(16, 12), tp.randint(0, 3, (16,))
    grads = {}
    for device in ("cpu", "mlu"):
        m = model.to(device)
        m.zero_grad()
        loss = tp.nn.functional.cross_entropy(m(x.to(device)), target.to(device))
        loss.backward()
        grads[device] = (loss.detach().cpu(), [p.grad.cpu() for p in m.parameters()])
    ref_loss, ref_grads = grads["cpu"]
    loss, got = grads["mlu"]
    assert abs(loss.item() - ref_loss.item()) < 1e-4
    for g, r in zip(got, ref_grads):
        assert (g - r).abs().max().item() < 1e-4


def test_profiler_reports_device_time():
    from tensorplay.profiler import ProfilerActivity, profile, supported_activities

    assert ProfilerActivity.PrivateUse1 in supported_activities()
    a = rand(512, 512).to("mlu")
    tp.mm(a, a)
    # Kernel-level records (when the activity library loads) and event pairs.
    for options in ({}, {"gpu_timing": True, "gpu_trace": False}):
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1],
                     **options) as prof:
            for _ in range(3):
                tp.mm(a, a).relu()
        rows = {row.key: row for row in prof.key_averages()}
        assert rows["mm"].count == 3 and rows["mm"].device_time_total > 0, options
        assert rows["relu"].device_time_total > 0, options
        assert "MLU" in prof.key_averages().table(sort_by="device_time_total", row_limit=5)
        if prof.device_trace:
            # Each kernel joined to the operation that launched it.
            kernels = [r for r in prof.kineto_results() if r.activity_type() == "gpu_kernel"]
            assert kernels and all(r.op_slot() is not None for r in kernels)
            assert rows["mm"].kernel_count >= 3, rows["mm"].kernel_count


def test_memory_statistics():
    tp.mlu.empty_cache()
    before = tp.mlu.memory_allocated()
    x = tp.empty(1 << 20, device="mlu")
    assert tp.mlu.memory_allocated() >= before + (4 << 20)
    assert tp.mlu.max_memory_allocated() >= tp.mlu.memory_allocated()
    del x
    assert tp.mlu.memory_allocated() == before
    assert tp.mlu.memory_reserved() >= before
    free, total = tp.mlu.mem_get_info()
    assert 0 < free <= total

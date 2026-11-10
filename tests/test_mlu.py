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


# -- indexing, scatter, scans, where -------------------------------------------

def indices(*shape, high, seed=7):
    tp.manual_seed(seed)
    return tp.randint(0, high, shape)


@pytest.mark.parametrize("dtype", DTYPES + [tp.int32])
def test_gather_and_scatter(dtype):
    # int32 values are exact, so the host comparison uses the float tolerance.
    tol = dtype if dtype in TOL else tp.float32
    if dtype == tp.int32:
        x = tp.arange(20, dtype=tp.float32).reshape(4, 5).to(dtype)
        src = tp.arange(20, dtype=tp.float32).reshape(4, 5).to(dtype)
    else:
        x = rand(4, 5, dtype=dtype, seed=21)
        src = rand(4, 5, dtype=dtype, seed=22)
    idx = indices(4, 5, high=5, seed=23)
    xm, sm, im = x.to("mlu"), src.to("mlu"), idx.to("mlu")
    close(tp.gather(xm, 1, im), tp.gather(x, 1, idx), tol)
    idx0 = indices(2, 5, high=4, seed=24)
    close(tp.gather(xm, 0, idx0.to("mlu")), tp.gather(x, 0, idx0), tol)
    rows = indices(3, high=4, seed=25)
    close(tp.index_select(xm, 0, rows.to("mlu")), tp.index_select(x, 0, rows), tol)
    close(tp.scatter_add(xm, 1, im, sm), tp.scatter_add(x, 1, idx, src), tol)
    close(tp.scatter(xm, 1, im, sm), tp.scatter(x, 1, idx, src), tol)
    d, ref = xm.clone(), x.clone()
    d.scatter_add_(1, im, sm)
    ref.scatter_add_(1, idx, src)
    close(d, ref, tol)


@pytest.mark.parametrize("include_self", (True, False))
def test_scatter_reduce(include_self):
    x = rand(4, 5, seed=26)
    src = rand(4, 5, seed=27)
    idx = indices(4, 5, high=5, seed=28)
    xm, sm, im = x.to("mlu"), src.to("mlu"), idx.to("mlu")
    for reduce in ("sum", "amax", "amin"):
        close(tp.scatter_reduce(xm, 1, im, sm, reduce, include_self=include_self),
              tp.scatter_reduce(x, 1, idx, src, reduce, include_self=include_self))


def test_scatter_reduce_unsupported_reduce_takes_the_host_path():
    x = rand(4, 5, seed=29)
    src = rand(4, 5, seed=30)
    idx = indices(4, 5, high=5, seed=31)
    close(tp.scatter_reduce(x.to("mlu"), 1, idx.to("mlu"), src.to("mlu"), "prod"),
          tp.scatter_reduce(x, 1, idx, src, "prod"))


def test_cumsum_and_cumprod():
    for dtype in DTYPES:
        x = rand(6, 7, dtype=dtype, seed=32)
        close(tp.cumsum(x.to("mlu"), 1), tp.cumsum(x, 1), dtype)
    x = rand(6, 7, seed=33)
    close(tp.cumsum(x.to("mlu"), 0), tp.cumsum(x, 0))
    close(tp.cumprod(x.to("mlu"), 0), tp.cumprod(x, 0))
    d = x.to("mlu")
    d.cumsum_(1)
    close(d, tp.cumsum(x, 1))
    ints = tp.arange(12, dtype=tp.int64).reshape(3, 4)
    close(tp.cumsum(ints.to("mlu"), 1), tp.cumsum(ints, 1))


def test_where():
    cond = rand(4, 5, seed=34) > 0
    a = rand(4, 5, seed=35)
    b = rand(4, 5, seed=36)
    close(tp.where(cond.to("mlu"), a.to("mlu"), b.to("mlu")), tp.where(cond, a, b))


def test_where_with_broadcasting():
    cond = rand(4, 1, seed=37) > 0
    a = rand(4, 5, seed=38)
    b = rand(5, seed=39)
    close(tp.where(cond.to("mlu"), a.to("mlu"), b.to("mlu")), tp.where(cond, a, b))


def test_gather_backward_accumulates_duplicate_indices():
    x = rand(4, 5, seed=40)
    idx = indices(4, 5, high=2, seed=41)
    xd = x.to("mlu").requires_grad_()
    tp.gather(xd, 1, idx.to("mlu")).sum().backward()
    xh = x.clone().requires_grad_()
    tp.gather(xh, 1, idx).sum().backward()
    assert xd.grad is not None
    close(xd.grad, xh.grad)


# -- convolutions ------------------------------------------------------------

CONV_DTYPES = [tp.float32, tp.float16, tp.bfloat16]


def transpose_ref(fn, x, w, bias, *params):
    """Host reference for a transposed convolution, accumulated in double
    precision.  The single-precision host deconvolution picks an accelerated
    library path whose result is not bit-stable from one call to the next; the
    double-precision path is a plain accumulation and answers the same every
    time, so it is the reference the device result is held to.
    """
    x = x.to(tp.float64)
    w = w.to(tp.float64)
    bias = None if bias is None else bias.to(tp.float64)
    return fn(x, w, bias, *params)


@pytest.mark.parametrize("dtype", CONV_DTYPES)
@pytest.mark.parametrize("groups", (1, 2))
def test_conv2d(dtype, groups):
    x = rand(2, 4, 8, 8, dtype=dtype, seed=50)
    w = rand(6, 4 // groups, 3, 3, dtype=dtype, seed=51)
    b = rand(6, dtype=dtype, seed=52)
    close(tp.conv2d(x.to("mlu"), w.to("mlu"), b.to("mlu"), [1, 1], [1, 1], [1, 1], groups),
          tp.conv2d(x, w, b, [1, 1], [1, 1], [1, 1], groups), dtype)


def test_conv2d_stride_pad_dilation():
    x = rand(2, 3, 12, 12, seed=53)
    w = rand(3, 3, 3, 3, seed=54)
    for stride, pad, dil in (([2, 2], [1, 1], [1, 1]),
                             ([1, 2], [2, 1], [1, 1]),
                             ([1, 1], [2, 2], [2, 2]),
                             ([2, 1], [0, 1], [1, 3])):
        close(tp.conv2d(x.to("mlu"), w.to("mlu"), None, stride, pad, dil, 1),
              tp.conv2d(x, w, None, stride, pad, dil, 1))


def test_conv1d_and_conv3d():
    x1 = rand(2, 3, 10, seed=55)
    w1 = rand(4, 3, 3, seed=56)
    close(tp.conv1d(x1.to("mlu"), w1.to("mlu"), None, [2], [1], [1], 1),
          tp.conv1d(x1, w1, None, [2], [1], [1], 1))
    x3 = rand(1, 2, 5, 5, 5, seed=57)
    w3 = rand(3, 2, 2, 2, 2, seed=58)
    close(tp.conv3d(x3.to("mlu"), w3.to("mlu"), None, [2, 1, 1], [1, 1, 1], [1, 1, 1], 1),
          tp.conv3d(x3, w3, None, [2, 1, 1], [1, 1, 1], [1, 1, 1], 1))


@pytest.mark.parametrize("dtype", CONV_DTYPES)
def test_conv_transpose2d(dtype):
    x = rand(2, 4, 6, 6, dtype=dtype, seed=59)
    w = rand(4, 2, 3, 3, dtype=dtype, seed=60)
    b = rand(2, dtype=dtype, seed=61)
    close(tp.conv_transpose2d(x.to("mlu"), w.to("mlu"), b.to("mlu"), [2, 2], [1, 1], [1, 1], 1),
          transpose_ref(tp.conv_transpose2d, x, w, b, [2, 2], [1, 1], [1, 1], 1), dtype)


def test_conv_transpose_output_padding_and_groups():
    x = rand(2, 4, 7, 7, seed=62)
    w = rand(4, 2, 3, 3, seed=63)
    for stride, pad, opad in (([2, 2], [1, 1], [0, 0]),
                              ([2, 2], [1, 1], [1, 1]),
                              ([3, 2], [2, 1], [1, 0])):
        close(tp.conv_transpose2d(x.to("mlu"), w.to("mlu"), None, stride, pad, opad, 1),
              transpose_ref(tp.conv_transpose2d, x, w, None, stride, pad, opad, 1))
    xg = rand(2, 6, 5, 5, seed=64)
    wg = rand(6, 2, 3, 3, seed=65)
    close(tp.conv_transpose2d(xg.to("mlu"), wg.to("mlu"), None, [2, 2], [1, 1], [1, 1], 3),
          transpose_ref(tp.conv_transpose2d, xg, wg, None, [2, 2], [1, 1], [1, 1], 3))


def test_conv_transpose1d_and_3d():
    x1 = rand(2, 3, 10, seed=66)
    w1 = rand(3, 4, 3, seed=67)
    close(tp.conv_transpose1d(x1.to("mlu"), w1.to("mlu"), None, [2], [1], [1], 1),
          transpose_ref(tp.conv_transpose1d, x1, w1, None, [2], [1], [1], 1))
    x3 = rand(1, 2, 4, 4, 4, seed=68)
    w3 = rand(2, 3, 2, 2, 2, seed=69)
    close(tp.conv_transpose3d(x3.to("mlu"), w3.to("mlu"), None, [2, 1, 1], [1, 1, 1], [1, 0, 0], 1),
          transpose_ref(tp.conv_transpose3d, x3, w3, None, [2, 1, 1], [1, 1, 1], [1, 0, 0], 1))


def test_conv2d_backward():
    x = rand(2, 3, 8, 8, seed=70)
    w = rand(5, 3, 3, 3, seed=71)
    b = rand(5, seed=72)
    xd = x.to("mlu").requires_grad_()
    wd = w.to("mlu").requires_grad_()
    bd = b.to("mlu").requires_grad_()
    y = tp.conv2d(xd, wd, bd, [1, 1], [1, 1], [1, 1], 1)
    (y * y).sum().backward()
    xh = x.clone().requires_grad_()
    wh = w.clone().requires_grad_()
    bh = b.clone().requires_grad_()
    yh = tp.conv2d(xh, wh, bh, [1, 1], [1, 1], [1, 1], 1)
    (yh * yh).sum().backward()
    close(xd.grad, xh.grad)
    close(wd.grad, wh.grad)
    close(bd.grad, bh.grad)


def test_conv_transpose2d_backward():
    x = rand(2, 4, 6, 6, seed=73)
    w = rand(4, 2, 3, 3, seed=74)
    b = rand(2, seed=75)
    xd = x.to("mlu").requires_grad_()
    wd = w.to("mlu").requires_grad_()
    bd = b.to("mlu").requires_grad_()
    y = tp.conv_transpose2d(xd, wd, bd, [2, 2], [1, 1], [1, 1], 1)
    (y * y).sum().backward()
    xh = x.to(tp.float64).requires_grad_()
    wh = w.to(tp.float64).requires_grad_()
    bh = b.to(tp.float64).requires_grad_()
    yh = tp.conv_transpose2d(xh, wh, bh, [2, 2], [1, 1], [1, 1], 1)
    (yh * yh).sum().backward()
    close(xd.grad, xh.grad)
    close(wd.grad, wh.grad)
    close(bd.grad, bh.grad)


# -- pooling -----------------------------------------------------------------

POOL_DTYPES = [tp.float32, tp.float16, tp.bfloat16]


def host_pool(fn, x, *params):
    """Host reference for a pooling call.  The host implements pooling only for
    the wide types, and a narrow value widens to float32 exactly, so the
    float32 host result keeps the same values and the same winning positions.
    """
    return fn(x.to(tp.float32), *params)


def same_indices(actual, expected):
    assert actual.device.type == "mlu" and actual.dtype == tp.int64
    assert tuple(actual.shape) == tuple(expected.shape), (actual.shape, expected.shape)
    assert tp.equal(actual.cpu(), expected.to(tp.int64)), (actual.cpu(), expected)


@pytest.mark.parametrize("dtype", POOL_DTYPES)
def test_max_pool2d_with_indices(dtype):
    x = rand(2, 3, 7, 7, dtype=dtype, seed=80)
    for kernel, stride, pad, dil, ceil in (([2, 2], [2, 2], [0, 0], [1, 1], False),
                                           ([3, 3], [2, 2], [1, 1], [1, 1], False),
                                           ([2, 3], [1, 2], [1, 1], [1, 1], False),
                                           ([3, 3], [1, 1], [2, 2], [2, 2], False),
                                           ([2, 2], [2, 2], [0, 0], [1, 1], True)):
        v, i = tp.max_pool2d_with_indices(x.to("mlu"), kernel, stride, pad, dil, ceil)
        vh, ih = host_pool(tp.max_pool2d_with_indices, x, kernel, stride, pad, dil, ceil)
        close(v.to(tp.float32), vh, dtype)
        same_indices(i, ih)


@pytest.mark.parametrize("dtype", POOL_DTYPES)
def test_avg_pool2d(dtype):
    # The padded geometries, with the window either counted whole or trimmed
    # to its covered part.
    x = rand(2, 3, 8, 8, dtype=dtype, seed=81)
    for ceil in (False, True):
        for include_pad in (True, False):
            close(tp.avg_pool2d(x.to("mlu"), [3, 3], [2, 2], [1, 1], ceil, include_pad),
                  tp.avg_pool2d(x.to(tp.float32), [3, 3], [2, 2], [1, 1], ceil, include_pad),
                  dtype)


def test_avg_pool2d_divisor_override_runs_on_host():
    x = rand(2, 3, 8, 8, seed=82)
    close(tp.avg_pool2d(x.to("mlu"), [3, 3], [2, 2], [1, 1], False, True, 4),
          tp.avg_pool2d(x, [3, 3], [2, 2], [1, 1], False, True, 4))


@pytest.mark.parametrize("dtype", POOL_DTYPES)
def test_max_pool3d_with_indices(dtype):
    x = rand(2, 2, 5, 6, 7, dtype=dtype, seed=83)
    v, i = tp.max_pool3d_with_indices(x.to("mlu"), [2, 2, 2], [2, 2, 1], [1, 1, 1], [1, 1, 1], False)
    vh, ih = host_pool(tp.max_pool3d_with_indices, x, [2, 2, 2], [2, 2, 1], [1, 1, 1], [1, 1, 1],
                       False)
    close(v.to(tp.float32), vh, dtype)
    same_indices(i, ih)


@pytest.mark.parametrize("dtype", POOL_DTYPES)
def test_avg_pool3d(dtype):
    x = rand(2, 2, 5, 6, 7, dtype=dtype, seed=84)
    for include_pad in (True, False):
        close(tp.avg_pool3d(x.to("mlu"), [2, 2, 2], [2, 2, 1], [1, 1, 1], False, include_pad),
              tp.avg_pool3d(x.to(tp.float32), [2, 2, 2], [2, 2, 1], [1, 1, 1], False, include_pad),
              dtype)


@pytest.mark.parametrize("dtype", POOL_DTYPES)
def test_adaptive_pool2d(dtype):
    # A mix of down- and up-scaling, including the rectangular windows whose
    # extent changes from one output position to the next.
    x = rand(2, 3, 7, 9, dtype=dtype, seed=85)
    for out in ([3, 3], [1, 1], [5, 4], [1, 9], [4, 7]):
        close(tp.adaptive_avg_pool2d(x.to("mlu"), out),
              tp.adaptive_avg_pool2d(x.to(tp.float32), out), dtype)
    for out in ([3, 3], [5, 4], [4, 7]):
        v, i = tp.adaptive_max_pool2d_with_indices(x.to("mlu"), out)
        vh, ih = host_pool(tp.adaptive_max_pool2d_with_indices, x, out)
        close(v.to(tp.float32), vh, dtype)
        same_indices(i, ih)


@pytest.mark.parametrize("dtype", POOL_DTYPES)
def test_adaptive_pool3d(dtype):
    x = rand(2, 2, 5, 7, 9, dtype=dtype, seed=86)
    for out in ([2, 3, 4], [1, 1, 1], [3, 4, 5]):
        close(tp.adaptive_avg_pool3d(x.to("mlu"), out),
              tp.adaptive_avg_pool3d(x.to(tp.float32), out), dtype)


def test_adaptive_max_pool3d_runs_on_host():
    # The card has no 3-D adaptive maximum; the operator still answers, from
    # the host.
    x = rand(2, 2, 5, 7, 9, seed=87)
    v, i = tp.adaptive_max_pool3d(x.to("mlu"), [2, 3, 4])
    vh, ih = tp.adaptive_max_pool3d(x, [2, 3, 4])
    close(v, vh)
    same_indices(i, ih)


@pytest.mark.parametrize("dtype", POOL_DTYPES)
def test_pool_unbatched(dtype):
    # A 2-D pool over a 3-D operand and a 3-D pool over a 4-D one run as the
    # batched call with a unit batch axis.
    x2 = rand(3, 7, 7, dtype=dtype, seed=88)
    close(tp.max_pool2d(x2.to("mlu"), [2, 2], [2, 2], [0, 0], [1, 1], False),
          tp.max_pool2d(x2.to(tp.float32), [2, 2], [2, 2], [0, 0], [1, 1], False), dtype)
    close(tp.avg_pool2d(x2.to("mlu"), [2, 2], [2, 2], [1, 1], False, False),
          tp.avg_pool2d(x2.to(tp.float32), [2, 2], [2, 2], [1, 1], False, False), dtype)
    close(tp.adaptive_avg_pool2d(x2.to("mlu"), [3, 3]),
          tp.adaptive_avg_pool2d(x2.to(tp.float32), [3, 3]), dtype)
    x3 = rand(3, 5, 6, 7, dtype=dtype, seed=89)
    close(tp.max_pool3d(x3.to("mlu"), [2, 2, 2], [2, 2, 2], [0, 0, 0], [1, 1, 1], False),
          tp.max_pool3d(x3.to(tp.float32), [2, 2, 2], [2, 2, 2], [0, 0, 0], [1, 1, 1], False), dtype)
    close(tp.adaptive_avg_pool3d(x3.to("mlu"), [2, 2, 2]),
          tp.adaptive_avg_pool3d(x3.to(tp.float32), [2, 2, 2]), dtype)


def test_pool1d():
    # The 1-D forms are expressed through the 2-D operators.
    x = rand(3, 4, 10, seed=90)
    close(tp.max_pool1d(x.to("mlu"), [3], [2], [1], [1], False),
          tp.max_pool1d(x, [3], [2], [1], [1], False))
    close(tp.avg_pool1d(x.to("mlu"), [3], [2], [1], False, True),
          tp.avg_pool1d(x, [3], [2], [1], False, True))


def _pool_gradient(name, x, *params, dtype=tp.float32):
    xd = x.to("mlu").requires_grad_()
    getattr(tp, name)(xd, *params).pow(2).sum().backward()
    xh = x.to(tp.float32).clone().requires_grad_()
    getattr(tp, name)(xh, *params).pow(2).sum().backward()
    close(xd.grad, xh.grad, dtype)


@pytest.mark.parametrize("dtype", POOL_DTYPES)
def test_pool_backward(dtype):
    x = rand(2, 3, 8, 8, dtype=dtype, seed=91)
    _pool_gradient("max_pool2d", x, [2, 2], [2, 2], [0, 0], [1, 1], False, dtype=dtype)
    _pool_gradient("avg_pool2d", x, [3, 3], [2, 2], [1, 1], False, False, dtype=dtype)
    _pool_gradient("adaptive_max_pool2d", x, [3, 3], dtype=dtype)
    _pool_gradient("adaptive_avg_pool2d", x, [3, 3], dtype=dtype)
    x3 = rand(2, 2, 5, 6, 7, dtype=dtype, seed=92)
    _pool_gradient("max_pool3d", x3, [2, 2, 2], [2, 2, 2], [0, 0, 0], [1, 1, 1], False, dtype=dtype)
    _pool_gradient("adaptive_avg_pool3d", x3, [2, 2, 2], dtype=dtype)


# -- batch normalization -----------------------------------------------------

BN_DTYPES = [tp.float32, tp.float16, tp.bfloat16]
BN_SHAPES = [(8, 4), (3, 5, 7), (2, 3, 4, 5), (2, 3, 4, 4, 3)]


def mlu(t):
    return None if t is None else t.to("mlu")


def bn_params(shape, dtype, affine, seed):
    """A batch-normalization sample: the input, the optional affine pair and the
    running statistics, all in ``dtype``."""
    c = shape[1]
    x = rand(*shape, dtype=dtype, seed=seed)
    w = rand(c, dtype=dtype, seed=seed + 1) if affine else None
    b = rand(c, dtype=dtype, seed=seed + 2) if affine else None
    rm = rand(c, dtype=dtype, seed=seed + 3).abs()
    rv = rand(c, dtype=dtype, seed=seed + 4).abs() + 0.5
    return x, w, b, rm, rv


@pytest.mark.parametrize("dtype", BN_DTYPES)
@pytest.mark.parametrize("shape", BN_SHAPES)
@pytest.mark.parametrize("affine", (True, False))
def test_batch_norm_training(shape, affine, dtype):
    # A training pass normalizes by the batch statistics and folds those into
    # the running statistics with the momentum.
    x, w, b, rm, rv = bn_params(shape, dtype, affine, seed=100)
    rmm, rvm = rm.to("mlu"), rv.to("mlu")
    y = tp.nn.functional.batch_norm(mlu(x), rmm, rvm, mlu(w), mlu(b), True, 0.1, 1e-5)
    yh = tp.nn.functional.batch_norm(x, rm, rv, w, b, True, 0.1, 1e-5)
    close(y, yh, dtype)
    close(rmm, rm, dtype)
    close(rvm, rv, dtype)


@pytest.mark.parametrize("dtype", BN_DTYPES)
@pytest.mark.parametrize("shape", BN_SHAPES)
def test_batch_norm_eval(shape, dtype):
    # An inference pass normalizes by the running statistics and leaves them
    # untouched.
    x, w, b, rm, rv = bn_params(shape, dtype, True, seed=110)
    rmm, rvm = rm.to("mlu"), rv.to("mlu")
    y = tp.nn.functional.batch_norm(mlu(x), rmm, rvm, mlu(w), mlu(b), False, 0.1, 1e-5)
    yh = tp.nn.functional.batch_norm(x, rm, rv, w, b, False, 0.1, 1e-5)
    close(y, yh, dtype)
    assert tp.equal(rmm.cpu(), rm) and tp.equal(rvm.cpu(), rv)


@pytest.mark.parametrize("dtype", BN_DTYPES)
def test_native_batch_norm_saves_statistics(dtype):
    # The native form also reports the batch statistics it normalized by.
    x, w, b, rm, rv = bn_params((2, 3, 4, 5), dtype, True, seed=120)
    rmm, rvm = rm.to("mlu"), rv.to("mlu")
    om, sm, si = tp.native_batch_norm(mlu(x), mlu(w), mlu(b), rmm, rvm, True, 0.1, 1e-5)
    oh, sh, sih = tp.native_batch_norm(x, w, b, rm, rv, True, 0.1, 1e-5)
    close(om, oh, dtype)
    close(sm, sh, dtype)
    close(si, sih, dtype)
    close(rmm, rm, dtype)
    close(rvm, rv, dtype)


def test_native_batch_norm_eval_returns_empty_statistics():
    x, w, b, rm, rv = bn_params((2, 3, 4, 5), tp.float32, True, seed=130)
    om, sm, si = tp.native_batch_norm(mlu(x), mlu(w), mlu(b), mlu(rm), mlu(rv), False, 0.1, 1e-5)
    oh, sh, sih = tp.native_batch_norm(x, w, b, rm, rv, False, 0.1, 1e-5)
    close(om, oh)
    assert sm.numel() == 0 and si.numel() == 0
    assert sh.numel() == 0 and sih.numel() == 0


@pytest.mark.parametrize("dtype", BN_DTYPES)
@pytest.mark.parametrize("training", (True, False))
def test_batch_norm_backward(training, dtype):
    # A training backward takes its statistics from a fresh training forward;
    # an inference backward uses the running statistics.
    x, w, b, rm, rv = bn_params((2, 3, 4, 5), dtype, True, seed=140)
    xm = x.to("mlu").requires_grad_()
    wm = w.to("mlu").requires_grad_()
    bm = b.to("mlu").requires_grad_()
    (tp.nn.functional.batch_norm(xm, mlu(rm), mlu(rv), wm, bm, training, 0.1,
                                 1e-5) ** 2).sum().backward()
    xh = x.to(tp.float32).requires_grad_()
    wh = w.to(tp.float32).requires_grad_()
    bh = b.to(tp.float32).requires_grad_()
    (tp.nn.functional.batch_norm(xh, rm.to(tp.float32).clone(), rv.to(tp.float32).clone(), wh, bh,
                                 training, 0.1, 1e-5) ** 2).sum().backward()
    close(xm.grad, xh.grad, dtype)
    close(wm.grad, wh.grad, dtype)
    close(bm.grad, bh.grad, dtype)


def test_batch_norm_module_training_and_eval():
    # The module drives the same operator end to end, over every rank it
    # accepts.
    for cls, shape in ((tp.nn.BatchNorm1d, (8, 4)), (tp.nn.BatchNorm1d, (3, 5, 7)),
                       (tp.nn.BatchNorm2d, (2, 3, 4, 5)), (tp.nn.BatchNorm3d, (2, 3, 3, 4, 4))):
        tp.manual_seed(150)
        m = cls(shape[1])
        md = cls(shape[1])
        md.load_state_dict(m.state_dict())
        md = md.to("mlu")
        x = rand(*shape, seed=151)
        m.train()
        md.train()
        close(md(x.to("mlu")), m(x))
        close(md.running_mean, m.running_mean)
        close(md.running_var, m.running_var)
        m.eval()
        md.eval()
        close(md(x.to("mlu")), m(x))


# -- autocast ----------------------------------------------------------------

AUTOCAST_DTYPES = [tp.float16, tp.bfloat16]


def test_autocast_is_available_on_the_device():
    assert tp.amp.is_autocast_available("mlu")
    assert tp.amp.is_autocast_available("cpu")


def wide(x, fast):
    """The host reference for an autocast call: the card rounds each operand to
    the fast type and accumulates from there, so widening the rounded values
    reproduces the same computation on the host."""
    return x.to(fast).to(tp.float32)


@pytest.mark.parametrize("fast", AUTOCAST_DTYPES)
def test_autocast_casts_matrix_ops(fast):
    x = rand(8, 16, seed=200)
    w = rand(16, 32, seed=201)
    with tp.autocast(device_type="mlu", dtype=fast):
        y = tp.mm(x.to("mlu"), w.to("mlu"))
    assert y.dtype == fast
    close(y, tp.mm(wide(x, fast), wide(w, fast)), fast)

    b = rand(4, 8, 16, seed=202)
    wb = w.reshape(1, 16, 32).expand(4, 16, 32).contiguous()
    with tp.autocast(device_type="mlu", dtype=fast):
        yb = tp.bmm(b.to("mlu"), wb.to("mlu"))
    assert yb.dtype == fast
    close(yb, tp.bmm(wide(b, fast), wide(wb, fast)), fast)

    bias = rand(32, seed=203)
    wt = w.t().contiguous()
    with tp.autocast(device_type="mlu", dtype=fast):
        yl = tp.nn.functional.linear(x.to("mlu"), wt.to("mlu"), bias.to("mlu"))
    assert yl.dtype == fast
    close(yl, tp.nn.functional.linear(wide(x, fast), wide(wt, fast), bias), fast)


@pytest.mark.parametrize("fast", AUTOCAST_DTYPES)
def test_autocast_casts_conv2d(fast):
    x = rand(2, 3, 8, 8, seed=210)
    w = rand(4, 3, 3, 3, seed=211)
    with tp.autocast(device_type="mlu", dtype=fast):
        y = tp.nn.functional.conv2d(x.to("mlu"), w.to("mlu"))
    assert y.dtype == fast
    close(y, tp.nn.functional.conv2d(wide(x, fast), wide(w, fast)), fast)


@pytest.mark.parametrize("fast", AUTOCAST_DTYPES)
def test_autocast_keeps_fp32_ops_in_fp32(fast):
    # The fp32 policy group runs uncast, so these keep their wide type and
    # match the plain host result.
    x = rand(4, 16, seed=220)
    with tp.autocast(device_type="mlu", dtype=fast):
        ln = tp.nn.functional.layer_norm(x.to("mlu"), [16])
        sm = tp.nn.functional.softmax(x.to("mlu"), -1)
    assert ln.dtype == tp.float32 and sm.dtype == tp.float32
    close(ln, tp.nn.functional.layer_norm(x, [16]))
    close(sm, tp.nn.functional.softmax(x, -1))


def test_autocast_disabled_subregion_runs_wide():
    x = rand(8, 16, seed=230)
    w = rand(16, 32, seed=231)
    with tp.autocast(device_type="mlu", dtype=tp.float16):
        with tp.autocast(device_type="mlu", enabled=False):
            y = tp.mm(x.to("mlu"), w.to("mlu"))
    assert y.dtype == tp.float32
    close(y, tp.mm(x, w))


def test_autocast_module_forward():
    tp.manual_seed(240)
    m = tp.nn.Linear(16, 8).to("mlu")
    x = rand(4, 16, seed=241)
    with tp.autocast(device_type="mlu", dtype=tp.float16):
        y = m(x.to("mlu"))
    assert y.dtype == tp.float16
    close(y, tp.nn.functional.linear(wide(x, tp.float16),
                                     wide(m.weight.cpu(), tp.float16),
                                     m.bias.cpu()), tp.float16)



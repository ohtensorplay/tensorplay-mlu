# tensorplay-mlu

Cambricon MLU backend for TensorPlay, built on the out-of-tree device slot
(`PrivateUse1`; see [Adding a device backend] in the TensorPlay docs).

```bash
pip install -e .                    # needs tensorplay and the Neuware toolkit
python -c "import tensorplay as tp; print(tp.ones(3, device='mlu') * 2)"
```

[Adding a device backend]: https://github.com/lexing-2026/TensorPlay/blob/main/docs/source/notes/backends.md

`import tensorplay` loads the backend through the `tensorplay.backends`
entry point when `$NEUWARE_HOME` (default `/usr/local/neuware`) holds the
toolkit.  The C++ sources under `tensorplay_mlu/csrc` compile against it the
first time and are cached under `~/.cache/tensorplay_extensions`.

## Layout

| File | Role |
| --- | --- |
| `MLURuntime.*` | errors, current device, one queue per device, per-thread CNNL handles |
| `MLUAllocator.*` | caching allocator over `cnrtMalloc`, memory statistics |
| `MLUHooks.cpp` | device count, synchronization, byte copies, memory info |
| `Cnnl.*` | tensor descriptors, workspaces, dense/strided copies, dtype rules |
| `DeviceName.cpp` | the product name the kernel library selects its kernels by (see below) |
| `CopyKernels.cpp` | `empty`, `*_like` factories, `copy_` (host/device, casts, broadcasting), `fill_`, `clone` |
| `BinaryKernels.cpp` | add/sub/mul/div with broadcasting, in-place and scalar forms |
| `UnaryKernels.cpp` | exp/log/sqrt/..., `pow` with a scalar exponent, activations and their backward kernels |
| `BlasKernels.cpp` | `mm`, `addmm`, `bmm`, `matmul`, `linear` (transposes read in place, cached algorithms; float32 uses TF32 only when `tensorplay.set_float32_matmul_precision` is not `"highest"`) |
| `ReduceKernels.cpp` | `sum`, `mean`, `amax`/`amin`, softmax and log-softmax with backward |
| `MLUProfiler.cpp` | profiler device time: a notifier pair around each operation |
| `MLUActivity.cpp` | profiler kernel-level records through CNPAPI (opened at the first traced session) |
| `Module.cpp` | C entry points: host fallbacks, memory statistics |

Operators without an MLU kernel run on the host through
`cpu_fallback::register_host_fallbacks`, so every model runs; the native
kernels cover the hot paths.  Data types CNNL does not compute in (integers,
float64) take the host path as well.

## Generic product names

CNNL selects its kernels and tiling tables by the product name the driver
reports.  Virtualized and rebranded cards can report a generic name (`mlu`;
CNNL logs "device name too short"), and CNNL then falls back to an
unknown-device configuration whose kernels fault the queue or return wrong
data without an error status: matrix products with 2 to 256 rows or above
2048 on a side, long reductions along the last axis, whole-tensor reductions
over 65536 elements, half precision elementwise operators above 32 MiB, and
every bfloat16 operator is refused.

`DeviceName.cpp` exports `cnDeviceGetName` ahead of the driver.  A name CNNL
recognizes (`MLU` followed by a digit) passes through; any other is replaced
by the series name for the chip's compute capability (`MLU590` for 5.x,
`MLU370` for 3.x).  Set `TP_MLU_DEVICE_NAME` to force a specific name.  The
replacement only takes effect when this library is loaded before anything
else that links the driver (such as another framework's MLU integration) in
the same process.

## Profiling

```python
from tensorplay.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1]) as prof:
    train_step()
print(prof.key_averages().table(sort_by="device_time_total", row_limit=10))
```

Each operation gets its MLU time (notifiers on the device's queue) and, when
`libcnpapi.so` loads, the kernels, copies and fills it launched
(`prof.kineto_results()`, exported traces).  `profile(use_device="mlu")` asks
for the same.  `tensorplay.utils.count_host_fallbacks()` lists the
operators that still run on the host.

## Tests and benchmark

The test suite compares every kernel against the host and needs a Cambricon
MLU with the Neuware toolkit:

```bash
pytest tests
```

`benchmark/bench_mlu_vs_torch.py` in the TensorPlay repository runs this
backend against torch + torch_mlu on the same card.

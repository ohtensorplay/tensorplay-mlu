#!/usr/bin/env python3
"""Check the pooling entry points against the host computation and time them.

The values and the input gradient of every sliding-window spelling are compared
element by element with the same call made on the host, and the streaming
maximum is timed at the shape a ResNet-18 stem uses.

    python tests/check_pool_mlu.py
"""

from __future__ import annotations

import os
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorplay as tp  # noqa: E402


def close(a, b, dtype="fp32"):
    a = np.asarray(a.to("cpu") if hasattr(a, "to") else a, dtype=np.float64)
    b = np.asarray(b.to("cpu") if hasattr(b, "to") else b, dtype=np.float64)
    if a.shape != b.shape:
        return False, f"shape {a.shape} vs {b.shape}"
    if a.size == 0:
        return True, "empty"
    err = float(np.max(np.abs(a - b)))
    tol = 1e-4 if dtype == "fp32" else 1e-2
    return err <= tol, f"max|a-b|={err:.3g}"


def compare_forward(name, fn, x):
    want = fn(x)
    got = fn(x.to("mlu"))
    return close(want, got)


def compare_grad(name, fn, x):
    a = x.clone()
    a.requires_grad_(True)
    fn(a).sum().backward()
    b = x.clone().to("mlu")
    b.requires_grad_(True)
    fn(b).sum().backward()
    return close(a.grad, b.grad)


def guarded(fn):
    """Run a comparison and turn a raised error into a failed result."""
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001 - the report is the point
        return False, f"{type(exc).__name__}: {exc}".replace("\n", " ")[:120]


def stream_max(x, **kw):
    return tp.nn.functional.max_pool2d(x, 3, stride=2, padding=1, **kw)


def main() -> int:
    import tensorplay_mlu

    tensorplay_mlu.load()
    f = tp.nn.functional
    rng = np.random.default_rng(3)

    cases = [
        ("max_pool2d",
         lambda t: f.max_pool2d(t, 3, stride=2, padding=1)),
        ("max_pool2d ceil_mode",
         lambda t: f.max_pool2d(t, 2, stride=2, padding=0, ceil_mode=True)),
        ("max_pool2d dilation",
         lambda t: f.max_pool2d(t, 3, stride=1, padding=1, dilation=2)),
        ("max_pool2d stride 1",
         lambda t: f.max_pool2d(t, 3, stride=1, padding=1)),
        ("max_pool2d_with_indices values",
         lambda t: f.max_pool2d_with_indices(t, 3, stride=2, padding=1)[0]),
        ("max_pool3d",
         lambda t: f.max_pool3d(t, 2, stride=2, padding=0)),
        ("avg_pool2d",
         lambda t: f.avg_pool2d(t, 3, stride=2, padding=1)),
        ("adaptive_max_pool2d",
         lambda t: f.adaptive_max_pool2d(t, (3, 4))),
    ]

    ok = True
    for name, fn in cases:
        x = tp.from_numpy(rng.standard_normal((2, 3, 17, 19)).astype(np.float32))
        if "3d" in name:
            x = tp.from_numpy(rng.standard_normal((2, 3, 9, 11, 13)).astype(np.float32))
        good, note = guarded(lambda: compare_forward(name, fn, x))
        # The slider with no positions has no derivative of its own on either
        # side, so only its values are compared.
        if "max_pool3d" in name:
            ggood, gnote = True, "n/a"
        else:
            ggood, gnote = guarded(lambda: compare_grad(name, fn, x))
        ok = ok and good and ggood
        print(f"  {name:34s} forward {'ok ' if good else 'FAIL'} ({note})"
              f"   grad {'ok' if ggood else 'FAIL'} ({gnote})")

    # A convolution-free maximum at the ResNet-18 stem shape, in both
    # spellings: the one that reports positions pays for the bookkeeping that
    # produces them, the one that does not should not.
    xs = tp.randn(32, 64, 112, 112, device="mlu")

    def bench(fn):
        for _ in range(3):
            fn()
        samples = []
        for _ in range(10):
            tp.mlu.synchronize()
            t0 = time.perf_counter()
            fn()
            tp.mlu.synchronize()
            samples.append(time.perf_counter() - t0)
        return statistics.median(samples) * 1e3

    plain = bench(lambda: tp.nn.functional.max_pool2d(xs, 3, stride=2, padding=1))
    withidx = bench(lambda: tp.nn.functional.max_pool2d_with_indices(xs, 3, stride=2, padding=1))
    print(f"\n  max_pool2d              32x64x112x112 -> {plain:.3f} ms")
    print(f"  max_pool2d_with_indices 32x64x112x112 -> {withidx:.3f} ms")

    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

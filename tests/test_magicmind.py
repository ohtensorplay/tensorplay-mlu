"""The MagicMind graph backend: settings parsing and region execution.

The settings checks need nothing but this package.  The region checks need a
Cambricon MLU, the Neuware toolkit, and the MagicMind SDK, and are skipped
when any of them is missing; every check compares the compiled result with
the same computation run eagerly.
"""

import os

import pytest

from tensorplay_mlu.compiler.settings import CompilationSettings


def _has_sdk() -> bool:
    try:
        import magicmind  # noqa: F401
    except ImportError:
        return False
    return True


# -- settings ----------------------------------------------------------------

def test_precision_short_and_long_spellings_agree():
    assert CompilationSettings(precision="fp32").precision_mode == "force_float32"
    assert (
        CompilationSettings(precision="force_float32").precision_mode
        == "force_float32"
    )
    assert CompilationSettings(precision="fp16").precision_mode == "force_float16"


def test_settings_from_options_mapping():
    settings = CompilationSettings.from_kwargs(
        {"options": {"precision": "fp16", "mlu_archs": ["mtp_372"]}}
    )
    assert settings.precision_mode == "force_float16"
    assert settings.mlu_archs == ("mtp_372",)


def test_settings_accept_frontend_keywords():
    settings = CompilationSettings.from_kwargs(
        {"mode": "default", "name": "region", "dynamic": False, "strict_native": True}
    )
    assert settings.precision == "fp32"


def test_settings_reject_unknown_option():
    with pytest.raises(TypeError):
        CompilationSettings.from_kwargs({"options": {"workspace": 1}})


def test_settings_reject_unknown_precision():
    with pytest.raises(ValueError):
        CompilationSettings(precision="int8")


def test_settings_reject_option_given_twice():
    with pytest.raises(TypeError):
        CompilationSettings.from_kwargs({"precision": "fp16", "options": {"precision": "fp16"}})


# -- backend registration and execution --------------------------------------

@pytest.mark.skipif(not _has_sdk(), reason="MagicMind SDK not installed")
def test_backend_is_discoverable_by_name():
    tp = pytest.importorskip("tensorplay")

    assert "magicmind" in tp.compiler.list_backends(include_unavailable=True)


@pytest.mark.skipif(not _has_sdk(), reason="MagicMind SDK not installed")
def test_compiled_region_matches_eager_on_mlu():
    tp = pytest.importorskip("tensorplay")
    tensorplay_mlu = pytest.importorskip("tensorplay_mlu")

    if not os.path.isfile(
        os.path.join(tensorplay_mlu.neuware_home(), "include", "cnnl.h")
    ):
        pytest.skip("Neuware toolkit not installed")
    tensorplay_mlu.load()
    if not tp.mlu.is_available():
        pytest.skip("no MLU device")

    def region(x, w, bias):
        h = tp.relu(x @ w + bias)
        return tp.softmax(h, 1)

    tp.manual_seed(0)
    x = tp.randn(8, 16, device="mlu")
    w = tp.randn(16, 32, device="mlu")
    bias = tp.randn(32, device="mlu")

    expected = region(x, w, bias)
    compiled = tp.compile(region, backend="magicmind")
    got = compiled(x, w, bias)

    assert got.device.type == "mlu"
    err = (
        (got.cpu().to(tp.float64) - expected.cpu().to(tp.float64)).abs().max().item()
    )
    assert err < 1e-3, f"max error {err}"

"""Bridge tests for the ``magicmind`` backend that need no SDK.

The bridge is the part this package owns: it assembles the build
configuration, hands the captured region to MagicMind's ONNX front end,
caches the serialized model, stages inputs, and drives the enqueue call.
Here the MagicMind bindings are replaced by a stand-in that speaks the
documented API, so the bridge runs for real on any machine -- no MLU, no
toolkit, no SDK.

What the stand-in cannot check is MagicMind itself; that needs the SDK and
lives in the MLU-only checks in ``test_magicmind.py``.  The real ONNX
hand-off is covered separately, with ``onnx.checker``.
"""

import json
import sys
import types

import pytest

np = pytest.importorskip("numpy")
tp = pytest.importorskip("tensorplay")

from tensorplay_mlu.compiler import _mm, runtime
from tensorplay_mlu.compiler.settings import CompilationSettings

# Declared by the stand-in ONNX front end; the real names come from the
# exported ONNX model, which the SDK-only check exercises.
_INPUTS = (("x", "FLOAT32", (8, 16)), ("w", "FLOAT32", (16, 32)), ("b", "FLOAT32", (32,)))
_OUTPUTS = (("y", "FLOAT32", (8, 32)),)

_DTYPES = (
    "FLOAT32",
    "FLOAT16",
    "BFLOAT16",
    "INT8",
    "INT16",
    "INT32",
    "INT64",
    "UINT8",
    "UINT16",
    "UINT32",
    "BOOL",
)


class _Ok:
    def ok(self):
        return True


class _Dims:
    def __init__(self, dims):
        self._dims = [int(d) for d in dims]

    def GetDims(self):
        return list(self._dims)


class _Namespace:
    pass


class _Network:
    def __init__(self):
        self.inputs = ()
        self.outputs = ()


class _Parser:
    def __init__(self, kind):
        self.kind = kind

    def parse(self, network, path):
        with open(path, "rb") as handle:
            self.seen_bytes = handle.read()
        network.inputs = _INPUTS
        network.outputs = _OUTPUTS
        return _Ok()


class _BuilderConfig:
    def __init__(self):
        self.fragments = []

    def parse_from_string(self, text):
        self.fragments.append(text)
        return _Ok()


class _Input:
    def __init__(self, staged):
        self._staged = staged
        self.array = None

    def from_numpy(self, array):
        self.array = array
        self._staged.append(("from_numpy", tuple(array.shape)))
        return _Ok()

    def to(self, device):
        self._staged.append(("to", None))
        return _Ok()


class _Output:
    def __init__(self, shape):
        self._shape = shape

    def asnumpy(self):
        return np.zeros(self._shape, dtype=np.float32)


class _Queue:
    def sync(self):
        return _Ok()


class _Device:
    def __init__(self, calls):
        self.id = 0
        self._calls = calls

    def active(self):
        self._calls.append(("active", self.id))
        return _Ok()

    def create_queue(self):
        return _Queue()


class _Context:
    def __init__(self, model, calls):
        self._model = model
        self._calls = calls
        self.staged = []

    def create_inputs(self):
        return [_Input(self.staged) for _ in self._model.inputs]

    def enqueue(self, inputs, outputs, queue):
        self._calls.append(("enqueue", len(inputs)))
        outputs.extend(_Output(shape) for _, _, shape in self._model.outputs)
        return _Ok()


class _Engine:
    def __init__(self, model, calls):
        self._model = model
        self._calls = calls

    def create_i_context(self):
        self._calls.append(("create_i_context", None))
        return _Context(self._model, self._calls)


class _Model:
    class EngineConfig:
        device_type = None

    def __init__(self):
        self.inputs = ()
        self.outputs = ()

    def serialize_to_file(self, path):
        with open(path, "w") as handle:
            json.dump({"inputs": self.inputs, "outputs": self.outputs}, handle)
        return _Ok()

    def deserialize_from_file(self, path):
        with open(path) as handle:
            meta = json.load(handle)
        self.inputs = tuple((n, d, tuple(s)) for n, d, s in meta["inputs"])
        self.outputs = tuple((n, d, tuple(s)) for n, d, s in meta["outputs"])
        return _Ok()

    def get_input_names(self):
        return [n for n, _, _ in self.inputs]

    def get_output_names(self):
        return [n for n, _, _ in self.outputs]

    def get_input_data_types(self):
        return [d for _, d, _ in self.inputs]

    def get_input_dimensions(self):
        return [_Dims(s) for _, _, s in self.inputs]

    def get_output_data_types(self):
        return [d for _, d, _ in self.outputs]

    def get_output_dimensions(self):
        return [_Dims(s) for _, _, s in self.outputs]

    def create_i_engine(self, config):
        engine_calls.append(("create_i_engine", config.device_type))
        return _Engine(self, engine_calls)


class _Builder:
    def build_model(self, name, network, config):
        builder_calls.append(("build_model", name, list(config.fragments)))
        model = _Model()
        model.inputs, model.outputs = network.inputs, network.outputs
        return model


class _System:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _install(monkeypatch):
    """Put a stand-in MagicMind in place of the real one."""
    calls = {"builder": builder_calls, "engine": engine_calls}

    runtime_mod = types.ModuleType("magicmind.python.runtime")
    runtime_mod.__path__ = []
    runtime_mod.Network = _Network
    runtime_mod.Parser = _Parser
    runtime_mod.BuilderConfig = _BuilderConfig
    runtime_mod.Builder = _Builder
    runtime_mod.Model = _Model
    runtime_mod.Device = lambda: _Device(engine_calls)
    runtime_mod.System = _System
    runtime_mod.DataType = data_type
    runtime_mod.ModelKind = model_kind

    parser_mod = types.ModuleType("magicmind.python.runtime.parser")
    parser_mod.Parser = _Parser
    runtime_mod.parser = parser_mod

    for name in ("magicmind", "magicmind.python"):
        package = types.ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)
    monkeypatch.setitem(sys.modules, "magicmind.python.runtime", runtime_mod)
    monkeypatch.setitem(sys.modules, "magicmind.python.runtime.parser", parser_mod)

    # The import point caches the module; drop it so the stand-in is picked up.
    monkeypatch.setattr(_mm, "_MODULE", None)
    monkeypatch.setattr(runtime, "_export_onnx", lambda gm, ei: b"onnx-bytes")
    return calls


builder_calls = []
engine_calls = []
data_type = _Namespace()
for _name in _DTYPES:
    setattr(data_type, _name, _name)
model_kind = _Namespace()
model_kind.kOnnx = "kOnnx"


def _region_inputs():
    return [
        types.SimpleNamespace(shape=(8, 16), dtype="float32"),
        types.SimpleNamespace(shape=(16, 32), dtype="float32"),
        types.SimpleNamespace(shape=(32,), dtype="float32"),
    ]


def test_build_plan_hands_the_region_to_magicmind(tmp_path, monkeypatch):
    builder_calls.clear()
    _install(monkeypatch)
    settings = CompilationSettings(cache_dir=str(tmp_path))

    plan = runtime.build_plan(
        types.SimpleNamespace(graph="region-graph"), _region_inputs(), settings
    )

    assert plan.input_names == ["x", "w", "b"]
    assert plan.output_names == ["y"]
    assert plan.io_shapes["x"] == (8, 16)
    assert plan.io_dtypes["x"] is tp.float32
    # The documented knobs are written as JSON fragments, never as attributes.
    _, name, fragments = builder_calls[0]
    assert name == "tensorplay_magicmind"
    parsed = [json.loads(f) for f in fragments]
    assert any(f.get("graph_shape_mutable") is False for f in parsed)
    assert any(
        f.get("precision_config", {}).get("precision_mode") == "force_float32"
        for f in parsed
    )


def test_build_plan_names_the_target_archs(tmp_path, monkeypatch):
    builder_calls.clear()
    _install(monkeypatch)
    settings = CompilationSettings(cache_dir=str(tmp_path), mlu_archs=("mtp_372",))

    runtime.build_plan(
        types.SimpleNamespace(graph="region-graph"), _region_inputs(), settings
    )

    _, _, fragments = builder_calls[0]
    assert any(json.loads(f).get("archs") == ["mtp_372"] for f in fragments)


def test_build_plan_reuses_the_cached_model(tmp_path, monkeypatch):
    builder_calls.clear()
    _install(monkeypatch)
    settings = CompilationSettings(cache_dir=str(tmp_path))
    graph, inputs = types.SimpleNamespace(graph="region-graph"), _region_inputs()

    runtime.build_plan(graph, inputs, settings)
    runtime.build_plan(graph, inputs, settings)

    assert len(builder_calls) == 1  # the second call read the cache
    assert len(list(tmp_path.glob("*.mm"))) == 1


def test_runner_stages_inputs_and_returns_model_outputs(tmp_path, monkeypatch):
    builder_calls.clear()
    engine_calls.clear()
    _install(monkeypatch)
    settings = CompilationSettings(cache_dir=str(tmp_path))
    plan = runtime.build_plan(
        types.SimpleNamespace(graph="region-graph"), _region_inputs(), settings
    )

    runner = runtime.Runner(plan, settings)
    x = tp.randn(8, 16)
    w = tp.randn(16, 32)
    b = tp.randn(32)
    out = runner(x, w, b)

    assert ("create_i_engine", "MLU") in engine_calls
    assert ("enqueue", 3) in engine_calls
    assert tuple(out.shape) == (8, 32)
    assert out.device.type == "cpu"  # inputs were on the host


def test_runner_rejects_the_wrong_number_of_inputs(tmp_path, monkeypatch):
    builder_calls.clear()
    _install(monkeypatch)
    settings = CompilationSettings(cache_dir=str(tmp_path))
    plan = runtime.build_plan(
        types.SimpleNamespace(graph="region-graph"), _region_inputs(), settings
    )

    with pytest.raises(TypeError):
        runtime.Runner(plan, settings)(tp.randn(8, 16))

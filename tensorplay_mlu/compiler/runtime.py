"""Model building, on-disk caching, and execution for the MagicMind backend.

The captured region is exported to ONNX and handed to the MagicMind ONNX
front end; the builder turns that network into an MLU model, which is
serialized to a content-addressed file.  The cache key covers the graph
structure, the sample shapes and dtypes, and the settings, so a rebuild only
happens when something that could change the binary changed.

A built model runs with MLU input buffers: host tensors are staged onto the
device for the call, and results return to whichever device the region's
inputs lived on.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import threading
from typing import Any, Callable

from ._mm import magicmind
from .settings import CompilationSettings

__all__ = ["ModelPlan", "Runner", "build_plan"]

log = logging.getLogger(__name__)

# Bump when the lowering changes so models built by an older conversion
# surface are never replayed against a graph they were not lowered for.
LOWERING_ABI = "1"


def _tensorplay():
    import tensorplay

    return tensorplay


def _current_device(tp: Any) -> int:
    """Index of the MLU device the region runs on."""

    module = getattr(tp, "mlu", None)
    if module is not None and hasattr(module, "current_device"):
        return int(module.current_device())
    return 0


def _dtype_names(mm: Any) -> dict[Any, Any]:
    """Map MagicMind data types onto TensorPlay dtypes.

    A data type the running MagicMind build does not expose is simply absent,
    so a model carrying it is reported as unsupported instead of crashing.
    """

    tp = _tensorplay()
    by_name = {
        "FLOAT32": tp.float32,
        "FLOAT16": tp.float16,
        "BFLOAT16": getattr(tp, "bfloat16", None),
        "INT8": tp.int8,
        "INT16": getattr(tp, "int16", None),
        "INT32": tp.int32,
        "INT64": tp.int64,
        "UINT8": tp.uint8,
        "UINT16": getattr(tp, "uint16", None),
        "UINT32": getattr(tp, "uint32", None),
        "BOOL": tp.bool,
    }
    mapping: dict[Any, Any] = {}
    for name, dtype in by_name.items():
        value = getattr(mm.DataType, name, None)
        if value is not None and dtype is not None:
            mapping[value] = dtype
    return mapping


def _graph_key(
    graph_module: Any,
    example_inputs: list[Any],
    settings: CompilationSettings,
    mm: Any,
) -> str:
    digest = hashlib.sha256()
    digest.update(f"lowering_abi={LOWERING_ABI}".encode())
    digest.update(str(graph_module.graph).encode())
    for sample in example_inputs:
        shape = tuple(int(d) for d in sample.shape)
        digest.update(f"|{shape}:{sample.dtype}".encode())
    digest.update(f"|precision={settings.precision_mode}".encode())
    digest.update(f"|archs={settings.mlu_archs}".encode())
    version = getattr(mm, "__version__", None) or getattr(mm, "version", None)
    digest.update(f"|magicmind={version}".encode())
    return digest.hexdigest()


def _export_onnx(graph_module: Any, example_inputs: list[Any]) -> bytes:
    from tensorplay import onnx as tp_onnx

    model = tp_onnx.export((graph_module, *example_inputs))
    return model.SerializeToString()


def _network_from_onnx(mm: Any, onnx_bytes: bytes) -> Any:
    from magicmind.python.runtime.parser import Parser

    network = mm.Network()
    parser = Parser(mm.ModelKind.kOnnx)
    directory = tempfile.mkdtemp(prefix="tensorplay-magicmind-")
    path = os.path.join(directory, "region.onnx")
    try:
        with open(path, "wb") as handle:
            handle.write(onnx_bytes)
        parser.parse(network, path)
    finally:
        try:
            os.remove(path)
            os.rmdir(directory)
        except OSError:
            pass
    return network


def _build_config(mm: Any, settings: CompilationSettings) -> Any:
    """The build configuration MagicMind consumes for one model.

    Every knob is written the way the builder documents it, as a JSON
    fragment handed to ``parse_from_string`` -- shapes are fixed (one
    specialization per sample shape, keyed into the cache), the target
    architectures are named, and the arithmetic precision is selected.
    """

    config = mm.BuilderConfig()
    assert config.parse_from_string('{"graph_shape_mutable":false}').ok()
    if settings.mlu_archs is not None:
        archs = json.dumps({"archs": list(settings.mlu_archs)})
        assert config.parse_from_string(archs).ok()
    precision = json.dumps(
        {"precision_config": {"precision_mode": settings.precision_mode}}
    )
    assert config.parse_from_string(precision).ok()
    return config


def _model_bytes(mm: Any, network: Any, settings: CompilationSettings) -> bytes:
    config = _build_config(mm, settings)
    builder = mm.Builder()
    model = builder.build_model("tensorplay_magicmind", network, config)
    if model is None:
        raise RuntimeError("MagicMind model build failed")
    directory = tempfile.mkdtemp(prefix="tensorplay-magicmind-")
    path = os.path.join(directory, "region.mm")
    try:
        assert model.serialize_to_file(path).ok()
        with open(path, "rb") as handle:
            return handle.read()
    finally:
        try:
            os.remove(path)
            os.rmdir(directory)
        except OSError:
            pass


def _load_or_build(settings: CompilationSettings, key: str, build: Callable[[], bytes]) -> bytes:
    if settings.cache_dir is None:
        return build()
    os.makedirs(settings.cache_dir, exist_ok=True)
    path = os.path.join(settings.cache_dir, f"{key}.mm")
    if os.path.exists(path):
        log.debug("magicmind model cache hit: %s", path)
        with open(path, "rb") as handle:
            return handle.read()
    blob = build()
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "wb") as handle:
        handle.write(blob)
    os.replace(tmp, path)
    return blob


class ModelPlan:
    """A serialized MagicMind model plus the metadata execution needs."""

    def __init__(
        self,
        blob: bytes,
        input_names: list[str],
        output_names: list[str],
        io_dtypes: dict[str, Any],
        io_shapes: dict[str, tuple[int, ...]],
    ) -> None:
        self.blob = blob
        self.input_names = input_names
        self.output_names = output_names
        self.io_dtypes = io_dtypes
        self.io_shapes = io_shapes


def _io_metadata(mm: Any, model: Any) -> tuple[list[str], list[str], dict[str, Any], dict[str, tuple[int, ...]]]:
    dtypes = _dtype_names(mm)
    input_names = list(model.get_input_names())
    output_names = list(model.get_output_names())
    names = input_names + output_names
    io_dtypes: dict[str, Any] = {}
    io_shapes: dict[str, tuple[int, ...]] = {}
    for name, data_type, dims in zip(
        input_names, model.get_input_data_types(), model.get_input_dimensions()
    ):
        io_dtypes[name] = dtypes.get(data_type)
        io_shapes[name] = tuple(int(d) for d in dims.GetDims())
    for name, data_type, dims in zip(
        output_names, model.get_output_data_types(), model.get_output_dimensions()
    ):
        io_dtypes[name] = dtypes.get(data_type)
        io_shapes[name] = tuple(int(d) for d in dims.GetDims())
    for name in names:
        if io_dtypes[name] is None:
            raise RuntimeError(
                f"the magicmind backend does not support the data type of "
                f"tensor {name!r}"
            )
    return input_names, output_names, io_dtypes, io_shapes


def build_plan(
    graph_module: Any,
    example_inputs: list[Any],
    settings: CompilationSettings,
) -> ModelPlan:
    """Lower the region, build (or load) the model, and read its I/O metadata."""

    mm = magicmind()
    onnx_bytes = _export_onnx(graph_module, example_inputs)
    network = _network_from_onnx(mm, onnx_bytes)
    key = _graph_key(graph_module, example_inputs, settings, mm)
    blob = _load_or_build(settings, key, lambda: _model_bytes(mm, network, settings))
    return _io_plan(mm, blob)


def _io_plan(mm: Any, blob: bytes) -> ModelPlan:
    model = mm.Model()
    _deserialize(mm, model, blob)
    input_names, output_names, io_dtypes, io_shapes = _io_metadata(mm, model)
    return ModelPlan(blob, input_names, output_names, io_dtypes, io_shapes)


def _deserialize(mm: Any, model: Any, blob: bytes) -> None:
    directory = tempfile.mkdtemp(prefix="tensorplay-magicmind-")
    path = os.path.join(directory, "region.mm")
    try:
        with open(path, "wb") as handle:
            handle.write(blob)
        assert model.deserialize_from_file(path).ok()
    finally:
        try:
            os.remove(path)
            os.rmdir(directory)
        except OSError:
            pass


def _to_numpy(value: Any):
    import numpy as np

    if isinstance(value, np.ndarray):
        return value
    tensor = value
    if getattr(tensor, "requires_grad", False):
        tensor = tensor.detach()
    device = getattr(tensor, "device", None)
    if device is not None and str(getattr(device, "type", device)) != "cpu":
        tensor = tensor.cpu()
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    return tensor.numpy()


class Runner:
    """Executable wrapper around one built MagicMind model."""

    def __init__(self, plan: ModelPlan, settings: CompilationSettings) -> None:
        tp = _tensorplay()
        mm = magicmind()

        self._plan = plan
        self._settings = settings
        self._tp = tp
        self._mm = mm
        self._name = "magicmind"
        # Visible through the compiled region's cache so callers (and tests)
        # can tell a MagicMind product from an interpreter fallback.
        self._tensorplay_codegen = "magicmind"

        model = mm.Model()
        _deserialize(mm, model, plan.blob)
        with mm.System():
            device = mm.Device()
            device.id = _current_device(tp)
            if not device.active().ok():
                raise RuntimeError("activating the MLU device for MagicMind failed")
            config = mm.Model.EngineConfig()
            config.device_type = "MLU"
            engine = model.create_i_engine(config)
            if engine is None:
                raise RuntimeError("creating a MagicMind engine failed")
            context = engine.create_i_context()
            if context is None:
                raise RuntimeError("creating a MagicMind context failed")
            queue = device.create_queue()
            if queue is None:
                raise RuntimeError("creating a MagicMind queue failed")
        self._model = model
        self._device = device
        self._engine = engine
        self._context = context
        self._queue = queue
        self._inputs = context.create_inputs()
        # One context and one queue per runner: calls are serialized so a
        # region reached from several threads stays correct.
        self._lock = threading.Lock()

    def __call__(self, *args: Any) -> Any:
        plan = self._plan
        if len(args) != len(plan.input_names):
            raise TypeError(
                f"{self._name} region expects {len(plan.input_names)} "
                f"input(s), got {len(args)}"
            )

        on_device = any(
            str(getattr(getattr(value, "device", None), "type", "cpu")) != "cpu"
            for value in args
        )
        target = "mlu" if on_device else "cpu"

        with self._lock:
            for index, value in enumerate(args):
                name = plan.input_names[index]
                tensor = value
                if str(getattr(getattr(tensor, "device", None), "type", "cpu")) != "cpu":
                    tensor = tensor.cpu()
                if getattr(tensor, "dtype", None) != plan.io_dtypes[name]:
                    tensor = tensor.to(plan.io_dtypes[name])
                self._inputs[index].from_numpy(_to_numpy(tensor))
                self._inputs[index].to(self._device)

            outputs: list[Any] = []
            assert self._context.enqueue(self._inputs, outputs, self._queue).ok()
            assert self._queue.sync().ok()

            results = []
            for index, name in enumerate(plan.output_names):
                tensor = self._tp.from_numpy(outputs[index].asnumpy())
                if target != "cpu":
                    tensor = tensor.to(target)
                results.append(tensor)
        return results[0] if len(results) == 1 else tuple(results)

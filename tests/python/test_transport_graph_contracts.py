"""CUDA graph preparation boundaries and dependency-lazy public imports."""

import importlib
import subprocess
import sys
from typing import Any, cast

import pytest

from dpt.transport.graph import _binding_identity, prepare_transport_graph
from dpt.transport.model import TransportError
from dpt.transport.rng import HistoryBatch
from helpers.transport_graph_inputs import graph_scene

wp: Any = importlib.import_module("warp")

# Binding fingerprints are shared by preparation and replay, including CPU
# arrays used to stage checked diagnostics.
# pyright: reportPrivateUsage=false


def test_graph_import_keeps_optional_runtime_dependencies_lazy() -> None:
    code = """
import sys
import dpt.transport.graph
assert not {'numpy', 'torch', 'warp'}.intersection(sys.modules)
"""
    subprocess.run([sys.executable, "-I", "-c", code], check=True)


def test_graph_binding_accepts_native_empty_arrays_and_detects_real_rebinding() -> None:
    workspace, density, _ = graph_scene("cpu")
    # The absent optional voxel-density map is a legal native Warp empty array.
    # Warp represents this buffer with a null pointer rather than integer zero.
    empty = workspace.transport.voxel_density
    assert empty.size == 0 and empty.ptr is None
    identity = _binding_identity(workspace, density)
    assert (0, tuple(empty.shape), empty.dtype, str(empty.device)) in identity
    assert _binding_identity(workspace, density) == identity

    # Keep the original allocation alive so allocator reuse cannot give a new
    # array its address. Same shape/dtype/device must still detect a new pointer.
    original = workspace.mean
    replacement = wp.empty_like(original)
    assert replacement.ptr != original.ptr
    workspace.mean = replacement
    assert _binding_identity(workspace, density) != identity
    workspace.mean = original
    assert _binding_identity(workspace, density) == identity


def test_graph_requires_cuda_without_running_warm_simulation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace, density, source = graph_scene("cpu")

    def unexpected(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("CPU request must fail before warm simulation")

    monkeypatch.setattr("dpt.transport.graph.simulate", unexpected)
    with pytest.raises(TransportError, match="requires a CUDA device"):
        prepare_transport_graph(
            density, source=source, batch=HistoryBatch(17, 0, 257), workspace=workspace
        )


@pytest.mark.parametrize("wrong", ["workspace", "batch", "source"])
def test_graph_rejects_invalid_typed_bindings(wrong: str) -> None:
    workspace, density, source = graph_scene("cpu")
    with pytest.raises(TransportError):
        prepare_transport_graph(
            density,
            source=cast(Any, object()) if wrong == "source" else source,
            batch=cast(Any, object()) if wrong == "batch" else HistoryBatch(17, 0, 257),
            workspace=cast(Any, object()) if wrong == "workspace" else workspace,
        )

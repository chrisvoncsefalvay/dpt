"""Independent geometry and analytic-box checks for the benchmark adapters."""

import importlib
import importlib.util
import math
import sys
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def reference():
    path = Path(__file__).resolve().parents[2] / "benchmarks" / "reference.py"
    spec = importlib.util.spec_from_file_location("dpt_reference_benchmark", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_vector_geometry_matches_every_pixel(reference):
    case = reference.Case(16, 5, 7, 1)
    for panel in case.panels:
        vector = reference.panel_vectors(panel)
        assert vector[:3] == panel.source_mm
        for row, column in ((0, 0), (3, 4), (6, 6)):
            expected = panel.pixel_centre(row, column)
            actual = tuple(
                vector[3 + k] + (column - 3) * vector[6 + k] + (row - 3) * vector[9 + k]
                for k in range(3)
            )
            assert actual == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize(
    ("source", "target", "expected"),
    [
        ((-20, 0, 0), (20, 0, 0), 16),
        ((-20, -20, 0), (20, 20, 0), 16 * math.sqrt(2)),
        ((-20, 10, 0), (20, 10, 0), 0),
        ((0, 0, 0), (4, 0, 0), 4),
    ],
)
def test_finite_segment_cube_oracle(reference, source, target, expected):
    assert reference.box_depth(source, target, 16) == pytest.approx(expected)


@pytest.mark.parametrize("kwargs", [{"size": 513}, {"views": 0}, {"detector": 257}, {"repeats": 0}])
def test_bounded_case_rejects_excessive_work(reference, kwargs):
    with pytest.raises(ValueError):
        reference.Case(**kwargs)


def test_dpt_multiview_host_results_are_independent_and_adjoint(reference):
    importlib.import_module("numpy")
    importlib.import_module("warp")
    result = reference.run("dpt", reference.Case(8, 3, 3, 1), "cpu")
    assert result["relative_adjoint_defect"] < 2e-5
    assert result["relative_forward_fd_error"] < 2e-4
    assert result["relative_backprojection_gradient_error"] < 2e-4
    assert result["warmup_calls_per_operation"] == 2
    assert len(result["warm_forward_seconds"]) == len(result["warm_backward_seconds"]) == 1


def test_timing_completes_before_clock_and_inside_each_interval(reference, monkeypatch):
    events = []
    ticks = iter((10.0, 12.0, 20.0, 23.0))

    def clock():
        events.append("clock")
        return next(ticks)

    def complete():
        events.append("complete")

    def call():
        events.append("call")
        return "result"

    def validate(result):
        events.append("validate")
        assert result == "result"

    monkeypatch.setattr(reference.time, "perf_counter", clock)
    result, durations = reference.timed(call, 2, synchronise=complete, validate=validate)
    assert result == "result"
    assert durations == [2.0, 3.0]
    assert events == ["complete", "clock", "call", "complete", "clock", "validate"] * 2


def test_timing_rejects_invalid_intermediate_result(reference):
    results = iter((None, "valid"))

    def validate(result):
        if result is None:
            raise RuntimeError("invalid intermediate result")

    with pytest.raises(RuntimeError, match="invalid intermediate result"):
        reference.timed(lambda: next(results), 2, synchronise=lambda: None, validate=validate)
    assert next(results) == "valid"


@pytest.mark.parametrize("failure_at", [1, 2])
def test_completion_failure_stops_timing(reference, failure_at):
    completions = 0
    calls = 0

    def complete():
        nonlocal completions
        completions += 1
        if completions == failure_at:
            raise RuntimeError("completion failed")

    def call():
        nonlocal calls
        calls += 1

    with pytest.raises(RuntimeError, match="completion failed"):
        reference.timed(call, 2, synchronise=complete)
    assert calls == failure_at - 1


def test_cpu_reference_completion_needs_no_cuda_import(reference, monkeypatch):
    def unexpected_import(name):
        pytest.fail(f"CPU completion imported {name}")

    monkeypatch.setattr(reference.importlib, "import_module", unexpected_import)
    reference.reference_completion("cpu")()


@pytest.mark.parametrize("is_cuda,is_primary", [(False, False), (True, False)])
def test_reference_completion_rejects_wrong_context(reference, monkeypatch, is_cuda, is_primary):
    from types import SimpleNamespace

    device = SimpleNamespace(is_cuda=is_cuda, is_primary=is_primary)
    wp = SimpleNamespace(get_device=lambda name: device)
    monkeypatch.setattr(reference.importlib, "import_module", lambda name: wp)
    with pytest.raises(RuntimeError, match="CUDA primary context"):
        reference.reference_completion("cuda:0")


def test_reference_completion_uses_selected_primary_context(reference, monkeypatch):
    from types import SimpleNamespace

    completed = []
    device = SimpleNamespace(is_cuda=True, is_primary=True)
    wp = SimpleNamespace(get_device=lambda name: device, synchronize_device=completed.append)
    monkeypatch.setattr(reference.importlib, "import_module", lambda name: wp)
    reference.reference_completion("cuda:0")()
    assert completed == [device]


def test_off_centre_gaussian_oracle_against_independent_midpoint_quadrature(reference):
    source, target, side = (-40.0, 1.7, 0.2), (40.0, -0.3, 1.1), 16
    centre, sigma = reference.gaussian_parameters(side)
    steps = 20000
    # Integrate on the full source-to-detector segment, explicitly masking the
    # cube; this oracle uses no interval solver or Gaussian antiderivative.
    total = 0.0
    for index in range(steps):
        t = (index + 0.5) / steps
        point = tuple(a + t * (b - a) for a, b in zip(source, target, strict=True))
        if all(-side / 2 < value < side / 2 for value in point):
            exponent = sum(
                ((value - c) / s) ** 2 for value, c, s in zip(point, centre, sigma, strict=True)
            )
            total += 0.02 * math.exp(-0.5 * exponent)
    expected = total * math.dist(source, target) / steps
    assert reference.gaussian_depth(source, target, side) == pytest.approx(expected, rel=1e-7)

"""Prepared evaluator epilogues transfer controls and delivered status together."""

import importlib
from contextlib import nullcontext
from typing import Any

import numpy as np
import pytest

from dpt._runtime import HostReadback, cpu_test_context
from dpt.contracts import ContractError, NumericalError
from dpt.material_reconstruction import MaterialReconstruction
from helpers import material_inputs as material_cases
from helpers.evaluators import _evaluate, _evaluator

wp: Any = importlib.import_module("warp")


@pytest.mark.parametrize("kind", ["primary", "spectral", "material"])
def test_prepared_completion_has_one_fp64_control_and_status_readback(
    kind, warp_device, monkeypatch
):
    with cpu_test_context() if warp_device == "cpu" else nullcontext():
        evaluator, parameters = _evaluator(kind, warp_device)
        expected = _evaluate(evaluator, parameters)
        reads: list[tuple[Any, int]] = []
        original = HostReadback.read

        def record(reader: HostReadback) -> Any:
            reads.append((reader.source.dtype, reader.source.size))
            return original(reader)

        waits: list[Any] = []
        original_wait = wp.synchronize_stream

        def record_wait(stream):
            waits.append(stream)
            original_wait(stream)

        monkeypatch.setattr(HostReadback, "read", record)
        monkeypatch.setattr(wp, "synchronize_stream", record_wait)
        actual = _evaluate(evaluator, parameters)
        if parameters is None:
            np.testing.assert_allclose(actual, expected, rtol=1e-12)
        else:
            np.testing.assert_allclose(actual.loss, expected.loss, rtol=1e-12)
            np.testing.assert_allclose(actual.gradient, expected.gradient, rtol=1e-7, atol=1e-9)
        assert sum(dtype == wp.float64 for dtype, _ in reads) == 1
        assert reads[-1][0] == wp.float64
        if kind in ("primary", "spectral"):
            # One entry snapshot guards pending unchecked work; one epilogue
            # carries all output controls and the newly produced diagnostics.
            assert len(reads) == 2
            assert waits == (
                [evaluator.context.stream] * 2 if evaluator.context.device.is_cuda else []
            )
        minimum_controls = (
            7 if kind == "primary" else 10 * len(evaluator._prepared) if kind == "spectral" else 3
        )
        assert reads[-1][1] > minimum_controls


def test_spectral_completion_delivers_all_new_failures_and_does_not_erase_pending_errors(
    warp_device, monkeypatch
):
    from dpt import spectral_recovery

    with cpu_test_context() if warp_device == "cpu" else nullcontext():
        evaluator, parameters = _evaluator("spectral", warp_device)
        expected = evaluator(parameters)
        previous_pose = evaluator._pose_device.numpy().copy()
        evaluator._prepared[-1].material.projection._status.fill_(8)
        with pytest.raises(NumericalError) as pending:
            evaluator(tuple(0.0 for _ in parameters))
        assert pending.value.__dict__["status_diagnostics"]
        np.testing.assert_array_equal(evaluator._pose_device.numpy(), previous_pose)
        assert evaluator(parameters) == expected

        original = spectral_recovery.material_projection_vjp
        calls = 0

        def produce_failures(*args, **kwargs):
            nonlocal calls
            original(*args, **kwargs)
            calls += 1
            if calls == len(evaluator._prepared):
                evaluator._prepared[0].spectral._status.fill_(1)
                evaluator._prepared[-1].material.projection._status.fill_(8)

        monkeypatch.setattr(spectral_recovery, "material_projection_vjp", produce_failures)
        with pytest.raises((ContractError, NumericalError)) as delivered:
            evaluator(parameters)
        assert len(delivered.value.__dict__["status_diagnostics"]) == 2
        assert all(code != 0 for _, code, _ in delivered.value.__dict__["status_diagnostics"])
        monkeypatch.setattr(spectral_recovery, "material_projection_vjp", original)
        repaired = evaluator(parameters)
        np.testing.assert_allclose(repaired.loss, expected.loss, rtol=1e-12)
        np.testing.assert_allclose(repaired.gradient, expected.gradient, rtol=1e-7, atol=1e-9)


def test_material_deferred_gradient_checks_keep_final_overflow_delivery_and_retry(
    warp_device, monkeypatch
):
    from dpt import material_reconstruction

    with cpu_test_context() if warp_device == "cpu" else nullcontext():
        inputs = material_cases._inputs("float64", regularisation=0)
        inputs["device"] = warp_device
        solver = MaterialReconstruction(**inputs)
        expected = solver.evaluate()
        original = material_reconstruction.material_projection_vjp
        checked_flags = []

        def accumulate_then_overflow(*args, **kwargs):
            checked_flags.append(kwargs["check_volume_gradient"])
            assert kwargs["accumulate"] is True and kwargs["validate"] is False
            original(*args, **kwargs)
            if len(checked_flags) == len(solver.views):
                # Reproduce an atomic accumulation overflow after the last
                # projector. Zero beta must still check the assembled gradient.
                solver.gradient.fill_(float("inf"))

        monkeypatch.setattr(
            material_reconstruction, "material_projection_vjp", accumulate_then_overflow
        )
        with pytest.raises(NumericalError) as failure:
            solver.evaluate()
        assert checked_flags == [False] * len(solver.views)
        assert any(
            owner == "MaterialReconstruction"
            for owner, _, _ in failure.value.__dict__["status_diagnostics"]
        )
        monkeypatch.setattr(material_reconstruction, "material_projection_vjp", original)
        assert solver.evaluate() == expected
        assert np.isfinite(solver.gradient.numpy()).all()

"""A checked call must observe rather than erase a preceding unchecked failure."""

import importlib
from functools import partial
from typing import Any

import pytest

from dpt._runtime import cpu_test_context
from dpt.contracts import ContractError, NumericalError
from dpt.geometry import DetectorGeometry, RigidTransform
from dpt.objectives import ObjectiveSpec, evaluate_objective, prepare_objective
from dpt.projection import ProjectionSpec, prepare_projection, project_optical_depth
from dpt.transmission import TransmissionSpec, prepare_transmission, transmission_vjp, transmit
from dpt.volumes import GridSpec

wp: Any = importlib.import_module("warp")


def test_transmission_keeps_unchecked_failure_until_explicit_clear():
    with cpu_test_context():
        w = prepare_transmission(TransmissionSpec(beam="scalar"), max_pixels=1, device="cpu")
        x = wp.zeros(1, dtype=wp.float32, device="cpu")
        out = wp.empty_like(x)
        seed = wp.array([3e38], dtype=wp.float32, device="cpu")
        transmission_vjp(
            x, n0=3e38, cotangent_counts=seed, out_grad_L=out, workspace=w, validate=False
        )
        x.assign([1.0])
        with pytest.raises((ContractError, NumericalError)):
            transmit(x, n0=1.0, out_T=out, workspace=w)
        with pytest.raises((ContractError, NumericalError)):
            w.check_status()
        w.clear_status()
        transmit(x, n0=1.0, out_T=out, workspace=w)


def test_objective_keeps_unchecked_failure_until_explicit_clear():
    with cpu_test_context():
        w = prepare_objective(ObjectiveSpec(), max_pixels=1, device="cpu")
        x = wp.array([float("nan")], dtype=wp.float32, device="cpu")
        y = wp.zeros(1, dtype=wp.float32, device="cpu")
        loss = wp.empty(1, dtype=wp.float64, device="cpu")
        evaluate_objective(x, y, workspace=w, out_loss=loss, validate=False)
        x.assign([1.0])
        with pytest.raises((ContractError, NumericalError)):
            evaluate_objective(x, y, workspace=w, out_loss=loss)
        with pytest.raises((ContractError, NumericalError)):
            w.check_status()


def test_projection_keeps_unchecked_failure_until_explicit_clear():
    with cpu_test_context():
        grid = GridSpec((2, 2, 2), (1.0, 1.0, 1.0), (-0.5, -0.5, -0.5))
        geom = DetectorGeometry(
            (-2.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 1.0), (1, 1)
        )
        w = prepare_projection(grid, geom, ProjectionSpec(8), device="cpu")
        x = wp.array([float("nan")] * 8, dtype=wp.float32, device="cpu")
        pose = wp.array(RigidTransform().packed(), dtype=wp.float64, device="cpu")
        out = wp.empty(1, dtype=wp.float32, device="cpu")
        project_optical_depth(x, pose, workspace=w, out_L=out, validate=False)
        x.assign([1.0] * 8)
        with pytest.raises((ContractError, NumericalError)):
            project_optical_depth(x, pose, workspace=w, out_L=out)
        with pytest.raises((ContractError, NumericalError)):
            w.check_status()


def test_spectral_keeps_unchecked_failure_until_explicit_clear():
    from dpt.materials import Provenance
    from dpt.spectral import SpectralSpec, prepare_spectral, spectral_signal

    provenance = Provenance("analytic:status", "0" * 64, "Apache-2.0", "test")
    with cpu_test_context():
        w = prepare_spectral(
            SpectralSpec(1, 1, (provenance,), provenance, provenance, "count", "test"),
            max_pixels=1,
            device="cpu",
        )
        paths = wp.zeros(1, dtype=wp.float32, device="cpu")
        coefficient = wp.ones(1, dtype=wp.float32, device="cpu")
        weights = wp.full(1, 3e38, dtype=wp.float32, device="cpu")
        response = wp.full(1, 3e38, dtype=wp.float32, device="cpu")
        out = wp.empty_like(paths)
        spectral_signal(
            paths, coefficient, weights, response, out_mean=out, workspace=w, validate=False
        )
        weights.fill_(1)
        response.fill_(1)
        with pytest.raises(NumericalError):
            spectral_signal(paths, coefficient, weights, response, out_mean=out, workspace=w)
        with pytest.raises(NumericalError):
            w.check_status()


def test_detector_keeps_unchecked_failure_until_explicit_clear():
    from dpt.detector import CalibrationSpec, calibrate, prepare_detector

    with cpu_test_context():
        w = prepare_detector(max_pixels=1, device="cpu")
        x = wp.full(1, 3e38, dtype=wp.float32, device="cpu")
        gain = wp.full(1, 3e38, dtype=wp.float32, device="cpu")
        exposure = wp.ones(1, dtype=wp.float32, device="cpu")
        offset = wp.zeros(1, dtype=wp.float32, device="cpu")
        out = wp.empty_like(x)
        spec = CalibrationSpec("test")
        calibrate(x, gain, exposure, offset, out_signal=out, spec=spec, workspace=w, validate=False)
        x.fill_(1)
        gain.fill_(1)
        with pytest.raises(NumericalError):
            calibrate(x, gain, exposure, offset, out_signal=out, spec=spec, workspace=w)
        with pytest.raises(NumericalError):
            w.check_status()


def test_material_projection_keeps_unchecked_failure_until_explicit_clear():
    from dpt.material_projection import prepare_material_projection, project_material_paths

    with cpu_test_context():
        grid = GridSpec((2, 2, 2), (1.0, 1.0, 1.0), (-0.5, -0.5, -0.5))
        geom = DetectorGeometry(
            (-2.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 1.0), (1, 1)
        )
        fields = wp.full(8, float("nan"), dtype=wp.float32, device="cpu")
        paths = wp.empty(1, dtype=wp.float32, device="cpu")
        pose = wp.array(RigidTransform().packed(), dtype=wp.float64, device="cpu")
        w = prepare_material_projection(
            grid, geom, ProjectionSpec(8), materials=1, fields=fields, out_paths=paths, device="cpu"
        )
        project_material_paths(pose, workspace=w, validate=False)
        fields.fill_(0.5)
        with pytest.raises(NumericalError):
            project_material_paths(pose, workspace=w)
        with pytest.raises(NumericalError):
            w.check_status()


@pytest.mark.parametrize("checked", [True, False])
def test_prediction_domain_error_class_matches_fused_and_direct(checked):
    from dpt.contracts import TrialDomainError
    from dpt.objectives import evaluate_primary_objective

    with cpu_test_context():
        for fused in (False, True):
            w = prepare_objective(
                ObjectiveSpec(kind="poisson", domain="counts", precision="float64"),
                max_pixels=1,
                device="cpu",
            )
            prediction = wp.full(1, -1.0, dtype=wp.float64, device="cpu")
            observation = wp.ones(1, dtype=wp.float32, device="cpu")
            loss = wp.empty(1, dtype=wp.float64, device="cpu")
            if fused:
                depth = wp.full(1, -1.0, dtype=wp.float64, device="cpu")
                seed = wp.empty_like(depth)
                call = partial(
                    evaluate_primary_objective,
                    depth,
                    observation,
                    out_prediction=prediction,
                    out_loss=loss,
                    out_depth_cotangent=seed,
                    workspace=w,
                    validate=checked,
                )
            else:
                call = partial(
                    evaluate_objective,
                    prediction,
                    observation,
                    out_loss=loss,
                    workspace=w,
                    validate=checked,
                )
            if checked:
                with pytest.raises(TrialDomainError):
                    call()
            else:
                call()
            with pytest.raises(TrialDomainError):
                w.check_status()

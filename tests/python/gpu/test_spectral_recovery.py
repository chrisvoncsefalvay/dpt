"""CUDA integration checks for composed gradients and shared views.

Fields and observations are analytic test fixtures.
"""

import hashlib
import importlib
import math
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import pytest

from dpt.contracts import ContractError, NumericalError
from dpt.detector import BlurSpec
from dpt.geometry import DetectorGeometry, RigidTransform
from dpt.materials import Provenance
from dpt.objectives import ObjectiveSpec
from dpt.recovery import PoseChart
from dpt.spectral import SpectralSpec
from dpt.spectral_inputs import read_spectral_inputs
from dpt.spectral_recovery import (
    CalibrationBlock,
    SpectralPoseEvaluator,
    SpectralPoseProblem,
    SpectralView,
)
from dpt.volumes import GridSpec

# pytest.approx supplies dynamic third-party stubs.
# pyright: reportUnknownMemberType=false
wp: Any = importlib.import_module("warp")
FIXTURE = Provenance(
    source="analytic: a(x,y,z)=0.15+0.02*x+0.03*y+0.01*z at grid sample centres",
    sha256=hashlib.sha256(
        b"a=.15+.02*x+.03*y+.01*z; mu=(.02,.04); photons=(100,200); zero observation"
    ).hexdigest(),
    rights="Original analytic integration fixture; Apache-2.0",
    description="Not anatomy or detector measurements",
)
pytestmark = pytest.mark.gpu


def test_cached_upload_keeps_actual_shape_when_views_have_equal_pixel_counts(
    tmp_path: Path,
) -> None:
    np: Any = importlib.import_module("numpy")
    archive = tmp_path / "analytic.npz"
    np.savez(
        archive,
        fields=np.zeros((1, 1, 1, 1), dtype=np.float32),
        coefficients=np.ones((1, 2), dtype=np.float32),
        weights=np.ones(2, dtype=np.float32),
        response=np.ones(2, dtype=np.float32),
        observation=np.zeros((2, 3), dtype=np.float32),
    )
    geometry = DetectorGeometry(
        (0.0, 0.0, -10.0),
        (0.0, 0.0, 10.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (1.0, 1.0),
        (2, 3),
    )
    first = {
        "name": "first",
        "geometry": asdict(geometry),
        "energies_kev": [40.0, 80.0],
        "coefficients": "coefficients",
        "coefficients_unit": "mm^-1",
        "coefficients_provenance": [asdict(FIXTURE)],
        "weights": "weights",
        "spectrum_provenance": asdict(FIXTURE),
        "response": "response",
        "response_provenance": asdict(FIXTURE),
        "output_unit": "signal",
        "input_description": "analytic shape validation only",
        "observation": "observation",
        "observation_provenance": asdict(FIXTURE),
        "calibration_group": "shared",
        "objective": asdict(ObjectiveSpec()),
    }
    metadata = {
        "schema_version": 1,
        "grid": asdict(GridSpec((1, 1, 1), (1.0, 1.0, 1.0))),
        "materials": 1,
        "fields": "fields",
        "fields_provenance": asdict(FIXTURE),
        "calibration": [asdict(CalibrationBlock("shared"))],
        "views": [
            first,
            {**first, "name": "second", "geometry": asdict(replace(geometry, shape=(3, 2)))},
        ],
    }
    with pytest.raises(ContractError, match="incompatible role shape"):
        read_spectral_inputs(archive, metadata, samples_per_ray=8)


def array(values: list[float] | tuple[float, ...]) -> Any:
    return wp.array(values, dtype=wp.float32, device="cuda:0")


def problem(*, spread: bool = False) -> SpectralPoseProblem:
    grid = GridSpec((4, 4, 4), (1.0, 1.0, 1.0), (-1.5, -1.5, -1.5))
    fields = array(
        [
            0.15 + 0.02 * x + 0.03 * y + 0.01 * z
            for z in range(4)
            for y in range(4)
            for x in range(4)
        ]
    )
    geometry = DetectorGeometry(
        (0.3, -0.7, -10.0), (-1.1, -0.8, 10.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.7, 0.8), (3, 4)
    )
    spec = SpectralSpec(
        1, 2, (FIXTURE,), FIXTURE, FIXTURE, "signal", "analytic fixed-density fractions"
    )
    first = SpectralView(
        "first",
        geometry,
        spec,
        array((0.02, 0.04)),
        array((100.0, 200.0)),
        array((1.0, 2.0)),
        array([0.0] * 12),
        "shared",
        FIXTURE,
        ObjectiveSpec(reduction="mean"),
        spatial_response=BlurSpec(3, 4, 1, 3, (0.125, 0.5, 0.375), FIXTURE) if spread else None,
    )
    second = replace(
        first,
        name="second",
        geometry=replace(geometry, source_mm=(-1.0, 0.7, -8.0)),
        objective_weight=0.25,
    )
    return SpectralPoseProblem(
        grid,
        1,
        fields,
        FIXTURE,
        (first, second),
        (CalibrationBlock("shared", fit_scale="exposure", fit_offset=True),),
        64,
    )


@pytest.mark.parametrize("spread", [False, True])
def test_joint_view_gradient_is_sum_in_one_shared_parameter_chart(spread: bool) -> None:
    data = problem(spread=spread)
    chart = PoseChart(anchor=RigidTransform(translation_mm=(0.13, -0.17, 0.19)))
    coordinates = (0.03, -0.02, 0.01, 0.1, -0.2, 0.3, 0.05, -0.4)
    joint = SpectralPoseEvaluator(data, chart)(coordinates)
    separate = [
        SpectralPoseEvaluator(replace(data, views=(view,)), chart)(coordinates)
        for view in data.views
    ]
    assert joint.loss == pytest.approx(math.fsum(value.loss for value in separate), rel=1e-12)
    assert joint.gradient == pytest.approx(
        tuple(math.fsum(value.gradient[j] for value in separate) for j in range(8)),
        rel=2e-6,
        abs=1e-8,
    )


def test_shared_prior_contributes_once_for_multiple_views() -> None:
    data = problem()
    chart = PoseChart(anchor=RigidTransform(translation_mm=(0.13, -0.17, 0.19)))
    point = (0.0,) * 6 + (0.5, -2.0)
    baseline = SpectralPoseEvaluator(data, chart)(point)
    regularised = replace(
        data,
        calibration=(
            replace(data.calibration[0], scale_prior_precision=3, offset_prior_precision=2),
        ),
    )
    value = SpectralPoseEvaluator(regularised, chart)(point)
    assert value.loss - baseline.loss == pytest.approx(4.375, rel=1e-10)
    assert value.gradient[-2] - baseline.gradient[-2] == pytest.approx(1.5, rel=1e-9)
    assert value.gradient[-1] - baseline.gradient[-1] == pytest.approx(-4.0, rel=1e-9)


def test_nuisance_derivatives_agree_with_independent_central_differences() -> None:
    evaluator = SpectralPoseEvaluator(problem(spread=True), PoseChart())
    point = (0.0,) * 6 + (0.05, -0.4)
    value = evaluator(point)
    for axis in (6, 7):
        errors: list[float] = []
        for step in (0.1, 0.03, 0.01):
            plus, minus = list(point), list(point)
            plus[axis] += step
            minus[axis] -= step
            derivative = (evaluator(tuple(plus)).loss - evaluator(tuple(minus)).loss) / (2 * step)
            errors.append(abs(derivative - value.gradient[axis]))
        assert min(errors) <= 0.003 * max(1.0, abs(value.gradient[axis]))


def test_rejected_trial_and_ambient_tape_do_not_reuse_pending_staging() -> None:
    evaluator = SpectralPoseEvaluator(problem(), PoseChart())
    zero = (0.0,) * evaluator.dimension
    first = evaluator(zero)
    invalid = list(zero)
    invalid[6] = 10000.0
    with pytest.raises(NumericalError):
        evaluator(tuple(invalid))
    with wp.Tape(), pytest.raises(ContractError, match="ambient tape"):
        evaluator(zero)
    repeated = evaluator(zero)
    assert repeated == first


@pytest.mark.parametrize(
    "role", ["fields", "coefficients", "weights", "response", "observation", "objective_weights"]
)
def test_composed_preparation_rejects_invalid_fixed_data_via_public_validators(role: str) -> None:
    data = problem()
    first = data.views[0]
    if role == "fields":
        data.fields.fill_(float("nan"))
    elif role == "objective_weights":
        first = replace(
            first,
            objective=replace(first.objective, weighted=True),
            objective_weights=array([-1.0] * first.geometry.pixels),
        )
    else:
        getattr(first, role).fill_(float("nan"))
    with pytest.raises(ContractError):
        SpectralPoseEvaluator(replace(data, views=(first,)), PoseChart())

"""Composed CUDA derivative checks against an independent sampled-field oracle."""

# pytest.approx annotations are incomplete; keep that external boundary local.
# pyright: reportUnknownMemberType=false

import importlib
import math
import struct
from typing import Any

import pytest

from dpt.geometry import DetectorGeometry, RigidTransform
from dpt.objectives import ObjectiveSpec
from dpt.recovery import PoseChart, PrimaryPoseEvaluator, PrimaryPoseProblem
from dpt.validation.projection import integrate_sampled_field
from dpt.volumes import GridSpec

wp: Any = importlib.import_module("warp")
pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("integration", ["midpoint", "cell_gauss"])
@pytest.mark.parametrize("precision", ["float32", "float64"])
@pytest.mark.parametrize("domain", ["counts", "log_transmission"])
def test_composed_loss_and_chart_vjp_against_independent_reference(
    domain: str, precision: Any, integration: Any
) -> None:
    grid = GridSpec((4, 5, 6), (1.1, 0.9, 0.7), (-2.5, -1.9, -1.2))
    geometry = DetectorGeometry(
        (-20, 0.13, 0.17), (20, -1.3, -1.1), (0, 1, 0), (0, 0, 1), (0.13, 0.17), (9, 11)
    )
    values = [
        struct.unpack(
            "f", struct.pack("f", 0.03 + 0.002 * x + 0.003 * y + 0.005 * z + 0.0007 * x * y)
        )[0]
        for z in range(4)
        for y in range(5)
        for x in range(6)
    ]
    samples, beam = 53, 100.0
    observations = [
        80.0 + (index % 7) if domain == "counts" else -0.4 - (index % 5) * 0.01
        for index in range(geometry.pixels)
    ]
    observations = [struct.unpack("f", struct.pack("f", value))[0] for value in observations]
    attenuation = wp.array(values, dtype=wp.float32, device="cuda:0")
    observed = wp.array(observations, dtype=wp.float32, device="cuda:0")
    chart = PoseChart(anchor=RigidTransform(translation_mm=(0.1, -0.1, 0.05)))
    objective = ObjectiveSpec(
        domain="counts" if domain == "counts" else "log_transmission", reduction="mean"
    )
    evaluator = PrimaryPoseEvaluator(
        PrimaryPoseProblem(
            grid,
            geometry,
            attenuation,
            observed,
            objective,
            beam,
            samples_per_ray=samples,
            precision=precision,
            integration=integration,
        ),
        chart,
    )
    point = (0.07, 0.03, -0.04, 0.8, -0.5, 0.3)
    result = evaluator(point)

    def reference(coordinates: tuple[float, ...]) -> float:
        pose = chart.pose(coordinates)
        depths = [
            integrate_sampled_field(
                values,
                grid,
                geometry,
                pose,
                row,
                column,
                midpoint_samples=samples if integration == "midpoint" else None,
            )
            for row in range(geometry.shape[0])
            for column in range(geometry.shape[1])
        ]
        predicted = [beam * math.exp(-value) if domain == "counts" else -value for value in depths]
        return (
            0.5
            * math.fsum((a - b) ** 2 for a, b in zip(predicted, observations, strict=True))
            / geometry.pixels
        )

    assert result.loss == pytest.approx(reference(point), rel=5e-6, abs=2e-8)
    # Differentiate an independent unrounded sampled-field reference, using
    # several scales to expose knots and compare the complete chain.
    for direction in (
        (0.7, -0.4, 0.3, 0.5, 0.2, -0.6),
        (-0.2, 0.6, 0.1, -0.3, 0.8, 0.5),
        (0.4, 0.1, -0.9, 0.2, -0.5, 0.6),
    ):
        predicted = math.fsum(a * b for a, b in zip(result.gradient, direction, strict=True))
        differences: list[float] = []
        for step in (1e-3, 3e-4, 1e-4):
            plus = tuple(a + step * b for a, b in zip(point, direction, strict=True))
            minus = tuple(a - step * b for a, b in zip(point, direction, strict=True))
            differences.append((reference(plus) - reference(minus)) / (2 * step))
        assert min(abs(value - predicted) for value in differences) < 2e-4 * max(
            1e-5, abs(predicted)
        )
    # A later trial cannot change the result at the same immutable input state.
    evaluator(tuple(value * 0.5 for value in point))
    repeated = evaluator(point)
    assert repeated.loss == result.loss
    assert repeated.gradient == result.gradient

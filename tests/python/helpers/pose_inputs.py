"""Shared fixtures and independent oracles: pose inputs."""

import importlib
import math
from typing import Any

from dpt.geometry import DetectorGeometry
from dpt.materials import Provenance
from dpt.recovery import PoseChart
from dpt.registration import RecoveryResult
from dpt.volumes import GridSpec

wp: Any = importlib.import_module("warp")


LANDMARKS = ((-2.0, 0.0, 0.0), (0.0, 2.0, 0.0), (0.0, 0.0, 2.0), (1.0, 1.0, 1.0))


PROVENANCE = Provenance(
    "analytic: four asymmetric Gaussian components sampled on a 9-cubed grid",
    "0" * 64,
    "Original regression fixture; Apache-2.0",
    "Known identity pose, unit gain/exposure and zero electronic offset",
)


CHART_SCALES = (0.1, 0.1, 0.1, 0.01, 0.01, 0.01)


def _chart(initial: tuple[float, ...], near_boundary: bool) -> PoseChart:
    if not near_boundary:
        return PoseChart(scales=CHART_SCALES)
    # The same physical perturbation starts at 99.50% of this explicitly
    # declared chart. The truth remains strictly inside, with room for the
    # coupled pose/calibration descent after rejecting out-of-chart trials.
    rotation = math.hypot(
        *(value * scale for value, scale in zip(initial[3:6], CHART_SCALES[3:], strict=True))
    )
    return PoseChart(scales=CHART_SCALES, rotation_radius_radians=1.005 * rotation)


def _boundary_backtracking(
    chart: PoseChart, initial: tuple[float, ...], result: RecoveryResult
) -> dict[str, float | int]:
    def rotation(parameters: tuple[float, ...]) -> float:
        return math.hypot(
            *(value * scale for value, scale in zip(parameters[3:6], chart.scales[3:], strict=True))
        )

    initial_fraction = rotation(initial) / chart.rotation_radius_radians
    assert 0.995 < initial_fraction < 1.0
    rejected = result.rejected_trials
    assert rejected
    for trial in rejected:
        assert "outside the declared fixed rotation chart" in trial.message
        assert all(math.isfinite(value) for value in trial.parameters)
        assert rotation(trial.parameters) >= chart.rotation_radius_radians
    assert rotation(result.parameters) < chart.rotation_radius_radians
    assert result.history[-1].iteration > 0
    assert result.history[-1].loss < result.history[0].loss
    return {
        "chart_radius_radians": chart.rotation_radius_radians,
        "initial_fraction_of_radius": initial_fraction,
        "rejected_chart_trials": len(rejected),
        "smallest_rejected_rotation_radians": min(rotation(trial.parameters) for trial in rejected),
        "accepted_rotation_radians": rotation(result.parameters),
        "evaluations": result.evaluations,
    }


def phantom() -> tuple[GridSpec, list[float]]:
    grid = GridSpec((9, 9, 9), (1.0, 1.0, 1.0), (-4.0, -4.0, -4.0))
    components = (
        (0.05, -1.3, -0.8, 0.7, 0.9, 1.2, 0.7),
        (0.035, 1.4, 0.9, -1.1, 1.1, 0.65, 1.3),
        (0.025, 0.2, -1.5, -1.4, 0.6, 0.8, 0.7),
        (0.04, -0.9, 1.2, 1.4, 0.6, 0.6, 0.9),
    )
    values = [
        0.002
        + math.fsum(
            amplitude
            * math.exp(-0.5 * (((x - cx) / sx) ** 2 + ((y - cy) / sy) ** 2 + ((z - cz) / sz) ** 2))
            for amplitude, cx, cy, cz, sx, sy, sz in components
        )
        for z in range(-4, 5)
        for y in range(-4, 5)
        for x in range(-4, 5)
    ]
    return grid, values


def geometries() -> tuple[DetectorGeometry, DetectorGeometry]:
    return (
        DetectorGeometry(
            (-30.0, 0.13, 0.17),
            (30.0, -5.5, -5.5),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 1.0),
            (12, 12),
        ),
        DetectorGeometry(
            (0.13, -30.0, 0.17),
            (-5.5, 30.0, -5.5),
            (1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 1.0),
            (12, 12),
        ),
    )

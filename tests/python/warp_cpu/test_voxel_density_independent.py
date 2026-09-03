"""Voxel gradients against independent exponential-flight transport and geometry."""

from __future__ import annotations

import importlib
import math
from typing import Any, Literal

import numpy as np
import pytest
from scipy import stats

from dpt._runtime import cpu_test_context
from dpt.transport.forward import prepare_transport, trace_histories
from dpt.transport.model import MaterialGrid, PlanarDetector, TransportSpec
from dpt.transport.rng import HistoryBatch
from dpt.transport.voxel_derivatives import prepare_voxel_density_vjp, voxel_density_vjp

wp: Any = importlib.import_module("warp")

# Fixed physical scene shared by the independently implemented ray-box oracle
# and the explicit production fixture. Every layer has the same material;
# different voxel multipliers are the three active log-density coordinates.
_VOXEL_DENSITY = np.array([0.5, 1.3, 1.0])
_ABSORPTION = 0.12 * 1.1 * _VOXEL_DENSITY
_SCATTERING = 0.60 * 1.1 * _VOXEL_DENSITY


def _chords(position, direction):
    """Intersect the ray with each whole layer box, rather than stepping cells."""
    result = []
    for cell in range(3):
        lower, upper = (-1.5, -1.5, float(cell)), (1.5, 1.5, float(cell + 1))
        enter, leave = 0.0, math.inf
        intersects = True
        for axis in range(3):
            if direction[axis] == 0.0:
                if not lower[axis] <= position[axis] < upper[axis]:
                    intersects = False
                    break
            else:
                first = (lower[axis] - position[axis]) / direction[axis]
                second = (upper[axis] - position[axis]) / direction[axis]
                enter = max(enter, min(first, second))
                leave = min(leave, max(first, second))
        if intersects and leave > enter:
            result.append((enter, leave, cell))
    return sorted(result)


def _independent_histories(estimator, count):
    """Complete-history means, scores and moments from NumPy's separate PCG64."""
    rng = np.random.Generator(np.random.PCG64(41001 if estimator == "analogue" else 41002))
    flight_rate = _ABSORPTION + _SCATTERING if estimator == "analogue" else _SCATTERING
    lengths = np.zeros((count, 3))
    collisions = np.zeros((count, 3), dtype=np.int64)
    values = np.zeros(count)
    event_counts = np.zeros(count, dtype=np.int64)
    reentries = np.zeros(count, dtype=np.int64)
    absorbed = np.zeros(count, dtype=bool)
    for history in range(count):
        position, direction = [0.0, 0.0, -1.0], [0.0, 0.0, 1.0]
        itinerary = []
        for _ in range(512):
            depth = float(rng.exponential())
            chords = _chords(position, direction)
            collision = None
            for near, far, cell in chords:
                if not itinerary or itinerary[-1] != cell:
                    itinerary.append(cell)
                length = far - near
                if depth < flight_rate[cell] * length:
                    length = depth / flight_rate[cell]
                    lengths[history, cell] += length
                    position = [
                        position[axis] + (near + length) * direction[axis] for axis in range(3)
                    ]
                    collisions[history, cell] += 1
                    event_counts[history] += 1
                    collision = cell
                    break
                lengths[history, cell] += length
                depth -= flight_rate[cell] * length
            if collision is None:
                if chords:
                    position = [
                        position[axis] + chords[-1][1] * direction[axis] for axis in range(3)
                    ]
                if direction[2] != 0:
                    distance = (4.0 - position[2]) / direction[2]
                    x, y = (position[axis] + distance * direction[axis] for axis in range(2))
                    if distance >= 0 and -3 <= x < 3 and -3 <= y < 3:
                        values[history] = (
                            1.0
                            if estimator == "analogue"
                            else math.exp(-float(lengths[history] @ _ABSORPTION))
                        )
                break
            if (
                estimator == "analogue"
                and rng.random() < _ABSORPTION[collision] / flight_rate[collision]
            ):
                absorbed[history] = True
                break
            # Isotropic elastic directions are uniform in a global frame too;
            # this deliberately shares no production angular rotation basis.
            cosine, azimuth = 2 * float(rng.random()) - 1, 2 * math.pi * float(rng.random())
            sine = math.sqrt(max(0.0, 1 - cosine * cosine))
            direction = [sine * math.cos(azimuth), sine * math.sin(azimuth), cosine]
        else:
            raise AssertionError("independent reference exhausted its declared event budget")
        reentries[history] = len(itinerary) - len(set(itinerary))
    # Collisions and lengths are summed by voxel BEFORE the detector weighting.
    # This includes the final escape segment and all correlated return visits.
    scores = collisions - lengths * (_ABSORPTION + _SCATTERING)
    gradient = values[:, None] * scores
    return np.column_stack((values, gradient)), event_counts, reentries, absorbed


@pytest.mark.parametrize("estimator", ["analogue", "continuous-absorption"])
def test_voxel_vjp_matches_independent_layered_scatter(
    estimator: Literal["analogue", "continuous-absorption"],
    warp_device: str,
    request: pytest.FixtureRequest,
) -> None:
    # Fixed seeds and cap; no retries until a favourable statistical comparison.
    count = 32768
    reference, event_counts, reentries, absorbed = _independent_histories(estimator, count)
    assert (
        np.count_nonzero(reentries) > 500
    )  # Actual cell exit and reentry, not merely repeated segments.
    assert np.count_nonzero(event_counts >= 3) > 500
    assert event_counts.max() < 512
    if estimator == "analogue":
        assert np.count_nonzero(absorbed) > 500

    def array(values, dtype=None):
        return wp.array(values, dtype=wp.float64 if dtype is None else dtype, device=warp_device)

    with cpu_test_context():
        spec = TransportSpec(
            MaterialGrid((-1.5, -1.5, 0.0), (3.0, 3.0, 1.0), (3, 1, 1), 1),
            PlanarDetector((-3.0, -3.0), (6.0, 6.0), (1, 1), 4.0),
            80.0,
            "independent fixed-layer isotropic scatter gradient fixture",
            estimator=estimator,
            max_events=512,
        )
        transport = prepare_transport(
            spec,
            material_ids=array([0, 0, 0], wp.int32),
            absorption=array([0.12]),
            scattering=array([0.60]),
            voxel_density=array(_VOXEL_DENSITY),
            max_histories=count,
            device=warp_device,
        )
        inputs = [
            array(np.tile([0.0, 0.0, -1.0], (count, 1)), wp.vec3d),
            array(np.tile([0.0, 0.0, 1.0], (count, 1)), wp.vec3d),
            array(np.ones(count)),
            array([1.1]),
        ]
        batch = HistoryBatch(60719 if estimator == "analogue" else 60720, 0, count)
        outputs = {
            f"out_{name}": wp.empty(count, dtype=dtype, device=warp_device)
            for name, dtype in (
                ("pixel", wp.int32),
                ("score", wp.float64),
                ("energy", wp.float64),
                ("events", wp.int32),
                ("status", wp.int32),
            )
        }
        trace_histories(*inputs, batch=batch, workspace=transport, **outputs)
        vjp = prepare_voxel_density_vjp(transport, slots_per_history=4)
        gradient = wp.empty(3, dtype=wp.float64, device=warp_device)
        variance = wp.empty_like(gradient)
        result = voxel_density_vjp(
            inputs[0],
            inputs[1],
            inputs[2],
            inputs[3],
            array([1.0]),
            batch=batch,
            workspace=vjp,
            out_gradient=gradient,
            out_variance_of_mean=variance,
            variance_law="iid-original-histories",
        )
        assert result.checked and result.batch == batch
        assert result.parameterization == "log-voxel-density"
        assert result.variance_law == "iid-original-histories-conditional-on-cotangent"
        assert result.grid_shape == (3, 1, 1)
        statuses = outputs["out_status"].numpy()
        assert np.all(statuses <= 1)
        actual_signal = outputs["out_score"].numpy()
        actual_mean = np.concatenate(([actual_signal.mean()], gradient.numpy()))
        actual_variance = np.concatenate(([actual_signal.var(ddof=1) / count], variance.numpy()))
    assert np.all(np.isfinite(actual_variance)) and np.all(actual_variance > 0)
    reference_mean = reference.mean(axis=0)
    reference_variance = reference.var(axis=0, ddof=1) / count
    total_variance = actual_variance + reference_variance
    degrees = total_variance**2 / ((actual_variance**2 + reference_variance**2) / (count - 1))
    # Eight predeclared contrasts: signal and three voxel gradients in two
    # estimators. This is a plug-in Welch t CI, with a 99% Bonferroni family.
    critical = stats.t.ppf(1 - 0.01 / (2 * 8), degrees)
    half_width = critical * np.sqrt(total_variance)
    difference = np.abs(actual_mean - reference_mean)
    assert np.all(difference <= half_width), (actual_mean, reference_mean, half_width)
    # A noisy CI cannot pass just because it includes zero. Require the whole
    # discrepancy interval inside the predeclared absolute effect-size margin.
    assert np.all(difference + half_width <= 0.02), (difference, half_width)
    # The margin is in expected photons/original history for the signal, and
    # expected photons/original history/unit log density for each gradient.
    # Every declared gradient effect exceeds twice this margin; zero or the
    # wrong sign cannot count as a precise match for this scene.
    assert np.all(reference_mean[1:] < -2 * 0.02)
    assert np.all(actual_mean[1:] < 0.0)
    request.node.user_properties.extend(
        (
            ("independent_reference_rng", "NumPy PCG64"),
            ("independent_reference_seed", 41001 if estimator == "analogue" else 41002),
            ("production_philox_seed", batch.seed),
            ("original_histories_per_independent_batch", count),
            ("bonferroni_family_confidence", 0.99),
            ("predeclared_simultaneous_contrasts", 8),
            ("whole_discrepancy_ci_precision_margin", 0.02),
            ("reference_histories_with_cell_reentry", int(np.count_nonzero(reentries))),
        )
    )
    for index, name in enumerate(("signal", "log_voxel_0", "log_voxel_1", "log_voxel_2")):
        request.node.user_properties.extend(
            (
                (f"{name}_reference_effect", float(reference_mean[index])),
                (f"{name}_production_estimate", float(actual_mean[index])),
                (
                    f"{name}_discrepancy_ci_lower",
                    float(actual_mean[index] - reference_mean[index] - half_width[index]),
                ),
                (
                    f"{name}_discrepancy_ci_upper",
                    float(actual_mean[index] - reference_mean[index] + half_width[index]),
                ),
            )
        )

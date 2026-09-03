"""Portable checks of resident projectors and scalar iteration policy kernels."""

import math

import numpy as np
import pytest

wp = pytest.importorskip("warp")
from dpt.kernels import material_iteration as controls  # noqa: E402
from dpt.kernels import material_iteration_projection as resident  # noqa: E402
from dpt.kernels import material_reconstruction as eager  # noqa: E402


def _launch(kernel, arguments, *, device: str, tiled=False, dim=1):
    if tiled:
        wp.launch_tiled(kernel, dim=dim, inputs=arguments, block_dim=256, device=device)
    else:
        wp.launch(kernel, dim=dim, inputs=arguments, device=device)


def _policy(**changes):
    policy = controls.IterationPolicy()
    values = {
        "maximum_backtracks": 4,
        "window": 4,
        "relative_mapping": 0,
        "use_secant": 0,
        "use_inertia": 0,
        "maximum_step": 32.0,
        "minimum_secant": 1e-16,
        "backtrack_factor": 0.5,
        "armijo": 0.1,
        "mapping_step": 1.0,
        "relative_tolerance": 0.2,
    }
    for name, value in (values | changes).items():
        setattr(policy, name, value)
    return policy


def _scalars(device: str, **values):
    numeric = np.zeros(controls.CONTROL_SIZE, dtype=np.float64)
    flags = np.zeros(controls.FLAG_SIZE, dtype=np.int32)
    for name, value in values.items():
        numeric[getattr(controls, name)] = value
    return wp.array(numeric, dtype=wp.float64, device=device), wp.array(flags, device=device)


@pytest.mark.parametrize("materials", range(1, 9))
@pytest.mark.parametrize("nonnegative", (False, True))
@pytest.mark.parametrize("scaled", (False, True))
def test_resident_projector_preserves_eager_arithmetic(materials, nonnegative, scaled, device: str):
    voxels = 3
    fields = wp.array(np.full(materials * voxels, 0.4 / materials, dtype=np.float32), device=device)
    gradient = wp.array(np.linspace(-3, 2, materials * voxels, dtype=np.float32), device=device)
    previous = wp.array(
        np.full(materials * voxels, 0.2 / materials, dtype=np.float32), device=device
    )
    scale = wp.array(np.array([0.5, 1.0, 2.0], dtype=np.float32), device=device)
    diagonal = wp.array(np.linspace(0.5, 2.0, materials), dtype=wp.float64, device=device)
    state, _ = _scalars(device=device, STEP=0.25, MOMENTUM=0.3)
    results = []
    for graph in (False, True):
        trial = wp.empty_like(fields)
        slope = wp.zeros(1, dtype=wp.float64, device=device)
        mapping = wp.zeros(1, dtype=wp.float64, device=device)
        status = wp.zeros(1, dtype=wp.int32, device=device)
        kernel = (
            resident.projected_step_tiled(materials, nonnegative)
            if graph
            else eager.projected_step_tiled(materials, nonnegative)
        )
        step = [state, controls.STEP] if graph else [0.25]
        momentum = controls.MOMENTUM if graph else 0.3
        _launch(
            kernel,
            [
                fields,
                gradient,
                voxels,
                *step,
                previous,
                momentum,
                scaled,
                scale,
                scaled,
                diagonal,
                trial,
                slope,
                mapping,
                status,
            ],
            device=device,
            tiled=True,
        )
        results.append(tuple(value.numpy().copy() for value in (trial, slope, mapping, status)))
    for actual, expected in zip(results[1], results[0], strict=True):
        np.testing.assert_array_equal(actual, expected)
    point = results[1][0].reshape(materials, voxels)
    assert np.all(point >= 0)
    if not nonnegative:
        assert np.all(point.sum(axis=0, dtype=np.float64) <= 1 + 2**-24)


@pytest.mark.parametrize("nonnegative", (False, True))
@pytest.mark.parametrize("scaled", (False, True))
def test_resident_coupled_metric_preserves_certificate(nonnegative, scaled, device: str):
    fields = wp.array(np.array([0.2, 0.4, 0.3, 0.1], dtype=np.float32), device=device)
    gradient = wp.array(np.array([-2, 1, -1, -3], dtype=np.float32), device=device)
    scale = wp.array(np.array([0.5, 2], dtype=np.float32), device=device)
    metric = wp.vec3d(2, 0.25, 1)
    determinant = 2 - 0.25**2
    inverse = wp.vec3d(1 / determinant, -0.25 / determinant, 2 / determinant)
    state, _ = _scalars(device=device, STEP=0.75, MOMENTUM=0.0)
    results = []
    for module in (eager, resident):
        trial = wp.empty_like(fields)
        slope, mapping = (wp.zeros(1, dtype=wp.float64, device=device) for _ in range(2))
        status = wp.zeros(1, dtype=wp.int32, device=device)
        kernel = (
            module.projected_metric_nonnegative_step_tiled
            if nonnegative
            else module.projected_metric_step_tiled
        )
        step = [state, controls.STEP] if module is resident else [0.75]
        momentum = controls.MOMENTUM if module is resident else 0.0
        _launch(
            kernel,
            [
                fields,
                gradient,
                2,
                *step,
                fields,
                momentum,
                scaled,
                scale,
                metric,
                inverse,
                trial,
                slope,
                mapping,
                status,
            ],
            device=device,
            tiled=True,
        )
        results.append(tuple(value.numpy().copy() for value in (trial, slope, mapping, status)))
    for actual, expected in zip(results[1], results[0], strict=True):
        np.testing.assert_array_equal(actual, expected)
    assert results[1][-1][0] == 0


def test_armijo_rejection_and_geometric_growth_have_original_rounding(device: str):
    state, flags = _scalars(
        device=device, STEP=1.0, LOSS=1.0, REFERENCE=1.0, SLOPE=-1.0, TRIAL_LOSS=1.2
    )
    ledger = wp.zeros(controls.LEDGER_WIDTH * 4, dtype=wp.float64, device=device)
    _launch(controls.decide_attempt, [state, flags, ledger, _policy()], device=device)
    assert state.numpy()[controls.STEP] == 0.5
    assert flags.numpy()[controls.BACKTRACKS] == 1
    assert ledger.numpy()[7] == 1
    # The exact quadratic q(x)=x²/2 at x=sqrt(2) has a decreasing
    # half-step with slope -1 and q(trial)=1/4.
    values = state.numpy().copy()
    values[controls.SLOPE], values[controls.TRIAL_LOSS] = -1.0, 0.25
    wp.copy(state, wp.array(values, dtype=wp.float64, device=device))
    _launch(controls.decide_attempt, [state, flags, ledger, _policy()], device=device)
    assert flags.numpy()[controls.ACCEPTED] == 1
    assert flags.numpy()[controls.BACKTRACKS] == 1
    assert flags.numpy()[controls.ATTEMPTS] == 2


def test_inertial_restart_retries_same_alpha_without_backtrack(device: str):
    state, flags = _scalars(
        device=device,
        STEP=0.5,
        MOMENTUM=0.25,
        LOSS=1.0,
        REFERENCE=1.0,
        SLOPE=0.0,
        TRIAL_LOSS=math.inf,
    )
    ledger = wp.zeros(controls.LEDGER_WIDTH * 5, dtype=wp.float64, device=device)
    _launch(controls.decide_attempt, [state, flags, ledger, _policy(use_inertia=1)], device=device)
    assert state.numpy()[controls.STEP] == 0.5
    assert state.numpy()[controls.MOMENTUM] == 0
    assert flags.numpy()[controls.BACKTRACKS] == 0
    assert flags.numpy()[controls.RESTART] == 2
    _launch(controls.decide_attempt, [state, flags, ledger, _policy(use_inertia=1)], device=device)
    assert state.numpy()[controls.STEP] == 0.25
    assert flags.numpy()[controls.BACKTRACKS] == 1


def test_gll_reference_admits_ascent_below_actual_window_maximum(device: str):
    state, flags = _scalars(
        device=device, STEP=1.0, LOSS=1.0, REFERENCE=1.5, SLOPE=-0.5, TRIAL_LOSS=1.2
    )
    ledger = wp.zeros(controls.LEDGER_WIDTH * 4, dtype=wp.float64, device=device)
    _launch(controls.decide_attempt, [state, flags, ledger, _policy()], device=device)
    assert flags.numpy()[controls.ACCEPTED] == 1
    assert 1.0 < ledger.numpy()[2] < 1.5 - 0.1 * 0.5


@pytest.mark.parametrize(
    "numerator,denominator,code,expected",
    (
        (4.0, 2.0, 0, 2.0),
        (-1.0, 2.0, 0, 0.25),
        (4.0, 0.0, 0, 0.25),
        (math.inf, 2.0, 0, 0.25),
        (4.0, 2.0, 2, 0.25),
        (100.0, 1.0, 0, 32.0),
    ),
)
def test_secant_is_bounded_proposal_or_preserves_geometric_step(
    numerator, denominator, code, expected, device: str
):
    state, flags = _scalars(device=device, STEP=0.25)
    total = wp.array([numerator, denominator], dtype=wp.float64, device=device)
    status = wp.array([code], dtype=wp.int32, device=device)
    _launch(controls.secant_result, [total, status, state, flags, _policy()], device=device)
    assert state.numpy()[controls.STEP] == expected
    assert status.numpy()[0] == 0


@pytest.mark.parametrize("scale,expected", ((0.0, 1.0), (4.0, 4.0)))
def test_relative_mapping_uses_initial_gradient_scale_and_combined_tolerance(
    scale, expected, device: str
):
    state, flags = _scalars(device=device, TOLERANCE=0.1)
    maximum = wp.array([scale], dtype=wp.float64, device=device)
    _launch(
        controls.relative_scale, [maximum, state, flags, _policy(relative_mapping=1)], device=device
    )
    trial = wp.array([0.5, -0.25, 0.0], dtype=wp.float64, device=device)
    _launch(
        controls.mapping_result, [trial, state, flags, _policy(relative_mapping=1)], device=device
    )
    assert state.numpy()[controls.NORMALISER] == expected
    assert state.numpy()[controls.MAPPING_STEP] == 1 / expected
    assert state.numpy()[controls.TOLERANCE] == 0.1 * expected + 0.2 * 0.5


@pytest.mark.parametrize(
    "code,fixed,fatal,numerical", ((1, 1, 1, 0), (2, 1, 0, 1), (8, 1, 0, 1), (4, 0, 0, 1))
)
def test_trial_status_distinguishes_fixed_input_from_rejectable_numerics(
    code, fixed, fatal, numerical, device: str
):
    _, flags = _scalars(device=device)
    source = wp.array([code], dtype=wp.int32, device=device)
    snapshots = wp.zeros(4, dtype=wp.int32, device=device)
    _launch(controls.collect_trial_status, [source, snapshots, 4, 0, flags, fixed], device=device)
    assert flags.numpy()[controls.FATAL] == fatal
    assert flags.numpy()[controls.NUMERICAL] == numerical
    assert snapshots.numpy()[0] == code
    assert source.numpy()[0] == 0


@pytest.mark.parametrize("numerical,fatal,expected", ((0, 0, 1), (1, 0, 0), (0, 1, 0), (1, 1, 0)))
def test_pending_trial_errors_prevent_evaluator_entry(numerical, fatal, expected, device: str):
    _, flags = _scalars(device=device)
    values = flags.numpy().copy()
    values[controls.EVALUATE] = 1
    values[controls.NUMERICAL] = numerical
    values[controls.FATAL] = fatal
    wp.copy(flags, wp.array(values, dtype=wp.int32, device=device))
    selected = wp.zeros(1, dtype=wp.int32, device=device)
    _launch(controls.gate, [flags, selected, 2], device=device)
    assert selected.numpy()[0] == expected

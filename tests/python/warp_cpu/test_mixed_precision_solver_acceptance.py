"""Independent represented-input references for mixed-precision solver controls."""

import hashlib
import importlib
import json
import math
from contextlib import nullcontext
from fractions import Fraction
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pytest

from dpt._reductions import prepare_reduction
from dpt._runtime import cpu_test_context, prepare_context
from dpt.pose_optimization import (
    NormalEquations,
    ParameterBounds,
    PoseLeastSquaresPolicy,
    recover_least_squares,
)

wp: Any = importlib.import_module("warp")
material: Any = importlib.import_module("dpt.kernels.material_reconstruction")
normal: Any = importlib.import_module("dpt.kernels.pose_optimization")

SOURCE_MODULES = (
    "dpt._runtime",
    "dpt._reductions",
    "dpt.kernels.reductions",
    "dpt.kernels.material_reconstruction",
    "dpt.kernels.pose_optimization",
    "dpt.pose_optimization",
)
TREE_ROWS = (129, 257, 1025)
TREE_LAYOUTS = ((128, 43), (256, 1))
CONTROL_VOXELS = (257, 513)
CONTROL_METRICS = ("identity", "coupled", "diagonal")
CONTROL_STEP = 2.0**-8
SOLVER_TOLERANCE = 2.0**-20
NORMAL_OPERATION_BUDGET = 32


def _fraction(value: Any) -> Fraction:
    return Fraction.from_float(float(value))


def _upper_binary64(value: Fraction) -> float:
    rounded = float(value)
    return math.nextafter(rounded, math.inf) if _fraction(rounded) < value else rounded


def _exact_sum(terms: list[Fraction]) -> float:
    """Certify exact binary64 products and any ordering of their additions."""
    denominator = max((term.denominator for term in terms), default=1)
    assert denominator & (denominator - 1) == 0
    absolute = sum((abs(term) for term in terms), Fraction()) * denominator
    assert absolute.denominator == 1 and absolute.numerator.bit_length() <= 53
    for term in terms:
        assert _fraction(float(term)) == term
    total = sum(terms, Fraction())
    result = float(total)
    assert _fraction(result) == total
    return result


def _source_hashes() -> dict[str, str]:
    hashes = {}
    for name in SOURCE_MODULES:
        module = importlib.import_module(name)
        filename = module.__file__
        assert isinstance(filename, str)
        hashes[name] = hashlib.sha256(Path(filename).read_bytes()).hexdigest()
    return hashes


@pytest.fixture(autouse=True)
def device_execution_guard(warp_device: str, monkeypatch: pytest.MonkeyPatch, record_property: Any):
    """Observe actual selected-device calls without changing numerical inputs/outputs."""
    before = _source_hashes()
    calls: list[dict[str, Any]] = []
    selected = wp.get_device(warp_device)
    if selected.is_cuda:
        assert isinstance(selected.uuid, str) and selected.uuid.startswith("GPU-")
    scope = cpu_test_context() if warp_device == "cpu" else nullcontext()
    with scope:
        for api in ("launch", "launch_tiled"):
            original = getattr(wp, api)

            def observe(*args: Any, _api=api, _original=original, **kwargs: Any):
                device = wp.get_device(kwargs.get("device"))
                assert device == selected and device.alias == warp_device
                if device.is_cuda:
                    assert device.uuid == selected.uuid
                arrays = []
                for group in ("inputs", "outputs"):
                    for value in kwargs.get(group, ()) or ():
                        if isinstance(value, wp.array):
                            assert value.device == device
                            arrays.append({"dtype": str(value.dtype), "size": int(value.size)})
                result = _original(*args, **kwargs)
                calls.append(
                    {
                        "api": _api,
                        "kernel": str((args[0] if args else kwargs["kernel"]).key),
                        "device": device.alias,
                        "device_uuid": device.uuid,
                        "dim": kwargs.get("dim"),
                        "arrays": arrays,
                    }
                )
                return result

            monkeypatch.setattr(wp, api, observe)
        yield calls
    assert calls and any(call["api"] == "launch_tiled" for call in calls)
    assert before == _source_hashes()
    record_property("mixed_precision_device_calls", json.dumps(calls, sort_keys=True))
    record_property("mixed_precision_source_sha256", json.dumps(before, sort_keys=True))


def _reduce(ctx: Any, source: Any, count: int) -> tuple[float, int]:
    tree = prepare_reduction(ctx, count)
    destination = wp.empty(1, dtype=wp.float64, device=ctx.device)
    status = wp.zeros(1, dtype=wp.int32, device=ctx.device)
    tree.finish(count, destination, status, source=source)
    assert all(partial.dtype == wp.float64 for partial in tree.partials)
    return float(destination.numpy()[0]), int(status.numpy()[0])


@pytest.mark.parametrize("rows", TREE_ROWS)
@pytest.mark.parametrize("tile,components", TREE_LAYOUTS)
def test_exact_dyadic_mixed_storage_tree(
    rows: int, tile: int, components: int, warp_device: str
) -> None:
    ctx = prepare_context(device=warp_device)
    pattern = (2.0**20, 2.0**-20, -(2.0**20))
    represented = np.array(
        [pattern[(row + component) % 3] for row in range(rows) for component in range(components)],
        dtype=np.float32,
    )
    source = wp.array(represented.astype(np.float64), dtype=wp.float64, device=ctx.device)
    assert source.dtype == wp.float64
    assert rows * 2**40 < 2**53
    tree = prepare_reduction(ctx, rows, tile=tile, components=components)
    status = wp.zeros(1, dtype=wp.int32, device=ctx.device)
    for count in (rows, tile - 1, 1, 0):
        count = min(count, rows)
        expected = [
            _exact_sum([_fraction(x) for x in represented[c : count * components : components]])
            for c in range(components)
        ]
        for dtype, host_dtype in ((wp.float64, np.float64), (wp.float32, np.float32)):
            destination = wp.empty(components, dtype=dtype, device=ctx.device)
            tree.finish(count, destination, status, components=components, source=source)
            np.testing.assert_array_equal(
                destination.numpy(), np.asarray(expected, dtype=host_dtype)
            )
            assert status.numpy().tolist() == [0]
    # This comparator is explicitly test-only; the production tree stays FP64.
    narrow = np.float32(np.float32(np.float32(pattern[0]) + np.float32(pattern[1])) + pattern[2])
    assert float(narrow) == 0.0
    assert _exact_sum([_fraction(x) for x in pattern]) == 2.0**-20


@pytest.mark.parametrize("voxels", CONTROL_VOXELS)
@pytest.mark.parametrize("metric", CONTROL_METRICS)
def test_material_controls_match_exact_represented_input_oracles(
    voxels: int, metric: str, warp_device: str, record_property: Any
) -> None:
    ctx = prepare_context(device=warp_device)
    indices = np.arange(voxels)
    fields = np.vstack((0.25 + (indices % 3) / 64, 0.375 + (indices % 5) / 128)).astype(np.float32)
    gradients = np.vstack(((indices % 5 - 2) / 32, (indices % 3 - 1) / 16)).astype(np.float32)
    gradients[0, -1] = 0.125
    displacement = np.vstack(((indices % 3 - 1) / 4096, (indices % 5 - 2) / 4096)).astype(
        np.float32
    )
    previous = fields - displacement
    old_gradient = gradients - np.float32(4) * displacement
    scales = np.array([0.5, 1.0, 2.0], dtype=np.float32)[indices % 3]
    arrays = {
        "fields": wp.array(fields.reshape(-1), dtype=wp.float32, device=ctx.device),
        "gradient": wp.array(gradients.reshape(-1), dtype=wp.float32, device=ctx.device),
        "previous": wp.array(previous.reshape(-1), dtype=wp.float32, device=ctx.device),
        "old_gradient": wp.array(old_gradient.reshape(-1), dtype=wp.float32, device=ctx.device),
        "scale": wp.array(scales, dtype=wp.float32, device=ctx.device),
    }
    diagonal = wp.array([2.0, 0.5], dtype=wp.float64, device=ctx.device)
    trial = wp.empty(2 * voxels, dtype=wp.float32, device=ctx.device)
    blocks = (voxels + 255) // 256
    slope_partials = wp.empty(blocks, dtype=wp.float64, device=ctx.device)
    mappings = wp.empty(blocks, dtype=wp.float64, device=ctx.device)
    status = wp.zeros(1, dtype=wp.int32, device=ctx.device)
    wp.launch_tiled(
        material.projected_step_tiled(2),
        dim=blocks,
        inputs=[
            arrays["fields"],
            arrays["gradient"],
            voxels,
            CONTROL_STEP,
            arrays["fields"],
            0.0,
            False,
            arrays["scale"],
            False,
            diagonal,
            trial,
            slope_partials,
            mappings,
            status,
        ],
        block_dim=256,
        device=ctx.device,
        record_tape=False,
    )
    expected_trial = [
        _fraction(f) - _fraction(CONTROL_STEP) * _fraction(g)
        for f, g in zip(fields.reshape(-1), gradients.reshape(-1), strict=True)
    ]
    expected_stored = np.array([float(x) for x in expected_trial], dtype=np.float32)
    assert all(
        _fraction(stored) == exact
        for stored, exact in zip(expected_stored, expected_trial, strict=True)
    )
    np.testing.assert_array_equal(trial.numpy(), expected_stored)
    slope_terms = [
        _fraction(g) * (_fraction(t) - _fraction(f))
        for g, t, f in zip(gradients.reshape(-1), expected_stored, fields.reshape(-1), strict=True)
    ]
    slope, code = _reduce(ctx, slope_partials, blocks)
    assert slope == _exact_sum(slope_terms) and code == 0
    maximum = wp.empty(1, dtype=wp.float64, device=ctx.device)
    wp.launch_tiled(
        material.max_partials,
        dim=1,
        inputs=[mappings, blocks, maximum],
        block_dim=256,
        device=ctx.device,
        record_tape=False,
    )
    assert float(maximum.numpy()[0]) == float(np.abs(gradients).max()) == 0.125
    assert status.numpy().tolist() == [0]

    count = 2 * voxels
    scalar_blocks = (count + 255) // 256
    numerator = wp.empty(scalar_blocks, dtype=wp.float64, device=ctx.device)
    denominator = wp.empty_like(numerator)
    coupled = metric == "coupled"
    diagonal_metric = metric == "diagonal"
    wp.launch_tiled(
        material.secant_products_tiled,
        dim=scalar_blocks,
        inputs=[
            count,
            arrays["fields"],
            arrays["gradient"],
            arrays["previous"],
            arrays["old_gradient"],
            voxels,
            coupled,
            wp.vec3d(2.0, -0.5, 1.0),
            True,
            arrays["scale"],
            diagonal_metric,
            diagonal,
            numerator,
            denominator,
        ],
        block_dim=256,
        device=ctx.device,
        record_tape=False,
    )
    num_terms, den_terms = [], []
    for voxel in range(voxels):
        s = [_fraction(fields[m, voxel]) - _fraction(previous[m, voxel]) for m in range(2)]
        y = [_fraction(gradients[m, voxel]) - _fraction(old_gradient[m, voxel]) for m in range(2)]
        hs = s
        if coupled:
            hs = [2 * s[0] - s[1] / 2, -s[0] / 2 + s[1]]
        elif diagonal_metric:
            hs = [2 * s[0], s[1] / 2]
        for axis in range(2):
            num_terms.append(s[axis] * hs[axis] * _fraction(scales[voxel]))
            den_terms.append(s[axis] * y[axis])
    actual_num, num_status = _reduce(ctx, numerator, scalar_blocks)
    actual_den, den_status = _reduce(ctx, denominator, scalar_blocks)
    assert actual_num == _exact_sum(num_terms) and num_status == 0
    assert actual_den == _exact_sum(den_terms) and den_status == 0
    ratio = float(sum(num_terms, Fraction()) / sum(den_terms, Fraction()))
    assert abs(actual_num / actual_den - ratio) <= math.ulp(ratio)
    assert all(array.dtype == wp.float32 for array in arrays.values())
    record_property(
        "control_reference",
        json.dumps(
            {
                "slope": slope,
                "numerator": actual_num,
                "denominator": actual_den,
                "BB_ratio": ratio,
                "status": 0,
                "metric": metric,
            },
            sort_keys=True,
        ),
    )


def test_material_prior_reduction_matches_exact_edge_sum(warp_device: str) -> None:
    ctx = prepare_context(device=warp_device)
    voxels = 257
    index = np.arange(voxels)
    fields = np.vstack((0.25 + (index % 3) / 64, 0.375 + (index % 5) / 128)).astype(np.float32)
    coefficients = (Fraction(1, 2), Fraction(2))
    source = wp.array(fields.reshape(-1), dtype=wp.float32, device=ctx.device)
    coefficient = wp.array([0.5, 0.0, 0.0, 2.0, 0.0, 0.0], dtype=wp.float64, device=ctx.device)
    gradient = wp.zeros(2 * voxels, dtype=wp.float32, device=ctx.device)
    blocks = (2 * voxels + 255) // 256
    partials = wp.empty(blocks, dtype=wp.float64, device=ctx.device)
    status = wp.zeros(1, dtype=wp.int32, device=ctx.device)
    wp.launch_tiled(
        material.regularise_tiled,
        dim=blocks,
        inputs=[
            2 * voxels,
            source,
            voxels,
            1,
            1,
            coefficient,
            False,
            wp.vec3d(1, 1, 1),
            True,
            gradient,
            partials,
            status,
        ],
        block_dim=256,
        device=ctx.device,
        record_tape=False,
    )
    terms = []
    expected_gradient = [[Fraction() for _ in range(voxels)] for _ in range(2)]
    for axis in range(2):
        for voxel in range(voxels - 1):
            difference = _fraction(fields[axis, voxel]) - _fraction(fields[axis, voxel + 1])
            terms.append(coefficients[axis] * difference * difference / 2)
            expected_gradient[axis][voxel] += coefficients[axis] * difference
            expected_gradient[axis][voxel + 1] -= coefficients[axis] * difference
    loss, code = _reduce(ctx, partials, blocks)
    assert loss == _exact_sum(terms) and code == 0
    np.testing.assert_array_equal(
        gradient.numpy().reshape(2, voxels),
        np.array([[float(x) for x in row] for row in expected_gradient], dtype=np.float32),
    )
    assert status.numpy().tolist() == [0]


def _normal_reference(
    signal: np.ndarray, observation: np.ndarray, jacobian: np.ndarray, weights: np.ndarray
) -> list[list[Fraction]]:
    terms: list[list[Fraction]] = [[] for _ in range(43)]
    for pixel in range(len(signal)):
        residual = _fraction(signal[pixel]) - _fraction(observation[pixel])
        weight = _fraction(weights[pixel])
        row = [_fraction(x) for x in jacobian[pixel]]
        terms[0].append(weight * residual * residual / 2)
        for axis in range(6):
            terms[1 + axis].append(weight * residual * row[axis])
            for other in range(6):
                terms[7 + axis * 6 + other].append(weight * row[axis] * row[other])
    return terms


class _ResidentNormal:
    def __init__(
        self, observation: np.ndarray, jacobian: np.ndarray, weights: np.ndarray, *, device: str
    ):
        self.ctx = prepare_context(device=device)
        self.count = len(observation)
        self.blocks = (self.count + 127) // 128
        self.tree = prepare_reduction(self.ctx, self.count, tile=128, components=43)
        self.signal = wp.empty(self.count, dtype=wp.float64, device=self.ctx.device)
        self.observation = wp.array(observation, dtype=wp.float32, device=self.ctx.device)
        self.jacobian = wp.array(jacobian.reshape(-1), dtype=wp.float64, device=self.ctx.device)
        self.weights = wp.array(weights, dtype=wp.float32, device=self.ctx.device)
        self.chart = wp.array(np.eye(6).reshape(-1), dtype=wp.float64, device=self.ctx.device)
        self.output = wp.empty(43, dtype=wp.float64, device=self.ctx.device)
        self.status = wp.zeros(1, dtype=wp.int32, device=self.ctx.device)

    def evaluate(self, signal: np.ndarray, *, weighted: bool) -> np.ndarray:
        self.signal.assign(signal)
        wp.launch_tiled(
            normal.normal_kernel(weighted),
            dim=self.blocks,
            inputs=[
                self.signal,
                self.observation,
                self.weights,
                self.jacobian,
                self.chart,
                self.count,
                1.0,
            ],
            outputs=[self.tree.partials[0]],
            block_dim=128,
            device=self.ctx.device,
            record_tape=False,
        )
        self.tree.finish(self.blocks, self.output, self.status, components=43)
        assert self.observation.dtype == self.weights.dtype == wp.float32
        assert self.signal.dtype == self.jacobian.dtype == self.output.dtype == wp.float64
        assert self.status.numpy().tolist() == [0]
        return self.output.numpy().copy()


@pytest.mark.parametrize("rows", [129, 257])
@pytest.mark.parametrize("weighted", [False, True])
def test_normal_controls_match_exact_contraction(
    rows: int, weighted: bool, warp_device: str
) -> None:
    index = np.arange(rows)
    observation = (1.0 + (index % 7) / 8).astype(np.float32)
    jacobian = np.zeros((rows, 6), dtype=np.float64)
    jacobian[index, index % 6] = 2.0 ** -(index % 6)
    weights = np.array([0.0, 0.5, 1.0, 2.0], dtype=np.float32)[index % 4]
    resident = _ResidentNormal(observation, jacobian, weights, device=warp_device)
    reference_weight = weights if weighted else np.ones(rows, dtype=np.float32)
    for residual in ((index % 5 - 2) / 32, np.full(rows, 2.0**-30)):
        signal = observation.astype(np.float64) + residual
        expected = [
            _exact_sum(terms)
            for terms in _normal_reference(signal, observation, jacobian, reference_weight)
        ]
        np.testing.assert_array_equal(resident.evaluate(signal, weighted=weighted), expected)


def _linear_reference(point: tuple[float, ...], rows: int) -> NormalEquations:
    hessian = (Fraction((rows + 1) // 2), Fraction(4 * (rows // 2)))
    displacement = (_fraction(point[0]) - Fraction(1, 2), _fraction(point[1]) - Fraction(1, 4))
    loss = sum((h * d * d / 2 for h, d in zip(hessian, displacement, strict=True)), Fraction())
    gradient = tuple(float(h * d) for h, d in zip(hessian, displacement, strict=True)) + (0.0,) * 4
    matrix = tuple(
        tuple(float(hessian[i]) if i == j and i < 2 else 0.0 for j in range(6)) for i in range(6)
    )
    return NormalEquations(float(loss), gradient, matrix)


@pytest.mark.parametrize("rows", [129, 257])
@pytest.mark.parametrize("method", ["gauss_newton", "levenberg_marquardt"])
def test_native_normal_and_reference_evaluators_converge_to_same_linear_solution(
    rows: int,
    method: Literal["gauss_newton", "levenberg_marquardt"],
    warp_device: str,
    record_property: Any,
) -> None:
    index = np.arange(rows)
    jacobian = np.zeros((rows, 6), dtype=np.float64)
    jacobian[index, index % 2] = np.where(index % 2 == 0, 1.0, 2.0)
    # These are measurements J*truth: both equal 1/2 for truth=(1/2,1/4).
    observation = np.full(rows, 0.5, dtype=np.float32)
    resident = _ResidentNormal(
        observation, jacobian, np.ones(rows, dtype=np.float32), device=warp_device
    )

    def native(point: tuple[float, ...]) -> NormalEquations:
        signal = np.array(
            [point[int(axis)] * (1.0 if axis == 0 else 2.0) for axis in index % 2], dtype=np.float64
        )
        controls = resident.evaluate(signal, weighted=False)
        return NormalEquations(
            float(controls[0]),
            tuple(float(x) for x in controls[1:7]),
            tuple(tuple(float(x) for x in row) for row in controls[7:].reshape(6, 6)),
        )

    blocks = (rows + 127) // 128
    # At most 127 first-tile additions and blocks-1 second-level additions.
    # q=32 covers two six-term chart contractions and scalar products; normal
    # scale16*rows bounds absolute elementary terms throughout the fixed box.
    operation_count = 127 + blocks - 1 + NORMAL_OPERATION_BUDGET
    unit = Fraction(1, 2**53)
    gamma = operation_count * unit / (1 - operation_count * unit)
    arithmetic_bound = _upper_binary64(gamma * 16 * rows)
    assert arithmetic_bound < SOLVER_TOLERANCE / 1024
    minimum_eigenvalue = (rows + 1) // 2
    parameter_bound = _upper_binary64(
        (_fraction(SOLVER_TOLERANCE) + _fraction(arithmetic_bound)) / minimum_eigenvalue
        + _fraction(math.ulp(0.5))
    )
    bounds = ParameterBounds((-1.0, -1.0) + (0.0,) * 4, (1.0, 1.0) + (0.0,) * 4)
    policy = PoseLeastSquaresPolicy(
        method=method,
        max_iterations=40,
        max_evaluations=160,
        gradient_tolerance=SOLVER_TOLERANCE,
    )
    initial = (0.125, -0.125) + (0.0,) * 4
    results = []
    for label, evaluator in (
        ("native", native),
        ("Fraction", lambda p: _linear_reference(p, rows)),
    ):
        result = recover_least_squares(evaluator, initial, bounds=bounds, policy=policy)
        assert result.stationary and result.evaluations <= 160
        reference = _linear_reference(result.parameters, rows)
        assert max(map(abs, reference.gradient)) <= SOLVER_TOLERANCE + arithmetic_bound
        assert (
            max(abs(result.parameters[0] - 0.5), abs(result.parameters[1] - 0.25))
            <= parameter_bound
        )
        assert result.parameters[2:] == (0.0,) * 4
        assert abs(result.evaluation.loss - reference.loss) <= arithmetic_bound
        assert all(not row.accepted or row.loss <= result.history[0].loss for row in result.history)
        results.append(
            {
                "evaluator": label,
                "termination": result.reason,
                "evaluations": result.evaluations,
                "parameters": result.parameters,
                "independent_gradient": reference.gradient,
                "loss": result.evaluation.loss,
                "arithmetic_bound": arithmetic_bound,
                "parameter_bound": parameter_bound,
            }
        )
    record_property("linear_convergence_reference", json.dumps(results, sort_keys=True))

"""Full-field log-density contracts, independent absorption and range oracles."""

# pyright: reportUnknownMemberType=false, reportUnknownParameterType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false
# pyright: reportMissingParameterType=false, reportPrivateUsage=false
# pyright: reportInvalidTypeForm=false
import importlib
import math
from decimal import Decimal, localcontext
from typing import Any, Literal

import numpy as np
import pytest

from dpt.contracts import ContractError, NumericalError
from dpt.transport import kernels as flight_kernels
from dpt.transport import voxel_kernels
from dpt.transport.derivatives import TransportParameter, derivative_histories
from dpt.transport.forward import prepare_transport, trace_histories
from dpt.transport.model import (
    IncompleteHistoryError,
    MaterialGrid,
    PlanarDetector,
    TransportError,
    TransportSpec,
)
from dpt.transport.rng import HistoryBatch
from dpt.transport.voxel_derivatives import prepare_voxel_density_vjp, voxel_density_vjp

wp: Any = importlib.import_module("warp")
Inputs = tuple[Any, Any, Any, Any]


def scene(
    device,
    count=32,
    estimator: Literal["analogue", "continuous-absorption"] = "continuous-absorption",
    *,
    theta=(0.5, 1.2, 0.7),
    scatter=0.0,
    scoring: Literal["photon-count", "energy-kev"] = "photon-count",
    absorption=(0.2, 0.3),
) -> tuple[Any, Inputs, Any, Any]:
    def arr(value, dtype=None):
        return wp.array(value, dtype=wp.float64 if dtype is None else dtype, device=device)

    spec = TransportSpec(
        MaterialGrid((-1.0, -1.0, 0.0), (2.0, 2.0, 1.0), (len(theta), 1, 1), 2),
        PlanarDetector((-20.0, -20.0), (40.0, 40.0), (1, 1), len(theta) + 1.0),
        80.0,
        "analytic piecewise-constant log-voxel-density fixture",
        estimator=estimator,
        scoring=scoring,
    )
    ids = np.arange(len(theta), dtype=np.int32) % 2
    transport = prepare_transport(
        spec,
        material_ids=arr(ids, wp.int32),
        absorption=arr(absorption),
        scattering=arr([scatter, scatter]),
        voxel_density=arr(theta),
        max_histories=count,
        device=device,
    )
    inputs: Inputs = (
        arr([[0.0, 0.0, -1.0]] * count, wp.vec3d),
        arr([[0.0, 0.0, 1.0]] * count, wp.vec3d),
        arr([1.0] * count),
        arr([1.3, 0.8]),
    )
    depth = np.asarray(theta) * np.asarray(absorption)[ids] * np.asarray([1.3, 0.8])[ids]
    return transport, inputs, arr, depth


def evaluate(
    transport,
    inputs: Inputs,
    arr,
    *,
    batch=None,
    active=None,
    cotangent=-2.0,
    amplitude=0.7,
    variance=True,
    slots=4,
):
    workspace = prepare_voxel_density_vjp(transport, slots_per_history=slots, active=active)
    gradient = wp.empty(
        len(transport.material_ids), dtype=wp.float64, device=transport.context.device
    )
    var = wp.empty_like(gradient) if variance else None
    result = voxel_density_vjp(
        *inputs,
        arr([cotangent]),
        batch=batch or HistoryBatch(751, 0, transport.max_histories),
        workspace=workspace,
        out_gradient=gradient,
        out_variance_of_mean=var,
        variance_law="iid-original-histories" if variance else None,
        source_amplitude=amplitude,
    )
    return workspace, result


@pytest.mark.parametrize("scoring", ["photon-count", "energy-kev"])
@pytest.mark.parametrize("vacuum", [False, True])
def test_independent_layer_chords_signed_vjp_and_inactive_vacuum(scoring, vacuum, warp_device):
    theta = (0.5, 0.0 if vacuum else 1.2, 0.7)
    transport, inputs, arr, depths = scene(warp_device, theta=theta, scoring=scoring)
    mask = arr([1, 0, 1], wp.int32) if vacuum else None
    workspace, result = evaluate(transport, inputs, arr, active=mask)
    energy = 80.0 if scoring == "energy-kev" else 1.0
    expected = -depths * math.exp(-float(depths.sum())) * -2.0 * 0.7 * energy
    np.testing.assert_allclose(result.gradient.numpy(), expected, rtol=2e-14, atol=1e-16)
    np.testing.assert_allclose(result.variance_of_mean.numpy(), 0.0, atol=1e-28)
    assert result.grid_shape == (3, 1, 1) and result.parameterization == "log-voxel-density"
    assert result.batch.count == 32 and result.checked
    assert result.variance_law == "iid-original-histories-conditional-on-cotangent"
    assert workspace.scratch_bytes == 20 * 32 * 4 + 28 * 3 + 4 + (12 if mask is None else 0)


def test_analogue_moments_match_original_history_binomial_variance(warp_device):
    count = 32768
    transport, inputs, arr, depths = scene(warp_device, count, "analogue")
    _, result = evaluate(transport, inputs, arr)
    probability = math.exp(-float(depths.sum()))
    factors = -depths * -2.0 * 0.7
    expected = factors * probability
    error = 7 * np.abs(factors) * math.sqrt(probability * (1 - probability) / count)
    assert (np.abs(result.gradient.numpy() - expected) <= error + 1e-14).all()
    # Every surviving history has the same independently derived layer score;
    # recover its observed binomial proportion from one gradient component.
    frequency = result.gradient.numpy()[0] / factors[0]
    empirical_variance = factors**2 * frequency * (1 - frequency) / (count - 1)
    np.testing.assert_allclose(result.variance_of_mean.numpy(), empirical_variance, rtol=2e-10)


@pytest.mark.parametrize("estimator", ["analogue", "continuous-absorption"])
def test_coalesced_voxel_sum_matches_existing_material_score_and_uncertainty(
    estimator, warp_device
):
    count = 4096
    transport, inputs, arr, _ = scene(warp_device, count, estimator, scatter=0.7)
    batch = HistoryBatch(3107, 2**35, count)
    workspace, result = evaluate(transport, inputs, arr, batch=batch, cotangent=1.0, amplitude=1.0)
    keys = workspace._arrays["keys"].numpy().reshape(count, 4)
    values = workspace._arrays["values"].numpy().reshape(count, 4)
    pixel = wp.empty(count, dtype=wp.int32, device=warp_device)
    status = wp.empty_like(pixel)
    derivative = wp.empty(count, dtype=wp.float64, device=warp_device)
    for material in (0, 1):
        derivative_histories(
            *inputs,
            parameter=TransportParameter("log-material-density", material),
            batch=batch,
            workspace=transport,
            out_pixel=pixel,
            out_derivative=derivative,
            out_status=status,
        )
        selected = np.isin(keys, [v for v in range(3) if v % 2 == material])
        per_history = np.where(selected, values, 0.0).sum(axis=1)
        np.testing.assert_allclose(per_history, derivative.numpy(), rtol=2e-13, atol=2e-14)
        np.testing.assert_allclose(
            result.gradient.numpy()[material::2].sum(), per_history.mean(), rtol=2e-12, atol=2e-14
        )
    for voxel in range(3):
        per_history = np.where(keys == voxel, values, 0.0).sum(axis=1)
        np.testing.assert_allclose(
            result.variance_of_mean.numpy()[voxel], np.var(per_history, ddof=1) / count, rtol=2e-12
        )
    # There is at most one slot for any history/voxel, even after re-entry.
    assert all(len(row[row >= 0]) == len(set(row[row >= 0])) for row in keys)


@pytest.mark.parametrize(
    "theta", [(0.0, 1.2, 0.7), (-1.0, 1.2, 0.7), (float("nan"), 1.2, 0.7), (float("inf"), 1.2, 0.7)]
)
def test_invalid_active_log_density_is_rejected(theta, warp_device):
    if theta[0] == 0:
        transport, _, _, _ = scene(warp_device, theta=theta)
        with pytest.raises(TransportError, match="invalid device"):
            prepare_voxel_density_vjp(transport, slots_per_history=4)
    else:
        with pytest.raises(TransportError, match="invalid device"):
            scene(warp_device, theta=theta)


@pytest.mark.parametrize("slots", [0, -2, 3, True, 2**31])
def test_sparse_budget_and_indexing_are_checked_before_allocation(slots, warp_device):
    transport, _, _, _ = scene(warp_device)
    with pytest.raises(TransportError):
        prepare_voxel_density_vjp(transport, slots_per_history=slots)
    with pytest.raises(TransportError, match="preparation budget"):
        prepare_voxel_density_vjp(transport, slots_per_history=4, max_bytes=1)


def test_sparse_exhaustion_invalidates_only_derivative_and_retains_sticky_error(warp_device):
    transport, inputs, arr, _ = scene(warp_device)
    workspace = prepare_voxel_density_vjp(transport, slots_per_history=1)
    gradient = wp.empty(3, dtype=wp.float64, device=warp_device)
    args: dict[str, Any] = dict(
        batch=HistoryBatch(17, 0, 32), workspace=workspace, out_gradient=gradient
    )
    with pytest.raises(IncompleteHistoryError, match="slots_per_history"):
        voxel_density_vjp(*inputs, arr([1.0]), **args)
    transport.check_status()  # Photon terminal codes and forward physics remain valid.
    with pytest.raises(IncompleteHistoryError, match="slots_per_history"):
        voxel_density_vjp(*inputs, arr([0.0]), **args)
    workspace.clear_status()
    gradient.fill_(0.0)
    with pytest.raises(IncompleteHistoryError):
        voxel_density_vjp(*inputs, arr([1.0]), validate=False, **args)
        workspace.check_status()


def test_shape_alias_stream_tape_and_structural_cotangent_contracts(warp_device):
    transport, inputs, arr, _ = scene(warp_device)
    workspace = prepare_voxel_density_vjp(transport, slots_per_history=4)
    batch = HistoryBatch(381, 0, 32).with_library_source("ParallelBeam")
    cotangent = arr([1.0])
    gradient = wp.empty(3, dtype=wp.float64, device=warp_device)
    args: dict[str, Any] = dict(batch=batch, workspace=workspace, out_gradient=gradient)
    with pytest.raises(ContractError, match="shape"):
        voxel_density_vjp(
            *inputs,
            cotangent,
            **{**args, "out_gradient": wp.empty(2, dtype=wp.float64, device=warp_device)},
        )
    with pytest.raises(ContractError, match="overlaps"):
        voxel_density_vjp(*inputs, cotangent, **{**args, "out_gradient": transport.voxel_density})
    with pytest.raises(ContractError, match="stream"):
        voxel_density_vjp(
            *inputs,
            cotangent,
            stream=wp.Stream(warp_device) if warp_device != "cpu" else object(),
            **args,
        )
    with wp.Tape(), pytest.raises(ContractError, match="tape"):
        voxel_density_vjp(*inputs, cotangent, **args)
    overlapping_source = HistoryBatch(381, 0, 32, "caller-new-label").with_library_source(
        "PointConeSource"
    )
    with pytest.raises(TransportError, match="disjoint"):
        voxel_density_vjp(*inputs, cotangent, cotangent_batch=overlapping_source, **args)
    independent = HistoryBatch(381, 32, 32).with_library_source("ParallelBeam")
    result = voxel_density_vjp(*inputs, cotangent, cotangent_batch=independent, **args)
    assert result.batch == batch and result.cotangent_batch == independent
    with pytest.raises(TransportError, match="variance"):
        voxel_density_vjp(*inputs, cotangent, out_variance_of_mean=wp.empty_like(gradient), **args)


def test_repeated_calls_reuse_all_owned_storage_without_allocating_arrays(warp_device, monkeypatch):
    transport, inputs, arr, _ = scene(warp_device)
    workspace = prepare_voxel_density_vjp(transport, slots_per_history=4)
    cotangent, gradient, variance = (
        arr([1.0]),
        wp.empty(3, dtype=wp.float64, device=warp_device),
        wp.empty(3, dtype=wp.float64, device=warp_device),
    )
    pointers = {name: value.ptr for name, value in workspace._arrays.items()}
    args: dict[str, Any] = dict(
        batch=HistoryBatch(518, 0, 32),
        workspace=workspace,
        out_gradient=gradient,
        out_variance_of_mean=variance,
        variance_law="iid-original-histories",
    )
    voxel_density_vjp(*inputs, cotangent, **args)
    allocations = []
    original = wp.array.__init__

    def monitored(array, *args, **kwargs):
        original(array, *args, **kwargs)
        if getattr(array, "deleter", None) is not None:
            allocations.append((str(array.device), array.capacity))

    monkeypatch.setattr(wp.array, "__init__", monitored)
    for _ in range(3):
        voxel_density_vjp(*inputs, cotangent, **args)
    assert allocations == []
    assert pointers == {name: value.ptr for name, value in workspace._arrays.items()}


def test_nonunderflow_derivative_uses_original_factors_and_log_attenuation(warp_device):
    transport, inputs, arr, _ = scene(warp_device, theta=(1.0,), absorption=(750.0, 0.3))
    inputs = (inputs[0], inputs[1], inputs[2], arr([1.0, 0.8]))
    workspace, result = evaluate(
        transport, inputs, arr, cotangent=1e300, amplitude=1.0, variance=False, slots=1
    )
    with localcontext() as context:
        context.prec = 80
        expected = float(-Decimal(750) * (-Decimal(750)).exp() * Decimal("1e300"))
    assert result.gradient.numpy()[0] != 0.0
    np.testing.assert_allclose(result.gradient.numpy()[0], expected, rtol=2e-13)
    # An overflowing unused primal can still have a representable zero VJP.
    inputs = (inputs[0], inputs[1], arr([1e300] * 32), inputs[3])
    zero = wp.empty(1, dtype=wp.float64, device=warp_device)
    voxel_density_vjp(
        *inputs,
        arr([0.0]),
        batch=HistoryBatch(18, 0, 32),
        workspace=workspace,
        out_gradient=zero,
        source_amplitude=1e300,
    )
    assert zero.numpy()[0] == 0.0


def test_independent_statistical_finite_difference_has_declared_precision(warp_device, request):
    count, confidence_z, h = 131072, 3.5, 0.25
    transport, inputs, arr, depths = scene(warp_device, count, "analogue")
    _, result = evaluate(
        transport, inputs, arr, batch=HistoryBatch(8041, 0, count), cotangent=1.0, amplitude=1.0
    )
    exact = -depths[0] * math.exp(-float(depths.sum()))
    precision = 0.45 * abs(exact)
    gradient, gradient_variance = result.gradient.numpy()[0], result.variance_of_mean.numpy()[0]
    for step, plus_seed, minus_seed in ((h, 8042, 8043), (h / 2, 8044, 8045)):
        means, variances, exact_means = [], [], []
        for sign, seed in ((1, plus_seed), (-1, minus_seed)):
            theta = (0.5 * math.exp(sign * step), 1.2, 0.7)
            physical, dynamic, _, varied_depth = scene(warp_device, count, "analogue", theta=theta)
            arrays = {
                name: wp.empty(count, dtype=dtype, device=warp_device)
                for name, dtype in (
                    ("out_pixel", wp.int32),
                    ("out_score", wp.float64),
                    ("out_energy", wp.float64),
                    ("out_events", wp.int32),
                    ("out_status", wp.int32),
                )
            }
            trace_histories(
                *dynamic, batch=HistoryBatch(seed, 0, count), workspace=physical, **arrays
            )
            scores = arrays["out_score"].numpy()
            means.append(float(scores.mean()))
            variances.append(float(np.var(scores, ddof=1) / count))
            exact_means.append(math.exp(-float(varied_depth.sum())))
        difference = (means[0] - means[1]) / (2 * step)
        exact_difference = (exact_means[0] - exact_means[1]) / (2 * step)
        sampling_sd = math.sqrt(gradient_variance + sum(variances) / (4 * step * step))
        radius = confidence_z * sampling_sd
        truncation = abs(exact_difference - exact)
        assert radius + truncation < precision
        assert abs(gradient - difference) <= radius + truncation
        # The bias-corrected statistical comparison excludes a zero or
        # wrong-sign finite difference for this declared nonzero effect.
        corrected_difference = difference - (exact_difference - exact)
        assert corrected_difference + radius < 0.0
        request.node.user_properties.extend(
            (
                (f"h_{step}_confidence_radius", radius),
                (f"h_{step}_truncation_bias", truncation),
                (f"h_{step}_fd_estimate", difference),
                (f"h_{step}_bias_corrected_ci_lower", corrected_difference - radius),
                (f"h_{step}_bias_corrected_ci_upper", corrected_difference + radius),
                (f"h_{step}_plus_batch_seed", plus_seed),
                (f"h_{step}_minus_batch_seed", minus_seed),
                ("independent_fd_history_count", count),
                ("analytic_gradient_effect", exact),
                ("precision_gate_effect_fraction", 0.45),
                ("gradient_batch_seed", 8041),
                ("normal_critical_z", confidence_z),
                ("gradient_units", "expected photons per original history per log-density"),
            )
        )
    assert abs(gradient - exact) <= confidence_z * math.sqrt(gradient_variance)


@wp.kernel
def _coalesced_range_fixture(
    active: wp.array(dtype=wp.int32),
    keys: wp.array(dtype=wp.int32),
    scores: wp.array(dtype=wp.float64),
    status: wp.array(dtype=wp.int32),
    out: wp.array(dtype=wp.float64),
):
    records = flight_kernels.VoxelRecords()
    records.enabled = 1
    records.capacity = 1
    records.history = 0
    records.active = active
    records.voxel = keys
    records.score = scores
    records.status = status
    # Repeated-visit operands isolate the range gate: weighting either
    # segment first overflows, whereas their complete-history score is finite.
    flight_kernels.record_voxel_score(records, 0, wp.float64(1.0))
    flight_kernels.record_voxel_score(
        records, 0, -wp.float64(1.0) + wp.float64(0.00000095367431640625)
    )
    out[0] = voxel_kernels.product5(
        wp.float64(1e308), wp.float64(10.0), wp.float64(1.0), wp.float64(1.0), scores[0]
    )


def test_repeated_voxel_score_cancellation_precedes_unrepresentable_segment_weights(warp_device):
    active = wp.ones(1, dtype=wp.int32, device=warp_device)
    keys = wp.full(1, -1, dtype=wp.int32, device=warp_device)
    scores = wp.zeros(1, dtype=wp.float64, device=warp_device)
    status = wp.zeros(1, dtype=wp.int32, device=warp_device)
    output = wp.empty(1, dtype=wp.float64, device=warp_device)
    wp.launch(
        _coalesced_range_fixture,
        dim=1,
        inputs=[active, keys, scores, status, output],
        device=warp_device,
    )
    assert status.numpy()[0] == 0 and keys.numpy()[0] == 0
    assert scores.numpy()[0] == 2.0**-20
    assert math.isinf(1e308 * 10.0)
    with localcontext() as context:
        context.prec = 80
        expected = float(Decimal("1e308") * Decimal(10) * (Decimal(2) ** -20))
    np.testing.assert_allclose(output.numpy()[0], expected, rtol=2e-15)


@pytest.mark.parametrize("cotangent", [float("nan"), float("inf")])
def test_nonfinite_cotangent_is_rejected_before_scoring(cotangent, warp_device):
    transport, inputs, arr, _ = scene(warp_device)
    with pytest.raises(TransportError, match="invalid device"):
        evaluate(transport, inputs, arr, cotangent=cotangent)


def test_unrepresentable_final_vjp_is_a_checked_numerical_failure(warp_device):
    transport, inputs, arr, _ = scene(warp_device)
    with pytest.raises(NumericalError, match="nonfinite"):
        evaluate(transport, inputs, arr, cotangent=1e308, amplitude=1e308, variance=False)


@pytest.mark.parametrize("values", [[2, 0, 1], [-1, 1, 1]])
def test_active_mask_rejects_nonbinary_membership(values, warp_device):
    transport, _, arr, _ = scene(warp_device)
    with pytest.raises(TransportError, match="invalid device"):
        prepare_voxel_density_vjp(transport, slots_per_history=4, active=arr(values, wp.int32))

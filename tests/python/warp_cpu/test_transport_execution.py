"""Portable transport replay, budget and sampling-law contracts.

Statistical tolerances use the independently known Bernoulli variance;
exact identity checks cover replay and chunking separately.
"""

import importlib
import json
import math
from collections.abc import Callable

# pytest.approx has untyped optional-dependency stubs.
# pyright: reportUnknownMemberType=false
from dataclasses import replace
from typing import Any, Literal

import numpy as np
import pytest
from scipy import stats

from dpt.contracts import ContractError, NumericalError
from dpt.transport.derivatives import TransportParameter, derivative_histories
from dpt.transport.estimators import (
    history_mean,
    history_moments,
    independent_squared_gradient,
    independent_squared_loss,
    prepare_estimators,
)
from dpt.transport.forward import TransportWorkspace, prepare_transport, trace_histories
from dpt.transport.model import (
    IncompleteHistoryError,
    MaterialGrid,
    PlanarDetector,
    TransportError,
    TransportSpec,
)
from dpt.transport.rng import HistoryBatch
from dpt.validation.transport import absorbing_layers

wp: Any = importlib.import_module("warp")
DEVICE = "cpu"


@pytest.fixture(autouse=True)
def _device_scope(warp_device: str):
    global DEVICE
    DEVICE = warp_device
    with wp.ScopedDevice(DEVICE):
        yield


def _array(values: Any, dtype: Any = None) -> Any:
    return wp.array(values, dtype=wp.float64 if dtype is None else dtype, device=DEVICE)


def _slab(
    histories: int,
    *,
    depths: tuple[float, ...] = (0.3, 0.7),
    scatter: float = 0.0,
    max_events: int = 4096,
    max_crossings: int = 65536,
) -> tuple[TransportWorkspace, list[Any], dict[str, Any]]:
    layers = len(depths)
    spec = TransportSpec(
        MaterialGrid((-10.0, -10.0, 0.0), (20.0, 20.0, 1.0), (layers, 1, 1), layers),
        PlanarDetector((-10.0, -10.0), (20.0, 20.0), (1, 1), float(layers + 1)),
        80.0,
        "analytic homogeneous absorption layers",
        max_events=max_events,
        max_crossings=max_crossings,
    )
    workspace = prepare_transport(
        spec,
        material_ids=_array(list(range(layers)), wp.int32),
        absorption=_array(list(depths)),
        scattering=_array([scatter] * layers),
        device=DEVICE,
        max_histories=histories,
    )
    inputs = [
        wp.full(histories, wp.vec3d(0.0, 0.0, -1.0), dtype=wp.vec3d, device=DEVICE),
        wp.full(histories, wp.vec3d(0.0, 0.0, 1.0), dtype=wp.vec3d, device=DEVICE),
        wp.full(histories, 1.0, dtype=wp.float64, device=DEVICE),
        _array([1.0] * layers),
    ]
    outputs = {
        f"out_{name}": wp.empty(histories, dtype=dtype, device=DEVICE)
        for name, dtype in (
            ("pixel", wp.int32),
            ("score", wp.float64),
            ("energy", wp.float64),
            ("events", wp.int32),
            ("status", wp.int32),
        )
    }
    return workspace, inputs, outputs


def test_layered_pure_absorption_mean_and_density_score(
    record_testsuite_property: Callable[[str, object], None],
) -> None:
    # Seven declared production-law gates per device (this survival gate plus
    # cosine/azimuth at three energies) have family alpha <= 7/16384 < 0.00043.
    # Exact tails calibrate the declared IID Bernoulli midpoint-grid model;
    # the recorded production Philox seed remains a deterministic test fixture.
    # No normal approximation or estimated SE determines acceptance.
    histories, alpha = 2**23, 2**-14
    workspace, inputs, outputs = _slab(histories)
    batch = HistoryBatch(92761, 2**40, histories)
    reference = absorbing_layers((0.3, 0.7))
    physical_probability = float(reference.transmission)
    uniform_grid = 2**32
    expected = math.floor(uniform_grid * physical_probability + 0.5) / uniform_grid
    assert abs(expected - physical_probability) <= 0.5 / uniform_grid
    lower = int(stats.binom.ppf(alpha / 2, histories, expected))
    upper = int(stats.binom.ppf(1 - alpha / 2, histories, expected))
    false_failure = float(
        stats.binom.cdf(lower - 1, histories, expected) + stats.binom.sf(upper, histories, expected)
    )
    assert false_failure <= alpha
    relative_halfwidth = max(expected - lower / histories, upper / histories - expected) / expected
    assert relative_halfwidth <= 0.003
    # The planned effect is a relative shift of the observable survival
    # probability, rather than the similarly sized extinction/free-path scale.
    shifted_probabilities = tuple(physical_probability * (1 + sign * 0.003) for sign in (-1, 1))
    powers = []
    for shifted in shifted_probabilities:
        grid_probability = math.floor(uniform_grid * shifted + 0.5) / uniform_grid
        exact_power = float(
            stats.binom.cdf(lower - 1, histories, grid_probability)
            + stats.binom.sf(upper, histories, grid_probability)
        )
        assert exact_power >= 0.99
        powers.append(exact_power)

    trace_histories(*inputs, batch=batch, workspace=workspace, **outputs)
    scores = outputs["out_score"].numpy()
    assert np.all((scores == 0) | (scores == 1))
    survivors = int(np.count_nonzero(scores))
    assert lower <= survivors <= upper
    measured = survivors / histories
    del scores
    mean = wp.empty(1, dtype=wp.float64, device=DEVICE)
    variance = wp.empty_like(mean)
    history_moments(
        outputs["out_pixel"],
        outputs["out_score"],
        outputs["out_status"],
        batch=batch,
        workspace=prepare_estimators(workspace),
        out_mean=mean,
        out_variance_of_mean=variance,
    )
    assert float(mean.numpy()[0]) == measured
    # Nonnegative FP64 terms admit the standard gamma_n summation bound for
    # any atomic order. Eight additional operations cover residual/scale,
    # square, normalisation and the output's sqrt/square, plus the host oracle.
    unit_roundoff = np.finfo(np.float64).eps / 2
    gamma_n = histories * unit_roundoff / (1 - histories * unit_roundoff)
    gamma_local = 8 * unit_roundoff / (1 - 8 * unit_roundoff)
    variance_roundoff_bound = (1 + gamma_n) * (1 + gamma_local) - 1
    conditional_variance = measured * (1 - measured) / (histories - 1)
    assert float(variance.numpy()[0]) == pytest.approx(
        conditional_variance, rel=variance_roundoff_bound
    )
    derivative = wp.empty(histories, dtype=wp.float64, device=DEVICE)
    derivative_histories(
        *inputs,
        parameter=TransportParameter("log-material-density", 0),
        batch=batch,
        workspace=workspace,
        out_pixel=outputs["out_pixel"],
        out_derivative=derivative,
        out_status=outputs["out_status"],
    )
    values = derivative.numpy()
    # The exact surviving-history score makes the derivative's statistical
    # gate identical to the already calibrated survival gate, not a new look.
    assert np.all((values == 0) | (np.abs(values + 0.3) < 2e-15))
    assert float(values.mean()) == pytest.approx(-0.3 * measured, rel=5e-14)
    del values, derivative
    # Scaling exponential free paths by s is equivalent in this absorbing slab
    # to dividing all extinction coefficients by s. These two actual history
    # runs instantiate the planned +/-0.3% survival mutants without changing
    # the production sampler, fixed seed or history identities.
    mutants = []
    for shifted, power in zip(shifted_probabilities, powers, strict=True):
        extinction_scale = -math.log(shifted)
        inputs[3].fill_(extinction_scale)
        trace_histories(*inputs, batch=batch, workspace=workspace, **outputs)
        mutated_survivors = int(np.count_nonzero(outputs["out_score"].numpy()))
        assert not lower <= mutated_survivors <= upper
        mutants.append(
            {
                "physical_survival_probability": shifted,
                "relative_survival_shift": shifted / physical_probability - 1,
                "flight_scale": 1 / extinction_scale,
                "actual_survivors": mutated_survivors,
                "planned_exact_IID_grid_power": power,
            }
        )
    record_testsuite_property(
        f"absorption-law-{DEVICE}",
        json.dumps(
            {
                "device": DEVICE,
                "histories": histories,
                "sampler_seed": batch.seed,
                "first_history": batch.first_history,
                "physical_probability": physical_probability,
                "midpoint_grid_probability": expected,
                "uniform_grid_bits": 32,
                "accept_count_interval": [lower, upper],
                "gate_alpha": alpha,
                "exact_IID_grid_null_failure": false_failure,
                "relative_acceptance_halfwidth": relative_halfwidth,
                "actual_survivors": survivors,
                "actual_mean": float(mean.numpy()[0]),
                "actual_variance_of_mean": float(variance.numpy()[0]),
                "variance_relative_roundoff_bound": variance_roundoff_bound,
                "variance_actual_relative_error": (
                    abs(float(variance.numpy()[0]) - conditional_variance) / conditional_variance
                ),
                "actual_mutants": mutants,
                "calibration": "IID 32-bit midpoint uniform null; recorded fixed Philox histories",
                "family_scope": "seven declared production null-law gates only",
                "family_null_alpha_upper_per_device": 7 * alpha,
            },
            sort_keys=True,
        ),
    )


def test_counter_identity_is_independent_of_launch_chunking() -> None:
    workspace, inputs, outputs = _slab(257)
    batch = HistoryBatch(451, 2**50, 257)
    trace_histories(*inputs, batch=batch, workspace=workspace, **outputs)
    expected = {name: value.numpy().copy() for name, value in outputs.items()}
    for first, count in ((0, 127), (127, 130)):
        chunk_inputs = [array[first : first + count] for array in inputs[:3]] + inputs[3:]
        chunk_outputs = {name: array[first : first + count] for name, array in outputs.items()}
        trace_histories(
            *chunk_inputs,
            batch=HistoryBatch(batch.seed, batch.first_history + first, count),
            workspace=workspace,
            **chunk_outputs,
        )
    for name, value in outputs.items():
        assert (value.numpy() == expected[name]).all()


def test_zero_extinction_and_absorption_outcomes_remain_distinct() -> None:
    workspace, inputs, outputs = _slab(32, depths=(0.0, 0.0))
    trace_histories(*inputs, batch=HistoryBatch(33, 0, 32), workspace=workspace, **outputs)
    assert (outputs["out_score"].numpy() == 1.0).all()
    assert (outputs["out_status"].numpy() == 0).all()
    assert (outputs["out_events"].numpy() == 0).all()


def test_crossing_budget_invalidates_entire_estimate() -> None:
    workspace, inputs, outputs = _slab(32, depths=(0.0, 0.0), max_crossings=1)
    with pytest.raises(IncompleteHistoryError):
        trace_histories(*inputs, batch=HistoryBatch(1, 0, 32), workspace=workspace, **outputs)
    assert (outputs["out_status"].numpy() == 3).all()
    with pytest.raises(IncompleteHistoryError):
        workspace.check_status()


def test_event_budget_never_turns_surviving_scatter_into_absorption() -> None:
    workspace, inputs, outputs = _slab(256, depths=(0.0,), scatter=1e6, max_events=1)
    with pytest.raises(IncompleteHistoryError):
        trace_histories(*inputs, batch=HistoryBatch(19, 0, 256), workspace=workspace, **outputs)
    assert (outputs["out_status"].numpy() == 2).any()
    assert not (outputs["out_status"].numpy() == 1).any()


def test_source_amplitude_derivative_survives_zero_amplitude() -> None:
    workspace, inputs, outputs = _slab(128, depths=(0.0,))
    derivative = wp.empty(128, dtype=wp.float64, device=DEVICE)
    derivative_histories(
        *inputs,
        parameter=TransportParameter("source-amplitude"),
        batch=HistoryBatch(8, 0, 128),
        workspace=workspace,
        out_pixel=outputs["out_pixel"],
        out_derivative=derivative,
        out_status=outputs["out_status"],
        source_amplitude=0.0,
    )
    assert (derivative.numpy() == 1.0).all()


def test_independent_loss_products_and_overlap_rejection() -> None:
    workspace, _, _ = _slab(2)
    output = wp.empty(1, dtype=wp.float64, device=DEVICE)
    batches = (HistoryBatch(3, 0, 2), HistoryBatch(3, 2, 2))
    independent_squared_gradient(
        _array([2.0]),
        _array([-0.5]),
        _array([1.0]),
        _array([3.0]),
        batches=batches,
        workspace=workspace,
        out_components=output,
    )
    assert float(output.numpy()[0]) == -1.5
    independent_squared_loss(
        _array([2.0]),
        _array([0.0]),
        _array([1.0]),
        _array([3.0]),
        batches=batches,
        workspace=workspace,
        out_components=output,
    )
    assert float(output.numpy()[0]) == -1.5  # unbiased estimates may be negative
    with pytest.raises(TransportError, match="disjoint"):
        independent_squared_loss(
            _array([2.0]),
            _array([0.0]),
            _array([1.0]),
            _array([3.0]),
            batches=(batches[0], batches[0]),
            workspace=workspace,
            out_components=output,
        )


def test_compton_energy_never_increases_and_missing_coefficients_fail() -> None:
    workspace, inputs, outputs = _slab(512, depths=(0.0,), scatter=1.0)
    spec = replace(
        workspace.spec,
        scattering_law="free-electron-compton",
        coefficient_energies_kev=(0.0, 80.0),
        coefficient_provenance="analytic constant-table model",
    )
    compton = prepare_transport(
        spec,
        material_ids=_array([0], wp.int32),
        absorption=_array([0.0, 0.0]),
        scattering=_array([1.0, 1.0]),
        device=DEVICE,
        max_histories=512,
    )
    trace_histories(*inputs, batch=HistoryBatch(67, 0, 512), workspace=compton, **outputs)
    energies = outputs["out_energy"].numpy()
    assert ((energies > 0) & (energies <= 80)).all()
    assert (energies < 80).any()
    narrow = prepare_transport(
        replace(spec, coefficient_energies_kev=(79.999, 80.0)),
        material_ids=_array([0], wp.int32),
        absorption=_array([0.0, 0.0]),
        scattering=_array([1.0, 1.0]),
        device=DEVICE,
        max_histories=512,
    )
    with pytest.raises(TransportError, match="energy support"):
        trace_histories(*inputs, batch=HistoryBatch(67, 0, 512), workspace=narrow, **outputs)


@pytest.mark.parametrize("baseline,perturbation", [(1.0, 1e-7), (1e154, 1e145)])
def test_centred_moments_preserve_small_variance_and_avoid_raw_square_overflow(
    baseline: float,
    perturbation: float,
) -> None:
    import statistics

    count = 1024
    workspace, _, _ = _slab(count, depths=(0.0,))
    samples = [baseline + (perturbation if index % 2 else -perturbation) for index in range(count)]
    mean = wp.empty(1, dtype=wp.float64, device=DEVICE)
    variance = wp.empty_like(mean)
    history_moments(
        _array([0] * count, wp.int32),
        _array(samples),
        _array([0] * count, wp.int32),
        batch=HistoryBatch(9, 0, count),
        workspace=prepare_estimators(workspace),
        out_mean=mean,
        out_variance_of_mean=variance,
    )
    assert float(mean.numpy()[0]) == pytest.approx(statistics.mean(samples), rel=1e-13)
    assert float(variance.numpy()[0]) == pytest.approx(
        statistics.variance(samples) / count, rel=1e-6
    )


def test_missing_scores_contribute_to_variance_denominator() -> None:
    workspace, _, _ = _slab(4, depths=(0.0,))
    mean = wp.empty(1, dtype=wp.float64, device=DEVICE)
    variance = wp.empty_like(mean)
    history_moments(
        _array([0, -1, 0, -1], wp.int32),
        _array([2.0, 0.0, 2.0, 0.0]),
        _array([0, 1, 0, 0], wp.int32),
        batch=HistoryBatch(6, 0, 4),
        workspace=prepare_estimators(workspace),
        out_mean=mean,
        out_variance_of_mean=variance,
    )
    assert float(mean.numpy()[0]) == 1.0
    assert float(variance.numpy()[0]) == pytest.approx(1 / 3)


@pytest.mark.parametrize("score", [1e308, 1e-320])
def test_mean_preserves_large_and_subnormal_scores(score: float) -> None:
    from dpt.transport.estimators import history_mean

    workspace, _, _ = _slab(4, depths=(0.0,))
    mean = wp.empty(1, dtype=wp.float64, device=DEVICE)
    history_mean(
        _array([0] * 4, wp.int32),
        _array([score] * 4),
        _array([0] * 4, wp.int32),
        batch=HistoryBatch(32, 0, 4),
        workspace=prepare_estimators(workspace),
        out_mean=mean,
    )
    assert float(mean.numpy()[0]) == score


def test_variance_rescales_after_accumulating_tiny_contributions() -> None:
    from decimal import Decimal, localcontext

    workspace, _, _ = _slab(4, depths=(0.0,))
    mean = wp.empty(1, dtype=wp.float64, device=DEVICE)
    variance = wp.empty_like(mean)
    history_moments(
        _array([0] * 4, wp.int32),
        _array([0.0, 0.0, 8e-162, 8e-162]),
        _array([0] * 4, wp.int32),
        batch=HistoryBatch(83, 0, 4),
        workspace=prepare_estimators(workspace),
        out_mean=mean,
        out_variance_of_mean=variance,
    )
    with localcontext() as context:
        context.prec = 80
        expected = float(Decimal.from_float(8e-162) ** 2 / 12)
    assert expected > 0
    assert float(variance.numpy()[0]) == expected


def test_energy_score_preserves_compensating_source_amplitude() -> None:
    original, inputs, outputs = _slab(4, depths=(0.0,))
    workspace = prepare_transport(
        replace(original.spec, scoring="energy-kev"),
        material_ids=_array([0], wp.int32),
        absorption=_array([0.0]),
        scattering=_array([0.0]),
        device=DEVICE,
        max_histories=4,
    )
    inputs[2] = _array([1e308] * 4)
    trace_histories(
        *inputs,
        batch=HistoryBatch(82, 0, 4),
        workspace=workspace,
        source_amplitude=1e-308,
        **outputs,
    )
    assert all(abs(float(value) - 80.0) < 1e-12 for value in outputs["out_score"].numpy())


def test_density_derivative_keeps_information_lost_by_primal_underflow() -> None:
    workspace, inputs, outputs = _slab(1024, depths=(2.0,))
    inputs[2] = _array([float.fromhex("0x0.0000000000001p-1022")] * 1024)
    derivative = wp.empty(1024, dtype=wp.float64, device=DEVICE)
    batch = HistoryBatch(55583, 0, 1024)
    trace_histories(
        *inputs,
        batch=batch,
        workspace=workspace,
        source_amplitude=0.4,
        **outputs,
    )
    assert (outputs["out_score"].numpy() == 0.0).all()
    derivative_histories(
        *inputs,
        parameter=TransportParameter("log-material-density", 0),
        batch=batch,
        workspace=workspace,
        source_amplitude=0.4,
        out_pixel=outputs["out_pixel"],
        out_derivative=derivative,
        out_status=outputs["out_status"],
    )
    values = derivative.numpy()
    assert (values < 0.0).any()
    assert all(
        value == 0.0 or value == -float.fromhex("0x0.0000000000001p-1022") for value in values
    )


@pytest.mark.parametrize(
    "histories,pixels", [(31, 3), (32, 1), (33, 65), (63, 3), (65, 3), (257, 3), (513, 65)]
)
def test_sparse_warp_tallies_preserve_signed_means_and_centred_variances(
    histories: int, pixels: int
) -> None:
    base, _, _ = _slab(histories)
    detector = PlanarDetector((-10.0, -10.0), (20.0 / pixels, 20.0), (pixels, 1), 3.0)
    workspace = prepare_transport(
        replace(base.spec, detector=detector),
        material_ids=base.material_ids,
        absorption=base.absorption,
        scattering=base.scattering,
        device=DEVICE,
        max_histories=histories,
    )
    # Interleaved peer masks, signed derivative scores, zero hits, misses and a
    # partial final warp exercise the reduction independently of photon physics.
    targets = [-1 if i % 5 == 0 else (i * 7 + i // 11) % pixels for i in range(histories)]
    pattern = (1.0, -0.5, 0.125, 0.0, 1e-100)
    values = [0.0 if p == -1 else pattern[i % len(pattern)] for i, p in enumerate(targets)]
    mean = wp.empty(pixels, dtype=wp.float64, device=DEVICE)
    variance = wp.empty_like(mean)
    history_moments(
        _array(targets, wp.int32),
        _array(values),
        wp.zeros(histories, dtype=wp.int32, device=DEVICE),
        batch=HistoryBatch(371, 0, histories),
        workspace=prepare_estimators(workspace),
        out_mean=mean,
        out_variance_of_mean=variance,
    )
    expected_mean = [
        math.fsum(v for t, v in zip(targets, values, strict=True) if t == p) / histories
        for p in range(pixels)
    ]
    expected_variance = [
        math.fsum(
            ((v if t == p else 0.0) - expected_mean[p]) ** 2
            for t, v in zip(targets, values, strict=True)
        )
        / (histories * (histories - 1))
        for p in range(pixels)
    ]
    assert mean.numpy().tolist() == pytest.approx(expected_mean, rel=2e-13, abs=1e-115)
    assert variance.numpy().tolist() == pytest.approx(expected_variance, rel=2e-13, abs=1e-215)


def test_all_missed_histories_have_empty_ballot_membership() -> None:
    histories = 17
    workspace, _, _ = _slab(histories)
    mean = wp.empty(1, dtype=wp.float64, device=DEVICE)
    variance = wp.empty_like(mean)
    history_moments(
        _array([-1] * histories, wp.int32),
        _array([0.0] * histories),
        _array([1] * histories, wp.int32),
        batch=HistoryBatch(371, 0, histories),
        workspace=prepare_estimators(workspace),
        out_mean=mean,
        out_variance_of_mean=variance,
    )
    assert mean.numpy().tolist() == [0.0]
    assert variance.numpy().tolist() == [0.0]


@pytest.mark.parametrize("histories", [2, 31, 32, 33, 63, 64, 65, 257])
def test_ballot_tallies_preserve_disjoint_sparse_peer_groups(histories: int) -> None:
    workspace, _, _ = _slab(histories, depths=(0.0,))
    moments = prepare_estimators(workspace)
    # Alternating missed warps and single-lane peer groups exercise zero masks,
    # sparse membership and launch tails independently of the transport sampler.
    targets = [0 if i // 32 % 2 == 0 and i % 32 in (0, 7, 23, 31) else -1 for i in range(histories)]
    values = [(-1.0 if i % 2 else 2.0) if p == 0 else 0.0 for i, p in enumerate(targets)]
    pixel = _array(targets, wp.int32)
    score = _array(values)
    status = _array([0 if p == 0 else 1 for p in targets], wp.int32)
    mean = wp.empty(1, dtype=wp.float64, device=DEVICE)
    variance = wp.empty_like(mean)
    batch = HistoryBatch(491, 0, histories)
    expected_mean = math.fsum(values) / histories
    expected_variance = math.fsum((value - expected_mean) ** 2 for value in values) / (
        histories * (histories - 1)
    )
    history_moments(
        pixel,
        score,
        status,
        batch=batch,
        workspace=moments,
        out_mean=mean,
        out_variance_of_mean=variance,
    )
    assert float(mean.numpy()[0]) == pytest.approx(expected_mean, rel=2e-14)
    assert float(variance.numpy()[0]) == pytest.approx(expected_variance, rel=2e-14)
    history_mean(pixel, score, status, batch=batch, workspace=moments, out_mean=mean)
    assert float(mean.numpy()[0]) == pytest.approx(expected_mean, rel=2e-14)


def test_numerical_transport_failure_uses_shared_error_type() -> None:
    workspace, inputs, outputs = _slab(33, depths=(1e308,))
    inputs[3] = _array([1e308])
    with pytest.raises(NumericalError):
        trace_histories(*inputs, batch=HistoryBatch(823, 0, 33), workspace=workspace, **outputs)
    assert (outputs["out_status"].numpy() == 4).all()


@pytest.mark.parametrize(
    "terminal,error_type",
    [(2, IncompleteHistoryError), (4, NumericalError), (5, ContractError), (7, ContractError)],
)
def test_history_mean_preserves_terminal_failure_type(
    terminal: int, error_type: type[Exception]
) -> None:
    workspace, _, _ = _slab(33)
    mean = wp.empty(1, dtype=wp.float64, device=DEVICE)
    with pytest.raises(error_type):
        history_mean(
            _array([-1] * 33, wp.int32),
            _array([0.0] * 33),
            _array([terminal] * 33, wp.int32),
            batch=HistoryBatch(47, 0, 33),
            workspace=prepare_estimators(workspace),
            out_mean=mean,
        )


def test_invalid_product_input_is_not_a_recoverable_numerical_trial() -> None:
    workspace, _, _ = _slab(2)
    with pytest.raises(ContractError):
        independent_squared_gradient(
            _array([math.nan]),
            _array([1.0]),
            _array([0.0]),
            _array([1.0]),
            batches=(HistoryBatch(9, 0, 2), HistoryBatch(9, 2, 2)),
            workspace=workspace,
            out_components=wp.empty(1, dtype=wp.float64, device=DEVICE),
        )


@pytest.mark.parametrize("block_dim", [64, 128, 256])
def test_exact_corner_crossings_preserve_vacuum_path(block_dim: Literal[64, 128, 256]) -> None:
    histories = 33
    spec = TransportSpec(
        MaterialGrid((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (2, 2, 2), 1),
        PlanarDetector((0.0, 0.0), (4.0, 4.0), (1, 1), 3.0),
        80.0,
        "exact vacuum cube fixture",
        max_crossings=2,
        block_dim=block_dim,
    )
    workspace = prepare_transport(
        spec,
        material_ids=_array([0] * 8, wp.int32),
        absorption=_array([0.0]),
        scattering=_array([0.0]),
        device=DEVICE,
        max_histories=histories,
    )
    direction = 1.0 / math.sqrt(3.0)
    out_pixel = wp.empty(histories, dtype=wp.int32, device=DEVICE)
    out_score = wp.empty(histories, dtype=wp.float64, device=DEVICE)
    out_status = wp.empty_like(out_pixel)
    trace_histories(
        _array([[-1.0, -1.0, -1.0]] * histories, wp.vec3d),
        _array([[direction] * 3] * histories, wp.vec3d),
        _array([1.0] * histories),
        _array([1.0]),
        batch=HistoryBatch(55, 0, histories),
        workspace=workspace,
        out_pixel=out_pixel,
        out_score=out_score,
        out_energy=wp.empty_like(out_score),
        out_events=wp.empty_like(out_pixel),
        out_status=out_status,
    )
    assert out_pixel.numpy().tolist() == [0] * histories
    assert out_score.numpy().tolist() == [1.0] * histories
    assert out_status.numpy().tolist() == [0] * histories

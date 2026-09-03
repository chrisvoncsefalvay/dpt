"""CPU-safe transport contracts and independent probability-space references."""

import math

# pytest.approx has untyped optional-dependency stubs.
# pyright: reportUnknownMemberType=false
from dataclasses import replace
from typing import Any

import pytest

from dpt.contracts import ContractError
from dpt.transport.derivatives import CAPABILITIES, TransportParameter
from dpt.transport.estimators import ReplicateEstimate, summarise_replicates
from dpt.transport.model import MaterialGrid, PlanarDetector, TransportError, TransportSpec
from dpt.transport.rng import HistoryBatch, require_independent
from dpt.validation.transport import (
    absorbing_layers,
    bernoulli_inverse,
    compton_energy,
    klein_nishina_total_ratio,
)


def _spec() -> TransportSpec:
    return TransportSpec(
        MaterialGrid((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (8, 8, 8), 2),
        PlanarDetector((0.0, 0.0), (1.0, 1.0), (8, 8), 10.0),
        80.0,
        "analytic test coefficients; no physical material claim",
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("energy_kev", 0.0),
        ("coefficient_provenance", " "),
        ("max_events", 0),
        ("max_crossings", True),
        ("max_angle_trials", 2**31),
        ("scattering_law", "rayleigh"),
        ("coefficient_energies_kev", (90.0, 100.0)),
        ("coefficient_energies_kev", (1.0, 80.0, 80.0)),
    ],
)
def test_reject_invalid_transport_policy(field: str, value: Any) -> None:
    with pytest.raises(TransportError):
        replace(_spec(), **{field: value})


def test_compton_requires_declared_energy_coefficients() -> None:
    with pytest.raises(TransportError, match="energy-dependent"):
        replace(_spec(), scattering_law="free-electron-compton")
    spec = replace(
        _spec(), scattering_law="free-electron-compton", coefficient_energies_kev=(0.1, 20.0, 80.0)
    )
    assert spec.energy_nodes == (0.1, 20.0, 80.0)


def test_unresolvable_grid_and_detector_geometry_rejected() -> None:
    with pytest.raises(TransportError, match="distinguishable"):
        MaterialGrid((1e100, 0.0, 0.0), (1.0, 1.0, 1.0), (2, 2, 2), 1)
    with pytest.raises(TransportError, match="above"):
        replace(_spec(), detector=replace(_spec().detector, z_mm=8.0))


def test_history_identity_wrap_overlap_and_source_reuse() -> None:
    with pytest.raises(TransportError, match="wrap"):
        HistoryBatch(42, 2**64 - 1, 2)
    left = HistoryBatch(42, 2**48, 1024, "source-left")
    require_independent(left, HistoryBatch(42, 2**48 + 1024, 1024, "source-right"))
    with pytest.raises(TransportError, match="disjoint"):
        require_independent(left, HistoryBatch(42, 2**48 + 1023, 1, "source-right"))
    with pytest.raises(TransportError, match="disjoint"):
        require_independent(left, HistoryBatch(88, 0, 1024, "source-left"))
    assert HistoryBatch(1, 0, 2).independent_of(HistoryBatch(1, 2, 2))


def test_capability_boundary_rejects_moving_geometry() -> None:
    supported = {entry.parameter for entry in CAPABILITIES if entry.supported}
    assert supported == {"log-material-density", "source-amplitude", "log-source-amplitude"}
    invalid: Any = "moving-geometry"
    with pytest.raises(TransportError, match="unsupported"):
        TransportParameter(invalid)
    with pytest.raises(TransportError):
        TransportParameter("log-material-density", True)


def test_original_history_uncertainty_does_not_count_reused_batches_twice() -> None:
    estimates = tuple(
        ReplicateEstimate(value, (HistoryBatch(11, index * 16, 16),))
        for index, value in enumerate((1.0, 2.0, 3.0))
    )
    summary = summarise_replicates(estimates)
    assert summary.mean == 2.0
    assert summary.standard_error == pytest.approx(math.sqrt(1 / 3))
    with pytest.raises(TransportError, match="disjoint"):
        summarise_replicates((estimates[0], estimates[0]))


def test_layer_reference_and_log_density_derivatives() -> None:
    combined = absorbing_layers((0.2, 0.3, 0.5))
    assert float(combined.transmission) == pytest.approx(math.exp(-1.0))
    assert float(sum(combined.log_density_derivative)) == pytest.approx(
        -float(combined.transmission)
    )
    assert float(combined.variance_per_history) == pytest.approx(math.exp(-1) * (1 - math.exp(-1)))


@pytest.mark.parametrize(
    "probability,amplitude,observed",
    [
        (0.0, 2.0, 1.0),
        (1.0, 2.0, 1.0),
        (0.2, 4.0, 0.3),
        (0.9, 0.0, 1.2),
    ],
)
def test_bernoulli_inverse_reference_self_test(
    probability: float,
    amplitude: float,
    observed: float,
) -> None:
    """Check the analytic reference identity used by production-estimator tests."""
    reference = bernoulli_inverse(probability, amplitude, observed)
    assert reference.independent_gradient_expectation == pytest.approx(
        reference.true_amplitude_gradient
    )
    assert (
        reference.reused_history_gradient_expectation - reference.true_amplitude_gradient
        == pytest.approx(amplitude * probability * (1 - probability))
    )


def test_compton_energy_and_thomson_limit() -> None:
    assert float(compton_energy(80.0, 1.0)) == 80.0
    backscatter = float(compton_energy(80.0, -1.0))
    assert 0 < backscatter < 80.0
    assert float(klein_nishina_total_ratio(1e-6)) == pytest.approx(1.0, rel=1e-8)
    assert 0 < float(klein_nishina_total_ratio(80.0)) < 1


def test_asymmetric_grid_storage_shape_matches_canonical_zyx() -> None:
    grid = MaterialGrid((10.0, 20.0, 30.0), (2.0, 3.0, 4.0), (5, 6, 7), 1)
    assert grid.shape_xyz == (7, 6, 5)
    assert grid.upper_mm == (24.0, 38.0, 50.0)


@pytest.mark.parametrize(
    "origin,extent", [((1e100, 0.0, 0.0), (1.0, 0.0)), ((0.0, 0.0, 0.0), (-1.0, 0.0))]
)
def test_parallel_source_rejects_unrepresentable_or_negative_support(
    origin: tuple[float, float, float],
    extent: tuple[float, float],
) -> None:
    from dpt.transport.source import ParallelBeam

    with pytest.raises(TransportError):
        ParallelBeam(origin, extent)


def test_runtime_list_arguments_are_frozen_by_model_contracts() -> None:
    origin: Any = [0.0, 0.0, 0.0]
    shape: Any = [2, 3, 4]
    grid = MaterialGrid(origin, (1.0, 1.0, 1.0), shape, 1)
    origin[0] = 999.0
    shape[0] = 999
    assert grid.origin_mm == (0.0, 0.0, 0.0)
    assert grid.shape == (2, 3, 4)
    energies: Any = [10.0, 80.0]
    spec = replace(_spec(), coefficient_energies_kev=energies)
    energies[0] = 70.0
    assert spec.coefficient_energies_kev == (10.0, 80.0)


@pytest.mark.parametrize(
    "field,value", [("energy_kev", True), ("coefficient_energies_kev", (False, 80.0))]
)
def test_boolean_physical_scalars_are_rejected(field: str, value: Any) -> None:
    with pytest.raises(TransportError):
        replace(_spec(), **{field: value})


def test_replicate_mean_and_error_preserve_binary64_range() -> None:
    equal = tuple(ReplicateEstimate(1e308, (HistoryBatch(91, i * 2, 2),)) for i in range(2))
    assert summarise_replicates(equal).mean == 1e308
    assert summarise_replicates(equal).standard_error == 0.0
    opposite = (
        ReplicateEstimate(1e308, (HistoryBatch(91, 0, 2),)),
        ReplicateEstimate(-1e308, (HistoryBatch(91, 2, 2),)),
    )
    assert summarise_replicates(opposite).standard_error == pytest.approx(1e308)


def test_transport_contract_errors_share_the_operator_hierarchy() -> None:
    from dpt.transport.source import ParallelBeam

    with pytest.raises(ContractError) as caught:
        ParallelBeam((0.0, 0.0, -1.0), (0.0, 0.0), weight=-1.0)
    assert isinstance(caught.value, TransportError)

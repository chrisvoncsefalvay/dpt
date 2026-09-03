"""Replica-axis replay agrees with scalar laws, counters and guarded checkpoints.

Counter identities preserve sampled paths; CUDA atomic detector reductions have
order-dependent binary64 rounding and do not promise bitwise repeatability.
"""

# pyright: reportUnknownMemberType=false, reportUnknownParameterType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false
# pyright: reportMissingParameterType=false, reportUntypedFunctionDecorator=false
import importlib
import math
from typing import Any, Literal

import numpy as np
import pytest

from dpt.contracts import ContractError
from dpt.geometry import DetectorGeometry
from dpt.registration import Vector
from dpt.stochastic_recovery import StochasticPolicy, recover_expected_signal
from dpt.transport.batched import prepare_transport_batched
from dpt.transport.derivatives import TransportParameter
from dpt.transport.forward import prepare_transport
from dpt.transport.inverse import TransportSquaredOracle, prepare_transport_inverse
from dpt.transport.model import (
    IncompleteHistoryError,
    MaterialGrid,
    OrientedDetector,
    PlanarDetector,
    TransportError,
    TransportSpec,
)
from dpt.transport.rng import HistoryBatch
from dpt.transport.source import ParallelBeam, PointConeSource

wp: Any = importlib.import_module("warp")


def fixture(
    device,
    *,
    dimension=5,
    count=257,
    estimator: Literal["analogue", "continuous-absorption"] = "continuous-absorption",
    compton=False,
    cone=False,
    symmetric=False,
    scattering=0.2,
    local_model=False,
    pixels=6,
    **kwargs,
):
    materials = max(1, dimension - 1)
    geometry = DetectorGeometry(
        (0.0, 0.0, -2.0),
        (-2.0 + 2.0 / pixels, 0.0, materials + 1.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (4.0 / pixels, 4.0),
        (1, pixels),
    )
    spec = TransportSpec(
        MaterialGrid((-2.0, -2.0, 0.0), (4.0, 4.0, 1.0), (materials, 1, 1), materials),
        OrientedDetector(geometry)
        if cone
        else PlanarDetector((-2.0, -2.0), (4.0, 4.0), (pixels, 1), materials + 1.0),
        80.0,
        "scalar versus replica-axis layered score fixture",
        estimator=estimator,
        scattering_law="free-electron-compton" if compton else "isotropic-elastic",
        coefficient_energies_kev=(0.0, 80.0) if compton else (),
        **kwargs,
    )

    def arr(values, dtype=None):
        return wp.array(values, dtype=wp.float64 if dtype is None else dtype, device=device)

    bins = len(spec.energy_nodes)
    workspace = prepare_transport(
        spec,
        material_ids=arr(list(range(materials)), wp.int32),
        absorption=arr([0.12 + 0.03 * i for i in range(materials) for _ in range(bins)]),
        scattering=arr([scattering] * (materials * bins)),
        max_histories=count,
        device=device,
    )
    selected = tuple(TransportParameter("log-material-density", i) for i in range(materials))
    if dimension > 1:
        selected += (TransportParameter("log-source-amplitude"),)
    oracle = prepare_transport_inverse(
        workspace,
        source=PointConeSource(geometry, weight=1.7)
        if cone
        else ParallelBeam((-0.7, -0.5, -2.0), (1.4, 1.0), weight=1.7),
        parameters=selected,
        observation=arr([0.03 + 0.01 * i for i in range(pixels)]),
        pixel_weights=arr([0.5 + 0.2 * i for i in range(pixels)]),
        base_density=(1.0,) * materials,
        fused_derivatives=True,
        symmetric_gradient=symmetric,
        local_model=local_model,
    )
    return oracle


def pairs(count=257, replicas=3):
    return tuple(
        (
            HistoryBatch(17 + 83 * row, 2**47 + row * 2**40, count),
            HistoryBatch(17 + 83 * row, 2**47 + row * 2**40 + count, count),
        )
        for row in range(replicas)
    )


@pytest.mark.parametrize("dimension", [1, 2, 5, 8])
@pytest.mark.parametrize("estimator", ["analogue", "continuous-absorption"])
@pytest.mark.parametrize("compton,cone", [(False, False), (True, True)])
def test_fixed_replicas_match_scalar_counter_replay_and_work(
    dimension, estimator, compton, cone, warp_device
):
    oracle = fixture(
        warp_device, dimension=dimension, estimator=estimator, compton=compton, cone=cone
    )
    batched = prepare_transport_batched(oracle, max_replicates=4)
    addresses = pairs()
    parameters = (0.1,) * dimension
    pointers = {key: value.ptr for key, value in batched._arrays.items()}
    host_pointers = batched._ids_host.ptr, batched._gradient_host.ptr, batched._loss_host.ptr
    before_work = batched.histories_traced
    before_download = batched.scalar_download_bytes
    before_upload = batched.parameter_upload_bytes
    actual = batched.gradient_replicates(parameters, addresses)
    assert batched.histories_traced - before_work == 3 * 257 * (1 + math.ceil(dimension / 4))
    assert batched.scalar_download_bytes - before_download == 8 * 3 * dimension
    assert batched.parameter_upload_bytes - before_upload == 32 * 3 + 8 * dimension
    expected = tuple(oracle.gradient_replicate(parameters, a, b) for a, b in addresses)
    np.testing.assert_allclose(actual, expected, rtol=5e-14, atol=5e-15)
    assert pointers == {key: value.ptr for key, value in batched._arrays.items()}
    assert host_pointers == (
        batched._ids_host.ptr,
        batched._gradient_host.ptr,
        batched._loss_host.ptr,
    )
    assert batched.host_staging_bytes == 4 * (32 + 8 * dimension + 8)
    assert not batched._upload_pending


@pytest.mark.parametrize("symmetric", [False, True])
def test_change_and_symmetric_gradient_match_scalar_products(symmetric, warp_device):
    oracle = fixture(warp_device, dimension=5, symmetric=symmetric, cone=True, pixels=515)
    batched = prepare_transport_batched(oracle)
    addresses = pairs(replicas=4)
    before, after = (0.0,) * 5, (0.02, -0.03, 0.01, 0.04, -0.02)
    download = batched.scalar_download_bytes
    work = batched.histories_traced
    actual = batched.gradient_replicates(before, addresses)
    assert batched.scalar_download_bytes - download == 8 * 4 * 5
    assert batched.histories_traced - work == (2 if symmetric else 1) * 4 * 257 * 3
    expected = tuple(oracle.gradient_replicate(before, a, b) for a, b in addresses)
    np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-15)
    download = batched.scalar_download_bytes
    actual_change = batched.change_replicates(before, after, addresses)
    assert batched.scalar_download_bytes - download == 8 * 4
    expected_change = tuple(oracle.change_replicate(before, after, a, b) for a, b in addresses)
    np.testing.assert_allclose(actual_change, expected_change, rtol=1e-10, atol=1e-13)
    assert batched.change_replicates(before, before, addresses) == (0.0,) * 4


def test_unequal_sizes_and_above_capacity_keep_scalar_fallback(warp_device, monkeypatch):
    oracle = fixture(warp_device, dimension=2)
    batched = prepare_transport_batched(oracle, max_replicates=2)
    original_gradient = TransportSquaredOracle.gradient_replicate
    calls: list[tuple[Vector, HistoryBatch, HistoryBatch, Vector]] = []

    def scalar_dispatch(
        self: TransportSquaredOracle,
        parameters: Vector,
        mean_batch: HistoryBatch,
        derivative_batch: HistoryBatch,
    ) -> Vector:
        result = original_gradient(self, parameters, mean_batch, derivative_batch)
        assert self is oracle
        calls.append((parameters, mean_batch, derivative_batch, result))
        return result

    monkeypatch.setattr(TransportSquaredOracle, "gradient_replicate", scalar_dispatch)
    different = (
        (HistoryBatch(12, 0, 17), HistoryBatch(12, 100, 31)),
        (HistoryBatch(13, 0, 45), HistoryBatch(13, 100, 91)),
    )
    above = pairs(replicas=3)
    for addresses in (different, above):
        calls.clear()
        actual = batched.gradient_replicates((0.0, 0.0), addresses)
        assert tuple(call[:3] for call in calls) == tuple(
            (
                (0.0, 0.0),
                first.with_library_source("parallel-rectangle-v1"),
                second.with_library_source("parallel-rectangle-v1"),
            )
            for first, second in addresses
        )
        # The fallback must return each dispatched tuple directly, in pair order.
        assert len(actual) == len(calls)
        for returned, call in zip(actual, calls, strict=True):
            assert returned is call[3]
        np.testing.assert_array_equal(actual, tuple(call[3] for call in calls))
        expected = tuple(
            original_gradient(oracle, (0.0, 0.0), first, second) for first, second in addresses
        )
        # Fresh canonical replays preserve histories but atomic addition order on
        # CUDA can shift the detector mean, and hence gradient, by a few ULPs.
        np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=2e-15)
    assert batched._batched_histories_traced == 0
    assert batched._batched_download_bytes == 0
    assert batched._batched_upload_bytes == 0
    calls.clear()
    assert batched.gradient_replicates((0.0, 0.0), ()) == ()
    assert calls == []


@pytest.mark.parametrize("dimension", [1, 2, 5, 8])
@pytest.mark.parametrize("symmetric", [False, True])
def test_plural_local_models_match_scalar_psd_proposals(dimension, symmetric, warp_device):
    oracle = fixture(
        warp_device, dimension=dimension, symmetric=symmetric, local_model=True, pixels=515
    )
    batched = prepare_transport_batched(oracle)
    addresses = pairs(replicas=3)
    assert batched.supports_local_model
    before_download, before_work = batched.scalar_download_bytes, batched.histories_traced
    actual = batched.model_replicates((0.0,) * dimension, addresses)
    assert batched.scalar_download_bytes - before_download == 8 * 3 * dimension * (dimension + 1)
    assert batched.histories_traced - before_work == (
        3 * 257 * (1 + math.ceil(dimension / 4)) * (2 if symmetric else 1)
    )
    expected = tuple(oracle.model_replicate((0.0,) * dimension, a, b) for a, b in addresses)
    for (gradient, metric), (expected_gradient, expected_metric) in zip(
        actual, expected, strict=True
    ):
        np.testing.assert_allclose(gradient, expected_gradient, rtol=1e-13, atol=1e-15)
        np.testing.assert_allclose(metric, expected_metric, rtol=1e-13, atol=1e-15)
        assert np.linalg.eigvalsh(metric).min() >= -1e-13


def test_plural_metric_matches_independent_layer_jacobian(warp_device):
    dimension = 8
    oracle = fixture(
        warp_device, dimension=dimension, count=17, local_model=True, scattering=0.0, pixels=1
    )
    batched = prepare_transport_batched(oracle)
    depth = np.array([0.12 + 0.03 * i for i in range(dimension - 1)])
    mean = 1.7 * math.exp(-math.fsum(depth))
    jacobian = np.append(-depth * mean, mean)
    expected_gradient = 0.5 * (mean - 0.03) * jacobian
    expected_metric = 0.5 * np.outer(jacobian, jacobian)
    for gradient, metric in batched.model_replicates((0.0,) * dimension, pairs(count=17)):
        np.testing.assert_allclose(gradient, expected_gradient, rtol=4e-14)
        np.testing.assert_allclose(metric, expected_metric, rtol=4e-14)


def test_large_or_unequal_local_models_retain_bounded_scalar_fallback(warp_device):
    oracle = fixture(warp_device, dimension=9, count=37, local_model=True)
    batched = prepare_transport_batched(oracle)
    actual = batched.model_replicates((0.0,) * 9, pairs(count=37, replicas=2))
    expected = tuple(
        oracle.model_replicate((0.0,) * 9, a, b) for a, b in pairs(count=37, replicas=2)
    )
    assert actual == expected
    assert batched._batched_histories_traced == 0
    assert "jacobian" not in batched._arrays
    smaller = prepare_transport_batched(fixture(warp_device, dimension=2, local_model=True))
    unequal = ((HistoryBatch(31, 0, 17), HistoryBatch(31, 100, 21)),)
    assert smaller.model_replicates((0.0, 0.0), unequal)
    assert smaller._batched_histories_traced == 0


def test_model_storage_is_bounded_and_prepared_host_buffers_are_reused(warp_device):
    oracle = fixture(warp_device, dimension=8, local_model=True, pixels=515)
    batched = prepare_transport_batched(oracle, max_replicates=3)
    required = batched.allocated_bytes - oracle.allocated_bytes
    assert prepare_transport_batched(oracle, max_replicates=3, max_bytes=required)
    with pytest.raises(TransportError, match="budget"):
        prepare_transport_batched(oracle, max_replicates=3, max_bytes=required - 1)
    pointers = {key: value.ptr for key, value in batched._arrays.items()}
    host = batched._model_host.ptr
    batched.model_replicates((0.0,) * 8, pairs())
    assert host == batched._model_host.ptr
    assert pointers == {key: value.ptr for key, value in batched._arrays.items()}
    assert batched.host_staging_bytes == 3 * (32 + 8 * 8 + 8 + 8 * 8 * 9)


def test_replica_domain_checks_include_source_and_inter_replica_overlap(warp_device):
    oracle = fixture(warp_device, dimension=2)
    batched = prepare_transport_batched(oracle)
    first, second = pairs(replicas=1)[0]
    with pytest.raises(TransportError, match="disjoint"):
        batched.gradient_replicates((0.0, 0.0), ((first, second), (first, second)))
    # Distinct nominal external sampler labels cannot hide shared library source words.
    fake = HistoryBatch(first.seed, first.first_history, first.count, source_namespace="fake")
    with pytest.raises(TransportError):
        batched.gradient_replicates((0.0, 0.0), ((first, fake),))
    with pytest.raises(TransportError, match="capacity"):
        batched.gradient_replicates((0.0, 0.0), pairs(count=258, replicas=1))
    with pytest.raises(TransportError, match="chart"):
        batched.gradient_replicates((0.0,), pairs(replicas=1))


def test_preparation_budget_counts_every_owned_device_array(warp_device):
    oracle = fixture(warp_device, dimension=8, pixels=515)
    batched = prepare_transport_batched(oracle, max_replicates=3)
    required = batched.allocated_bytes - oracle.allocated_bytes
    assert prepare_transport_batched(oracle, max_replicates=3, max_bytes=required)
    with pytest.raises(TransportError, match="budget"):
        prepare_transport_batched(oracle, max_replicates=3, max_bytes=required - 1)
    for invalid in (0, True, 1025):
        with pytest.raises(ContractError):
            prepare_transport_batched(oracle, max_replicates=invalid)


@pytest.mark.parametrize("local_model", [False, True])
def test_batched_budget_failure_is_fatal_and_status_stays_sticky(local_model, warp_device):
    oracle = fixture(
        warp_device,
        dimension=2,
        count=37,
        scattering=1e6,
        max_events=1,
        local_model=local_model,
    )
    batched = prepare_transport_batched(oracle)
    method = batched.model_replicates if local_model else batched.gradient_replicates
    with pytest.raises(IncompleteHistoryError):
        method((0.0, 0.0), pairs(count=37))
    assert not batched._upload_pending
    with pytest.raises(IncompleteHistoryError):
        oracle.workspace.check_status()
    with pytest.raises(IncompleteHistoryError):
        batched.change_replicates((0.0, 0.0), (0.0, 0.0), pairs(count=37))
    with pytest.raises(IncompleteHistoryError):
        oracle.workspace.consume_status()
    oracle.workspace.check_status()


@pytest.mark.parametrize("local_model", [False, True])
def test_controller_uses_plural_backend_and_reports_actual_work(local_model, warp_device):
    oracle = fixture(warp_device, dimension=2, count=32, local_model=local_model)
    batched = prepare_transport_batched(oracle, max_replicates=4)
    policy = StochasticPolicy(
        iterations=1,
        replicates=4,
        initial_batch=32,
        maximum_batch=32,
        unique_history_budget=4096,
        proposal="auto",
    )
    result = recover_expected_signal(batched, (0.0, 0.0), policy=policy, seed=1918)
    assert result.histories_traced == batched.histories_traced
    assert batched._batched_histories_traced > 0
    assert oracle.histories_traced == 0
    assert batched.supports_local_model == local_model
    assert result.proposal_policy == ("quadratic" if local_model else "linear")


@pytest.mark.parametrize("pixels", [515, 65537])
def test_replica_reduction_tree_matches_independent_host_arithmetic(pixels, warp_device):
    """Cross both pixel tile boundaries and distinct packed replicate rows."""
    kernels = importlib.import_module("dpt.transport.batched_kernels")
    rows = importlib.import_module("dpt.transport.recovery_kernels")
    replicas = 3
    rng = np.random.default_rng(77421)
    a = rng.normal(size=(replicas, pixels))
    b = rng.normal(size=(replicas, pixels))
    observed, weights = rng.normal(size=pixels), rng.uniform(size=pixels)

    def arr(value):
        return wp.array(value.ravel(), dtype=wp.float64, device=warp_device)

    inputs = [arr(value) for value in (a, b, observed, weights)]
    status = wp.zeros(1, dtype=wp.int32, device=warp_device)
    output = wp.zeros(replicas * 7, dtype=wp.float64, device=warp_device)
    tiles = (pixels + 255) // 256
    for loss in (0, 1):
        partial = wp.empty(replicas * tiles, dtype=wp.float64, device=warp_device)
        wp.launch_tiled(
            kernels.objective_tiles,
            dim=replicas * tiles,
            block_dim=256,
            inputs=[*inputs, pixels, tiles, loss, partial, status],
            device=warp_device,
            record_tape=False,
        )
        count = tiles
        while count > 1:
            following_count = (count + 255) // 256
            following = wp.empty(replicas * following_count, dtype=wp.float64, device=warp_device)
            wp.launch_tiled(
                rows.sum_gram_tiles,
                dim=replicas * following_count,
                block_dim=256,
                inputs=[partial, count, following_count, following],
                device=warp_device,
                record_tape=False,
            )
            partial, count = following, following_count
        wp.launch(
            kernels.finish_values,
            dim=replicas,
            inputs=[partial, 7, 3, output],
            device=warp_device,
            record_tape=False,
        )
        expected = [
            math.fsum(
                float(w * (x - y) * ((z - y) / 2 if loss else z))
                for x, z, y, w in zip(row_a, row_b, observed, weights, strict=True)
            )
            for row_a, row_b in zip(a, b, strict=True)
        ]
        np.testing.assert_allclose(output.numpy().reshape(replicas, 7)[:, 3], expected, rtol=3e-14)
        assert status.numpy()[0] == 0


@pytest.mark.parametrize("pixels", [515, 65537])
def test_packed_replica_gram_matches_independent_matrix_algebra(pixels, warp_device):
    kernels = importlib.import_module("dpt.transport.batched_model_kernels")
    reductions = importlib.import_module("dpt.transport.recovery_kernels")
    replicas, dimension = 3, 8
    rng = np.random.default_rng(87219)
    jacobian = rng.normal(size=(replicas, dimension, pixels))
    weights = rng.uniform(size=pixels)
    gradients = rng.normal(size=(replicas, dimension))

    def arr(value):
        return wp.array(value.ravel(), dtype=wp.float64, device=warp_device)

    stored = wp.empty(replicas * dimension * pixels, dtype=wp.float64, device=warp_device)
    for column in range(dimension):
        wp.launch(
            kernels.store_column,
            dim=replicas * pixels,
            inputs=[arr(jacobian[:, column, :]), column, pixels, dimension, stored],
            device=warp_device,
            record_tape=False,
        )
    np.testing.assert_array_equal(stored.numpy().reshape(jacobian.shape), jacobian)
    status = wp.zeros(1, dtype=wp.int32, device=warp_device)
    rows = replicas * dimension**2
    count = (pixels + 255) // 256
    previous = wp.empty(rows * count, dtype=wp.float64, device=warp_device)
    wp.launch_tiled(
        kernels.gram_tiles,
        dim=rows * count,
        block_dim=256,
        inputs=[stored, arr(weights), pixels, dimension, count, previous, status],
        device=warp_device,
        record_tape=False,
    )
    while count > 1:
        next_count = (count + 255) // 256
        following = wp.empty(rows * next_count, dtype=wp.float64, device=warp_device)
        wp.launch_tiled(
            reductions.sum_gram_tiles,
            dim=rows * next_count,
            block_dim=256,
            inputs=[previous, count, next_count, following],
            device=warp_device,
            record_tape=False,
        )
        previous, count = following, next_count
    expected = np.einsum("rmp,p,rnp->rmn", jacobian, weights, jacobian)
    np.testing.assert_allclose(
        previous.numpy().reshape(replicas, dimension, dimension), expected, rtol=4e-13, atol=5e-11
    )
    width = dimension * (dimension + 1)
    checkpoint = wp.empty(replicas * width, dtype=wp.float64, device=warp_device)
    wp.launch(
        kernels.finish_model,
        dim=replicas * width,
        inputs=[arr(gradients), previous, dimension, checkpoint],
        device=warp_device,
        record_tape=False,
    )
    packed = checkpoint.numpy().reshape(replicas, width)
    np.testing.assert_array_equal(packed[:, :dimension], gradients)
    np.testing.assert_array_equal(packed[:, dimension:], previous.numpy().reshape(replicas, -1))
    assert status.numpy()[0] == 0

"""Actual CUDA Jacobian contractions, capacity guards and deterministic classification."""

# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false
import importlib
import math
from typing import Any

import pytest

from dpt.transport.derivatives import TransportParameter
from dpt.transport.forward import prepare_transport
from dpt.transport.inverse import prepare_transport_inverse
from dpt.transport.model import MaterialGrid, PlanarDetector, TransportError, TransportSpec
from dpt.transport.rng import HistoryBatch
from dpt.transport.source import ParallelBeam

wp: Any = importlib.import_module("warp")
np: Any = importlib.import_module("numpy")
DEVICE = "cpu"


@pytest.fixture(autouse=True)
def _device_scope(warp_device: str):
    global DEVICE
    DEVICE = warp_device
    with wp.ScopedDevice(DEVICE):
        yield


def _oracle(
    parameters: int,
    *,
    local_model: bool = True,
    budget: int = 2**28,
    extent: float = 0.0,
    scattering: float = 0.0,
) -> Any:
    spec = TransportSpec(
        MaterialGrid((-1.0, -1.0, 0.0), (2.0, 2.0, 1.0), (parameters, 1, 1), parameters),
        PlanarDetector((-1.0, -1.0), (2.0, 2.0), (1, 1), parameters + 1.0),
        80.0,
        "declared mathematical layered local-model fixture",
        estimator="continuous-absorption",
    )
    workspace = prepare_transport(
        spec,
        material_ids=wp.array(list(range(parameters)), dtype=wp.int32, device=DEVICE),
        absorption=wp.array([0.1] * parameters, dtype=wp.float64, device=DEVICE),
        scattering=wp.array([scattering] * parameters, dtype=wp.float64, device=DEVICE),
        device=DEVICE,
        max_histories=17,
    )
    return prepare_transport_inverse(
        workspace,
        source=ParallelBeam((0.0, 0.0, -1.0), (extent, extent)),
        parameters=tuple(TransportParameter("log-material-density", i) for i in range(parameters)),
        observation=wp.array([0.2], dtype=wp.float64, device=DEVICE),
        pixel_weights=wp.array([2.0], dtype=wp.float64, device=DEVICE),
        base_density=(1.0,) * parameters,
        local_model=local_model,
        local_model_max_bytes=budget,
    )


@pytest.mark.parametrize("parameters", [1, 2, 8])
def test_end_to_end_model_matches_independent_layer_integral(parameters: int) -> None:
    oracle = _oracle(parameters)
    allocation = oracle.allocated_bytes
    pointers = {name: value.ptr for name, value in oracle._arrays.items()}
    gradient, curvature = oracle.model_replicate(
        (0.0,) * parameters, HistoryBatch(731, 0, 17), HistoryBatch(731, 17, 17)
    )
    mean = math.exp(-0.1 * parameters)
    expected_gradient = 2 * (mean - 0.2) * -0.1 * mean
    for value in gradient:
        assert value == pytest.approx(expected_gradient, abs=2e-14)
    for row in curvature:
        assert row == pytest.approx((2 * (0.1 * mean) ** 2,) * parameters, abs=2e-14)
    assert oracle.deterministic_sampling
    assert oracle.allocated_bytes == allocation
    assert {name: value.ptr for name, value in oracle._arrays.items()} == pointers
    assert oracle.scalar_download_bytes == 8 * (parameters + parameters**2)
    assert oracle.histories_traced == 17 * (1 + parameters)


@pytest.mark.parametrize("parameters", [1, 2, 8])
@pytest.mark.parametrize("pixels", [1, 257, 65537])
def test_gpu_gram_tree_against_independent_dense_algebra(parameters: int, pixels: int) -> None:
    from dpt.transport import recovery_kernels as kernels

    # These are explicit algebra operands, not synthetic transport measurements.
    x = np.arange(pixels, dtype=np.float64)
    jacobian = np.stack([np.sin((i + 1) * (x + 0.5) / 17) for i in range(parameters)])
    weights = 0.5 + (x % 11) / 11
    expected = (jacobian * weights) @ jacobian.T
    device_j = wp.array(jacobian.ravel(), dtype=wp.float64, device=DEVICE)
    device_w = wp.array(weights, dtype=wp.float64, device=DEVICE)
    status = wp.zeros(1, dtype=wp.int32, device=DEVICE)
    count = (pixels + 255) // 256
    previous = wp.empty(parameters**2 * count, dtype=wp.float64, device=DEVICE)
    wp.launch_tiled(
        kernels.gram_tiles,
        dim=parameters**2 * count,
        block_dim=256,
        inputs=[device_j, device_w, pixels, parameters, count, previous, status],
    )
    while count > 1:
        next_count = (count + 255) // 256
        output = wp.empty(parameters**2 * next_count, dtype=wp.float64, device=DEVICE)
        wp.launch_tiled(
            kernels.sum_gram_tiles,
            dim=parameters**2 * next_count,
            block_dim=256,
            inputs=[previous, count, next_count, output],
        )
        previous, count = output, next_count
    actual = previous.numpy().reshape(parameters, parameters)
    assert int(status.numpy()[0]) == 0
    assert np.allclose(actual, expected, rtol=2e-12, atol=1e-10)
    assert np.linalg.matrix_rank(actual) == min(parameters, pixels)


def test_model_preparation_capacity_and_memory_are_explicit() -> None:
    with pytest.raises(TransportError, match="at most 16"):
        _oracle(17)
    with pytest.raises(TransportError, match="preparation budget"):
        _oracle(2, budget=1)
    oracle = _oracle(1, local_model=False)
    with pytest.raises(TransportError, match="local_model=True"):
        oracle.model_replicate((0.0,), HistoryBatch(4, 0, 2), HistoryBatch(4, 2, 2))


def test_sampling_classification_uses_static_source_and_all_scattering_coefficients() -> None:
    assert _oracle(1).deterministic_sampling
    assert not _oracle(1, extent=0.5).deterministic_sampling
    assert not _oracle(1, scattering=0.001).deterministic_sampling


@pytest.mark.parametrize("parameters", [2, 8])
def test_identifiable_random_source_columns_have_diagonal_expected_metric(parameters: int) -> None:
    count = 2**16
    spec = TransportSpec(
        MaterialGrid((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (1, 1, parameters), parameters),
        PlanarDetector((0.0, 0.0), (1.0, 1.0), (parameters, 1), 2.0),
        80.0,
        "declared independent mathematical columns; random uniform source",
        estimator="continuous-absorption",
    )
    workspace = prepare_transport(
        spec,
        material_ids=wp.array(list(range(parameters)), dtype=wp.int32, device=DEVICE),
        absorption=wp.array([0.3] * parameters, dtype=wp.float64, device=DEVICE),
        scattering=wp.zeros(parameters, dtype=wp.float64, device=DEVICE),
        device=DEVICE,
        max_histories=count,
    )
    oracle = prepare_transport_inverse(
        workspace,
        source=ParallelBeam((0.0, 0.0, -1.0), (float(parameters), 1.0)),
        parameters=tuple(TransportParameter("log-material-density", i) for i in range(parameters)),
        observation=wp.array([0.02] * parameters, dtype=wp.float64, device=DEVICE),
        pixel_weights=wp.ones(parameters, dtype=wp.float64, device=DEVICE),
        base_density=(1.0,) * parameters,
        local_model=True,
    )
    gradient, curvature = oracle.model_replicate(
        (0.0,) * parameters, HistoryBatch(31903, 0, count), HistoryBatch(31903, count, count)
    )
    assert not oracle.deterministic_sampling
    probability = 1 / parameters
    transmission = math.exp(-0.3)
    mean = transmission * probability
    expected_gradient = (mean - 0.02) * -0.3 * mean
    # Independent A and B multinomial marginals. This analytic product variance
    # uses their known marginal binomial probabilities, not observed error bars.
    variance = transmission**2 * probability * (1 - probability) / count
    gradient_variance = 0.3**2 * ((mean - 0.02) ** 2 * variance + mean**2 * variance + variance**2)
    for i, actual in enumerate(gradient):
        assert abs(actual - expected_gradient) <= 7 * math.sqrt(gradient_variance)
        for j, entry in enumerate(curvature[i]):
            if i != j:
                assert entry == 0.0
            else:
                # Jacobian sampling makes E[B] larger by its variance.
                expected = 0.3**2 * (mean**2 + variance)
                # A seven-sigma bound on the underlying marginal proportion,
                # transformed through the square (plus the known bias term).
                delta = 7 * math.sqrt(variance)
                allowance = 0.3**2 * (2 * mean * delta + delta**2 + variance)
                assert abs(entry - expected) <= allowance
    assert np.linalg.matrix_rank(np.array(curvature)) == parameters


def test_sampling_certificate_cannot_be_assigned_by_caller() -> None:
    oracle = _oracle(1)
    with pytest.raises(AttributeError):
        oracle.deterministic_sampling = False

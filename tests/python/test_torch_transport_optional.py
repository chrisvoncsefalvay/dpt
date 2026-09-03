"""Expected transport metadata, identity laws and preparation boundaries."""

import importlib

import pytest

pytest.importorskip("torch")

from dpt._runtime import cpu_test_context
from dpt.contracts import ContractError
from dpt.interop.torch_transport import (
    ExpectedTransportEstimator,
    TransportAutogradSpec,
    prepare_expected_transport,
)
from dpt.transport.forward import prepare_transport
from dpt.transport.model import MaterialGrid, PlanarDetector, TransportError, TransportSpec
from dpt.transport.rng import HistoryBatch
from dpt.transport.source import ParallelBeam


@pytest.mark.parametrize("origin", ["forward-batch", "deterministic", "external-batch"])
def test_actual_library_source_tokens_and_cotangent_origins(origin):
    source = ParallelBeam((0.0, 0.0, -1.0), (1.0, 1.0))
    first, second = (
        HistoryBatch(11, 0, 7, "same random source"),
        HistoryBatch(11, 7, 9, "same random source"),
    )
    external = HistoryBatch(13, 0, 4).with_library_source("parallel-rectangle-v1")
    estimate = ExpectedTransportEstimator(
        first, second, origin, external if origin == "external-batch" else None
    ).bind_source(source)
    assert estimate.forward_batch.source_identity is not None
    assert estimate.forward_batch.source_identity.sampler == "parallel-rectangle-v1"
    assert estimate.derivative_batch.independent_of(estimate.forward_batch)
    assert estimate.cotangent_batch == (
        estimate.forward_batch
        if origin == "forward-batch"
        else external
        if origin == "external-batch"
        else None
    )
    assert first.source_identity is None and second.source_identity is None


@pytest.mark.parametrize("origin", ["forward-batch", "deterministic", "external-batch"])
def test_overlapping_transport_addresses_cannot_be_relabelled_independent(origin):
    source = ParallelBeam((0.0, 0.0, -1.0), (0.0, 0.0))
    first, second = HistoryBatch(11, 0, 7, "A"), HistoryBatch(11, 6, 9, "B")
    with pytest.raises(TransportError, match="disjoint"):
        ExpectedTransportEstimator(
            first, second, origin, HistoryBatch(13, 0, 4) if origin == "external-batch" else None
        ).bind_source(source)


def test_external_cotangent_source_overlap_is_rejected():
    first = HistoryBatch(11, 0, 7)
    second = HistoryBatch(13, 0, 9)
    external = HistoryBatch(13, 0, 9, "relabeled").with_library_source("another-name")
    with pytest.raises(TransportError, match="disjoint"):
        ExpectedTransportEstimator(first, second, "external-batch", external).bind_source(
            ParallelBeam((0.0, 0.0, -1.0), (0.0, 0.0))
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"active_voxels": (-1,)},
        {"active_materials": (True,)},
        {"active_voxels": (1, 1)},
        {"active_source_amplitude": 1},
        {"voxel_parameterization": "voxel-density"},
        {"source_parameterization": "log-source-amplitude"},
    ],
)
def test_unsupported_coordinate_and_active_chart_contracts(kwargs):
    with pytest.raises(ContractError):
        TransportAutogradSpec(**kwargs)


def test_cotangent_identity_must_match_declared_origin():
    first, second = HistoryBatch(11, 0, 7), HistoryBatch(13, 0, 9)
    with pytest.raises(ContractError, match="identity"):
        ExpectedTransportEstimator(first, second, "external-batch")
    with pytest.raises(ContractError, match="external_cotangent_batch"):
        ExpectedTransportEstimator(first, second, "deterministic", HistoryBatch(17, 0, 1))


@pytest.mark.parametrize("invalid", [None, "voxel-id", "material-id", "slots", "budget"])
def test_native_preparation_contracts(invalid):
    wp = importlib.import_module("warp")
    with cpu_test_context():
        spec = TransportSpec(
            MaterialGrid((-1.0, -1.0, 0.0), (2.0, 2.0, 1.0), (1, 1, 1), 1),
            PlanarDetector((-1.0, -1.0), (2.0, 2.0), (1, 1), 2.0),
            80.0,
            "independent Torch adapter preparation contract",
            estimator="continuous-absorption",
        )
        native = prepare_transport(
            spec,
            material_ids=wp.array([0], dtype=wp.int32, device="cpu"),
            absorption=wp.array([0.2], dtype=wp.float64, device="cpu"),
            scattering=wp.zeros(1, dtype=wp.float64, device="cpu"),
            max_histories=7,
            device="cpu",
        )
        kwargs = {}
        if invalid == "voxel-id":
            kwargs["parameters"] = TransportAutogradSpec(active_voxels=(1,))
        elif invalid == "material-id":
            kwargs["parameters"] = TransportAutogradSpec(active_materials=(1,))
        elif invalid == "slots":
            kwargs["slots_per_history"] = 3
        elif invalid == "budget":
            kwargs["max_snapshot_bytes"] = 1
        source = ParallelBeam((0.0, 0.0, -1.0), (0.0, 0.0))
        if invalid is None:
            config = prepare_expected_transport(native, source=source)
            assert config.transport_spec == spec and config.max_histories == 7
            assert config.context.device.is_cpu
        else:
            with pytest.raises(ContractError):
                prepare_expected_transport(native, source=source, **kwargs)

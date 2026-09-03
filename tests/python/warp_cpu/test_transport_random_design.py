"""Resident full-path designs against independent CPU oracles and CUDA selectors."""

# Warp annotations and resident arrays are runtime DSL objects.
# pyright: reportInvalidTypeForm=false, reportUnknownMemberType=false
# pyright: reportUnknownParameterType=false, reportUnknownVariableType=false
# pyright: reportMissingParameterType=false, reportUnknownArgumentType=false
# pyright: reportUntypedFunctionDecorator=false, reportPrivateUsage=false
import importlib
import math
from typing import Any

import numpy as np
import pytest

from dpt.kernels.random import random4
from dpt.transport import (
    HistoryBatch,
    MaterialGrid,
    ParallelBeam,
    PlanarDetector,
    SamplingDesign,
    SamplingIdentity,
    TransportError,
    TransportSpec,
    design_replicate_moments,
    prepare_simulation,
    prepare_transport_sampling,
    sampling_uniforms,
    simulate,
)
from dpt.transport.derivatives import TransportParameter, derivative_histories
from dpt.transport.forward import prepare_transport, trace_histories
from dpt.transport.physical import PhysicalTransport, xraylib_atomic_scattering
from dpt.transport.readout import TransportReadout, prepare_transport_readout, trace_readout
from dpt.transport.sampling_kernels import RandomDesign, inspect_random_design
from dpt.transport.source import sample_parallel_beam
from dpt.transport.source_spectrum import SourceSpectrum

wp: Any = importlib.import_module("warp")


@wp.kernel
def _iid_reference(
    seed: wp.uint64,
    first: wp.uint64,
    event: wp.uint32,
    domain: wp.uint32,
    out: wp.array(dtype=wp.vec4d),
):
    i = wp.tid()
    out[i] = random4(seed, first + wp.uint64(i), event, domain)


def _workspace(device, count, absorption=0.355, scattering=0.0, *, events=4, trials=4):
    spec = TransportSpec(
        MaterialGrid((-1.0, -1.0, 0.0), (2.0, 2.0, 2.0), (1, 1, 1), 1),
        PlanarDetector((-3.0, -4.0), (2.0, 4.0), (3, 2), 3.0),
        80.0,
        "homogeneous all-orders reference cube",
        max_events=events,
        max_angle_trials=trials,
    )
    return prepare_transport(
        spec,
        material_ids=wp.array([0], dtype=wp.int32, device=device),
        absorption=wp.array([absorption], dtype=wp.float64, device=device),
        scattering=wp.array([scattering], dtype=wp.float64, device=device),
        max_histories=count,
        device=device,
    )


@pytest.mark.parametrize("law", ["stratified", "scrambled-sobol"])
def test_resident_all_domains_exact_host_coordinates_and_iid_bitwise(law, warp_device):
    count = 128
    transport = _workspace(warp_device, count)
    design = SamplingDesign(law, 4, 4, 2)
    batch = prepare_transport_sampling(
        transport, batch=HistoryBatch(7263, 2**63 + 7, count), design=design
    )
    assert batch.sampling is not None
    out = wp.empty(count, dtype=wp.vec4d, device=warp_device)
    reference = wp.empty_like(out)
    roles = [0, 1, 2, 3, 0x40000000, 0x40000001, 0x80000003, 0xC0000003]
    for event, domain in [(0x80000000, 0), (0x80000002, 0), *((3, role + 65536) for role in roles)]:
        wp.launch(
            inspect_random_design,
            dim=count,
            inputs=[
                wp.uint64(batch.seed),
                wp.uint64(batch.first_history),
                wp.uint32(event),
                wp.uint32(domain),
                transport._parameters.sampling,
                out,
            ],
            device=warp_device,
        )
        expected = sampling_uniforms(batch.sampling, event=event, domain=domain)
        np.testing.assert_array_equal(out.numpy(), expected)
        wp.launch(
            _iid_reference,
            dim=count,
            inputs=[
                wp.uint64(batch.seed),
                wp.uint64(batch.first_history),
                wp.uint32(event),
                wp.uint32(domain),
                reference,
            ],
            device=warp_device,
        )
        wp.launch(
            inspect_random_design,
            dim=count,
            inputs=[
                wp.uint64(batch.seed),
                wp.uint64(batch.first_history),
                wp.uint32(event),
                wp.uint32(domain),
                RandomDesign(),
                out,
            ],
            device=warp_device,
        )
        np.testing.assert_array_equal(out.numpy(), reference.numpy())


@pytest.mark.parametrize("law", ["stratified", "scrambled-sobol"])
def test_source_coordinates_shards_and_prepared_identity_checks(law, warp_device):
    count = 128
    transport = _workspace(warp_device, count)
    batch = prepare_transport_sampling(
        transport, batch=HistoryBatch(823, 4096, count), design=SamplingDesign(law, 4, 4)
    )
    assert batch.sampling is not None
    position = wp.empty(count, dtype=wp.vec3d, device=warp_device)
    direction = wp.empty_like(position)
    weight = wp.empty(count, dtype=wp.float64, device=warp_device)
    source = ParallelBeam((-0.7, -0.4, -1.0), (1.4, 0.8))
    token = sample_parallel_beam(
        source,
        batch=batch,
        workspace=transport,
        out_position=position,
        out_direction=direction,
        out_weight=weight,
    )
    assert token.sampling == batch.sampling
    original = position.numpy().copy()
    for rank in range(3):
        shard = batch.shard(3, rank)
        sample_parallel_beam(
            source,
            batch=shard.batch,
            workspace=transport,
            out_position=position[: shard.batch.count],
            out_direction=direction[: shard.batch.count],
            out_weight=weight[: shard.batch.count],
        )
        np.testing.assert_array_equal(
            position.numpy()[: shard.batch.count],
            original[shard.offset : shard.offset + shard.batch.count],
        )
    other = HistoryBatch(824, 4096, count).with_sampling(
        SamplingIdentity(batch.sampling.design, 824, 4096, count)
    )
    with pytest.raises(TransportError, match="prepared randomisation"):
        sample_parallel_beam(
            source,
            batch=other,
            workspace=transport,
            out_position=position,
            out_direction=direction,
            out_weight=weight,
        )


@pytest.mark.parametrize("law", ["stratified", "scrambled-sobol"])
def test_resident_path_absorption_mean_moments_and_variance_comparison(
    law, warp_device, record_testsuite_property
):
    count, replicates = 256, 32
    transport = _workspace(warp_device, count)
    simulation = prepare_simulation(transport, max_histories=count)
    density = wp.ones(1, dtype=wp.float64, device=warp_device)
    source = ParallelBeam((0.0, 0.0, -1.0), (0.0, 0.0))
    design = SamplingDesign(law, 4, 4)
    values, controls, batches = [], [], []
    for seed in range(301, 301 + replicates):
        batch = prepare_transport_sampling(
            transport, batch=HistoryBatch(seed, 0, count), design=design
        )
        result = simulate(density, source=source, batch=batch, workspace=simulation)
        assert result.sampling_law == law and result.estimated_variance_of_mean is None
        assert result.status_histogram.total == count
        values.append(float(np.sum(result.mean.numpy())))
        batches.append(result.batch)
        control = simulate(
            density, source=source, batch=HistoryBatch(seed, 0, count), workspace=simulation
        )
        assert control.estimated_variance_of_mean is not None
        controls.append(float(np.sum(control.mean.numpy())))
    mean, variance = design_replicate_moments(np.array(values), batches=tuple(batches))
    exact = math.exp(-0.71)
    assert abs(float(mean) - exact) < 1 / count
    design_variance = float(np.var(values, ddof=1))
    iid_variance = float(np.var(controls, ddof=1))
    assert float(variance) == pytest.approx(design_variance / replicates, rel=2e-14)
    assert design_variance < iid_variance / 16
    record_testsuite_property(
        f"{law}_{warp_device}_absorption",
        str(
            {
                "count": count,
                "replicates": replicates,
                "exact": exact,
                "mean": float(mean),
                "design_variance": design_variance,
                "iid_variance": iid_variance,
            }
        ),
    )


@pytest.mark.parametrize("law", ["stratified", "scrambled-sobol"])
def test_all_orders_multiple_scatter_against_independent_numpy_transport(
    law, warp_device, record_testsuite_property
):
    # Reuse a scientifically independent NumPy RNG/geometry all-orders oracle.
    from helpers.transport_scattering_reference import independent_isotropic_cube

    count, replicates, reference_count = 2048, 16, 2**17
    albedo = 0.6
    reference = independent_isotropic_cube(albedo, reference_count)
    transport = _workspace(
        warp_device, count, 1.2 * (1 - albedo), 1.2 * albedo, events=32, trials=32
    )
    simulation = prepare_simulation(transport, max_histories=count)
    density = wp.ones(1, dtype=wp.float64, device=warp_device)
    source = ParallelBeam((0.0, 0.0, -1.0), (0.0, 0.0))
    means, batches = [], []
    multiple = 0
    for seed in range(901, 901 + replicates):
        batch = prepare_transport_sampling(
            transport, batch=HistoryBatch(seed, 0, count), design=SamplingDesign(law, 32, 32)
        )
        result = simulate(density, source=source, batch=batch, workspace=simulation)
        means.append(result.mean.numpy().copy())
        batches.append(result.batch)
        multiple += int(np.sum(simulation._histories["out_events"].numpy() >= 3))
    mean, variance = design_replicate_moments(np.array(means), batches=tuple(batches))
    reference_variance = reference * (1 - reference) / (reference_count - 1)
    allowance = 6 * np.sqrt(variance + reference_variance)
    assert bool(np.all(np.abs(mean - reference) < allowance))
    assert multiple > 1024
    record_testsuite_property(
        f"{law}_{warp_device}_all_orders",
        str(
            {
                "count": count,
                "replicates": replicates,
                "reference_count": reference_count,
                "mean": mean.tolist(),
                "reference": reference.tolist(),
                "variance": variance.tolist(),
                "allowance": allowance.tolist(),
                "multiple_scatter_histories": multiple,
            }
        ),
    )


@pytest.mark.parametrize("law", ["stratified", "scrambled-sobol"])
def test_design_splitting_next_event_and_roulette_complete_replicates(
    law, warp_device, record_testsuite_property
):
    count, replicates = 1024, 16
    transport = _workspace(warp_device, count, 0.48, 0.72, events=32, trials=32)
    readout = prepare_transport_readout(
        transport,
        spec=TransportReadout(
            next_event=True,
            split_branches=2,
            roulette_threshold=0.6,
            roulette_survival=0.5,
            tally_strategy="deterministic",
        ),
    )
    position = wp.array(np.tile((0.0, 0.0, -1.0), (count, 1)), dtype=wp.vec3d, device=warp_device)
    direction = wp.array(np.tile((0.0, 0.0, 1.0), (count, 1)), dtype=wp.vec3d, device=warp_device)
    weight = wp.ones(count, dtype=wp.float64, device=warp_device)
    density = wp.ones(1, dtype=wp.float64, device=warp_device)
    out = wp.empty(readout.columns, dtype=wp.float64, device=warp_device)
    within = wp.empty_like(out)
    values, controls, batches = [], [], []
    for seed in range(1731, 1731 + replicates):
        batch = prepare_transport_sampling(
            transport, batch=HistoryBatch(seed, 0, count), design=SamplingDesign(law, 32, 32, 2)
        )
        trace_readout(
            position, direction, weight, density, batch=batch, workspace=readout, out_mean=out
        )
        values.append(out.numpy().copy())
        batches.append(batch)
        with pytest.raises(TransportError, match="replicate variance"):
            trace_readout(
                position,
                direction,
                weight,
                density,
                batch=batch,
                workspace=readout,
                out_mean=out,
                out_variance_of_mean=within,
            )
        trace_readout(
            position,
            direction,
            weight,
            density,
            batch=HistoryBatch(seed, 0, count),
            workspace=readout,
            out_mean=out,
            out_variance_of_mean=within,
        )
        controls.append(out.numpy().copy())
    mean, variance = design_replicate_moments(np.array(values), batches=tuple(batches))
    iid = np.asarray(controls)
    allowance = 6 * np.sqrt(variance + np.var(iid, axis=0, ddof=1) / replicates)
    assert bool(np.all(abs(mean - np.mean(iid, axis=0)) < allowance))
    record_testsuite_property(
        f"{law}_{warp_device}_readout",
        str(
            {
                "count": count,
                "replicates": replicates,
                "branches": 2,
                "mean": mean.tolist(),
                "variance": variance.tolist(),
                "iid_mean": np.mean(iid, axis=0).tolist(),
                "allowance": allowance.tolist(),
            }
        ),
    )


@pytest.mark.parametrize("law", ["stratified", "scrambled-sobol"])
def test_polyenergetic_woodcock_design_against_beer_and_density_score(
    law, warp_device, record_testsuite_property
):
    # Actual pinned provider data support the atomic record. Beer-law and the
    # null-event marked Poisson score give a separate exact expectation.
    pytest.importorskip("xraylib")
    count, replicates = 1024, 16
    table = xraylib_atomic_scattering(13, (0.0, 0.1, 1.0, 10.0))
    physical = PhysicalTransport(
        (table,),
        (0.0, 0.0),
        majorant_mm_inverse=(2.0, 2.0),
        spectrum=SourceSpectrum((40.0, 80.0), (0.25, 0.75), "two prescribed source lines"),
    )
    spec = TransportSpec(
        MaterialGrid((-1.0, -1.0, 0.0), (2.0, 2.0, 1.0), (1, 1, 1), 1),
        PlanarDetector((-1.0, -1.0), (2.0, 2.0), (1, 1), 2.0),
        80.0,
        "analytic polyenergetic marked-Poisson slab",
        scoring="energy-kev",
        coefficient_energies_kev=(40.0, 80.0),
        max_events=32,
        max_angle_trials=32,
    )
    transport = prepare_transport(
        spec,
        material_ids=wp.array([0], dtype=wp.int32, device=warp_device),
        absorption=wp.array([0.5, 1.0], dtype=wp.float64, device=warp_device),
        scattering=wp.zeros(2, dtype=wp.float64, device=warp_device),
        max_histories=count,
        device=warp_device,
        physical=physical,
    )
    position = wp.array(np.tile((0.0, 0.0, -1.0), (count, 1)), dtype=wp.vec3d, device=warp_device)
    direction = wp.array(np.tile((0.0, 0.0, 1.0), (count, 1)), dtype=wp.vec3d, device=warp_device)
    weight = wp.ones(count, dtype=wp.float64, device=warp_device)
    density = wp.ones(1, dtype=wp.float64, device=warp_device)
    outputs = {
        "out_" + name: wp.empty(count, dtype=dtype, device=warp_device)
        for name, dtype in (
            ("pixel", wp.int32),
            ("score", wp.float64),
            ("energy", wp.float64),
            ("events", wp.int32),
            ("status", wp.int32),
        )
    }
    derivative = wp.empty(count, dtype=wp.float64, device=warp_device)
    values, batches = [], []
    null_history_count = 0
    for seed in range(2117, 2117 + replicates):
        batch = prepare_transport_sampling(
            transport, batch=HistoryBatch(seed, 0, count), design=SamplingDesign(law, 32, 32)
        )
        trace_histories(
            position, direction, weight, density, batch=batch, workspace=transport, **outputs
        )
        null_history_count += int(np.sum(outputs["out_events"].numpy() > 1))
        derivative_histories(
            position,
            direction,
            weight,
            density,
            batch=batch,
            workspace=transport,
            parameter=TransportParameter("log-material-density", 0),
            out_pixel=outputs["out_pixel"],
            out_derivative=derivative,
            out_status=outputs["out_status"],
        )
        values.append(
            (float(np.mean(outputs["out_score"].numpy())), float(np.mean(derivative.numpy())))
        )
        batches.append(batch)
    mean, variance = design_replicate_moments(np.array(values), batches=tuple(batches))
    exact = np.array(
        [
            0.25 * 40 * math.exp(-0.5) + 0.75 * 80 * math.exp(-1),
            -0.25 * 40 * 0.5 * math.exp(-0.5) - 0.75 * 80 * math.exp(-1),
        ]
    )
    assert bool(np.all(abs(mean - exact) < 6 * np.sqrt(variance)))
    assert null_history_count > 1024
    assert table.provider is not None
    record_testsuite_property(
        f"{law}_{warp_device}_Woodcock",
        str(
            {
                "count": count,
                "replicates": replicates,
                "mean": mean.tolist(),
                "exact": exact.tolist(),
                "variance": variance.tolist(),
                "multiple_candidate_histories": null_history_count,
                "provider": table.provider.version,
                "table_sha256": table.provenance.sha256,
            }
        ),
    )

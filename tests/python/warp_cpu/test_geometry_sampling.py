"""Native complete geometry coordinates, replay and independent canonical controls."""

# Warp annotations and resident arrays are runtime DSL objects.
# pyright: reportInvalidTypeForm=false, reportUnknownMemberType=false
# pyright: reportUnknownParameterType=false, reportUnknownVariableType=false
# pyright: reportMissingParameterType=false, reportUnknownArgumentType=false
# pyright: reportUntypedFunctionDecorator=false, reportPrivateUsage=false
import dataclasses
import importlib
from typing import Any

import numpy as np
import pytest

from dpt._runtime import prepare_context
from dpt.transport.geometry_sampling import (
    AUXILIARY_DOMAIN,
    GeometryHostSampler,
    GeometrySamplingDesign,
    GeometrySamplingIdentity,
    prepare_geometry_sampling,
)
from dpt.transport.geometry_sampling_kernels import (
    inspect_geometry_auxiliary,
    inspect_geometry_physical,
)
from dpt.transport.model import MaterialGrid, PlanarDetector, TransportError, TransportSpec
from dpt.transport.rng import HistoryBatch
from dpt.transport.sampling import sampling_uniforms
from dpt.transport.sampling_identity import SamplingDesign, SamplingIdentity
from dpt.transport.sampling_kernels import RandomDesign, inspect_random_design

wp: Any = importlib.import_module("warp")


def _spec(layout):
    return TransportSpec(
        MaterialGrid((-1.0, -1.0, 0.0), (2.0, 2.0, 2.0), (1, 1, 1), 1),
        PlanarDetector((-3.0, -3.0), (2.0, 2.0), (3, 3), 3.0),
        80.0,
        "geometry random-coordinate reference",
        (1.0, 80.0),
        scattering_law="bound-electron-compton"
        if layout == "bound-compton-v3"
        else "free-electron-compton",
        compton_sampler="packed-uniform-rejection"
        if layout == "packed-compton-v2"
        else "legacy-rejection",
        max_events=3,
        max_angle_trials=3,
    )


def _read(context, out):
    if context.device.is_cuda:
        context.wp.synchronize_stream(context.stream)
    return out.numpy().copy()


@pytest.mark.parametrize("layout", ("legacy-v1", "packed-compton-v2", "bound-compton-v3"))
@pytest.mark.parametrize("law", ("iid", "stratified", "scrambled-sobol"))
def test_all_auxiliary_native_coordinates_and_complete_strata(law, layout, warp_device):
    context = prepare_context(device=warp_device, contract_error=TransportError)
    batch = HistoryBatch(7263, 2**63 + 7, 64).with_library_source("parallel-beam")
    design = GeometrySamplingDesign(law, 3, 3, 4, layout)
    prepared = prepare_geometry_sampling(
        context, spec=_spec(layout), batch=batch, design=design, max_fluorescence_lines=2
    )
    provider = GeometryHostSampler(prepared.identity)
    expected_bytes = (
        0 if law == "iid" else 4 * design.dimensions * (33 if law == "scrambled-sobol" else 2)
    )
    assert prepared.allocated_bytes == expected_bytes
    assert prepared.physical_record.dimensions == design.physical_dimensions
    assert prepared.physical_record.words.ptr == prepared.auxiliary_record.physical.words.ptr
    assert prepared.physical_record.shift.ptr == prepared.auxiliary_record.physical.shift.ptr
    out = wp.empty(batch.count, dtype=wp.vec4d, device=warp_device)
    requests = [(event, role) for event in range(3) for role in range(3)] + [(0, 3), (0, 4)]
    for event, role in requests:
        domain = AUXILIARY_DOMAIN | role
        with context.scope():
            wp.launch(
                inspect_geometry_auxiliary,
                dim=batch.count,
                inputs=[
                    wp.uint64(batch.seed),
                    wp.uint64(batch.first_history),
                    wp.uint32(event),
                    wp.uint32(domain),
                    prepared.auxiliary_record,
                    out,
                ],
                device=context.device,
                stream=context.stream,
            )
        actual = _read(context, out)
        expected = np.array(
            [
                [provider.uniform((history, 0, 0), event, domain, lane) for lane in range(4)]
                for history in range(batch.first_history, batch.first_history + batch.count)
            ]
        )
        np.testing.assert_array_equal(actual, expected)
        active_width = (4, 3, 3, 3, 2)[role]
        np.testing.assert_array_equal(actual[:, active_width:], 0.5)
        if law != "iid":
            # A one-dimensional digital net and an odd affine stratum permutation
            # each contain exactly one sample in every dyadic stratum.
            for lane in range(active_width):
                np.testing.assert_array_equal(
                    np.sort(np.floor(actual[:, lane] * batch.count)), np.arange(batch.count)
                )
        for rank in range(3):
            shard = prepared.identity.shard(3, rank)
            local = prepared.assert_identity(shard)
            with context.scope():
                wp.launch(
                    inspect_geometry_auxiliary,
                    dim=local.count,
                    inputs=[
                        wp.uint64(local.seed),
                        wp.uint64(local.first_history),
                        wp.uint32(event),
                        wp.uint32(domain),
                        prepared.auxiliary_record,
                        out[: local.count],
                    ],
                    device=context.device,
                    stream=context.stream,
                )
            np.testing.assert_array_equal(
                _read(context, out)[: local.count],
                actual[shard.histories.offset : shard.histories.offset + local.count],
            )


@pytest.mark.parametrize("layout", ("legacy-v1", "packed-compton-v2", "bound-compton-v3"))
@pytest.mark.parametrize("law", ("iid", "stratified", "scrambled-sobol"))
def test_physical_prefix_bitwise_and_conditional_lineage(law, layout, warp_device):
    context = prepare_context(device=warp_device, contract_error=TransportError)
    batch = HistoryBatch(823, 4096, 64)
    design = GeometrySamplingDesign(law, 3, 3, 2, layout)
    prepared = prepare_geometry_sampling(
        context, spec=_spec(layout), batch=batch, design=design, max_fluorescence_lines=0
    )
    provider = GeometryHostSampler(prepared.identity)
    out = wp.empty(batch.count, dtype=wp.vec4d, device=warp_device)
    reference = wp.empty_like(out)
    compact = (
        None
        if law == "iid"
        else SamplingIdentity(
            SamplingDesign(law, 3, 3, 1, layout), batch.seed, batch.first_history, batch.count
        )
    )
    roles = [0, 1, 2, 3, 0x40000000, 0x40000001, 0x80000000, 0xC0000002]
    if layout == "bound-compton-v3":
        roles.extend((0x10000000, 0x20000000))
    for branch, offset, event, domain in [
        (0, 0, 0x80000000, 0),
        (0, 0, 0x80000002, 0),
        *[(branch, 0 if branch < 3 else 6, 2, role) for branch in range(5) for role in roles],
    ]:
        with context.scope():
            wp.launch(
                inspect_geometry_physical,
                dim=batch.count,
                inputs=[
                    wp.uint64(batch.seed),
                    wp.uint64(batch.first_history),
                    branch,
                    wp.uint32(offset),
                    wp.uint32(event),
                    wp.uint32(domain),
                    prepared.auxiliary_record,
                    out,
                ],
                device=context.device,
                stream=context.stream,
            )
        actual = _read(context, out)
        expected = np.array(
            [
                [
                    provider.uniform((history, branch, offset), event, domain, lane)
                    for lane in range(4)
                ]
                for history in range(batch.first_history, batch.first_history + batch.count)
            ]
        )
        np.testing.assert_array_equal(actual, expected)
        if law == "iid":
            # Kind zero is the unmodified default Philox implementation.
            with context.scope():
                wp.launch(
                    inspect_random_design,
                    dim=batch.count,
                    inputs=[
                        wp.uint64(batch.seed),
                        wp.uint64(batch.first_history),
                        wp.uint32(offset + event),
                        wp.uint32(domain | branch << 16),
                        RandomDesign(),
                        reference,
                    ],
                    device=context.device,
                    stream=context.stream,
                )
            np.testing.assert_array_equal(actual, _read(context, reference))
        elif branch == 0:
            assert compact is not None
            np.testing.assert_array_equal(
                actual, sampling_uniforms(compact, event=event, domain=domain)
            )


@pytest.mark.parametrize("law", ("iid", "stratified", "scrambled-sobol"))
def test_invalid_native_addresses_and_prepared_binding(law, warp_device):
    context = prepare_context(device=warp_device, contract_error=TransportError)
    batch = HistoryBatch(17, 101, 64)
    design = GeometrySamplingDesign(law, 3, 3, 2)
    prepared = prepare_geometry_sampling(
        context, spec=_spec("legacy-v1"), batch=batch, design=design, max_fluorescence_lines=0
    )
    out = wp.empty(batch.count, dtype=wp.vec4d, device=warp_device)
    for seed, first, event, domain in (
        (18, 101, 0, AUXILIARY_DOMAIN),
        (17, 165, 0, AUXILIARY_DOMAIN),
        (17, 101, 3, AUXILIARY_DOMAIN),
        (17, 101, 1, AUXILIARY_DOMAIN | 3),
        (17, 101, 0, AUXILIARY_DOMAIN | 5),
    ):
        with context.scope():
            wp.launch(
                inspect_geometry_auxiliary,
                dim=batch.count,
                inputs=[
                    wp.uint64(seed),
                    wp.uint64(first),
                    wp.uint32(event),
                    wp.uint32(domain),
                    prepared.auxiliary_record,
                    out,
                ],
                device=context.device,
                stream=context.stream,
            )
        assert np.all(np.isnan(_read(context, out)))
    for branch, offset, event, domain in (
        (0, 1, 0, 0),
        (3, 4, 0, 0),
        (5, 0, 0, 0),
        (0, 0, 3, 0),
        (3, 6, 3, 0),
        (0, 0, 0, AUXILIARY_DOMAIN),
        (0, 0, 0, 0x10000000),
        (1, 0, 0x80000000, 0),
    ):
        with context.scope():
            wp.launch(
                inspect_geometry_physical,
                dim=batch.count,
                inputs=[
                    wp.uint64(batch.seed),
                    wp.uint64(batch.first_history),
                    branch,
                    wp.uint32(offset),
                    wp.uint32(event),
                    wp.uint32(domain),
                    prepared.auxiliary_record,
                    out,
                ],
                device=context.device,
                stream=context.stream,
            )
        assert np.all(np.isnan(_read(context, out)))
    with pytest.raises(TransportError, match="complete design"):
        prepared.assert_identity(GeometrySamplingIdentity(design, HistoryBatch(18, 101, 64)))
    with pytest.raises(TransportError, match="channel slots"):
        prepare_geometry_sampling(
            context, spec=_spec("legacy-v1"), batch=batch, design=design, max_fluorescence_lines=1
        )
    with pytest.raises(TransportError, match="actual event guard"):
        prepare_geometry_sampling(
            context,
            spec=dataclasses.replace(_spec("legacy-v1"), max_events=2),
            batch=batch,
            design=design,
            max_fluorescence_lines=0,
        )
    with pytest.raises(TransportError, match="execution policy"):
        prepare_geometry_sampling(
            context,
            spec=_spec("packed-compton-v2"),
            batch=batch,
            design=design,
            max_fluorescence_lines=0,
        )

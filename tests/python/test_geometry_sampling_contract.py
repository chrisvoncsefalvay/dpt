"""Complete geometry sampling identities and independently checked coordinate maps."""

import dataclasses

import pytest

from dpt.transport.geometry_sampling import (
    AUXILIARY_DOMAIN,
    GeometryHostSampler,
    GeometrySamplingDesign,
    GeometrySamplingIdentity,
    GeometrySourceIdentity,
)
from dpt.transport.model import TransportError
from dpt.transport.rng import HistoryBatch
from dpt.transport.sampling_identity import SamplingDesign, SamplingIdentity


@pytest.mark.parametrize("layout", ("legacy-v1", "packed-compton-v2", "bound-compton-v3"))
@pytest.mark.parametrize("law", ("iid", "stratified", "scrambled-sobol"))
def test_complete_auxiliary_map_and_unchanged_prefix(layout, law):
    design = GeometrySamplingDesign(law, 3, 3, 4, layout)
    identity = GeometrySamplingIdentity(
        design, HistoryBatch(17, 91, 32).with_library_source("parallel-beam")
    )
    auxiliary = set()
    for event in range(3):
        for role, width in ((0, 4), (1, 3), (2, 3)):
            for lane in range(width):
                auxiliary.add(design.auxiliary_dimension(event, AUXILIARY_DOMAIN | role, lane))
    for role, width in ((3, 3), (4, 2)):
        for lane in range(width):
            auxiliary.add(design.auxiliary_dimension(0, AUXILIARY_DOMAIN | role, lane))
    assert auxiliary == set(range(design.physical_dimensions, design.dimensions))
    assert design.counter_span == 9 and design.max_branches == 7
    assert identity.source_identity == GeometrySourceIdentity(design, identity.batch)
    assert len(identity.identity) == 64
    if law != "iid":
        projection = identity.physical_projection
        assert projection is not None
        assert projection.design.dimensions == design.physical_dimensions
        compact = SamplingDesign(law, 3, 3, 1, layout)
        for event in range(3):
            for domain, width in (
                (0, 4),
                (1, 1),
                (2, 3),
                (3, 1),
                (0x40000000, 2),
                (0x40000001, 1),
                (0x80000000, 3),
                (0xC0000000, 3),
            ):
                for lane in range(width):
                    assert projection.design.dimension(event, domain, lane) == compact.dimension(
                        event, domain, lane
                    )


def test_philox_published_zero_vector_and_original_lineage():
    # Random123's Philox4x32-10 known-answer zero counter and zero key.
    expected = (0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8)
    identity = GeometrySamplingIdentity(
        GeometrySamplingDesign("iid", 4, 8, 2), HistoryBatch(0, 0, 16)
    )
    provider = GeometryHostSampler(identity)
    assert tuple(provider.uniform((0, 0, 0), 0, 0, lane) for lane in range(4)) == tuple(
        (word + 0.5) / 2**32 for word in expected
    )
    assert provider.identity == identity.identity and provider.layout == "legacy-v1"
    for history in range(16):
        pair = tuple(provider.uniform((history, 3, 8), 1, 0, lane) for lane in range(4))
        assert pair == tuple(provider.uniform((history, 3, 8), 1, 0, lane) for lane in range(4))
        assert all(0 < draw < 1 for draw in pair)
    assert provider.uniform((0, 0, 0), 0, AUXILIARY_DOMAIN | 4, 3) == 0.5


@pytest.mark.parametrize("layout", ("legacy-v1", "packed-compton-v2", "bound-compton-v3"))
@pytest.mark.parametrize("law", ("stratified", "scrambled-sobol"))
def test_shards_preserve_complete_identity_and_are_dependent(layout, law):
    design = GeometrySamplingDesign(law, 2, 4, 2, layout)
    identity = GeometrySamplingIdentity(design, HistoryBatch(19, 1, 16))
    shards = [identity.shard(3, rank) for rank in range(3)]
    assert all(shard.identity is identity for shard in shards)
    assert sum(shard.batch.count for shard in shards) == 16
    assert [
        history
        for shard in shards
        for history in range(
            shard.batch.first_history, shard.batch.first_history + shard.batch.count
        )
    ] == list(range(1, 17))
    other_range = GeometrySamplingIdentity(design, HistoryBatch(19, 101, 16))
    assert not identity.independent_of(other_range)
    assert identity.independent_of(GeometrySamplingIdentity(design, HistoryBatch(20, 1, 16)))
    assert identity.identity != other_range.identity


@pytest.mark.parametrize("law", ("stratified", "scrambled-sobol"))
def test_total_dimension_limit_and_complete_count(law):
    with pytest.raises(TransportError, match="complete geometry sampling requires"):
        GeometrySamplingDesign(law, 10, 8, 2)
    with pytest.raises(TransportError, match="power-of-two"):
        GeometrySamplingIdentity(GeometrySamplingDesign(law, 2, 8, 2), HistoryBatch(1, 0, 15))
    bound = HistoryBatch(1, 0, 16).with_sampling(
        SamplingIdentity(SamplingDesign(law, 2, 8), 1, 0, 16)
    )
    with pytest.raises(TransportError, match="complete unbound"):
        GeometrySamplingIdentity(GeometrySamplingDesign(law, 2, 8, 2), bound)


@pytest.mark.parametrize(
    "lineage,event,domain,lane",
    (
        ((0, 0, 1), 0, 0, 0),
        ((0, 3, 3), 0, 0, 0),
        ((0, 5, 0), 0, 0, 0),
        ((0, 0, 0), 4, 0, 0),
        ((0, 0, 0), 0, 0x10000, 0),
        ((0, 0, 0), 0, 0x10000000, 0),
        ((0, 1, 0), 0, AUXILIARY_DOMAIN, 0),
        ((0, 0, 0), 1, AUXILIARY_DOMAIN | 3, 0),
        ((0, 0, 0), 0, AUXILIARY_DOMAIN | 5, 0),
        ((0, 0, 0), 0, AUXILIARY_DOMAIN, 4),
        ((16, 0, 0), 0, 0, 0),
        ((0, 1, 0), 0x80000000, 0, 0),
    ),
)
def test_invalid_lineage_roles_and_guards_reject(lineage, event, domain, lane):
    provider = GeometryHostSampler(
        GeometrySamplingIdentity(GeometrySamplingDesign("iid", 4, 8, 2), HistoryBatch(0, 0, 16))
    )
    with pytest.raises(TransportError):
        provider.uniform(lineage, event, domain, lane)


def test_identity_binds_source_and_actual_guard():
    design = GeometrySamplingDesign("iid", 4, 8, 2)
    batch = HistoryBatch(17, 4, 16, "source-a")
    identity = GeometrySamplingIdentity(design, batch)
    assert (
        identity.identity
        != GeometrySamplingIdentity(
            design, dataclasses.replace(batch, source_namespace="source-b")
        ).identity
    )
    assert (
        identity.identity
        != GeometrySamplingIdentity(dataclasses.replace(design, max_events=5), batch).identity
    )
    assert (
        identity.identity
        != GeometrySamplingIdentity(design, batch.with_library_source("parallel-beam")).identity
    )
    with pytest.raises(TransportError, match="signed int32"):
        GeometrySamplingDesign("iid", 10000, 65536, 4093)
    with pytest.raises(TransportError):
        GeometrySamplingDesign("iid", True, 8, 2)

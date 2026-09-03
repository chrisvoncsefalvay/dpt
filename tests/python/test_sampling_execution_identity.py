"""Opt-in execution policies and complete versioned coordinate budgets."""

import itertools
from typing import Any, cast

import pytest

from dpt.contracts import ContractError
from dpt.detector import GaussianReadNoiseExecution, ObservationIdentity
from dpt.transport import SamplingDesign, SamplingIdentity, TransportError, TransportExecution


def test_legacy_execution_defaults_and_noise_identity_modes():
    assert TransportExecution().compton_sampler == "legacy-rejection"
    assert GaussianReadNoiseExecution().algorithm == "scalar-box-muller"
    assert ObservationIdentity(19, 4).read_noise_algorithm == "scalar-box-muller"
    assert ObservationIdentity(19, 4) != ObservationIdentity(
        19, 4, read_noise_algorithm="box-muller-pairs"
    )
    with pytest.raises(TransportError):
        TransportExecution(compton_sampler=cast(Any, "packed"))
    with pytest.raises(ContractError):
        GaussianReadNoiseExecution(cast(Any, "pairs"))
    with pytest.raises(ContractError):
        ObservationIdentity(19, 4, read_noise_algorithm=cast(Any, "pairs"))


@pytest.mark.parametrize("trials", [1, 2, 3, 8, 9])
def test_packed_layout_all_bounded_lanes_unique_and_legacy_addresses_preserved(trials):
    legacy = SamplingDesign("scrambled-sobol", 7, trials, 3)
    packed = SamplingDesign("scrambled-sobol", 7, trials, 3, "packed-compton-v2")
    assert legacy.layout == "legacy-v1"
    assert legacy.dimensions == 3 + 21 * (12 + 6 * trials)
    assert packed.dimensions == 3 + 21 * (12 + 4 * ((trials + 1) // 2) + 3 * trials)
    roles = [(0, 4), (1, 1), (2, 3), (3, 1), (0x40000000, 2), (0x40000001, 1)]
    roles.extend((0x80000000 + block, 4) for block in range((trials + 1) // 2))
    roles.extend((0xC0000000 + trial, 3) for trial in range(trials))
    coordinates = [0, 1, 2]
    for branch, event in itertools.product(range(3), range(7)):
        for domain, width in roles:
            coordinates.extend(
                packed.dimension(event, domain + branch * 65536, lane) for lane in range(width)
            )
        for trial, lane in itertools.product(range(trials), range(3)):
            assert legacy.dimension(event, 0x80000000 + branch * 65536 + trial, lane) == (
                3 + (branch * 7 + event) * (12 + 6 * trials) + 12 + 3 * trial + lane
            )
            assert legacy.dimension(event, 0xC0000000 + branch * 65536 + trial, lane) == (
                3 + (branch * 7 + event) * (12 + 6 * trials) + 12 + 3 * trials + 3 * trial + lane
            )
    assert sorted(coordinates) == list(range(packed.dimensions))
    with pytest.raises(TransportError):
        packed.dimension(0, 0x80000000 + (trials + 1) // 2, 0)
    with pytest.raises(TransportError):
        legacy.dimension(0, 0x80000000, 3)
    assert SamplingIdentity(legacy, 19, 4, 128) != SamplingIdentity(packed, 19, 4, 128)

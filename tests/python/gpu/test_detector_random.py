# Warp annotations are runtime DSL expressions, not Python static types.
# pyright: reportInvalidTypeForm=false, reportUnknownMemberType=false, reportUnknownParameterType=false
"""Published Philox known-answer vectors, rather than self-generated RNG expectations.

Reference: Random123 tests/kat_vectors, Philox4x32-10 zero/max/pi cases:
https://github.com/DEShawResearch/random123/blob/main/tests/kat_vectors.
The constants are mathematical algorithm outputs. No vendor source is bundled.
"""

import importlib
from typing import Any

import pytest

from dpt.validation.detector import poisson_log_mass

wp: Any = importlib.import_module("warp")
random_kernels: Any = importlib.import_module("dpt.kernels.random")
detector_kernels: Any = importlib.import_module("dpt.kernels.detector")
pytestmark = pytest.mark.gpu


@wp.kernel(module="unique", module_options={"enable_backward": False, "fast_math": False})
def log_mass_probe(count: wp.float64, rate: wp.float64, output: wp.array(dtype=wp.float64)):
    output[0] = detector_kernels.poisson_log_probability(count, rate)


@pytest.mark.parametrize(
    "count,rate",
    [
        (0, 10.0),
        (1, 10.0),
        (15, 10.0),
        (16, 10.0),
        (16, 16.0),
        (17, 17.0),
        (18, 21.999999999),
        (18, 22.0),
        (18, 22.000000001),
        (22, 17.999999999),
        (22, 18.0),
        (22, 18.000000001),
        (1000, 999.9),
        (10**9, 1e9),
        (10**9 + 1, 1e9),
        (10**9 - 1, 1e9),
        (10**9 - 400000, 1e9),
        (10**9 + 400000, 1e9),
    ],
)
def test_poisson_acceptance_log_mass_against_high_precision_definition(
    count: int, rate: float
) -> None:
    output = wp.empty(1, dtype=wp.float64, device="cuda:0")
    wp.launch(
        log_mass_probe,
        dim=1,
        inputs=[wp.float64(count), wp.float64(rate)],
        outputs=[output],
        device="cuda:0",
    )
    assert float(output.numpy()[0]) == pytest.approx(
        float(poisson_log_mass(count, rate)), rel=0, abs=5e-13
    )


@wp.kernel(module="unique", module_options={"enable_backward": False})
def draw_vectors(
    seed: wp.uint64,
    identity: wp.uint64,
    event: wp.uint32,
    domain: wp.uint32,
    output: wp.array(dtype=wp.float64),
):
    result = random_kernels.random4(seed, identity, event, domain)
    for i in range(4):
        output[i] = result[i]


@pytest.mark.parametrize(
    "seed,identity,event,domain,expected",
    [
        (0, 0, 0, 0, (0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8)),
        (
            2**64 - 1,
            2**64 - 1,
            2**32 - 1,
            2**32 - 1,
            (0x408F276D, 0x41C83B0E, 0xA20BC7C6, 0x6D5451FD),
        ),
        (
            0x299F31D0A4093822,
            0x85A308D3243F6A88,
            0x13198A2E,
            0x03707344,
            (0xD16CFE09, 0x94FDCCEB, 0x5001E420, 0x24126EA1),
        ),
    ],
)
def test_philox_published_words(
    seed: int,
    identity: int,
    event: int,
    domain: int,
    expected: tuple[int, ...],
) -> None:
    output = wp.empty(4, dtype=wp.float64, device="cuda:0")
    wp.launch(
        draw_vectors,
        dim=1,
        inputs=[wp.uint64(seed), wp.uint64(identity), wp.uint32(event), wp.uint32(domain)],
        outputs=[output],
        device="cuda:0",
    )
    uniforms = tuple(float(value) for value in output.numpy())
    assert all(0 < value < 1 for value in uniforms)
    assert tuple(int(value * 2**32) for value in uniforms) == expected

"""Corrected cached reciprocals retain division accuracy across FP64 exponent ranges."""

import importlib

# pyright: reportInvalidTypeForm=false, reportUnknownParameterType=false
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false, reportUntypedFunctionDecorator=false
import math
import random
from contextlib import nullcontext
from typing import Any

from dpt._runtime import cpu_test_context, load_kernels

wp: Any = importlib.import_module("warp")


def test_corrected_cached_reciprocals_match_division_within_one_ulp(warp_device: str) -> None:
    quotient = load_kernels("dpt.kernels.projection").crossing_quotient

    @wp.kernel(module="unique", enable_backward=False)
    def evaluate(
        numerator: wp.array(dtype=wp.float64),
        denominator: wp.array(dtype=wp.float64),
        inverse: wp.array(dtype=wp.float64),
        output: wp.array(dtype=wp.float64),
    ):
        index = wp.tid()
        output[index] = quotient(numerator[index], denominator[index], inverse[index])

    generator = random.Random(117)
    pairs = [
        (0.0, 3.0),
        (1.0, 3.0),
        (1e-300, 1e300),
        (1e300, 1e-300),
        (7e-321, 1e-320),
        (1.0, 1e308),
    ]
    for exponent in (-1000, -500, -100, -1, 0, 100, 500, 1000):
        for _ in range(64):
            pairs.append(
                (
                    math.ldexp(generator.uniform(-1.0, 1.0), exponent),
                    math.ldexp(generator.uniform(0.1, 1.0), exponent),
                )
            )
    inverses = []
    for _, direction in pairs:
        reciprocal = 1.0 / direction
        valid = (
            math.isfinite(reciprocal)
            and abs(reciprocal) >= 2.0**-1022
            and abs(direction) >= 2.0**-1022
        )
        inverses.append(reciprocal if valid else 0.0)
    with cpu_test_context() if warp_device == "cpu" else nullcontext():
        numerator = wp.array([x for x, _ in pairs], dtype=wp.float64, device=warp_device)
        denominator = wp.array([y for _, y in pairs], dtype=wp.float64, device=warp_device)
        inverse = wp.array(inverses, dtype=wp.float64, device=warp_device)
        output = wp.empty(len(pairs), dtype=wp.float64, device=warp_device)
        wp.launch(
            evaluate,
            dim=len(pairs),
            inputs=[numerator, denominator, inverse],
            outputs=[output],
            device=warp_device,
        )
    for actual, (delta, direction) in zip(output.numpy().tolist(), pairs, strict=True):
        expected = delta / direction
        if math.isfinite(expected):
            assert abs(actual - expected) <= math.ulp(expected), (
                delta,
                direction,
                actual,
                expected,
            )
        else:
            assert actual == expected

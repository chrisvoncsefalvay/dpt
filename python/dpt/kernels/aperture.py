"""Range-preserving weighted subray transmission and explicit count cotangents."""

# pyright: reportInvalidTypeForm=false, reportUnknownParameterType=false
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false, reportUntypedFunctionDecorator=false
# pyright: reportMissingImports=false

from functools import cache

import warp as wp

from dpt.kernels.numerics import attenuated_product4

OPTIONS = {"fast_math": False, "fuse_fp": False, "enable_backward": False}
wp.set_module_options(OPTIONS)


@cache
def get_validate(nonnegative: bool):
    @wp.kernel(module="unique", module_options=OPTIONS)
    def check(values: wp.array(dtype=wp.float64), status: wp.array(dtype=wp.int32)):
        value = values[wp.tid()]
        if not wp.isfinite(value):
            wp.atomic_or(status, 0, 1)
        if wp.static(nonnegative):
            if value < wp.float64(0.0):
                wp.atomic_or(status, 0, 1)

    return check


@cache
def _get_input_check(check_seed: bool):
    @wp.kernel(module="unique", module_options=OPTIONS)
    def check(
        beam: wp.array(dtype=wp.float64),
        seeds: wp.array(dtype=wp.float64),
        status: wp.array(dtype=wp.int32),
    ):
        pixel = wp.tid()
        value = beam[pixel]
        if not wp.isfinite(value) or value < wp.float64(0.0):
            wp.atomic_or(status, 0, 1)
        if wp.static(check_seed):
            if not wp.isfinite(seeds[pixel]):
                wp.atomic_or(status, 0, 1)

    return check


@wp.kernel
def forward(
    depths: wp.array(dtype=wp.float64),
    beam: wp.array(dtype=wp.float64),
    offsets: wp.array(dtype=wp.int32),
    weights: wp.array(dtype=wp.float64),
    counts: wp.array(dtype=wp.float64),
    status: wp.array(dtype=wp.int32),
):
    pixel = wp.tid()
    value = wp.float64(0.0)
    beam_value = beam[pixel]
    if not wp.isfinite(beam_value) or beam_value < wp.float64(0.0):
        wp.atomic_or(status, 0, 1)
    for ray in range(offsets[pixel], offsets[pixel + 1]):
        depth = depths[ray]
        if not wp.isfinite(depth):
            wp.atomic_or(status, 0, 2)
        elif depth < wp.float64(0.0):
            wp.atomic_or(status, 0, 8)
        value += attenuated_product4(
            depth, beam_value, weights[ray], wp.float64(1.0), wp.float64(1.0)
        )
    counts[pixel] = value
    if not wp.isfinite(value):
        wp.atomic_or(status, 0, 2)


@cache
def get_vjp(write_beam: bool, accumulate: bool):
    @wp.kernel(module="unique", module_options=OPTIONS)
    def backward(
        depths: wp.array(dtype=wp.float64),
        beam: wp.array(dtype=wp.float64),
        seeds: wp.array(dtype=wp.float64),
        offsets: wp.array(dtype=wp.int32),
        weights: wp.array(dtype=wp.float64),
        depth_seeds: wp.array(dtype=wp.float64),
        beam_gradient: wp.array(dtype=wp.float64),
        status: wp.array(dtype=wp.int32),
    ):
        pixel = wp.tid()
        total = wp.float64(0.0)
        beam_value = beam[pixel]
        seed_value = seeds[pixel]
        if (
            not wp.isfinite(beam_value)
            or beam_value < wp.float64(0.0)
            or not wp.isfinite(seed_value)
        ):
            wp.atomic_or(status, 0, 1)
        for ray in range(offsets[pixel], offsets[pixel + 1]):
            depth = depths[ray]
            weight = weights[ray]
            derivative = -attenuated_product4(
                depth, beam_value, weight, seed_value, wp.float64(1.0)
            )
            depth_seeds[ray] = derivative
            if not wp.isfinite(derivative):
                wp.atomic_or(status, 0, 2)
            if wp.static(write_beam):
                total += attenuated_product4(
                    depth, weight, seed_value, wp.float64(1.0), wp.float64(1.0)
                )
        if wp.static(write_beam):
            if wp.static(accumulate):
                total += beam_gradient[pixel]
            beam_gradient[pixel] = total
            if not wp.isfinite(total):
                wp.atomic_or(status, 0, 2)

    return backward

"""Spectral primary signal and explicit first-order products.

Paths are material-major (M,P); coefficients are (M,K); weights/response are
shared (K) or energy-major (K,P). FP64 registers hold depths, products and sums.
There is no pixel-by-energy retained tensor. Global parameter gradients use
bounded split reductions; only diagnostic flags use atomics.
"""

# Warp annotations are executable DSL expressions; host interfaces remain strict.
# The optional GPU import is resolved only when an operator is prepared.
# pyright: reportInvalidTypeForm=false, reportUnknownParameterType=false
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false, reportUntypedFunctionDecorator=false
# pyright: reportMissingImports=false, reportUntypedClassDecorator=false

from functools import cache

import warp as wp

from dpt.kernels.numerics import (
    attenuated_product4,
    attenuated_product4_from_state,
    attenuation_state,
)
from dpt.kernels.spectral_common import (
    compensated_sum_update,
    energy_pixel_index,
    material_major_index,
)

STRICT = {"fast_math": False, "fuse_fp": True, "enable_backward": False}
wp.set_module_options(STRICT)


@wp.func
def checked_store(value: wp.float64, status: wp.array(dtype=wp.int32)) -> wp.float32:
    rounded = wp.float32(value)
    if not wp.isfinite(rounded):
        wp.atomic_or(status, 0, 2)
    return rounded


@cache
def get_value_check(wide: bool, nonnegative: bool):
    dtype = wp.float64 if wide else wp.float32

    @wp.kernel(module="unique", module_options=STRICT)
    def check(values: wp.array(dtype=dtype), status: wp.array(dtype=wp.int32)):
        index = wp.tid()
        invalid = not wp.isfinite(values[index])
        if wp.static(nonnegative):
            invalid = invalid or values[index] < dtype(0.0)
        if invalid:
            wp.atomic_or(status, 0, 1)

    return check


check_nonnegative = get_value_check(False, True)
check_finite = get_value_check(False, False)


@cache
def get_signal_store(wide: bool):
    dtype = wp.float64 if wide else wp.float32

    @wp.func
    def store(value: wp.float64, status: wp.array(dtype=wp.int32)) -> dtype:
        result = dtype(value)
        if not wp.isfinite(result):
            wp.atomic_or(status, 0, 2)
        return result

    return store


@cache
def get_optical_depth(wide: bool, arithmetic32: bool = False):
    dtype = wp.float64 if wide else wp.float32

    @wp.func
    def depth_at(
        paths: wp.array(dtype=dtype),
        coefficients: wp.array(dtype=wp.float32),
        pixel: int,
        energy: int,
        pixels: int,
        materials: int,
        energies: int,
    ) -> wp.float64:
        if wp.static(arithmetic32):
            narrow_depth = wp.float32(0.0)
            for material in range(materials):
                narrow_depth = (
                    narrow_depth
                    + wp.float32(paths[material_major_index(material, pixel, pixels)])
                    * coefficients[material_major_index(material, energy, energies)]
                )
            return wp.float64(narrow_depth)
        else:
            depth = wp.float64(0.0)
            for material in range(materials):
                depth = depth + wp.float64(
                    paths[material_major_index(material, pixel, pixels)]
                ) * wp.float64(coefficients[material_major_index(material, energy, energies)])
            return depth

    return depth_at


@cache
def get_term(wide: bool, arithmetic32: bool = False):
    @wp.func
    def term(
        depth: wp.float64, a: wp.float64, b: wp.float64, c: wp.float64, d: wp.float64
    ) -> wp.float64:
        if (
            a == wp.float64(0.0)
            or b == wp.float64(0.0)
            or c == wp.float64(0.0)
            or d == wp.float64(0.0)
        ):
            return wp.float64(0.0)
        if wp.static(wide):
            return attenuated_product4(depth, a, b, c, d)
        if wp.static(arithmetic32):
            return (((a * wp.float64(wp.exp(-wp.float32(depth)))) * b) * c) * d
        return (((a * wp.exp(-depth)) * b) * c) * d

    return term


@cache
def get_attenuation(wide: bool, arithmetic32: bool = False):
    @wp.func
    def attenuation(depth: wp.float64) -> wp.vec2d:
        if wp.static(wide):
            return attenuation_state(depth)
        if wp.static(arithmetic32):
            return wp.vec2d(wp.float64(wp.exp(-wp.float32(depth))), wp.float64(0.0))
        return wp.vec2d(wp.exp(-depth), wp.float64(0.0))

    return attenuation


@cache
def get_term_from_attenuation(wide: bool):
    @wp.func
    def term(
        state: wp.vec2d, a: wp.float64, b: wp.float64, c: wp.float64, d: wp.float64
    ) -> wp.float64:
        if wp.static(wide):
            return attenuated_product4_from_state(state, a, b, c, d)
        if (
            a == wp.float64(0.0)
            or b == wp.float64(0.0)
            or c == wp.float64(0.0)
            or d == wp.float64(0.0)
        ):
            return wp.float64(0.0)
        return (((a * state[0]) * b) * c) * d

    return term


@cache
def get_forward(
    materials: int,
    energies: int,
    shared_weights: bool,
    shared_response: bool,
    wide: bool = False,
    arithmetic32: bool = False,
):
    dtype = wp.float64 if wide else wp.float32
    optical_depth = get_optical_depth(wide, arithmetic32)
    term_at = get_term(wide, arithmetic32)
    store_signal = get_signal_store(wide)

    @wp.kernel(module="unique", module_options=STRICT)
    def forward(
        paths: wp.array(dtype=dtype),
        coefficients: wp.array(dtype=wp.float32),
        weights: wp.array(dtype=wp.float32),
        response: wp.array(dtype=wp.float32),
        pixels: int,
        mean: wp.array(dtype=dtype),
        status: wp.array(dtype=wp.int32),
    ):
        pixel = wp.tid()
        total = wp.float64(0.0)
        compensation = wp.float64(0.0)
        for energy in range(energies):
            wi = energy_pixel_index(energy, pixel, pixels)
            ri = wi
            if wp.static(shared_weights):
                wi = energy
            if wp.static(shared_response):
                ri = energy
            depth = optical_depth(paths, coefficients, pixel, energy, pixels, materials, energies)
            contribution = term_at(
                depth,
                wp.float64(weights[wi]),
                wp.float64(response[ri]),
                wp.float64(1.0),
                wp.float64(1.0),
            )
            # All terms are non-negative, but compensation retains small bins
            # when a broad response places many decades in the same sum.
            if wp.static(arithmetic32):
                total += contribution
            else:
                total, compensation = compensated_sum_update(total, compensation, contribution)
        mean[pixel] = store_signal(total, status)

    return forward


@cache
def get_pixel_vjp(
    materials: int,
    energies: int,
    shared_weights: bool,
    shared_response: bool,
    write_paths: bool,
    write_weights: bool,
    write_response: bool,
    wide: bool = False,
    arithmetic32: bool = False,
):
    dtype = wp.float64 if wide else wp.float32
    optical_depth = get_optical_depth(wide, arithmetic32)
    attenuation_at = get_attenuation(wide, arithmetic32)
    term_at = get_term_from_attenuation(wide)
    store_signal = get_signal_store(wide)
    gradient_vector = wp.types.vector(length=materials, dtype=wp.float64)

    @wp.kernel(module="unique", module_options=STRICT)
    def vjp(
        paths: wp.array(dtype=dtype),
        coefficients: wp.array(dtype=wp.float32),
        weights: wp.array(dtype=wp.float32),
        response: wp.array(dtype=wp.float32),
        seed: wp.array(dtype=dtype),
        pixels: int,
        grad_paths: wp.array(dtype=dtype),
        grad_weights: wp.array(dtype=wp.float32),
        grad_response: wp.array(dtype=wp.float32),
        status: wp.array(dtype=wp.int32),
    ):
        pixel = wp.tid()
        path_gradient = gradient_vector()
        path_compensation = gradient_vector()
        for energy in range(energies):
            wi = energy_pixel_index(energy, pixel, pixels)
            ri = wi
            if wp.static(shared_weights):
                wi = energy
            if wp.static(shared_response):
                ri = energy
            depth = optical_depth(paths, coefficients, pixel, energy, pixels, materials, energies)
            attenuation = attenuation_at(depth)
            if wp.static(write_weights):
                grad_weights[wi] = checked_store(
                    term_at(
                        attenuation,
                        wp.float64(seed[pixel]),
                        wp.float64(response[ri]),
                        wp.float64(1.0),
                        wp.float64(1.0),
                    ),
                    status,
                )
            if wp.static(write_response):
                grad_response[ri] = checked_store(
                    term_at(
                        attenuation,
                        wp.float64(seed[pixel]),
                        wp.float64(weights[wi]),
                        wp.float64(1.0),
                        wp.float64(1.0),
                    ),
                    status,
                )
            if wp.static(write_paths):
                for material in range(materials):
                    term = term_at(
                        attenuation,
                        -wp.float64(seed[pixel]),
                        wp.float64(weights[wi]),
                        wp.float64(response[ri]),
                        wp.float64(coefficients[material_major_index(material, energy, energies)]),
                    )
                    updated, next_compensation = compensated_sum_update(
                        path_gradient[material], path_compensation[material], term
                    )
                    path_compensation[material] = next_compensation
                    path_gradient[material] = updated
        if wp.static(write_paths):
            for material in range(materials):
                grad_paths[material_major_index(material, pixel, pixels)] = store_signal(
                    path_gradient[material], status
                )

    return vjp


@cache
def get_shared_partials(
    materials: int,
    energies: int,
    shared_weights: bool,
    shared_response: bool,
    kind: int,
    wide: bool = False,
    arithmetic32: bool = False,
):
    """kind=0 weights or 1 response; coefficients have their own shared kernel."""
    dtype = wp.float64 if wide else wp.float32
    optical_depth = get_optical_depth(wide, arithmetic32)
    term_at = get_term(wide, arithmetic32)
    if kind not in (0, 1):
        raise ValueError("shared partial kind must select weights or response")

    @wp.kernel(module="unique", module_options=STRICT)
    def partials(
        paths: wp.array(dtype=dtype),
        coefficients: wp.array(dtype=wp.float32),
        weights: wp.array(dtype=wp.float32),
        response: wp.array(dtype=wp.float32),
        seed: wp.array(dtype=dtype),
        pixels: int,
        groups: int,
        output: wp.array(dtype=wp.float64),
        status: wp.array(dtype=wp.int32),
    ):
        group, parameter, lane = wp.tid()
        energy = parameter
        total = wp.float64(0.0)
        compensation = wp.float64(0.0)
        pixel = group * wp.block_dim() + lane
        while pixel < pixels:
            wi = energy_pixel_index(energy, pixel, pixels)
            ri = wi
            if wp.static(shared_weights):
                wi = energy
            if wp.static(shared_response):
                ri = energy
            depth = optical_depth(paths, coefficients, pixel, energy, pixels, materials, energies)
            if wp.static(kind == 0):
                term = term_at(
                    depth,
                    wp.float64(seed[pixel]),
                    wp.float64(response[ri]),
                    wp.float64(1.0),
                    wp.float64(1.0),
                )
            else:
                term = term_at(
                    depth,
                    wp.float64(seed[pixel]),
                    wp.float64(weights[wi]),
                    wp.float64(1.0),
                    wp.float64(1.0),
                )
            total, compensation = compensated_sum_update(total, compensation, term)
            # The final stride may exceed int32 even when every input index fits.
            # Stop before adding it; pixels - pixel is non-negative and bounded.
            if pixels - pixel <= groups * wp.block_dim():
                break
            pixel = pixel + groups * wp.block_dim()
        values = wp.tile(total)
        result = wp.tile_sum(values)
        wp.tile_store(output, result, offset=parameter * groups + group)

    return partials


@cache
def get_coefficient_partials(
    materials: int,
    energies: int,
    shared_weights: bool,
    shared_response: bool,
    wide: bool = False,
    arithmetic32: bool = False,
):
    """Reuse each energy's depth across its material cotangents, without a P*K tensor."""
    dtype = wp.float64 if wide else wp.float32
    optical_depth = get_optical_depth(wide, arithmetic32)
    attenuation_at = get_attenuation(wide, arithmetic32)
    term_at = get_term_from_attenuation(wide)
    accumulator = wp.types.vector(length=materials, dtype=wp.float64)

    @wp.kernel(module="unique", module_options=STRICT)
    def coefficient_partials(
        paths: wp.array(dtype=dtype),
        coefficients: wp.array(dtype=wp.float32),
        weights: wp.array(dtype=wp.float32),
        response: wp.array(dtype=wp.float32),
        seed: wp.array(dtype=dtype),
        pixels: int,
        groups: int,
        output: wp.array(dtype=wp.float64),
        status: wp.array(dtype=wp.int32),
    ):
        group, energy, lane = wp.tid()
        totals = accumulator()
        compensations = accumulator()
        pixel = group * wp.block_dim() + lane
        while pixel < pixels:
            wi = energy_pixel_index(energy, pixel, pixels)
            ri = wi
            if wp.static(shared_weights):
                wi = energy
            if wp.static(shared_response):
                ri = energy
            depth = optical_depth(paths, coefficients, pixel, energy, pixels, materials, energies)
            attenuation = attenuation_at(depth)
            for material in range(materials):
                term = term_at(
                    attenuation,
                    -wp.float64(seed[pixel]),
                    wp.float64(weights[wi]),
                    wp.float64(response[ri]),
                    wp.float64(paths[material_major_index(material, pixel, pixels)]),
                )
                updated, next_compensation = compensated_sum_update(
                    totals[material], compensations[material], term
                )
                compensations[material] = next_compensation
                totals[material] = updated
            # The final stride may exceed int32 even when every input index fits.
            # Stop before adding it; pixels - pixel is non-negative and bounded.
            if pixels - pixel <= groups * wp.block_dim():
                break
            pixel = pixel + groups * wp.block_dim()
        # Each parameter retains the original lane tree and split-sum order.
        for material in range(materials):
            values = wp.tile(totals[material])
            result = wp.tile_sum(values)
            wp.tile_store(
                output,
                result,
                offset=(material_major_index(material, energy, energies)) * groups + group,
            )

    return coefficient_partials


@cache
def get_nuisance_partials(
    materials: int,
    energies: int,
    shared_weights: bool,
    shared_response: bool,
    write_weights: bool,
    write_response: bool,
    write_coefficients: bool,
    wide: bool = False,
    arithmetic32: bool = False,
):
    """Fuse requested shared products without factoring their range-sensitive sums."""
    dtype = wp.float64 if wide else wp.float32
    optical_depth = get_optical_depth(wide, arithmetic32)
    attenuation_at = get_attenuation(wide, arithmetic32)
    term_at = get_term_from_attenuation(wide)
    accumulator = wp.types.vector(length=materials + 2, dtype=wp.float64)
    response_offset = int(write_weights) * energies
    coefficient_offset = (int(write_weights) + int(write_response)) * energies

    @wp.kernel(module="unique", module_options=STRICT)
    def nuisance_partials(
        paths: wp.array(dtype=dtype),
        coefficients: wp.array(dtype=wp.float32),
        weights: wp.array(dtype=wp.float32),
        response: wp.array(dtype=wp.float32),
        seed: wp.array(dtype=dtype),
        pixels: int,
        groups: int,
        output: wp.array(dtype=wp.float64),
        status: wp.array(dtype=wp.int32),
    ):
        group, energy, lane = wp.tid()
        totals = accumulator()
        compensations = accumulator()
        pixel = group * wp.block_dim() + lane
        while pixel < pixels:
            wi = energy_pixel_index(energy, pixel, pixels)
            ri = wi
            if wp.static(shared_weights):
                wi = energy
            if wp.static(shared_response):
                ri = energy
            depth = optical_depth(paths, coefficients, pixel, energy, pixels, materials, energies)
            attenuation = attenuation_at(depth)
            if wp.static(write_weights):
                term = term_at(
                    attenuation,
                    wp.float64(seed[pixel]),
                    wp.float64(response[ri]),
                    wp.float64(1.0),
                    wp.float64(1.0),
                )
                updated, next_compensation = compensated_sum_update(
                    totals[0], compensations[0], term
                )
                compensations[0] = next_compensation
                totals[0] = updated
            if wp.static(write_response):
                term = term_at(
                    attenuation,
                    wp.float64(seed[pixel]),
                    wp.float64(weights[wi]),
                    wp.float64(1.0),
                    wp.float64(1.0),
                )
                updated, next_compensation = compensated_sum_update(
                    totals[1], compensations[1], term
                )
                compensations[1] = next_compensation
                totals[1] = updated
            if wp.static(write_coefficients):
                for material in range(materials):
                    index = material + 2
                    term = term_at(
                        attenuation,
                        -wp.float64(seed[pixel]),
                        wp.float64(weights[wi]),
                        wp.float64(response[ri]),
                        wp.float64(paths[material_major_index(material, pixel, pixels)]),
                    )
                    updated, next_compensation = compensated_sum_update(
                        totals[index], compensations[index], term
                    )
                    compensations[index] = next_compensation
                    totals[index] = updated
            if pixels - pixel <= groups * wp.block_dim():
                break
            pixel = pixel + groups * wp.block_dim()
        if wp.static(write_weights):
            result = wp.tile_sum(wp.tile(totals[0]))
            wp.tile_store(output, result, offset=energy * groups + group)
        if wp.static(write_response):
            result = wp.tile_sum(wp.tile(totals[1]))
            wp.tile_store(output, result, offset=(response_offset + energy) * groups + group)
        if wp.static(write_coefficients):
            for material in range(materials):
                result = wp.tile_sum(wp.tile(totals[material + 2]))
                wp.tile_store(
                    output,
                    result,
                    offset=(coefficient_offset + material_major_index(material, energy, energies))
                    * groups
                    + group,
                )

    return nuisance_partials


@cache
def get_finish_shared(output64: bool):
    """One reduction implementation with explicitly chosen scalar destination precision."""
    output_dtype = wp.float64 if output64 else wp.float32

    @wp.kernel(module="unique", module_options=STRICT)
    def finish_shared(
        partials: wp.array(dtype=wp.float64),
        groups: int,
        output: wp.array(dtype=output_dtype),
        status: wp.array(dtype=wp.int32),
    ):
        parameter = wp.tid()
        total = wp.float64(0.0)
        compensation = wp.float64(0.0)
        for group in range(groups):
            total, compensation = compensated_sum_update(
                total, compensation, partials[parameter * groups + group]
            )
        if wp.static(output64):
            output[parameter] = total
            if not wp.isfinite(total):
                wp.atomic_or(status, 0, 2)
        else:
            output[parameter] = checked_store(total, status)

    return finish_shared


finish_shared = get_finish_shared(False)


@wp.kernel
def check_probability(values: wp.array(dtype=wp.float32), status: wp.array(dtype=wp.int32)):
    p = wp.tid()
    if not wp.isfinite(values[p]) or values[p] < wp.float32(0.0) or values[p] > wp.float32(1.0):
        wp.atomic_or(status, 0, 1)


@cache
def get_bin_counts(
    materials: int,
    energies: int,
    shared_weights: bool,
    shared_efficiency: bool,
    wide: bool = False,
    arithmetic32: bool = False,
):
    dtype = wp.float64 if wide else wp.float32
    optical_depth = get_optical_depth(wide, arithmetic32)
    term_at = get_term(wide, arithmetic32)
    store_signal = get_signal_store(wide)

    @wp.kernel(module="unique", module_options=STRICT)
    def counts(
        paths: wp.array(dtype=dtype),
        coefficients: wp.array(dtype=wp.float32),
        weights: wp.array(dtype=wp.float32),
        efficiency: wp.array(dtype=wp.float32),
        pixels: int,
        output: wp.array(dtype=dtype),
        status: wp.array(dtype=wp.int32),
    ):
        energy, p = wp.tid()
        wi = energy_pixel_index(energy, p, pixels)
        ri = wi
        if wp.static(shared_weights):
            wi = energy
        if wp.static(shared_efficiency):
            ri = energy
        depth = optical_depth(paths, coefficients, p, energy, pixels, materials, energies)
        rate = term_at(
            depth,
            wp.float64(weights[wi]),
            wp.float64(efficiency[ri]),
            wp.float64(1.0),
            wp.float64(1.0),
        )
        output[energy_pixel_index(energy, p, pixels)] = store_signal(rate, status)

    return counts


@wp.func_native("""
if (!wp::isfinite(tau)) return wp::vec_t<2, double>(NAN, 0.0);
// Four finite binary64 factors cannot rescue a magnitude past these limits.
if (tau > 8192.0) return wp::vec_t<2, double>(0.0, 0.0);
if (tau < -8192.0) return wp::vec_t<2, double>(INFINITY, 0.0);
const int exponent = static_cast<int>(floor(tau * 1.4426950408889634074));
const double residual = fma(double(exponent), 0.69314718055994530942, -tau)
    + double(exponent) * 2.3190468138462995584e-17;
return wp::vec_t<2, double>(exp(residual), double(exponent));
""")
def log_attenuation_state(tau: wp.float64) -> wp.vec2d:
    """Range-reduce the signed shifted depth used in logarithmic reverse."""
    ...


@wp.func
def log_normalized_product(
    state: wp.vec2d, a: wp.float64, b: wp.float64, c: wp.float64, d: wp.float64
) -> wp.float64:
    """Original factors times exp(-depth-log_mean), reusing one attenuation state."""
    if a == wp.float64(0.0) or b == wp.float64(0.0) or c == wp.float64(0.0) or d == wp.float64(0.0):
        return wp.float64(0.0)
    return attenuated_product4_from_state(state, a, b, c, d)


@cache
def get_log_mean(
    materials: int, energies: int, shared_weights: bool, shared_response: bool, wide: bool
):
    dtype = wp.float64 if wide else wp.float32
    optical_depth = get_optical_depth(wide)

    @wp.func
    def evaluate(
        paths: wp.array(dtype=dtype),
        coefficients: wp.array(dtype=wp.float32),
        weights: wp.array(dtype=wp.float32),
        response: wp.array(dtype=wp.float32),
        pixel: int,
        pixels: int,
    ) -> wp.vec2d:
        reference_depth = wp.float64(wp.inf)
        positive = False
        for energy in range(energies):
            wi, ri = (
                energy_pixel_index(energy, pixel, pixels),
                energy_pixel_index(energy, pixel, pixels),
            )
            if wp.static(shared_weights):
                wi = energy
            if wp.static(shared_response):
                ri = energy
            if weights[wi] > wp.float32(0.0) and response[ri] > wp.float32(0.0):
                positive = True
                depth = optical_depth(
                    paths, coefficients, pixel, energy, pixels, materials, energies
                )
                reference_depth = wp.min(reference_depth, depth)
        if reference_depth == wp.float64(wp.inf):
            if positive:
                return wp.vec2d(wp.float64(0.0), wp.float64(wp.nan))
            return wp.vec2d(wp.float64(0.0), wp.float64(-wp.inf))
        total, compensation = wp.float64(0.0), wp.float64(0.0)
        for energy in range(energies):
            wi, ri = (
                energy_pixel_index(energy, pixel, pixels),
                energy_pixel_index(energy, pixel, pixels),
            )
            if wp.static(shared_weights):
                wi = energy
            if wp.static(shared_response):
                ri = energy
            if weights[wi] > wp.float32(0.0) and response[ri] > wp.float32(0.0):
                depth = optical_depth(
                    paths, coefficients, pixel, energy, pixels, materials, energies
                )
                # Shift depths before adding the much smaller parameter logs.
                # FP32 weights/responses bound this scaled sum away from FP64
                # overflow and give at least one representable positive term.
                value = (
                    wp.log(wp.float64(weights[wi]))
                    + wp.log(wp.float64(response[ri]))
                    - (depth - reference_depth)
                )
                total, compensation = compensated_sum_update(total, compensation, wp.exp(value))
        return wp.vec2d(reference_depth, wp.log(total))

    return evaluate


@cache
def get_log_forward(
    materials: int,
    energies: int,
    shared_weights: bool,
    shared_response: bool,
    wide: bool = False,
    arithmetic32: bool = False,
    output64: bool | None = None,
    retain_normalizer: bool = False,
):
    dtype = wp.float64 if wide else wp.float32
    output_dtype = wp.float64 if (wide if output64 is None else output64) else wp.float32
    log_mean = get_log_mean(materials, energies, shared_weights, shared_response, wide)

    @wp.kernel(module="unique", module_options=STRICT)
    def forward(
        paths: wp.array(dtype=dtype),
        coefficients: wp.array(dtype=wp.float32),
        weights: wp.array(dtype=wp.float32),
        response: wp.array(dtype=wp.float32),
        pixels: int,
        mean: wp.array(dtype=output_dtype),
        status: wp.array(dtype=wp.int32),
    ):
        p = wp.tid()
        normalizer = log_mean(paths, coefficients, weights, response, p, pixels)
        value = normalizer[1] - normalizer[0]
        rounded = output_dtype(value)
        # The extended real log(0)=-inf is an intentional primal result.
        # Every other nonfinite value is an invalid arithmetic result.
        if value != wp.float64(-wp.inf) and not wp.isfinite(rounded):
            wp.atomic_or(status, 0, 2)
        if wp.static(retain_normalizer):
            mean[2 * p] = output_dtype(normalizer[0])
            mean[2 * p + 1] = output_dtype(normalizer[1])
        else:
            mean[p] = rounded

    return forward


@cache
def get_log_pixel_vjp(
    materials: int,
    energies: int,
    shared_weights: bool,
    shared_response: bool,
    write_paths: bool,
    write_weights: bool,
    write_response: bool,
    wide: bool = False,
    arithmetic32: bool = False,
):
    dtype = wp.float64 if wide else wp.float32
    optical_depth = get_optical_depth(wide, arithmetic32)
    term_at = log_normalized_product
    store_signal = get_signal_store(wide)
    gradient_vector = wp.types.vector(length=materials, dtype=wp.float64)

    @wp.kernel(module="unique", module_options=STRICT)
    def vjp(
        paths: wp.array(dtype=dtype),
        coefficients: wp.array(dtype=wp.float32),
        weights: wp.array(dtype=wp.float32),
        response: wp.array(dtype=wp.float32),
        seed: wp.array(dtype=dtype),
        log_mean: wp.array(dtype=wp.float64),
        pixels: int,
        grad_paths: wp.array(dtype=dtype),
        grad_weights: wp.array(dtype=wp.float32),
        grad_response: wp.array(dtype=wp.float32),
        status: wp.array(dtype=wp.int32),
    ):
        pixel = wp.tid()
        path_gradient = gradient_vector()
        path_compensation = gradient_vector()
        reference_depth = wp.float64(log_mean[2 * pixel])
        log_scaled_mean = wp.float64(log_mean[2 * pixel + 1])
        for energy in range(energies):
            wi = energy_pixel_index(energy, pixel, pixels)
            ri = wi
            if wp.static(shared_weights):
                wi = energy
            if wp.static(shared_response):
                ri = energy
            depth = optical_depth(paths, coefficients, pixel, energy, pixels, materials, energies)
            attenuation = log_attenuation_state((depth - reference_depth) + log_scaled_mean)
            if not wp.isfinite(reference_depth) or not wp.isfinite(log_scaled_mean):
                wp.atomic_or(status, 0, 2)
            if wp.static(write_weights):
                grad_weights[wi] = checked_store(
                    term_at(
                        attenuation,
                        wp.float64(seed[pixel]),
                        wp.float64(response[ri]),
                        wp.float64(1.0),
                        wp.float64(1.0),
                    ),
                    status,
                )
            if wp.static(write_response):
                grad_response[ri] = checked_store(
                    term_at(
                        attenuation,
                        wp.float64(seed[pixel]),
                        wp.float64(weights[wi]),
                        wp.float64(1.0),
                        wp.float64(1.0),
                    ),
                    status,
                )
            if wp.static(write_paths):
                for material in range(materials):
                    term = term_at(
                        attenuation,
                        -wp.float64(seed[pixel]),
                        wp.float64(weights[wi]),
                        wp.float64(response[ri]),
                        wp.float64(coefficients[material_major_index(material, energy, energies)]),
                    )
                    updated, next_compensation = compensated_sum_update(
                        path_gradient[material], path_compensation[material], term
                    )
                    path_compensation[material] = next_compensation
                    path_gradient[material] = updated
        if wp.static(write_paths):
            for material in range(materials):
                grad_paths[material_major_index(material, pixel, pixels)] = store_signal(
                    path_gradient[material], status
                )

    return vjp


@cache
def get_log_nuisance_partials(
    materials: int,
    energies: int,
    shared_weights: bool,
    shared_response: bool,
    write_weights: bool,
    write_response: bool,
    write_coefficients: bool,
    wide: bool = False,
    arithmetic32: bool = False,
):
    """Fuse requested shared products without factoring their range-sensitive sums."""
    dtype = wp.float64 if wide else wp.float32
    optical_depth = get_optical_depth(wide, arithmetic32)
    term_at = log_normalized_product
    accumulator = wp.types.vector(length=materials + 2, dtype=wp.float64)
    response_offset = int(write_weights) * energies
    coefficient_offset = (int(write_weights) + int(write_response)) * energies

    @wp.kernel(module="unique", module_options=STRICT)
    def nuisance_partials(
        paths: wp.array(dtype=dtype),
        coefficients: wp.array(dtype=wp.float32),
        weights: wp.array(dtype=wp.float32),
        response: wp.array(dtype=wp.float32),
        seed: wp.array(dtype=dtype),
        log_mean: wp.array(dtype=wp.float64),
        pixels: int,
        groups: int,
        output: wp.array(dtype=wp.float64),
        status: wp.array(dtype=wp.int32),
    ):
        group, energy, lane = wp.tid()
        totals = accumulator()
        compensations = accumulator()
        pixel = group * wp.block_dim() + lane
        while pixel < pixels:
            wi = energy_pixel_index(energy, pixel, pixels)
            ri = wi
            if wp.static(shared_weights):
                wi = energy
            if wp.static(shared_response):
                ri = energy
            depth = optical_depth(paths, coefficients, pixel, energy, pixels, materials, energies)
            reference_depth = wp.float64(log_mean[2 * pixel])
            log_scaled_mean = wp.float64(log_mean[2 * pixel + 1])
            attenuation = log_attenuation_state((depth - reference_depth) + log_scaled_mean)
            if not wp.isfinite(reference_depth) or not wp.isfinite(log_scaled_mean):
                wp.atomic_or(status, 0, 2)
            if wp.static(write_weights):
                term = term_at(
                    attenuation,
                    wp.float64(seed[pixel]),
                    wp.float64(response[ri]),
                    wp.float64(1.0),
                    wp.float64(1.0),
                )
                updated, next_compensation = compensated_sum_update(
                    totals[0], compensations[0], term
                )
                compensations[0] = next_compensation
                totals[0] = updated
            if wp.static(write_response):
                term = term_at(
                    attenuation,
                    wp.float64(seed[pixel]),
                    wp.float64(weights[wi]),
                    wp.float64(1.0),
                    wp.float64(1.0),
                )
                updated, next_compensation = compensated_sum_update(
                    totals[1], compensations[1], term
                )
                compensations[1] = next_compensation
                totals[1] = updated
            if wp.static(write_coefficients):
                for material in range(materials):
                    index = material + 2
                    term = term_at(
                        attenuation,
                        -wp.float64(seed[pixel]),
                        wp.float64(weights[wi]),
                        wp.float64(response[ri]),
                        wp.float64(paths[material_major_index(material, pixel, pixels)]),
                    )
                    updated, next_compensation = compensated_sum_update(
                        totals[index], compensations[index], term
                    )
                    compensations[index] = next_compensation
                    totals[index] = updated
            if pixels - pixel <= groups * wp.block_dim():
                break
            pixel = pixel + groups * wp.block_dim()
        if wp.static(write_weights):
            result = wp.tile_sum(wp.tile(totals[0]))
            wp.tile_store(output, result, offset=energy * groups + group)
        if wp.static(write_response):
            result = wp.tile_sum(wp.tile(totals[1]))
            wp.tile_store(output, result, offset=(response_offset + energy) * groups + group)
        if wp.static(write_coefficients):
            for material in range(materials):
                result = wp.tile_sum(wp.tile(totals[material + 2]))
                wp.tile_store(
                    output,
                    result,
                    offset=(coefficient_offset + material_major_index(material, energy, energies))
                    * groups
                    + group,
                )

    return nuisance_partials

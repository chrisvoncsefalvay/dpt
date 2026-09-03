"""Strict mixed-precision transmission kernels, with no launch-time allocation.

The host boundary owns domain, shape, stream and alias validation. Every launch
must use ``record_tape=False``: the public operator supplies its explicit VJP.
Mask bits select transmission (1), counts (2), log transmission (4), removed
primary fraction (8). Beam modes are absent (0), fixed scalar (1), device scalar
(2) and per-pixel (3). Unused array arguments may be empty device placeholders.

Status is an int32 array with at least one element: bit 0 is invalid input,
bit 1 is a non-finite gradient. The caller resets it before checked execution.
Only exceptional lanes update status; numerical reductions never use atomics.
"""

from functools import cache

import warp as wp

TILE_SIZE = 256
STRICT_OPTIONS = {"fast_math": False, "fuse_fp": True, "enable_backward": False}


@wp.func_native("""
#if defined(__CUDACC__)
return -expm1(-value);
#else
// Warp's freestanding CPU CRT omits expm1/log1p declarations. Its Clang
// builtins resolve the same platform double functions without a CRT edit.
return -__builtin_expm1(-value);
#endif
""")
def _removed_primary_native(value: wp.float64) -> wp.float64: ...


@wp.func_native("""
#if defined(__CUDACC__)
return -log1p(-value);
#else
return -__builtin_log1p(-value);
#endif
""")
def _optical_depth_native(value: wp.float64) -> wp.float64: ...


@wp.func
def removed_primary(optical_depth: wp.float32) -> wp.float32:
    """Cancellation-safe decrement with one binary64 expm1 in the weak range."""
    x = wp.float64(optical_depth)
    if x == wp.float64(0.0):
        return wp.float32(0.0)
    if x > wp.float64(0.0) and x <= wp.float64(2.98023223876953125e-8):
        # |(1-exp(-x))-x| <= x*x/2 < half a binary32 spacing, including
        # the smaller spacing immediately below powers of two. Returning the
        # original input also preserves every positive binary32 subnormal.
        return optical_depth
    if x <= wp.float64(0.5):
        return wp.float32(_removed_primary_native(x))
    return wp.float32(wp.float64(1.0) - wp.exp(-x))


@wp.func
def optical_depth_from_decrement(decrement: wp.float32) -> wp.float32:
    """Cancellation-safe binary64 log1p, retaining the original far branch."""
    x = wp.float64(decrement)
    if x == wp.float64(0.0):
        return wp.float32(0.0)
    if x > wp.float64(0.0) and x <= wp.float64(2.98023223876953125e-8):
        # -log(1-x)-x <= x*x/(2*(1-x)); this is below half a binary32
        # spacing throughout this interval, even for subnormal inputs.
        return decrement
    if x <= wp.float64(0.5):
        return wp.float32(_optical_depth_native(x))
    return wp.float32(-wp.log(wp.float64(1.0) - x))


@wp.func_native("return -value;")
def log_transmission_exact(value: wp.float32) -> wp.float32:
    """Use IEEE unary negation; Warp 1.17's generic neg is ``0 - value``."""
    ...


@wp.func
def transmission_factor(optical_depth: wp.float32) -> wp.float64:
    """Keep exponent range until all requested physical products are formed."""
    if optical_depth <= wp.float32(64.0):
        # Here exp(-L) is normal in FP32. Promotion retains its range, while
        # keeping the common case on the single-precision exponential path.
        return wp.float64(wp.exp(-optical_depth))
    return wp.exp(-wp.float64(optical_depth))


@cache
def get_forward_kernel(mask: int, beam_mode: int):
    """Return a specialised pointwise kernel, launched with ``dim=P``.

    Inputs: L, beam_array, beam_scalar. Outputs: T, counts, log_T, removed.
    Static selection eliminates unused array reads, stores and mathematics.
    """
    if mask < 1 or mask > 15 or beam_mode not in (0, 1, 2, 3):
        raise ValueError("Invalid output mask or beam mode")
    if mask & 2 and beam_mode == 0:
        raise ValueError("Counts require an open beam")
    if not mask & 2:
        # Warp hashes captured static beam comparisons even when the entire
        # count branch is eliminated. Canonicalise this unused specialization
        # axis so each count-free mask shares one compiled module.
        beam_mode = 0

    @wp.kernel(module="unique", module_options=STRICT_OPTIONS)
    def forward(
        optical_depth: wp.array(dtype=wp.float32),
        beam: wp.array(dtype=wp.float32),
        beam_scalar: wp.float32,
        transmission: wp.array(dtype=wp.float32),
        counts: wp.array(dtype=wp.float32),
        log_transmission: wp.array(dtype=wp.float32),
        removed: wp.array(dtype=wp.float32),
    ):
        p = wp.tid()
        depth = optical_depth[p]
        if wp.static(mask & 3 != 0):
            factor = transmission_factor(depth)
            if wp.static(mask & 1 != 0):
                transmission[p] = wp.float32(factor)
            if wp.static(mask & 2 != 0):
                illumination = beam_scalar
                if wp.static(beam_mode == 2):
                    illumination = beam[0]
                elif wp.static(beam_mode == 3):
                    illumination = beam[p]
                # In particular, never multiply illumination by stored T.
                count = wp.float32(wp.float64(illumination) * factor)
                if count == wp.float32(0.0):
                    count = wp.float32(0.0)
                counts[p] = count
        if wp.static(mask & 4 != 0):
            log_transmission[p] = log_transmission_exact(depth)
        if wp.static(mask & 8 != 0):
            decrement = removed_primary(depth)
            if decrement == wp.float32(0.0):
                decrement = wp.float32(0.0)
            removed[p] = decrement

    return forward


@wp.func
def checked_gradient(value: wp.float64, status: wp.array(dtype=wp.int32)) -> wp.float32:
    """Round once and flag overflow, including after tape accumulation."""
    rounded = wp.float32(value)
    if not wp.isfinite(rounded):
        wp.atomic_or(status, 0, 2)
    return rounded


@wp.func
def weighted_depth_adjoint(
    factor: wp.float64,
    illumination: wp.float32,
    seed_transmission: wp.float32,
    seed_counts: wp.float32,
    seed_log: wp.float32,
    seed_removed: wp.float32,
) -> wp.float64:
    """Differentiate the real-valued map, retaining range before rounding."""
    optical = (wp.float64(seed_removed) - wp.float64(seed_transmission)) * factor
    count = wp.float64(seed_counts) * wp.float64(illumination) * factor
    return optical - count - wp.float64(seed_log)


@cache
def get_vjp_kernel(
    seed_mask: int, beam_mode: int, write_depth: bool, write_beam: bool, accumulate: bool
):
    """Pointwise VJP: inputs L, beam, scalar, seed_T/counts/log/removed;
    outputs grad_L, grad_beam, status. Launch ``dim=P``.

    ``write_beam`` is only for per-pixel beams. Broadcast beams use the tiled
    reduction below. ``accumulate`` is reserved for the tape adapter.
    """
    if seed_mask < 0 or seed_mask > 15 or beam_mode not in (0, 1, 2, 3):
        raise ValueError("Invalid seed mask or beam mode")
    if seed_mask & 2 and beam_mode == 0:
        raise ValueError("Count cotangents require an open beam")
    if write_beam and beam_mode != 3:
        raise ValueError("Pointwise beam gradients require a per-pixel beam")
    if not seed_mask & 2:
        beam_mode = 0

    @wp.kernel(module="unique", module_options=STRICT_OPTIONS)
    def vjp(
        optical_depth: wp.array(dtype=wp.float32),
        beam: wp.array(dtype=wp.float32),
        beam_scalar: wp.float32,
        seed_transmission: wp.array(dtype=wp.float32),
        seed_counts: wp.array(dtype=wp.float32),
        seed_log: wp.array(dtype=wp.float32),
        seed_removed: wp.array(dtype=wp.float32),
        grad_depth: wp.array(dtype=wp.float32),
        grad_beam: wp.array(dtype=wp.float32),
        status: wp.array(dtype=wp.int32),
    ):
        p = wp.tid()
        factor = wp.float64(0.0)
        if wp.static(seed_mask & 11 != 0):
            factor = transmission_factor(optical_depth[p])
        a = wp.float32(0.0)
        b = wp.float32(0.0)
        c = wp.float32(0.0)
        r = wp.float32(0.0)
        if wp.static(seed_mask & 1 != 0):
            a = seed_transmission[p]
        if wp.static(seed_mask & 2 != 0):
            b = seed_counts[p]
        if wp.static(seed_mask & 4 != 0):
            c = seed_log[p]
        if wp.static(seed_mask & 8 != 0):
            r = seed_removed[p]
        if wp.static(write_depth):
            illumination = wp.float32(0.0)
            if wp.static(seed_mask & 2 != 0):
                illumination = beam_scalar
                if wp.static(beam_mode == 2):
                    illumination = beam[0]
                elif wp.static(beam_mode == 3):
                    illumination = beam[p]
            derivative = weighted_depth_adjoint(factor, illumination, a, b, c, r)
            if wp.static(accumulate):
                derivative = derivative + wp.float64(grad_depth[p])
            grad_depth[p] = checked_gradient(derivative, status)
        if wp.static(write_beam):
            derivative_beam = wp.float64(b) * factor
            if wp.static(accumulate):
                derivative_beam = derivative_beam + wp.float64(grad_beam[p])
            grad_beam[p] = checked_gradient(derivative_beam, status)

    return vjp


@wp.func
def beam_contribution(depth: wp.float32, seed: wp.float32) -> wp.float64:
    return wp.float64(seed) * transmission_factor(depth)


@cache
def get_beam_partial_kernel():
    """Inputs L, seed_counts; output FP64 partials.

    Use ``launch_tiled(dim=ceil(P/256), block_dim=256)``. Tile loads zero-pad
    their bounds, so padded seeds contribute zero even though exp(-0) is one.
    The separate pass deliberately avoids retaining an 8P-byte contribution
    image or contending for a global scalar. Its traffic is accounted separately.
    """

    @wp.kernel(module="unique", module_options=STRICT_OPTIONS)
    def partials(
        optical_depth: wp.array(dtype=wp.float32),
        seed_counts: wp.array(dtype=wp.float32),
        output: wp.array(dtype=wp.float64),
    ):
        block = wp.tid()
        depths = wp.tile_load(optical_depth, shape=TILE_SIZE, offset=block * TILE_SIZE)
        seeds = wp.tile_load(seed_counts, shape=TILE_SIZE, offset=block * TILE_SIZE)
        contributions = wp.tile_map(beam_contribution, depths, seeds)
        total = wp.tile_sum(contributions)
        wp.tile_store(output, total, offset=block)

    return partials


@cache
def get_inverse_kernel():
    """Inputs decrement; outputs optical depth; launch ``dim=P``."""

    @wp.kernel(module="unique", module_options=STRICT_OPTIONS)
    def inverse(delta: wp.array(dtype=wp.float32), depth: wp.array(dtype=wp.float32)):
        p = wp.tid()
        value = optical_depth_from_decrement(delta[p])
        if value == wp.float32(0.0):
            value = wp.float32(0.0)
        depth[p] = value

    return inverse


@cache
def get_inverse_vjp_kernel(accumulate: bool):
    """Inputs delta, seed_L; outputs grad_delta, status; launch ``dim=P``."""

    @wp.kernel(module="unique", module_options=STRICT_OPTIONS)
    def inverse_vjp(
        delta: wp.array(dtype=wp.float32),
        seed: wp.array(dtype=wp.float32),
        gradient: wp.array(dtype=wp.float32),
        status: wp.array(dtype=wp.int32),
    ):
        p = wp.tid()
        derivative = wp.float64(seed[p]) / (wp.float64(1.0) - wp.float64(delta[p]))
        if wp.static(accumulate):
            derivative = derivative + wp.float64(gradient[p])
        gradient[p] = checked_gradient(derivative, status)

    return inverse_vjp


@wp.kernel(module="unique", module_options=STRICT_OPTIONS)
def finite_nonnegative(values: wp.array(dtype=wp.float32), status: wp.array(dtype=wp.int32)):
    """Inputs values; outputs status; launch dim=values.shape[0]."""
    value = values[wp.tid()]
    if not wp.isfinite(value) or value < wp.float32(0.0):
        wp.atomic_or(status, 0, 1)


@wp.kernel(module="unique", module_options=STRICT_OPTIONS)
def finite_seed(values: wp.array(dtype=wp.float32), status: wp.array(dtype=wp.int32)):
    """Inputs cotangents; outputs status; launch dim=values.shape[0]."""
    if not wp.isfinite(values[wp.tid()]):
        wp.atomic_or(status, 0, 1)


@wp.kernel(module="unique", module_options=STRICT_OPTIONS)
def decrement_domain(values: wp.array(dtype=wp.float32), status: wp.array(dtype=wp.int32)):
    """Inputs decrements; outputs status; launch dim=values.shape[0]."""
    value = values[wp.tid()]
    if not wp.isfinite(value) or value < wp.float32(0.0) or value >= wp.float32(1.0):
        wp.atomic_or(status, 0, 1)


@wp.kernel(module="unique", module_options={"enable_backward": True})
def dependency_marker(
    depth: wp.array(dtype=wp.float32),
    beam: wp.array(dtype=wp.float32),
    transmission: wp.array(dtype=wp.float32),
    counts: wp.array(dtype=wp.float32),
    log_transmission: wp.array(dtype=wp.float32),
    removed: wp.array(dtype=wp.float32),
):
    """Zero-dimensional tape bookkeeping only; never executes a CUDA thread."""
    pass

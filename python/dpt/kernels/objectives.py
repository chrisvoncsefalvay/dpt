"""Fused objective/seed evaluation with a fixed FP64 reduction tree.

The image-space derivative is caller-owned. Only one scalar partial per tile is
retained: no per-pixel loss image or global floating-point atomic is needed.
"""

# Warp annotations are executable DSL expressions; host interfaces remain strict.
# The optional GPU import is resolved only when an operator is prepared.
# pyright: reportInvalidTypeForm=false, reportUnknownParameterType=false
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false, reportUntypedFunctionDecorator=false
# pyright: reportMissingImports=false, reportUntypedClassDecorator=false

from functools import cache

import warp as wp

from dpt.kernels.numerics import attenuated_product4, product3, product4

OPTIONS = {"fast_math": False, "fuse_fp": True, "enable_backward": False}


@cache
def observation_validation_kernel(
    poisson: bool, weighted: bool, masked: bool = False, integer_counts: bool = True
):
    @wp.kernel(module="unique", module_options=OPTIONS)
    def validate(
        observation: wp.array(dtype=wp.float32),
        weights: wp.array(dtype=wp.float32),
        valid: wp.array(dtype=wp.uint8),
        status: wp.array(dtype=wp.int32),
    ):
        p = wp.tid()
        active = bool(True)  # noqa: UP018 - Warp runtime Boolean.
        if wp.static(masked):
            if valid[p] > wp.uint8(1):
                wp.atomic_or(status, 0, 1)
            active = valid[p] == wp.uint8(1)
        if active:
            value = observation[p]
            invalid = not wp.isfinite(value)
            if wp.static(poisson):
                invalid = invalid or value < 0.0
                if wp.static(integer_counts):
                    invalid = invalid or wp.floor(value) != value
            if wp.static(weighted):
                invalid = invalid or not wp.isfinite(weights[p]) or weights[p] < 0.0
            if invalid:
                wp.atomic_or(status, 0, 1)

    return validate


@cache
def validation_kernel(
    poisson: bool,
    weighted: bool,
    masked: bool = False,
    precision: str = "float32",
    integer_counts: bool = True,
):
    value_dtype = wp.float64 if precision == "float64" else wp.float32

    @wp.kernel(module="unique", module_options=OPTIONS)
    def validate(
        prediction: wp.array(dtype=value_dtype),
        observation: wp.array(dtype=wp.float32),
        weights: wp.array(dtype=wp.float32),
        valid: wp.array(dtype=wp.uint8),
        status: wp.array(dtype=wp.int32),
    ):
        p = wp.tid()
        active = bool(True)  # noqa: UP018 - Warp runtime Boolean.
        if wp.static(masked):
            if valid[p] > wp.uint8(1):
                wp.atomic_or(status, 0, 1)
            active = valid[p] == wp.uint8(1)
        if active:
            a, b = prediction[p], observation[p]
            invalid = not wp.isfinite(b)
            domain = bool(False)  # noqa: UP018 - Warp runtime Boolean.
            if not wp.isfinite(a):
                wp.atomic_or(status, 0, 2)
            if wp.static(poisson):
                invalid = invalid or b < 0.0
                if wp.static(integer_counts):
                    invalid = invalid or wp.floor(b) != b
                domain = wp.isfinite(a) and (a < 0.0 or (a == 0.0 and b > 0.0))
            if domain:
                wp.atomic_or(status, 0, 8)
            if wp.static(weighted):
                invalid = invalid or not wp.isfinite(weights[p]) or weights[p] < 0.0
            if invalid:
                wp.atomic_or(status, 0, 1)

    return validate


@wp.func
def poisson_half_deviance(predicted: wp.float64, observed: wp.float64) -> wp.float64:
    """Canonical cancellation-safe Poisson term shared by both compositions."""
    if observed == wp.float64(0.0):
        return predicted
    difference = predicted - observed
    loss = wp.float64(0.0)
    # Half-deviance differs from the negative log likelihood
    # only by observation constants. This form retains a small
    # residual near equality without evaluating log(0) for y=0.
    if wp.abs(difference) <= wp.float64(0.25) * observed:
        ratio = difference / observed
        # log1p(r) = 2*atanh(s), s=r/(2+r). Removing its linear
        # term analytically avoids the r-log1p(r) cancellation throughout
        # the near-equality interval, including the former |r|=1e-3 seam.
        s = ratio / (wp.float64(2.0) + ratio)
        square = s * s
        series = wp.float64(1.0) / wp.float64(23.0)
        for term in range(10):
            series = wp.float64(1.0) / wp.float64(21 - 2 * term) + square * series
        loss = observed * (ratio * s - wp.float64(2.0) * s * square * series)
    else:
        quotient = observed / predicted
        if quotient > wp.float64(0.0) and wp.isfinite(quotient):
            # A single logarithm avoids subtracting two large, almost equal
            # logarithms around the near/far boundary. Extreme ratios retain
            # the separate-log form so neither quotient range is required.
            loss = difference + observed * wp.log(quotient)
        else:
            loss = difference + observed * (wp.log(observed) - wp.log(predicted))
    return loss


@cache
def evaluation_kernel(
    poisson: bool,
    weighted: bool,
    gradient: bool,
    masked: bool = False,
    precision: str = "float32",
):
    value_dtype = wp.float64 if precision == "float64" else wp.float32

    @wp.kernel(module="unique", module_options=OPTIONS)
    def evaluate(
        prediction: wp.array(dtype=value_dtype),
        observation: wp.array(dtype=wp.float32),
        weights: wp.array(dtype=wp.float32),
        valid: wp.array(dtype=wp.uint8),
        size: int,
        normalisation: wp.float64,
        seed: wp.array(dtype=value_dtype),
        partials: wp.array(dtype=wp.float64),
        status: wp.array(dtype=wp.int32),
    ):
        block, lane = wp.tid()
        p = block * wp.block_dim() + lane
        loss = wp.float64(0.0)
        if p < size:
            active = bool(True)  # noqa: UP018 - Warp runtime Boolean.
            if wp.static(masked):
                if valid[p] > wp.uint8(1):
                    wp.atomic_or(status, 0, 1)
                active = valid[p] == wp.uint8(1)
            if active:
                predicted = wp.float64(prediction[p])
                if not wp.isfinite(predicted):
                    wp.atomic_or(status, 0, 2)
                elif wp.static(poisson) and predicted < wp.float64(0.0):
                    wp.atomic_or(status, 0, 8)
                observed = wp.float64(observation[p])
                weight = normalisation
                if wp.static(weighted):
                    weight *= wp.float64(weights[p])
                difference = predicted - observed
                derivative = difference
                loss = wp.float64(0.5) * difference * difference
                if wp.static(poisson):
                    if observed == wp.float64(0.0):
                        loss = predicted
                        derivative = wp.float64(1.0)
                    else:
                        if wp.isfinite(predicted) and predicted <= wp.float64(0.0):
                            wp.atomic_or(status, 0, 8)
                        derivative = difference / predicted
                        loss = poisson_half_deviance(predicted, observed)
                if wp.static(precision == "float64" and not poisson):
                    # A small fixed weight can keep the final squared loss
                    # representable even when the unweighted square overflows.
                    loss = (wp.float64(0.5) * weight * difference) * difference
                else:
                    loss *= weight
                if not wp.isfinite(loss):
                    wp.atomic_or(status, 0, 2)
                if wp.static(gradient):
                    weighted_derivative = weight * derivative
                    if wp.static(precision == "float64" and poisson):
                        if not wp.isfinite(derivative):
                            # FP32 observations/weights bound this numerator;
                            # do not overflow a removable unweighted quotient.
                            weighted_derivative = (weight * difference) / predicted
                    result = value_dtype(weighted_derivative)
                    seed[p] = result
                    if not wp.isfinite(result):
                        wp.atomic_or(status, 0, 2)
            elif wp.static(gradient):
                seed[p] = value_dtype(0.0)
        total = wp.tile_sum(wp.tile(loss))
        wp.tile_store(partials, total, offset=block)

    return evaluate


@cache
def primary_evaluation_kernel(
    poisson: bool, counts: bool, weighted: bool, masked: bool, integer_counts: bool = True
):
    """Retain the primary signal and depth cotangent in FP64 for pose fitting."""

    @wp.kernel(module="unique", module_options=OPTIONS)
    def evaluate(
        depth: wp.array(dtype=wp.float64),
        beam: wp.float64,
        observation: wp.array(dtype=wp.float32),
        weights: wp.array(dtype=wp.float32),
        valid: wp.array(dtype=wp.uint8),
        size: int,
        normalisation: wp.float64,
        prediction: wp.array(dtype=wp.float64),
        depth_seed: wp.array(dtype=wp.float64),
        partials: wp.array(dtype=wp.float64),
        status: wp.array(dtype=wp.int32),
    ):
        block, lane = wp.tid()
        p = block * wp.block_dim() + lane
        loss = wp.float64(0.0)
        if p < size:
            optical = depth[p]
            predicted = -optical
            if wp.static(counts):
                predicted = beam * wp.exp(-optical)
                if optical > wp.float64(700.0) and beam > wp.float64(0.0):
                    # Range reduction keeps CUDA exp away from the subnormal
                    # boundary and rounds the original beam factor only once.
                    # Keep the direct product in the ordinary near-equality range.
                    predicted = attenuated_product4(
                        optical, beam, wp.float64(1.0), wp.float64(1.0), wp.float64(1.0)
                    )
            prediction[p] = predicted
            if not wp.isfinite(optical):
                wp.atomic_or(status, 0, 2)
            elif optical < wp.float64(0.0):
                wp.atomic_or(status, 0, 8)
            if not wp.isfinite(predicted):
                wp.atomic_or(status, 0, 2)
            active = bool(True)  # noqa: UP018 - Warp runtime Boolean.
            if wp.static(masked):
                if valid[p] > wp.uint8(1):
                    wp.atomic_or(status, 0, 1)
                active = valid[p] == wp.uint8(1)
            derivative = wp.float64(0.0)
            if active:
                observed = wp.float64(observation[p])
                invalid = not wp.isfinite(observed)
                if wp.static(poisson):
                    invalid = invalid or observed < wp.float64(0.0)
                    if wp.static(integer_counts):
                        invalid = invalid or wp.floor(observed) != observed
                if invalid:
                    wp.atomic_or(status, 0, 1)
                difference = predicted - observed
                loss = wp.float64(0.5) * difference * difference
                derivative = -difference
                if wp.static(counts):
                    derivative *= predicted
                if wp.static(poisson):
                    if (
                        wp.isfinite(optical)
                        and predicted == wp.float64(0.0)
                        and observed > wp.float64(0.0)
                    ):
                        wp.atomic_or(status, 0, 8)
                    loss = poisson_half_deviance(predicted, observed)
                    if (
                        observed > wp.float64(0.0)
                        and predicted < wp.float64(2.2250738585072014e-308)
                        and wp.abs(difference) > wp.float64(0.25) * observed
                    ):
                        # Avoid taking log of a rounded subnormal count.
                        loss = difference + observed * (wp.log(observed) - wp.log(beam) + optical)
                    # Compose (1-N/lambda)*(-lambda) before rounding/division.
                    derivative = -difference
                weight = normalisation
                if wp.static(weighted):
                    weight *= wp.float64(weights[p])
                loss *= weight
                derivative *= weight
                if wp.static(not poisson):
                    # Fixed weighting can rescue a finite squared loss or
                    # depth derivative after an unweighted product overflows.
                    if not wp.isfinite(loss):
                        loss = product4(wp.float64(0.5), weight, difference, difference)
                    if not wp.isfinite(derivative):
                        derivative = -weight * difference
                        if wp.static(counts):
                            derivative = product3(-weight, difference, predicted)
            depth_seed[p] = derivative
            if not wp.isfinite(loss) or not wp.isfinite(derivative):
                wp.atomic_or(status, 0, 2)
        wp.tile_store(partials, wp.tile_sum(wp.tile(loss)), offset=block)

    return evaluate

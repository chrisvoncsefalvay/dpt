"""Reserved spectrum-domain selection and likelihood-score readout."""

import warp as wp

from ._constants import INPUT_MASK, NUMERICAL_FAILURE


@wp.func
def _spectrum_index(cdf: wp.array(dtype=wp.float64), draw: wp.float64) -> int:
    low = int(0)  # noqa: UP018, RUF046 - Warp requires a dynamic loop variable.
    high = cdf.shape[0] - 1
    while low < high:
        mid = (low + high) // 2
        if draw < cdf[mid]:
            high = mid
        else:
            low = mid + 1
    return low


@wp.kernel
def spectrum_logit_derivative(
    pixel: wp.array(dtype=wp.int32),
    score: wp.array(dtype=wp.float64),
    selected: wp.array(dtype=wp.int32),
    bins: int,
    active: int,
    probability: wp.float64,
    out: wp.array(dtype=wp.float64),
    status: wp.array(dtype=wp.int32),
):
    h = wp.tid()
    out[h] = wp.float64(0.0)
    if selected[h] < 0 or selected[h] >= bins or not wp.isfinite(score[h]):
        wp.atomic_or(status, 0, INPUT_MASK)
    elif pixel[h] >= 0:
        out[h] = score[h] * (wp.float64(int(selected[h] == active)) - probability)
        if not wp.isfinite(out[h]):
            wp.atomic_or(status, 0, 1 << NUMERICAL_FAILURE)

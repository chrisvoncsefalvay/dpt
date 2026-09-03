"""Preserve sampler bits and history masks while consolidating kernel state."""

# Warp decorators and executable array types are outside host type checking.
# pyright: reportInvalidTypeForm=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false
# pyright: reportUnknownParameterType=false, reportUntypedFunctionDecorator=false
# Warp's vec4d indexing is resolved by its kernel compiler.
# pyright: reportIndexIssue=false
import importlib
from typing import Any

import numpy as np
import pytest

from dpt.kernels.random import random4
from dpt.transport.model import HistoryStatus

wp: Any = importlib.import_module("warp")
kernels: Any = importlib.import_module("dpt.transport.kernels")
batched: Any = importlib.import_module("dpt.transport.batched_kernels")

wp.set_module_options({"fast_math": False, "fuse_fp": False, "enable_backward": False})


@wp.func
def _legacy_compton(
    energy: wp.float64,
    seed: wp.uint64,
    history: wp.uint64,
    event: wp.uint32,
    max_trials: int,
) -> wp.vec4d:
    # Frozen pre-consolidation arithmetic and random-word mapping. The existing
    # independent KN quadrature tests establish the law; this guards identity.
    result = wp.vec4d(wp.float64(0.0), wp.float64(0.0), energy, wp.float64(6.0))
    for trial in range(max_trials):
        angular = random4(seed, history, event, wp.uint32(2147483648) + wp.uint32(trial))
        cosine = wp.float64(2.0) * angular[0] - wp.float64(1.0)
        ratio = wp.float64(1.0) / (
            wp.float64(1.0) + energy / wp.float64(510.99895069) * (wp.float64(1.0) - cosine)
        )
        acceptance = wp.float64(0.5) * (
            ratio * ratio * ratio + ratio - ratio * ratio * (wp.float64(1.0) - cosine * cosine)
        )
        if angular[1] < acceptance:
            result = wp.vec4d(
                cosine,
                wp.float64(6.2831853071795864769) * angular[2],
                energy * ratio,
                wp.float64(0.0),
            )
            break
    return result


@wp.kernel
def _compare_compton(
    energy: wp.float64,
    max_trials: int,
    reference: wp.array(dtype=wp.vec4d),
    packed: wp.array(dtype=wp.vec4d),
    typed: wp.array(dtype=wp.vec4d),
    typed_status: wp.array(dtype=wp.int32),
):
    index = wp.tid()
    seed = wp.uint64(0xFEDCBA9876543210)
    history = wp.uint64(1099511627776) + wp.uint64(index)
    event = wp.uint32(17 + index % 8)
    reference[index] = _legacy_compton(energy, seed, history, event, max_trials)
    packed[index] = kernels.compton_scatter(energy, seed, history, event, max_trials)
    result = kernels._compton_event(energy, seed, history, event, max_trials)
    typed[index] = wp.vec4d(result.cosine, result.azimuth, result.energy, wp.float64(result.status))
    typed_status[index] = result.status


@pytest.mark.parametrize("energy", [1.0, 80.0, 1000.0, 1e6])
@pytest.mark.parametrize("max_trials", [0, 1, 32])
def test_typed_compton_preserves_random_words_and_packed_output(
    energy: float, max_trials: int, warp_device: str
) -> None:
    count = 129
    arrays = [wp.empty(count, dtype=wp.vec4d, device=warp_device) for _ in range(3)]
    status = wp.empty(count, dtype=wp.int32, device=warp_device)
    wp.launch(
        _compare_compton,
        dim=count,
        inputs=[energy, max_trials, *arrays, status],
        device=warp_device,
    )
    expected, packed, typed = [array.numpy() for array in arrays]
    np.testing.assert_array_equal(expected.view(np.uint64), packed.view(np.uint64))
    np.testing.assert_array_equal(expected.view(np.uint64), typed.view(np.uint64))
    np.testing.assert_array_equal(status.numpy(), expected[:, 3].astype(np.int32))
    if max_trials == 0:
        assert np.all(status.numpy() == int(HistoryStatus.ANGLE_BUDGET))


_HISTORY_CASES = (
    [
        # Invalid terminal values, ordinary escape/absorption and every failure code.
        (0, 1.0, terminal)
        for terminal in range(-1, 8)
    ]
    + [(-1, 0.0, terminal) for terminal in range(7)]
    + [
        (-2, 0.0, 0),
        (2, 0.0, 0),
        (-1, 1.0, 0),
        (0, 0.0, 1),
        (0, -3.0, 0),
        (0, float("nan"), 0),
        (0, float("inf"), 0),
        (0, -float("inf"), 0),
        # A terminal failure owns the diagnostic even if its other outputs are bad.
        (-2, float("nan"), 2),
        (2, float("inf"), 6),
    ]
)


@pytest.mark.parametrize(("target", "value", "terminal"), _HISTORY_CASES)
def test_scalar_and_batched_tally_keep_terminal_masks_and_participation(
    target: int, value: float, terminal: int, warp_device: str
) -> None:
    if 2 <= terminal <= 6:
        expected_mask = 1 << terminal
        expected_scale = 0.0
    elif (
        target < -1
        or target >= 2
        or not np.isfinite(value)
        or terminal < 0
        or terminal > 6
        or (target == -1 and value != 0.0)
        or (terminal == 1 and target != -1)
    ):
        expected_mask = 1
        expected_scale = 0.0
    else:
        expected_mask = 0
        expected_scale = abs(value) if target >= 0 else 0.0
    pixel = wp.array([target], dtype=wp.int32, device=warp_device)
    score = wp.array([value], dtype=wp.float64, device=warp_device)
    history_status = wp.array([terminal], dtype=wp.int32, device=warp_device)
    for kernel, extra in (
        (kernels.find_tally_scale, []),
        (kernels.find_tally_scale_plain, []),
        (batched.find_tally_scale, [1]),
    ):
        scale = wp.zeros(2, dtype=wp.float64, device=warp_device)
        status = wp.zeros(1, dtype=wp.int32, device=warp_device)
        wp.launch(
            kernel,
            dim=1,
            inputs=[pixel, score, history_status, 2, *extra, scale, status],
            device=warp_device,
        )
        assert int(status.numpy()[0]) == expected_mask
        assert float(scale.numpy().sum()) == expected_scale

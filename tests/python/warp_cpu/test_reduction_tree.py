"""Portable multi-row regressions for the production FP64 reduction tree."""

import importlib
import math
from contextlib import nullcontext
from typing import Any

import pytest

from dpt._reductions import prepare_reduction
from dpt._runtime import cpu_test_context, prepare_context
from dpt.objectives import (
    ObjectiveSpec,
    evaluate_objective,
    prepare_objective,
    reduce_objective_components,
)
from dpt.transmission import TransmissionSpec, prepare_transmission, transmission_vjp

np: Any = importlib.import_module("numpy")
wp: Any = importlib.import_module("warp")
approx: Any = pytest.approx  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]


@pytest.mark.parametrize("tile", [128, 256])
@pytest.mark.parametrize("components", [1, 6, 12])
def test_packed_tree_reduces_every_row_and_respects_reused_logical_count(
    warp_device: str, tile: int, components: int
) -> None:
    scope = cpu_test_context() if warp_device == "cpu" else nullcontext()
    with scope:
        ctx = prepare_context(device=warp_device)
        capacity = tile * tile + 1
        tree = prepare_reduction(ctx, capacity, tile=tile, components=components)
        values = [float((index % 17) - 8) for index in range(capacity * components)]
        source = wp.array(values, dtype=wp.float64, device=ctx.device)
        output = wp.empty(components, dtype=wp.float64, device=ctx.device)
        status = wp.zeros(1, dtype=wp.int32, device=ctx.device)
        for size in (capacity, tile + 1, tile, tile - 1, 2, 1, 0):
            tree.finish(size, output, status, components=components, source=source)
            expected = [
                math.fsum(values[c : size * components : components]) for c in range(components)
            ]
            assert output.numpy().tolist() == expected
            assert status.numpy().tolist() == [0]


@pytest.mark.parametrize("size", [2, 255, 256, 257, 65537])
def test_multi_pixel_objective_loss_and_seed_match_closed_form(warp_device: str, size: int) -> None:
    scope = cpu_test_context() if warp_device == "cpu" else nullcontext()
    with scope:
        workspace = prepare_objective(ObjectiveSpec(), max_pixels=size, device=warp_device)
        prediction = wp.array(
            np.arange(1, size + 1, dtype=np.float32), dtype=wp.float32, device=warp_device
        )
        observation = wp.zeros(size, dtype=wp.float32, device=warp_device)
        loss = wp.empty(1, dtype=wp.float64, device=warp_device)
        seed = wp.empty(size, dtype=wp.float32, device=warp_device)
        evaluate_objective(
            prediction, observation, out_loss=loss, out_cotangent=seed, workspace=workspace
        )
        assert float(loss.numpy()[0]) == size * (size + 1) * (2 * size + 1) / 12
        np.testing.assert_array_equal(seed.numpy(), prediction.numpy())
        assert workspace._status.numpy().tolist() == [0]


@pytest.mark.parametrize("size", [255, 256, 257, 65537])
def test_component_reduction_sums_all_rows(warp_device: str, size: int) -> None:
    scope = cpu_test_context() if warp_device == "cpu" else nullcontext()
    with scope:
        workspace = prepare_objective(ObjectiveSpec(), max_pixels=size, device=warp_device)
        values = [float((i % 19) - 8) for i in range(size)]
        components = wp.array(values, dtype=wp.float64, device=warp_device)
        loss = wp.empty(1, dtype=wp.float64, device=warp_device)
        reduce_objective_components(components, out_loss=loss, workspace=workspace)
        assert float(loss.numpy()[0]) == math.fsum(values)


@pytest.mark.parametrize("size", [255, 256, 257, 65537])
def test_active_device_scalar_beam_gradient_reduces_all_pixels(warp_device: str, size: int) -> None:
    scope = cpu_test_context() if warp_device == "cpu" else nullcontext()
    with scope:
        workspace = prepare_transmission(
            TransmissionSpec(beam="device-scalar", active_beam=True),
            max_pixels=size,
            device=warp_device,
        )
        depths = np.array([float(i % 5) / 8 for i in range(size)], dtype=np.float32)
        seeds = np.array([float((i % 9) - 3) for i in range(size)], dtype=np.float32)
        optical_depth = wp.array(depths, dtype=wp.float32, device=warp_device)
        beam = wp.ones(1, dtype=wp.float32, device=warp_device)
        seed = wp.array(seeds, dtype=wp.float32, device=warp_device)
        gradient = wp.full(1, 17.0, dtype=wp.float32, device=warp_device)
        transmission_vjp(
            optical_depth,
            beam,
            cotangent_counts=seed,
            out_grad_n0=gradient,
            workspace=workspace,
        )
        expected = np.float32(
            math.fsum(float(s) * math.exp(-float(x)) for s, x in zip(seeds, depths, strict=True))
        )
        assert float(gradient.numpy()[0]) == approx(float(expected), rel=2e-7)
        assert workspace._status.numpy().tolist() == [0]

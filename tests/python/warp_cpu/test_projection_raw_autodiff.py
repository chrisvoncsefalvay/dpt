"""An independent backward-enabled Warp projector checks the explicit midpoint VJP."""

import importlib

# Warp DSL annotations are evaluated at runtime.
# pyright: reportInvalidTypeForm=false, reportUnknownParameterType=false
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false, reportUntypedFunctionDecorator=false
import math
from collections.abc import Iterator
from contextlib import nullcontext
from typing import Any

import pytest

from dpt._runtime import cpu_test_context
from dpt.geometry import RayBundleGeometry, RigidTransform, compose_pose
from dpt.projection import ProjectionSpec, prepare_projection, project_optical_depth, projection_vjp
from dpt.volumes import GridSpec

wp: Any = importlib.import_module("warp")


@wp.kernel(module="unique", module_options={"enable_backward": True, "fast_math": False})
def _raw_midpoint(
    field: wp.array(dtype=wp.float32),
    pose: wp.array(dtype=wp.float64),
    source: wp.vec3d,
    endpoint: wp.vec3d,
    inverse_grid: wp.mat33d,
    grid_origin: wp.vec3d,
    shape: wp.vec3i,
    samples: int,
    output: wp.array(dtype=wp.float64),
):
    # No production projection functions participate in this raw tape graph.
    rotation = wp.mat33d(
        pose[0], pose[1], pose[2], pose[3], pose[4], pose[5], pose[6], pose[7], pose[8]
    )
    translation = wp.vec3d(pose[9], pose[10], pose[11])
    start = inverse_grid * (wp.transpose(rotation) * (source - translation) - grid_origin)
    direction = inverse_grid * (wp.transpose(rotation) * (endpoint - source))
    # Fixture directions have no zero components. This explicit nested min/max
    # expression supplies an independent differentiable intersection graph.
    first = wp.cw_div(wp.vec3d(-0.5) - start, direction)
    last = wp.cw_div(
        wp.vec3d(
            wp.float64(shape[0]) - wp.float64(0.5),
            wp.float64(shape[1]) - wp.float64(0.5),
            wp.float64(shape[2]) - wp.float64(0.5),
        )
        - start,
        direction,
    )
    lower = wp.max(
        wp.float64(0.0),
        wp.max(
            wp.min(first[0], last[0]), wp.max(wp.min(first[1], last[1]), wp.min(first[2], last[2]))
        ),
    )
    upper = wp.min(
        wp.float64(1.0),
        wp.min(
            wp.max(first[0], last[0]), wp.min(wp.max(first[1], last[1]), wp.max(first[2], last[2]))
        ),
    )
    step = (upper - lower) / wp.float64(samples)
    ray_scale = wp.length(endpoint - source) * step
    total = wp.float64(0.0)
    for sample in range(samples):
        point = start + (lower + (wp.float64(sample) + wp.float64(0.5)) * step) * direction
        index = wp.vec3i()
        fraction = wp.vec3d()
        for axis in range(3):
            coordinate = wp.clamp(point[axis], wp.float64(0.0), wp.float64(shape[axis] - 1))
            index[axis] = int(wp.floor(coordinate))
            fraction[axis] = coordinate - wp.float64(index[axis])
        for corner in range(8):
            location = wp.vec3i()
            weight = wp.float64(1.0)
            for axis in range(3):
                bit = (corner >> axis) & 1
                location[axis] = wp.min(index[axis] + bit, shape[axis] - 1)
                if bit == 0:
                    weight *= wp.float64(1.0) - fraction[axis]
                else:
                    weight *= fraction[axis]
            # Scale before the dynamic loop accumulation. Warp's backward
            # replay does not retain the accumulated primal for a product
            # placed after the loop; differentiating that product would omit
            # the total*d(step) intersection-bound term.
            total += (
                ray_scale
                * weight
                * wp.float64(field[(location[2] * shape[1] + location[1]) * shape[0] + location[0]])
            )
    output[0] = total


@pytest.fixture(autouse=True)
def _kernel_context(warp_device: str) -> Iterator[None]:
    with cpu_test_context() if warp_device == "cpu" else nullcontext():
        yield


@pytest.mark.parametrize("samples", [1, 17, 47])
@pytest.mark.parametrize("signed", [False, True])
def test_explicit_volume_and_storage_pose_vjp_matches_independent_raw_warp_tape(
    samples: int, signed: bool, warp_device: str
) -> None:
    angle = 0.2
    grid = GridSpec(
        (3, 4, 5),
        (1.2, 0.8, 1.1),
        (-2.0, -1.0, -1.0),
        (
            math.cos(angle),
            -math.sin(angle),
            0.0,
            math.sin(angle),
            math.cos(angle),
            0.0,
            0.0,
            0.0,
            1.0,
        ),
    )
    bundle = RayBundleGeometry(((-7.0, -2.1, -1.3),), ((8.0, 1.4, 1.7),))
    spec = ProjectionSpec(
        samples,
        active_volume=True,
        precision="float64",
        field_domain="signed_linear" if signed else "physical",
    )
    workspace = prepare_projection(grid, bundle, spec, device=warp_device)
    transform = compose_pose(RigidTransform(), (0.13, -0.08, 0.07, 0.02, -0.03, 0.01))
    values = [0.05 + (i + 1) / 256 * ((-1.0) ** i if signed else 1.0) for i in range(grid.voxels)]
    field = wp.array(values, dtype=wp.float32, device=warp_device, requires_grad=True)
    pose = wp.array(transform.packed(), dtype=wp.float64, device=warp_device, requires_grad=True)
    raw_output = wp.empty(1, dtype=wp.float64, device=warp_device, requires_grad=True)
    seeds = wp.array([-0.75], dtype=wp.float64, device=warp_device)
    inverse_grid = wp.mat33d(
        *(grid.orientation[3 * j + i] / grid.spacing_mm[i] for i in range(3) for j in range(3))
    )
    with wp.Tape() as raw_tape:
        wp.launch(
            _raw_midpoint,
            dim=1,
            inputs=[
                field,
                pose,
                wp.vec3d(*bundle.sources_mm[0]),
                wp.vec3d(*bundle.endpoints_mm[0]),
                inverse_grid,
                wp.vec3d(*grid.origin_mm),
                wp.vec3i(*reversed(grid.shape)),
                samples,
            ],
            outputs=[raw_output],
            device=warp_device,
        )
    raw_tape.backward(grads={raw_output: seeds})
    output = wp.empty(1, dtype=wp.float64, device=warp_device)
    field_gradient = wp.empty(grid.voxels, dtype=wp.float64, device=warp_device)
    pose_gradient = wp.empty(12, dtype=wp.float64, device=warp_device)
    project_optical_depth(field, pose, workspace=workspace, out_L=output)
    projection_vjp(
        field,
        pose,
        workspace=workspace,
        cotangent=seeds,
        out_mu=field_gradient,
        out_pose_matrix=pose_gradient,
        volume_gradient_precision="float64",
    )
    assert output.numpy().tolist() == pytest.approx(
        raw_output.numpy().tolist(), rel=3e-13, abs=3e-14
    )
    assert field_gradient.numpy().tolist() == pytest.approx(
        field.grad.numpy().tolist(), rel=3e-6, abs=3e-7
    )
    assert pose_gradient.numpy().tolist() == pytest.approx(
        pose.grad.numpy().tolist(), rel=3e-12, abs=3e-13
    )

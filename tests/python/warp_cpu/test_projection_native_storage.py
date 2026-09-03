"""Native half storage and FP32 cell arithmetic retain bounded numerical errors."""

# pyright: reportUnknownMemberType=false

import importlib
import math
import struct
from typing import Any, Literal, cast

import pytest

from dpt.contracts import ContractError, TrialDomainError
from dpt.geometry import RayBundleGeometry, RigidTransform
from dpt.projection import ProjectionSpec, prepare_projection, project_optical_depth, projection_vjp
from dpt.validation.projection import _clip_segment
from dpt.volumes import GridSpec

wp: Any = importlib.import_module("warp")


def _f32(value: float) -> float:
    return struct.unpack("f", struct.pack("f", value))[0]


def _values(field: Any, storage: str) -> list[float]:
    if storage == "bfloat16":
        return [struct.unpack("f", struct.pack("I", int(bits) << 16))[0] for bits in field.numpy()]
    return [float(value) for value in field.numpy()]


def _metadata() -> tuple[GridSpec, RayBundleGeometry]:
    grid = GridSpec((3, 4, 5), (1.2, 0.8, 1.1), (-2.0, -1.0, -1.0))
    rays = RayBundleGeometry(
        ((-8.0, 0.1, 0.2), (0.1, 0.2, -8.0), (-8.0, 9.0, 9.0)),
        ((8.0, 0.8, 0.7), (0.3, 0.6, 8.0), (8.0, 9.0, 9.0)),
    )
    return grid, rays


def _float32_oracle(
    values: list[float], grid: GridSpec, rays: RayBundleGeometry, samples: int
) -> tuple[list[float], list[list[float]]]:
    """Scalar rounded/FMA interpolation and an independent voxel weight matrix."""
    xyz = tuple(reversed(grid.shape))
    image = []
    matrix: list[list[float]] = []
    for ray in range(rays.pixels):
        start, direction, lower, upper, length = _clip_segment(
            grid, RigidTransform(), *rays.segment(ray)
        )
        coefficients = [0.0] * grid.voxels
        total = 0.0
        if upper > lower:
            step = (upper - lower) / samples
            entry = [_f32(start[axis] + lower * direction[axis]) for axis in range(3)]
            delta = [_f32(step * direction[axis]) for axis in range(3)]
            for sample in range(samples):
                offset = _f32(float(sample) + 0.5)
                point = [_f32(entry[axis] + offset * delta[axis]) for axis in range(3)]
                if any(point[axis] < -0.5 or point[axis] > xyz[axis] - 0.5 for axis in range(3)):
                    continue
                bounded = [_f32(min(max(point[axis], 0.0), xyz[axis] - 1)) for axis in range(3)]
                low = [min(math.floor(bounded[axis]), max(xyz[axis] - 2, 0)) for axis in range(3)]
                fraction = [_f32(bounded[axis] - float(low[axis])) for axis in range(3)]
                value = 0.0
                for corner in range(8):
                    weight = 1.0
                    indices = []
                    for axis in range(3):
                        high = (corner >> axis) & 1
                        factor = fraction[axis] if high else _f32(1.0 - fraction[axis])
                        weight = _f32(weight * factor)
                        indices.append(min(low[axis] + high, xyz[axis] - 1))
                    voxel = (indices[2] * xyz[1] + indices[1]) * xyz[0] + indices[0]
                    # Warp fuses the multiply/add into one FP32 rounding.
                    value = _f32(value + weight * values[voxel])
                    coefficients[voxel] += length * step * weight
                total += value
            total *= length * step
        image.append(total)
        matrix.append(coefficients)
    return image, matrix


@pytest.mark.parametrize("storage", ["float16", "bfloat16"])
@pytest.mark.parametrize("integration", ["midpoint", "cell_gauss"])
def test_native_half_matches_decoded_float32_baseline_without_expansion(
    storage: Any, integration: Any, warp_device: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    grid, rays = _metadata()
    original = [0.07 + 0.003 * (voxel % 11) for voxel in range(grid.voxels)]
    half = wp.array(original, dtype=getattr(wp, storage), device=warp_device)
    pose = wp.array(RigidTransform().packed(), dtype=wp.float64, device=warp_device)
    baseline = wp.array(_values(half, storage), dtype=wp.float32, device=warp_device)
    outputs = []
    gradients = []
    for field, precision in ((half, storage), (baseline, "float32")):
        workspace = prepare_projection(
            grid,
            rays,
            ProjectionSpec(
                31,
                active_pose=False,
                active_volume=True,
                precision="float64",
                integration=integration,
                storage_precision=cast(Literal["float32", "float16", "bfloat16"], precision),
            ),
            device=warp_device,
        )
        output = wp.empty(rays.pixels, dtype=wp.float64, device=warp_device)
        cotangent = wp.array([1.5, -0.25, 3.0], dtype=wp.float64, device=warp_device)
        gradient = wp.empty(grid.voxels, dtype=wp.float64, device=warp_device)
        if precision == storage:
            allocated: list[tuple[Any, Any]] = []
            allocate = wp.empty

            def counted(
                shape: Any,
                *args: Any,
                allocate: Any = allocate,
                allocated: Any = allocated,
                **kwargs: Any,
            ) -> Any:
                allocated.append((shape, kwargs.get("dtype")))
                return allocate(shape, *args, **kwargs)

            with monkeypatch.context() as patch:
                patch.setattr(wp, "empty", counted)
                inputs = workspace.prepare_inputs(field, pose)
                project_optical_depth(inputs, workspace=workspace, out_L=output)
                projection_vjp(
                    inputs,
                    workspace=workspace,
                    cotangent=cotangent,
                    out_mu=gradient,
                    volume_gradient_precision="float64",
                )
            assert inputs.device_bytes == 2 * grid.voxels + 96
            assert ((grid.voxels,), wp.float32) not in allocated
            assert inputs._mu.dtype == getattr(wp, storage)
        else:
            project_optical_depth(field, pose, workspace=workspace, out_L=output)
            projection_vjp(
                field,
                pose,
                workspace=workspace,
                cotangent=cotangent,
                out_mu=gradient,
                volume_gradient_precision="float64",
            )
        outputs.append(output.numpy().tolist())
        gradients.append(gradient.numpy().tolist())
    assert outputs[0] == pytest.approx(outputs[1], rel=3e-13, abs=3e-14)
    assert gradients[0] == pytest.approx(gradients[1], rel=3e-13, abs=3e-14)


@pytest.mark.parametrize("storage", ["float32", "float16", "bfloat16"])
@pytest.mark.parametrize("samples", [17, 97])
def test_fast_cell_arithmetic_matches_independent_rounded_oracle_and_vjp(
    storage: Any, samples: int, warp_device: str
) -> None:
    grid, rays = _metadata()
    field = wp.array(
        [0.07 + 0.003 * (voxel % 11) for voxel in range(grid.voxels)],
        dtype=getattr(wp, storage),
        device=warp_device,
    )
    decoded = _values(field, storage)
    expected, matrix = _float32_oracle(decoded, grid, rays, samples)
    pose = wp.array(RigidTransform().packed(), dtype=wp.float64, device=warp_device)
    workspace = prepare_projection(
        grid,
        rays,
        ProjectionSpec(
            samples,
            active_pose=False,
            active_volume=True,
            precision="float64",
            storage_precision=storage,
            sampling_precision="fast_float32",
        ),
        device=warp_device,
    )
    output = wp.empty(rays.pixels, dtype=wp.float64, device=warp_device)
    cotangent_values = [1.5, -0.25, 3.0]
    cotangent = wp.array(cotangent_values, dtype=wp.float64, device=warp_device)
    gradient = wp.empty(grid.voxels, dtype=wp.float64, device=warp_device)
    project_optical_depth(field, pose, workspace=workspace, out_L=output)
    projection_vjp(
        field,
        pose,
        workspace=workspace,
        cotangent=cotangent,
        out_mu=gradient,
        volume_gradient_precision="float64",
    )
    assert output.numpy().tolist() == pytest.approx(expected, rel=3e-13, abs=3e-14)
    transpose = [
        math.fsum(row[voxel] * seed for row, seed in zip(matrix, cotangent_values, strict=True))
        for voxel in range(grid.voxels)
    ]
    assert gradient.numpy().tolist() == pytest.approx(transpose, rel=3e-13, abs=3e-14)
    # FP32 cell roundoff makes the forward approximate linearity. Measure its
    # adjoint error against an independently assembled rounded weight matrix.
    lhs = math.fsum(
        value * seed for value, seed in zip(output.numpy(), cotangent_values, strict=True)
    )
    rhs = math.fsum(value * slope for value, slope in zip(decoded, transpose, strict=True))
    assert abs(lhs - rhs) / max(abs(lhs), 1e-12) < 4e-7
    strict = prepare_projection(
        grid,
        rays,
        ProjectionSpec(
            samples,
            active_pose=False,
            precision="float64",
            storage_precision=storage,
        ),
        device=warp_device,
    )
    reference = wp.empty(rays.pixels, dtype=wp.float64, device=warp_device)
    project_optical_depth(field, pose, workspace=strict, out_L=reference)
    error = max(
        abs(float(a) - float(b)) for a, b in zip(output.numpy(), reference.numpy(), strict=True)
    )
    assert error / max(abs(float(value)) for value in reference.numpy()) < 2e-6


@pytest.mark.parametrize("storage", ["float16", "bfloat16"])
def test_half_validation_retains_domain_checks_and_warm_support(
    storage: Any, warp_device: str
) -> None:
    grid, rays = _metadata()
    workspace = prepare_projection(
        grid,
        rays,
        ProjectionSpec(
            17,
            active_pose=False,
            active_volume=True,
            storage_precision=storage,
            sampling_precision="fast_float32",
        ),
        device=warp_device,
    )
    assert workspace.warm()
    field = wp.array([-0.125] * grid.voxels, dtype=getattr(wp, storage), device=warp_device)
    pose = wp.array(RigidTransform().packed(), dtype=wp.float64, device=warp_device)
    with pytest.raises(TrialDomainError):
        workspace.prepare_inputs(field, pose)
    with pytest.raises(ContractError, match="storage_precision"):
        ProjectionSpec(1, storage_precision=cast(Any, "int16"))


def test_fast_sampling_retains_fixed_pose_midpoint_contract() -> None:
    with pytest.raises(ContractError, match="fixed pose"):
        ProjectionSpec(1, sampling_precision="fast_float32")
    with pytest.raises(ContractError, match="midpoint"):
        ProjectionSpec(
            1, active_pose=False, integration="cell_gauss", sampling_precision="fast_float32"
        )


@pytest.mark.parametrize("storage", ["float16", "bfloat16"])
@pytest.mark.parametrize("integration", ["midpoint", "cell_gauss"])
def test_native_half_pose_derivative_matches_analytic_linear_field(
    storage: Any, integration: Any, warp_device: str
) -> None:
    grid = GridSpec((2, 2, 2), (1.0, 1.0, 1.0))
    rays = RayBundleGeometry(((0.25, 0.25, -2.0),), ((0.25, 0.25, 3.0),))
    workspace = prepare_projection(
        grid,
        rays,
        ProjectionSpec(31, precision="float64", storage_precision=storage, integration=integration),
        device=warp_device,
    )
    field = wp.array(
        [0.125 + 0.03125 * (voxel % 2) for voxel in range(8)],
        dtype=getattr(wp, storage),
        device=warp_device,
    )
    pose = wp.array(RigidTransform().packed(), dtype=wp.float64, device=warp_device)
    cotangent = wp.ones(1, dtype=wp.float64, device=warp_device)
    gradient = wp.empty(6, dtype=wp.float64, device=warp_device)
    projection_vjp(field, pose, cotangent=cotangent, workspace=workspace, out_pose_tangent=gradient)
    assert float(gradient.numpy()[0]) == pytest.approx(-0.0625, abs=3e-14)
    output = wp.empty(1, dtype=wp.float64, device=warp_device)
    step = 2.0**-12
    depths = []
    for sign in (-1.0, 1.0):
        changed = wp.array(
            RigidTransform(translation_mm=(sign * step, 0.0, 0.0)).packed(),
            dtype=wp.float64,
            device=warp_device,
        )
        project_optical_depth(field, changed, workspace=workspace, out_L=output)
        depths.append(float(output.numpy()[0]))
    assert (depths[1] - depths[0]) / (2 * step) == pytest.approx(-0.0625, abs=3e-12)

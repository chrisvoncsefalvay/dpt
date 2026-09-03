"""Shared realised coordinates retain strict sampling and transpose semantics."""

from contextlib import nullcontext
from functools import cache

import numpy as np
import pytest
import warp as wp

from dpt._runtime import cpu_test_context
from dpt.contracts import NumericalError
from dpt.geometry import RayBundleGeometry, RigidTransform, compose_pose
from dpt.kernels import projection as strict
from dpt.projection import ProjectionSpec, prepare_projection, projection_vjp
from dpt.volumes import GridSpec


@pytest.fixture(autouse=True)
def context(warp_device):
    with cpu_test_context() if warp_device == "cpu" else nullcontext():
        yield


def evaluate(shape, samples, device, *, matrix=False, double=False, joint=True, seed_scale=1.0):
    nz, ny, nx = shape
    grid = GridSpec(shape, (1.2, 0.8, 1.1), (-0.3, -0.2, -0.1))
    middle = ((ny - 1) * 0.4, (nz - 1) * 0.55)
    rays = RayBundleGeometry(
        ((-4.0, *middle), (nx * 1.2 + 3.0, *middle), (-4.0, -0.6, 0.3)),
        ((nx * 1.2 + 3.0, *middle), (-4.0, *middle), (nx * 1.2 + 3.0, 0.7, 0.9)),
    )
    workspace = prepare_projection(
        grid, rays, ProjectionSpec(samples, precision="float64", active_volume=True), device=device
    )
    field = wp.array(
        np.linspace(0.01, 0.12, grid.voxels, dtype=np.float32), dtype=wp.float32, device=device
    )
    pose = wp.array(
        compose_pose(RigidTransform(), (0.03, -0.02, 0.01, 0.02, -0.01, 0.015)).packed(),
        dtype=wp.float64,
        device=device,
    )
    seed = wp.array(np.array([0.7, -0.3, 0.0]) * seed_scale, dtype=wp.float64, device=device)
    volume = wp.empty(grid.voxels, dtype=wp.float64 if double else wp.float32, device=device)
    pose_gradient = wp.empty(12 if matrix else 6, dtype=wp.float64, device=device)
    pose_output = {"out_pose_matrix" if matrix else "out_pose_tangent": pose_gradient}
    options = {"volume_gradient_precision": "float64" if double else "float32"}
    if joint:
        projection_vjp(
            field,
            pose,
            cotangent=seed,
            workspace=workspace,
            out_mu=volume,
            **pose_output,
            **options,
        )
    else:
        projection_vjp(field, pose, cotangent=seed, workspace=workspace, out_mu=volume, **options)
        projection_vjp(field, pose, cotangent=seed, workspace=workspace, **pose_output)
    return volume.numpy(), pose_gradient.numpy()


@pytest.mark.parametrize("shape", [(1, 1, 1), (1, 3, 5), (3, 1, 5), (3, 4, 1), (3, 4, 5)])
@pytest.mark.parametrize("samples", [17, 97, 1024])
@pytest.mark.parametrize("matrix", [False, True])
def test_joint_matches_separate_products(shape, samples, matrix, warp_device):
    expected = evaluate(shape, samples, warp_device, matrix=matrix, joint=False)
    actual = evaluate(shape, samples, warp_device, matrix=matrix, joint=True)
    for got, want in zip(actual, expected, strict=True):
        if warp_device == "cpu":
            np.testing.assert_array_equal(got, want)
        else:
            np.testing.assert_allclose(got, want, rtol=2e-5, atol=1e-9)


@pytest.mark.parametrize("scale", [1e-35, -1e-35, 1e40, -1e40, 1e280])
def test_wide_destination_keeps_extreme_signed_seed_order(scale, warp_device):
    expected = evaluate((3, 4, 5), 97, warp_device, double=True, joint=False, seed_scale=scale)
    actual = evaluate((3, 4, 5), 97, warp_device, double=True, joint=True, seed_scale=scale)
    for got, want in zip(actual, expected, strict=True):
        assert np.isfinite(got).all()
        np.testing.assert_allclose(got, want, rtol=2e-14, atol=0)


@pytest.mark.parametrize("joint", [False, True])
def test_fp32_destination_still_reports_finite_increment_overflow(joint, warp_device):
    with pytest.raises(NumericalError):
        evaluate((3, 4, 5), 17, warp_device, joint=joint, seed_scale=1e40)


@cache
def coordinate_probe(double, rounded):
    dtype = wp.float64 if double else wp.float32
    old_scatter = strict.get_scatter_field(double, False, rounded)
    reused_scatter = strict.get_scatter_realised(double, rounded)

    @wp.kernel(module="unique", module_options=strict.OPTIONS)
    def kernel(
        field: wp.array(dtype=wp.float32),
        points: wp.array(dtype=wp.vec3d),
        shape: wp.vec3i,
        seed: wp.float64,
        reference_values: wp.array2d(dtype=wp.float64),
        actual_values: wp.array2d(dtype=wp.float64),
        reference_gradient: wp.array(dtype=dtype),
        actual_gradient: wp.array(dtype=dtype),
        reference_status: wp.array(dtype=wp.int32),
        actual_status: wp.array(dtype=wp.int32),
    ):
        index = wp.tid()
        point = points[index]
        expected = strict.sample_field(field, point, shape)
        actual = strict.Sample()
        old_scatter(reference_gradient, point, shape, seed, reference_status)
        if not strict.finite_point(point):
            wp.atomic_or(actual_status, 0, 2)
        if strict.inside_support(point, shape):
            lower = strict.cell_lower(point, shape)
            fraction = strict.cell_fraction(point, lower, shape)
            actual = strict.sample_cell_fraction(
                strict.load_cell(field, lower, shape), point, shape, fraction
            )
            reused_scatter(actual_gradient, lower, fraction, shape, seed, actual_status)
        reference_values[index, 0] = expected.value
        actual_values[index, 0] = actual.value
        for axis in range(3):
            reference_values[index, axis + 1] = expected.gradient[axis]
            actual_values[index, axis + 1] = actual.gradient[axis]

    return kernel


@pytest.mark.parametrize("shape", [(2, 2, 2), (1, 2, 2), (2, 1, 2), (1, 1, 1)])
@pytest.mark.parametrize("double,rounded", [(False, False), (True, False), (True, True)])
def test_faces_singletons_nonfinite_and_weighted_lerp(shape, double, rounded, warp_device):
    count = int(np.prod(shape))
    field_values = np.resize(np.array([2.0**100, 2.0**-100], dtype=np.float32), count)
    field = wp.array(field_values, dtype=wp.float32, device=warp_device)
    coordinates = [
        -0.5,
        np.nextafter(-0.5, -np.inf),
        -0.0,
        0.5,
        1.0,
        1.5,
        np.nextafter(1.5, np.inf),
        np.nan,
        np.inf,
        -np.inf,
        np.finfo(float).max,
    ]
    points = wp.array([(x, 0.0, 0.0) for x in coordinates], dtype=wp.vec3d, device=warp_device)
    values = [
        wp.zeros((len(coordinates), 4), dtype=wp.float64, device=warp_device) for _ in range(2)
    ]
    gradients = [
        wp.zeros(count, dtype=wp.float64 if double else wp.float32, device=warp_device)
        for _ in range(2)
    ]
    statuses = [wp.zeros(1, dtype=wp.int32, device=warp_device) for _ in range(2)]
    wp.launch(
        coordinate_probe(double, rounded),
        dim=len(coordinates),
        inputs=[field, points, wp.vec3i(*shape), wp.float64(-0.75)],
        outputs=[*values, *gradients, *statuses],
        device=warp_device,
    )
    np.testing.assert_array_equal(values[0].numpy(), values[1].numpy())
    np.testing.assert_array_equal(gradients[0].numpy(), gradients[1].numpy())
    np.testing.assert_array_equal(statuses[0].numpy(), [2])
    np.testing.assert_array_equal(statuses[1].numpy(), [2])
    if shape[0] == 2:
        assert values[1].numpy()[4, 0] == 2.0**-100
        assert values[1].numpy()[4, 1] == 0

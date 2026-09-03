"""Real material-path composition gates, to execute only in an authorised CUDA pass."""

import importlib
import math
from typing import Any, Literal

import pytest

from dpt.contracts import ContractError, TrialDomainError
from dpt.geometry import DetectorGeometry, RigidTransform
from dpt.material_projection import (
    material_projection_vjp,
    prepare_material_projection,
    project_material_paths,
)
from dpt.projection import ProjectionSpec
from dpt.volumes import GridSpec

wp: Any = importlib.import_module("warp")
pytestmark = pytest.mark.gpu


def _setup(
    *,
    active_fields: bool = True,
    fractions: tuple[float, float] = (0.25, 0.5),
    field_domain: Literal["fractions", "nonnegative"] = "fractions",
    precision: Literal["float32", "float64"] = "float32",
) -> Any:
    grid = GridSpec((2, 2, 3), (1.0, 1.0, 1.0), (-1.0, -0.5, -0.5))
    geometry = DetectorGeometry(
        (-10.0, 0.2, 0.1), (10.0, -0.2, -0.1), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (0.2, 0.2), (2, 3)
    )
    fields = wp.array(
        [fractions[0]] * grid.voxels + [fractions[1]] * grid.voxels,
        dtype=wp.float32,
        device="cuda:0",
    )
    path_dtype = wp.float64 if precision == "float64" else wp.float32
    paths = wp.empty(2 * geometry.pixels, dtype=path_dtype, device="cuda:0")
    seeds = wp.array(
        [1.0] * geometry.pixels + [-0.5] * geometry.pixels, dtype=path_dtype, device="cuda:0"
    )
    gradients = (
        wp.full(2 * grid.voxels, 123.0, dtype=wp.float32, device="cuda:0")
        if active_fields
        else None
    )
    return prepare_material_projection(
        grid,
        geometry,
        ProjectionSpec(31, active_volume=active_fields, precision=precision),
        materials=2,
        fields=fields,
        out_paths=paths,
        cotangent_paths=seeds,
        out_fields=gradients,
        field_domain=field_domain,
    )


@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_material_layout_and_mm_paths(precision: Literal["float32", "float64"]) -> None:
    workspace = _setup(precision=precision)
    pose = wp.array(RigidTransform().packed(), dtype=wp.float64, device="cuda:0")
    project_material_paths(pose, workspace=workspace)
    paths = workspace.paths.numpy().tolist()
    for pixel in range(workspace.projection.geometry.pixels):
        assert math.isclose(paths[pixel] * 2.0, paths[6 + pixel], rel_tol=2e-7)
    assert (
        workspace.field_views[1].ptr - workspace.fields.ptr == 4 * workspace.projection.grid.voxels
    )
    assert (
        workspace.path_views[1].ptr - workspace.paths.ptr
        == (8 if precision == "float64" else 4) * workspace.projection.geometry.pixels
    )


@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_material_vjp_overwrites_each_field_and_sums_pose(
    precision: Literal["float32", "float64"],
) -> None:
    workspace = _setup(precision=precision)
    pose = wp.array(RigidTransform().packed(), dtype=wp.float64, device="cuda:0")
    gradient = wp.empty(6, dtype=wp.float64, device="cuda:0")
    material_projection_vjp(pose, workspace=workspace, out_pose_tangent=gradient)
    # constant .25 and .5 fields weighted 1 and -.5 cancel their pose derivative.
    assert max(abs(float(x)) for x in gradient.numpy()) < 1e-12
    fields = workspace.grad_fields.numpy().tolist()
    for index in range(workspace.projection.grid.voxels):
        assert math.isclose(fields[index] * -0.5, fields[12 + index], rel_tol=2e-5, abs_tol=1e-7)
    material_projection_vjp(pose, workspace=workspace, out_pose_tangent=gradient, accumulate=True)
    second = workspace.grad_fields.numpy().tolist()
    assert all(
        math.isclose(a * 2, b, rel_tol=2e-5, abs_tol=1e-7)
        for a, b in zip(fields, second, strict=True)
    )


def test_invalid_material_simplex_rejected_before_projection() -> None:
    workspace = _setup(fractions=(0.75, 0.5))
    pose = wp.array(RigidTransform().packed(), dtype=wp.float64, device="cuda:0")
    workspace.paths.fill_(42.0)
    with pytest.raises(TrialDomainError, match="trial domain"):
        project_material_paths(pose, workspace=workspace)
    assert workspace.paths.numpy().tolist() == [42.0] * 12


def test_material_tape_recording_is_not_silently_ignored() -> None:
    workspace = _setup(active_fields=False)
    pose = wp.array(RigidTransform().packed(), dtype=wp.float64, device="cuda:0")
    with wp.Tape(), pytest.raises(ContractError, match="ambient"):
        project_material_paths(pose, workspace=workspace)


@pytest.mark.parametrize("fractions", [(0.6, 0.4), (0.2, 0.8), (0.5, 0.5), (0.3, 0.7), (0.1, 0.8)])
def test_rounded_partitions_are_validated_without_renormalising(
    fractions: tuple[float, float],
) -> None:
    workspace = _setup(fractions=fractions)
    pose = wp.array(RigidTransform().packed(), dtype=wp.float64, device="cuda:0")
    original = workspace.fields.numpy().tobytes()
    workspace.validate_inputs()
    project_material_paths(pose, workspace=workspace)
    assert workspace.fields.numpy().tobytes() == original
    assert all(math.isfinite(float(value)) for value in workspace.paths.numpy())


@pytest.mark.parametrize(
    "fractions", [(0.6, 0.400001), (1.0 + 2.0**-23, 0.0), (-1e-9, 1.0), (math.nan, 0.0)]
)
def test_fraction_tolerance_does_not_accept_overfill_or_invalid_components(
    fractions: tuple[float, float],
) -> None:
    workspace = _setup(fractions=fractions)
    with pytest.raises(TrialDomainError, match="trial domain"):
        workspace.validate_inputs()


@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_nonnegative_equivalent_basis_has_no_fraction_cap_and_preserves_adjoint(
    precision: Literal["float32", "float64"],
) -> None:
    workspace = _setup(fractions=(2.0, 1.5), field_domain="nonnegative", precision=precision)
    pose = wp.array(RigidTransform().packed(), dtype=wp.float64, device="cuda:0")
    project_material_paths(pose, workspace=workspace)
    material_projection_vjp(pose, workspace=workspace)
    paths, fields, seeds = (
        workspace.paths.numpy(),
        workspace.fields.numpy(),
        workspace.cotangent_paths.numpy(),
    )
    gradient = workspace.grad_fields.numpy()
    assert math.isclose(float(gradient @ fields), float(seeds @ paths), rel_tol=2e-6)
    geometry = workspace.projection.geometry
    for row in range(geometry.shape[0]):
        for col in range(geometry.shape[1]):
            target = tuple(
                geometry.origin_mm[i]
                + col * geometry.spacing_mm[0] * geometry.u[i]
                + row * geometry.spacing_mm[1] * geometry.v[i]
                for i in range(3)
            )
            direction = tuple(target[i] - geometry.source_mm[i] for i in range(3))
            length = 3.0 * math.sqrt(sum(v * v for v in direction)) / abs(direction[0])
            pixel = row * geometry.shape[1] + col
            assert math.isclose(
                float(paths[pixel]),
                2.0 * length,
                rel_tol=2e-13 if precision == "float64" else 2e-7,
            )
    workspace.fields.fill_(-0.01)
    with pytest.raises(TrialDomainError):
        workspace.validate_inputs()


def test_fp64_material_path_seed_validation_precedes_gradient_writes() -> None:
    workspace = _setup(precision="float64")
    pose = wp.array(RigidTransform().packed(), dtype=wp.float64, device="cuda:0")
    workspace.cotangent_paths.fill_(float("nan"))
    original = workspace.grad_fields.numpy().tobytes()
    with pytest.raises(ContractError, match="cotangents"):
        material_projection_vjp(pose, workspace=workspace)
    assert workspace.grad_fields.numpy().tobytes() == original

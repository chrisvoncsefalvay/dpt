"""Material-channel traversal equivalence and supplied-table ingestion boundaries."""

import importlib
from dataclasses import asdict
from typing import Any

import numpy as np
import pytest

from dpt._runtime import cpu_test_context, prepare_context
from dpt.contracts import ContractError, TrialDomainError
from dpt.geometry import DetectorGeometry, RayBundleGeometry, RigidTransform
from dpt.material_projection import (
    material_projection_vjp,
    prepare_material_projection,
    project_material_paths,
)
from dpt.materials import MaterialBasis, MaterialTable
from dpt.projection import ProjectionSpec
from dpt.spectral import Spectrum
from dpt.spectral_inputs import read_spectral_inputs
from dpt.volumes import GridSpec

wp: Any = importlib.import_module("warp")


@pytest.mark.parametrize("precision", ["float32", "float64"])
@pytest.mark.parametrize("integration", ["midpoint", "cell_gauss"])
def test_fused_material_channels_equal_serial_canonical_paths(precision, integration, warp_device):
    with cpu_test_context():
        m = 3
        grid = GridSpec((3, 4, 5), (0.7, 1.1, 0.9), (-1.4, -1.5, -1.0))
        geometry = DetectorGeometry(
            (-12.0, -0.1, 0.2),
            (12.0, -0.6, -0.5),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.4, 0.3),
            (4, 5),
        )
        host = np.random.default_rng(51).uniform(0.01, 0.2, (m, *grid.shape)).astype(np.float32)
        fields = wp.array(host, dtype=wp.float32, device=warp_device)
        dtype = wp.float64 if precision == "float64" else wp.float32
        paths = [wp.empty((m, geometry.pixels), dtype=dtype, device=warp_device) for _ in range(2)]
        pose = wp.array(RigidTransform().packed(), dtype=wp.float64, device=warp_device)
        spec = ProjectionSpec(31, precision=precision, integration=integration)
        for execution, output in zip(("serial", "fused"), paths, strict=True):
            workspace = prepare_material_projection(
                grid,
                geometry,
                spec,
                materials=m,
                fields=fields,
                out_paths=output,
                device=warp_device,
                channel_execution=execution,
            )
            assert workspace.fields.ptr == fields.ptr
            assert workspace.paths.ptr == output.ptr
            project_material_paths(pose, workspace=workspace)
        np.testing.assert_array_equal(paths[0].numpy(), paths[1].numpy())
        assert np.all(paths[1].numpy() > 0)


def test_checked_material_failure_consumes_only_its_delivered_diagnostics(warp_device):
    with cpu_test_context():
        grid = GridSpec((2, 2, 2), (1.0, 1.0, 1.0), (-0.5, -0.5, -0.5))
        geometry = DetectorGeometry(
            (-10.0, 0.0, 0.0),
            (10.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 1.0),
            (1, 1),
        )
        fields = wp.full(grid.voxels, 2.0, dtype=wp.float32, device=warp_device)
        output = wp.full(1, 99.0, dtype=wp.float32, device=warp_device)
        workspace = prepare_material_projection(
            grid,
            geometry,
            ProjectionSpec(8),
            materials=1,
            fields=fields,
            out_paths=output,
            device=warp_device,
        )
        pose = wp.array(RigidTransform().packed(), dtype=wp.float64, device=warp_device)
        with pytest.raises(TrialDomainError):
            project_material_paths(pose, workspace=workspace)
        assert output.numpy()[0] == 99
        fields.fill_(0.5)
        project_material_paths(pose, workspace=workspace)
        assert output.numpy()[0] == pytest.approx(1.0)
        # An unchecked failure remains pending until explicitly acknowledged.
        workspace.status.fill_(8)
        with pytest.raises(TrialDomainError):
            project_material_paths(pose, workspace=workspace)
        with pytest.raises(TrialDomainError):
            workspace.check_status()
        with pytest.raises(TrialDomainError):
            workspace.consume_status()
        workspace.check_status()


def test_point_preparation_uploads_basis_and_integrated_spectrum(warp_device):
    with cpu_test_context():
        context = prepare_context(device=warp_device)
        table = MaterialTable("a", (10.0, 20.0), (1.0, 2.0), "mm^-1")
        basis = MaterialBasis((table,), "declared fixed-density basis")
        coefficients = basis.to_device((10.0, 20.0), context, edge_side="above")
        fluence = Spectrum(
            (10.0, 20.0), (2.0, 3.0), "density", quadrature_weights_kev=(5.0, 7.0)
        ).to_device(context)
        np.testing.assert_array_equal(coefficients.numpy(), [[1.0, 2.0]])
        np.testing.assert_array_equal(fluence.numpy(), [10.0, 21.0])


def test_fused_material_paths_support_explicit_ray_bundles(warp_device):
    with cpu_test_context():
        grid = GridSpec((2, 2, 2), (1.0, 1.0, 1.0), (-0.5, -0.5, -0.5))
        rays = RayBundleGeometry(
            ((-10.0, 0.0, 0.0), (-10.0, 0.2, 0.3)), ((10.0, 0.0, 0.0), (10.0, 0.2, 0.3))
        )
        fields = wp.array(
            [[0.25] * grid.voxels, [0.5] * grid.voxels], dtype=wp.float32, device=warp_device
        )
        output = wp.empty((2, 2), dtype=wp.float64, device=warp_device)
        workspace = prepare_material_projection(
            grid,
            rays,
            ProjectionSpec(8, precision="float64", integration="cell_gauss"),
            materials=2,
            fields=fields,
            out_paths=output,
            device=warp_device,
            channel_execution="fused",
        )
        pose = wp.array(RigidTransform().packed(), dtype=wp.float64, device=warp_device)
        project_material_paths(pose, workspace=workspace)
        np.testing.assert_allclose(output.numpy(), [[0.5, 0.5], [1.0, 1.0]], rtol=2e-15)


@pytest.mark.parametrize("precision", ["float32", "float64"])
@pytest.mark.parametrize("integration", ["midpoint", "cell_gauss"])
def test_fused_material_vjp_matches_serial_and_independent_differences(
    precision, integration, warp_device
):
    with cpu_test_context():
        m = 3
        grid = GridSpec((3, 4, 5), (0.7, 1.1, 0.9), (-1.4, -1.5, -1.0))
        geometry = DetectorGeometry(
            (-12.0, -0.1, 0.2),
            (12.0, -0.6, -0.5),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.4, 0.3),
            (4, 5),
        )
        rng = np.random.default_rng(519)
        host = rng.uniform(0.04, 0.2, (m, grid.voxels)).astype(np.float32)
        fields = wp.array(host, dtype=wp.float32, device=warp_device)
        dtype = wp.float64 if precision == "float64" else wp.float32
        seeds = wp.array(rng.normal(size=(m, geometry.pixels)), dtype=dtype, device=warp_device)
        pose = wp.array(RigidTransform().packed(), dtype=wp.float64, device=warp_device)
        spec = ProjectionSpec(31, active_volume=True, precision=precision, integration=integration)
        outputs = []
        workspaces = []
        for execution in ("serial", "fused"):
            paths = wp.empty((m, geometry.pixels), dtype=dtype, device=warp_device)
            gradient = wp.full((m, grid.voxels), 123.0, dtype=wp.float32, device=warp_device)
            workspace = prepare_material_projection(
                grid,
                geometry,
                spec,
                materials=m,
                fields=fields,
                out_paths=paths,
                cotangent_paths=seeds,
                out_fields=gradient,
                device=warp_device,
                channel_execution=execution,
            )
            grad_pose = wp.full(6, 123.0, dtype=wp.float64, device=warp_device)
            material_projection_vjp(pose, workspace=workspace, out_pose_tangent=grad_pose)
            outputs.append((gradient.numpy().copy(), grad_pose.numpy().copy()))
            workspaces.append(workspace)
            material_projection_vjp(
                pose, workspace=workspace, out_pose_tangent=grad_pose, accumulate=True
            )
            np.testing.assert_allclose(gradient.numpy(), 2 * outputs[-1][0], rtol=2e-6, atol=2e-7)
            np.testing.assert_allclose(
                grad_pose.numpy(), 2 * outputs[-1][1], rtol=2e-12, atol=2e-13
            )
        np.testing.assert_allclose(outputs[1][0], outputs[0][0], rtol=2e-6, atol=2e-7)
        np.testing.assert_allclose(outputs[1][1], outputs[0][1], rtol=2e-12, atol=2e-13)
        assert np.linalg.norm(outputs[1][1]) > 0.01
        direction = rng.normal(size=host.shape).astype(np.float32)
        epsilon = 0.01
        plus, minus = (
            (host + epsilon * direction).astype(np.float32),
            (host - epsilon * direction).astype(np.float32),
        )
        values = []
        for trial in (plus, minus):
            fields.assign(trial)
            project_material_paths(pose, workspace=workspaces[1])
            values.append(
                float(
                    np.sum(
                        workspaces[1].paths.numpy().astype(np.float64) * seeds.numpy().reshape(-1)
                    )
                )
            )
        represented_direction = (plus.astype(np.float64) - minus.astype(np.float64)) / (2 * epsilon)
        transpose_dot = float(np.sum(outputs[1][0] * represented_direction))
        assert (values[0] - values[1]) / (2 * epsilon) == pytest.approx(
            transpose_dot, rel=2e-5, abs=2e-6
        )
        fields.assign(host)
        if precision == "float64":
            values = []
            for displacement in (1e-5, -1e-5):
                pose.assign(RigidTransform(translation_mm=(displacement, 0.0, 0.0)).packed())
                project_material_paths(pose, workspace=workspaces[1])
                values.append(
                    float(np.sum(workspaces[1].paths.numpy() * seeds.numpy().reshape(-1)))
                )
            assert (values[0] - values[1]) / 2e-5 == pytest.approx(
                outputs[1][1][0], rel=2e-7, abs=2e-8
            )


@pytest.mark.parametrize("sampling", ["point", "uniform-bin-average"])
def test_archive_ingestion_can_explicitly_prepare_supplied_tables(tmp_path, sampling, warp_device):
    with cpu_test_context():
        grid = GridSpec((2, 2, 2), (1.0, 1.0, 1.0), (-0.5, -0.5, -0.5))
        geometry = DetectorGeometry(
            (-10.0, 0.0, 0.0),
            (10.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 1.0),
            (1, 1),
        )
        archive = tmp_path / "arrays.npz"
        np.savez(
            archive,
            fields=np.full((1, *grid.shape), 0.5, dtype=np.float32),
            weights=np.array([2.0, 3.0], dtype=np.float32),
            response=np.ones(2, dtype=np.float32),
            observation=np.zeros(1, dtype=np.float32),
        )
        rule = (
            {"kind": "point", "edge_side": "above"}
            if sampling == "point"
            else {
                "kind": "uniform-bin-average",
                "bin_edges_kev": [10.0, 20.0, 30.0],
                "weighting": "uniform-energy",
            }
        )
        metadata = {
            "schema_version": 1,
            "grid": asdict(grid),
            "materials": 1,
            "fields": "fields",
            "calibration": [{"name": "group"}],
            "views": [
                {
                    "name": "view",
                    "geometry": asdict(geometry),
                    "energies_kev": [15.0, 25.0],
                    "material_tables": [
                        {
                            "name": "linear",
                            "energies_kev": [10.0, 30.0],
                            "coefficients": [1.0, 3.0],
                            "units": "mm^-1",
                            "interpolation": "linear",
                        }
                    ],
                    "coefficient_sampling": rule,
                    "weights": "weights",
                    "response": "response",
                    "output_unit": "ADU",
                    "input_description": "declared fixed-density volume fractions",
                    "observation": "observation",
                    "calibration_group": "group",
                    "objective": {"kind": "squared_error", "domain": "signal"},
                }
            ],
        }
        problem, _ = read_spectral_inputs(archive, metadata, samples_per_ray=8, device=warp_device)
        assert problem.fields_provenance is None
        assert problem.views[0].observation_provenance is None
        np.testing.assert_allclose(problem.views[0].coefficients.numpy(), [1.5, 2.5], rtol=1e-7)
        metadata["views"][0]["coefficient_sampling"] = {"kind": "point"}
        with pytest.raises(ContractError, match="edge_side"):
            read_spectral_inputs(archive, metadata, samples_per_ray=8, device=warp_device)

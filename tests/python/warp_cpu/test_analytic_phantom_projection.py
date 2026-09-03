"""Continuum discretisation checks and a complete synthetic reconstruction."""

import importlib
from typing import Any, cast

import pytest

from dpt.examples.quickstart import reconstruct_sphere
from dpt.geometry import RigidTransform, Vector3
from dpt.phantoms import Ellipsoid, circular_cone_beam, sample_ellipsoids
from dpt.projection import ProjectionSpec, prepare_projection, project_optical_depth
from dpt.volumes import GridSpec

np: Any = importlib.import_module("numpy")
wp: Any = importlib.import_module("warp")


def test_ellipsoid_projection_grid_refinement_against_analytic_chords(warp_device: str) -> None:
    model = Ellipsoid((2.83, 2.41, 2.67), 0.017, (0.13, -0.11, 0.17))
    geometry = circular_cone_beam(
        [0.29],
        source_radius_mm=30,
        detector_radius_mm=30,
        shape_rows_columns=(5, 5),
        spacing_column_row_mm=(0.7, 0.7),
    )[0]
    exact = np.array(
        [
            model.optical_depth(
                geometry.source_mm,
                cast(
                    Vector3,
                    tuple(
                        geometry.origin_mm[i] + c * 0.7 * geometry.u[i] + r * 0.7 * geometry.v[i]
                        for i in range(3)
                    ),
                ),
            )
            for r in range(5)
            for c in range(5)
        ]
    )
    errors: list[float] = []
    for n in (16, 32, 64):
        spacing = 8.0 / n
        origin = -4.0 + spacing / 2
        grid = GridSpec((n, n, n), (spacing, spacing, spacing), (origin, origin, origin))
        values = sample_ellipsoids(grid, (model,))
        workspace = prepare_projection(
            grid,
            geometry,
            ProjectionSpec(32, precision="float64", integration="cell_gauss"),
            device=warp_device,
        )
        volume = wp.array(values.reshape(-1), dtype=wp.float32, device=warp_device)
        pose = wp.array(RigidTransform().packed(), dtype=wp.float64, device=warp_device)
        depth = wp.empty(geometry.pixels, dtype=wp.float64, device=warp_device)
        project_optical_depth(volume, pose, out_L=depth, workspace=workspace)
        error = float(np.sqrt(np.mean((depth.numpy() - exact) ** 2)))
        errors.append(error)
        # Binary indicator sampling has a boundary discretisation error; an
        # exact integrator cannot remove it. Keep its physical length bound.
        assert error < 2 * spacing * model.attenuation_mm_inverse
    assert errors[-1] < errors[0] / 3, errors


def test_synthetic_count_volume_reconstruction_without_external_files(
    warp_device: str, tmp_path: Any
) -> None:
    result = reconstruct_sphere(device=warp_device, size=8, views=4, iterations=8, seed=41)
    summary = result.summary
    assert cast(float, summary["final_loss"]) < 0.5 * cast(float, summary["initial_loss"])
    assert cast(float, summary["final_rmse"]) < 0.8 * cast(float, summary["initial_rmse"])
    assert bool(np.all(result.reconstruction >= 0)) and bool(np.all(result.reconstruction <= 1))
    assert len(result.counts) == 4 and result.reconstruction.shape == (8, 8, 8)
    path = result.write(tmp_path / "synthetic")
    from dpt.examples._common import load_case

    case = load_case(path)
    np.testing.assert_array_equal(
        case.array("reconstruction", units="volume fraction"), result.reconstruction
    )


def test_synthetic_generator_rejects_counts_before_binary32_rounding(
    warp_device: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dpt.contracts import ContractError
    from dpt.examples import quickstart

    def oversized_sample(*args: Any, **kwargs: Any) -> None:
        kwargs["out_counts"].fill_(2**24 + 1)

    monkeypatch.setattr(quickstart, "sample_poisson_counts", oversized_sample)
    with pytest.raises(ContractError, match="exact binary32 observation range"):
        quickstart.reconstruct_sphere(device=warp_device, size=8, views=2, iterations=1)

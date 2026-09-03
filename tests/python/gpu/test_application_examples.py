"""Actual CUDA checks for multiview Poisson composition and finite-view Fisher."""

# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false, reportMissingTypeArgument=false
# pyright: reportUnknownParameterType=false, reportMissingParameterType=false

import hashlib
import importlib
import json
import math
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest

from dpt._runtime import prepare_context
from dpt.examples._common import ExampleInputError, load_case
from dpt.examples.acquisition_design import candidate_information
from dpt.examples.reconstruction import Reconstruction
from dpt.examples.registration import PreparedView, SharedPoseObjective, prepare_views
from dpt.geometry import DetectorGeometry, RigidTransform, compose_pose
from dpt.objectives import ObjectiveSpec
from dpt.recovery import PoseChart, PrimaryPoseEvaluator, PrimaryPoseProblem
from dpt.validation.projection import integrate_sampled_field
from dpt.volumes import GridSpec

np: Any = importlib.import_module("numpy")
wp: Any = importlib.import_module("warp")
torch: Any = importlib.import_module("torch")
pytestmark = pytest.mark.gpu


def _cold_cli(module: str, case: Path, output: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [sys.executable, "-m", module, "--case", str(case), "--output", str(output)],
        capture_output=True,
        text=True,
        timeout=float(os.environ.get("DPT_TEST_CLI_TIMEOUT", "180")),
        env={**os.environ, "WARP_CACHE_PATH": str(wp.config.kernel_cache_dir)},
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert json.loads((output / "configuration.json").read_text()) == json.loads(case.read_text())
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["device_class"] == "cuda"
    return metadata


def _inputs():
    grid = GridSpec((4, 5, 6), (1.1, 0.9, 0.7), (-2.5, -1.9, -1.2))
    field = np.array(
        [
            0.025 + 0.003 * x + 0.002 * y + 0.004 * z + 0.0007 * x * y
            for z in range(4)
            for y in range(5)
            for x in range(6)
        ],
        dtype=np.float32,
    )
    geometries = [
        DetectorGeometry(
            (-20.0, 0.13, 0.17), (20.0, -1.3, -1.1), (0, 1, 0), (0, 0, 1), (0.21, 0.19), (3, 5)
        ),
        DetectorGeometry(
            (0.17, -20.0, 0.21), (1.1, 20.0, -1.0), (-1, 0, 0), (0, 0, 1), (0.17, 0.23), (3, 5)
        ),
    ]
    return grid, field, geometries


@pytest.mark.parametrize("integration", ["midpoint", "cell_gauss"])
def test_shared_nonzero_chart_poisson_direction_matches_independent_cpu_oracle(
    integration: Any,
) -> None:
    grid, field, geometries = _inputs()
    device_field = wp.array(field, dtype=wp.float32, device="cuda:0")
    chart = PoseChart(
        anchor=compose_pose(RigidTransform(), (0.1, -0.08, 0.03, 0.03, -0.02, 0.01)),
        scales=(0.9, 1.1, 0.8, 0.02, 0.03, 0.01),
    )
    point = (0.07, 0.03, -0.04, 0.08, -0.05, 0.03)
    samples, beam = 67, 100.0
    prepared, observed_all, masks = [], [], []
    for i, geom in enumerate(geometries):
        observed = np.array([80 + (p + i) % 7 for p in range(geom.pixels)], dtype=np.float32)
        mask = np.ones(geom.pixels, dtype=np.float32)
        mask[i::5] = 0.0
        observed[mask == 0] = 0.0
        observed_all.append(observed)
        masks.append(mask)
        evaluator = PrimaryPoseEvaluator(
            PrimaryPoseProblem(
                grid,
                geom,
                device_field,
                wp.array(observed, dtype=wp.float32, device="cuda:0"),
                ObjectiveSpec(kind="poisson", domain="counts", reduction="sum", weighted=True),
                beam,
                wp.array(mask, dtype=wp.float32, device="cuda:0"),
                samples,
                integration=integration,
            ),
            chart,
        )
        prepared.append(PreparedView(f"v{i}", evaluator, int(mask.sum())))
    objective = SharedPoseObjective(tuple(prepared), chart)
    result = objective(point)
    singles = [view.evaluator(point) for view in prepared]
    assert result.loss == math.fsum(v.loss for v in singles)
    assert result.gradient == tuple(math.fsum(v.gradient[k] for v in singles) for k in range(6))

    def reference(coords):
        total = []
        for geom, observed, mask in zip(geometries, observed_all, masks, strict=True):
            for pixel in range(geom.pixels):
                if mask[pixel] == 0:
                    continue
                row, col = divmod(pixel, geom.shape[1])
                depth = integrate_sampled_field(
                    field.tolist(),
                    grid,
                    geom,
                    chart.pose(coords),
                    row,
                    col,
                    midpoint_samples=samples if integration == "midpoint" else None,
                    validate=False,
                )
                mean = beam * math.exp(-depth)
                target = float(observed[pixel])
                total.append(mean - target + target * math.log(target / mean))
        return math.fsum(total)

    assert result.loss == pytest.approx(reference(point), rel=2e-6, abs=1e-5)
    direction = (0.7, -0.4, 0.3, 0.5, 0.2, -0.6)
    predicted = math.fsum(a * b for a, b in zip(result.gradient, direction, strict=True))
    errors = []
    for h in (1e-3, 3e-4, 1e-4):
        plus = tuple(a + h * b for a, b in zip(point, direction, strict=True))
        minus = tuple(a - h * b for a, b in zip(point, direction, strict=True))
        errors.append(abs((reference(plus) - reference(minus)) / (2 * h) - predicted))
    assert min(errors) < 5e-4 * max(1.0, abs(predicted))
    objective(tuple(x * 0.4 for x in point))
    assert objective(point) == result


@pytest.mark.parametrize(
    "precision,integration", [("float32", "midpoint"), ("float64", "cell_gauss")]
)
def test_candidate_fisher_matches_independent_finite_difference_outer_products(
    precision: Any, integration: Any
) -> None:
    grid, field, geometries = _inputs()
    geom = geometries[0]
    pose = compose_pose(RigidTransform(), (0.13, -0.08, 0.04, 0.02, -0.03, 0.01))
    scales = np.array([0.9, 1.1, 0.8, 0.02, 0.03, 0.01])
    beam = np.full(geom.shape, 100.0, dtype=np.float32)
    beam[0, 0] = 0.0
    stream = torch.cuda.current_stream()
    ctx = prepare_context(device="cuda:0", stream=wp.stream_from_torch(stream))
    with torch.cuda.stream(stream), ctx.scope():
        devfield = wp.array(field, dtype=wp.float32, device="cuda:0")
        devpose = wp.array(pose.packed(), dtype=wp.float64, device="cuda:0")
        actual, diag = candidate_information(
            np=np,
            torch=torch,
            ctx=ctx,
            grid=grid,
            geometry=geom,
            attenuation=devfield,
            pose=devpose,
            beam_host=beam,
            scales=torch.as_tensor(scales, dtype=torch.float64, device="cuda:0"),
            samples_per_ray=67,
            precision=precision,
            integration=integration,
        )
    expected = np.zeros((6, 6))
    for pixel in range(geom.pixels):
        row, col = divmod(pixel, geom.shape[1])

        def depth(transform, row=row, col=col):
            return integrate_sampled_field(
                field.tolist(),
                grid,
                geom,
                transform,
                row,
                col,
                midpoint_samples=67 if integration == "midpoint" else None,
                validate=False,
            )

        derivative = []
        for axis in range(6):
            delta = np.zeros(6)
            delta[axis] = scales[axis] * 1e-4
            derivative.append(
                (depth(compose_pose(pose, tuple(delta))) - depth(compose_pose(pose, tuple(-delta))))
                / 2e-4
            )
        mean = float(beam[row, col]) * math.exp(-depth(pose))
        expected += mean * np.outer(derivative, derivative)
    np.testing.assert_allclose(actual, expected, rtol=4e-4, atol=1e-9)
    assert diag["zero_beam_pixels"] == 1


def test_supplied_case_uploads_masked_multiview_inputs_and_detects_changed_bytes(
    tmp_path: Path,
) -> None:
    grid, field, geometries = _inputs()
    arrays = {}

    def stored(name, value, units):
        path = tmp_path / f"{name}.npy"
        np.save(path, value, allow_pickle=False)
        arrays[name] = {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "units": units,
            "source": "explicit software correctness fixture",
            "rights": "original test fixture; Apache-2.0",
        }
        return name

    attenuation = stored("attenuation", field.reshape(grid.shape), "mm^-1")
    views = []
    for i, geom in enumerate(geometries):
        observed = np.full(geom.shape, 80.0, dtype=np.float32)
        mask = np.ones(geom.shape, dtype=np.float32)
        mask[0, 0] = 0
        observed[0, 0] = 0
        views.append(
            {
                "id": f"v{i}",
                "geometry": asdict(geom),
                "observation": stored(f"obs{i}", observed, "counts"),
                "mask": stored(f"mask{i}", mask, "dimensionless"),
                "open_beam_counts": 100.0,
            }
        )
    config = {
        "schema_version": 1,
        "arrays": arrays,
        "registration": {
            "grid": asdict(grid),
            "attenuation": attenuation,
            "samples_per_ray": 67,
            "views": views,
            "chart": asdict(PoseChart()),
            "policy": {"max_iterations": 2},
            "evaluation": {
                "reference_pose": asdict(RigidTransform()),
                "landmarks": stored("landmarks", np.array([[0.0, 0.0, 0.0]]), "mm"),
                "source": "declared software fixture reference",
                "uncertainty": "exact chosen fixture pose; not an anatomical result",
            },
        },
    }
    path = tmp_path / "case.json"
    path.write_text(json.dumps(config))
    case = load_case(path)
    chart = PoseChart()
    prepared = prepare_views(case, case.config["registration"], chart, "cuda:0")
    result = SharedPoseObjective(prepared, chart)((0.0,) * 6)
    assert math.isfinite(result.loss) and all(math.isfinite(x) for x in result.gradient)
    assert [v.included_pixels for v in prepared] == [14, 14]
    _cold_cli("dpt.examples.registration", path, tmp_path / "cold-registration")
    np.save(tmp_path / "obs0.npy", np.zeros((3, 5), dtype=np.float32), allow_pickle=False)
    with pytest.raises(ExampleInputError, match="changed or does not match"):
        case.array("obs0", shape=(3, 5), units="counts")


def test_supplied_attenuation_driver_gradient_constraints_and_final_predictions(
    tmp_path: Path,
) -> None:
    grid, field, geometries = _inputs()
    arrays = {}

    def stored(name, values, units):
        path = tmp_path / f"{name}.npy"
        np.save(path, values, allow_pickle=False)
        arrays[name] = {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "units": units,
            "source": "explicit software correctness fixture",
            "rights": "original test fixture; Apache-2.0",
        }
        return name

    initial = stored("mu", field.reshape(grid.shape), "mm^-1")
    views = []
    observations = []
    for i, geom in enumerate(geometries):
        observed = np.full(geom.shape, 120.0, dtype=np.float32)
        observations.append(observed)
        views.append(
            {
                "geometry": asdict(geom),
                "counts": stored(f"counts{i}", observed, "counts"),
                "open_beam": stored(
                    f"beam{i}", np.full(geom.shape, 100.0, dtype=np.float32), "counts"
                ),
            }
        )
    beta = 0.4
    samples = 67
    config = {
        "schema_version": 1,
        "arrays": arrays,
        "reconstruction": {
            "grid": asdict(grid),
            "fixed_pose": asdict(RigidTransform()),
            "initial_volume": initial,
            "samples_per_ray": samples,
            "views": views,
            "solver": {
                "iterations": 8,
                "initial_step_mm_inverse_squared": 0.001,
                "backtracking_factor": 0.5,
                "armijo": 1e-4,
                "maximum_backtracks": 30,
                "gradient_mapping_tolerance_mm": 1e-6,
                "regularisation_mm": beta,
            },
        },
    }
    path = tmp_path / "case.json"
    path.write_text(json.dumps(config))
    _cold_cli("dpt.examples.reconstruction", path, tmp_path / "cold-reconstruction")
    problem = Reconstruction(load_case(path), device="cuda:0", torch=torch, wp=wp)
    with torch.cuda.stream(problem.torch_stream), torch.no_grad():
        actual = problem.evaluate(problem.mu, problem.mu_wp, gradient=True)
        gradient = problem.gradient.cpu().numpy().copy()

        def reference(values):
            loss = []
            for geom, observed in zip(geometries, observations, strict=True):
                for p in range(geom.pixels):
                    row, col = divmod(p, geom.shape[1])
                    depth = integrate_sampled_field(
                        values.tolist(),
                        grid,
                        geom,
                        RigidTransform(),
                        row,
                        col,
                        midpoint_samples=samples,
                        validate=False,
                    )
                    mean = 100 * math.exp(-depth)
                    target = float(observed[row, col])
                    loss.append(mean - target + target * math.log(target / mean))
            shaped = values.reshape(grid.shape)
            for axis, spacing in enumerate(reversed(grid.spacing_mm)):
                loss.append(
                    0.5
                    * beta
                    * math.prod(grid.spacing_mm)
                    / spacing**2
                    * float(np.sum(np.diff(shaped, axis=axis) ** 2))
                )
            return math.fsum(loss)

        assert actual == pytest.approx(reference(field.astype(np.float64)), rel=2e-6)
        direction = np.sin(np.arange(field.size) * 0.7) * 0.01
        predicted = float(gradient.astype(np.float64) @ direction)
        expected = (
            reference(field.astype(np.float64) + 1e-3 * direction)
            - reference(field.astype(np.float64) - 1e-3 * direction)
        ) / 2e-3
        assert predicted == pytest.approx(expected, rel=3e-4, abs=1e-5)
        result = problem.solve()
        final = problem.mu.cpu().numpy().copy()
        assert np.isfinite(final).all() and np.min(final) >= 0
        assert result["accepted_steps"] > 0 and result["final_objective"] < actual
        assert np.any(final == 0)
        for geom, view in zip(geometries, problem.views, strict=True):
            predicted_image = view["prediction"].numpy().copy()
            expected_image = []
            for p in range(geom.pixels):
                row, col = divmod(p, geom.shape[1])
                depth = integrate_sampled_field(
                    final.tolist(),
                    grid,
                    geom,
                    RigidTransform(),
                    row,
                    col,
                    midpoint_samples=samples,
                    validate=False,
                )
                expected_image.append(100 * math.exp(-depth))
            np.testing.assert_allclose(predicted_image, expected_image, rtol=2e-6, atol=1e-5)


def test_acquisition_cli_initialises_fresh_runtime_and_preserves_exact_tie(tmp_path: Path) -> None:
    grid, field, geometries = _inputs()
    geom = geometries[0]
    arrays = {}

    def stored(name, values, units):
        path = tmp_path / f"{name}.npy"
        np.save(path, values, allow_pickle=False)
        arrays[name] = {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "units": units,
            "source": "explicit software correctness fixture",
            "rights": "original test fixture; Apache-2.0",
        }
        return name

    attenuation = stored("mu", field.reshape(grid.shape), "mm^-1")
    prior = stored("prior", np.eye(6, dtype=np.float64), "dimensionless")
    targets = stored("targets", np.array([[1.0, 2.0, 3.0]], dtype=np.float64), "mm")
    beam = stored("beam", np.full(geom.shape, 100.0, dtype=np.float32), "photons/pixel")
    costly_beam = stored(
        "costly-beam", np.full(geom.shape, 200.0, dtype=np.float32), "photons/pixel"
    )
    candidate = {
        "geometry": asdict(geom),
        "open_beam": beam,
        "cost": 1500.0,
        "feasibility_description": "identical feasible software fixture acquisition",
    }
    config = {
        "schema_version": 1,
        "arrays": arrays,
        "acquisition_design": {
            "grid": asdict(grid),
            "pose": asdict(RigidTransform()),
            "attenuation": attenuation,
            "prior_precision": prior,
            "targets_object_mm": targets,
            "parameter_scales": [1.0, 1.0, 1.0, 0.02, 0.02, 0.02],
            "samples_per_ray": 67,
            "max_candidate_pixels": geom.pixels,
            "maximum_cost": 1500.0,
            "cost_unit": "expected_detector_photons",
            "current_information_description": "explicit unit Gaussian software fixture prior",
            "candidates": [dict(candidate, name=name) for name in ("zeta", "alpha")]
            + [dict(candidate, name="over-budget", open_beam=costly_beam, cost=3000.0)],
        },
    }
    path = tmp_path / "case.json"
    path.write_text(json.dumps(config))
    output = tmp_path / "cold-acquisition"
    _cold_cli("dpt.examples.acquisition_design", path, output)
    report = json.loads((output / "design.json").read_text())
    assert report["selected_candidate"] == "alpha"
    assert report["excluded_by_budget"] == ["over-budget"]
    rows = report["candidates"]
    assert rows[0]["mean_target_variance_mm2"] == rows[1]["mean_target_variance_mm2"]
    assert rows[0]["mean_target_variance_mm2"] < report["current_mean_target_variance_mm2"]
    matrices = np.load(output / "candidate_information.npy", allow_pickle=False)
    np.testing.assert_array_equal(matrices[0], matrices[1])

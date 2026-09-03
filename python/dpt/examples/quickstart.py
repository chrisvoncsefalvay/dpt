"""Simulate independent count views and reconstruct an analytic sphere on CPU or CUDA."""

from __future__ import annotations

import argparse
import importlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dpt.contracts import ContractError, integer
from dpt.detector import ObservationIdentity, prepare_detector, sample_poisson_counts
from dpt.examples.make_case import CaseArray, write_case
from dpt.geometry import RigidTransform
from dpt.material_projection import prepare_material_projection, project_material_paths
from dpt.material_reconstruction import (
    MaterialReconstruction,
    MaterialReconstructionSettings,
    MaterialReconstructionView,
)
from dpt.phantoms import Ellipsoid, circular_cone_beam, sample_ellipsoids
from dpt.projection import ProjectionSpec
from dpt.spectral import SpectralSpec, prepare_spectral, spectral_signal
from dpt.volumes import GridSpec


@dataclass(frozen=True, slots=True)
class QuickstartResult:
    """Synthetic truth, fitted fractions and per-view Poisson observations.

    ``truth`` and ``reconstruction`` are contiguous FP32 arrays with shape
    ``(size, size, size)`` in z/y/x order. Each member of ``counts`` has shape
    ``(1, size, size)`` in detector-band/row/column order and holds integer
    counts exactly represented in FP32. ``summary`` records the experiment,
    initial/final objective, fraction RMSE and solver termination.
    """

    summary: dict[str, object]
    truth: Any
    reconstruction: Any
    counts: tuple[Any, ...]

    def write(self, directory: str | Path) -> Path:
        """Save the synthetic inputs, reconstruction and measured solve results."""
        provenance = ("DPT analytic sphere quickstart", "Apache-2.0")
        arrays = {
            "truth": CaseArray(self.truth, "volume fraction", *provenance),
            "reconstruction": CaseArray(self.reconstruction, "volume fraction", *provenance),
            **{
                f"counts_{i}": CaseArray(image, "counts", *provenance)
                for i, image in enumerate(self.counts)
            },
        }
        return write_case(directory, {"quickstart": self.summary}, arrays)


def reconstruct_sphere(
    *, device: str = "cpu", size: int = 16, views: int = 6, iterations: int = 12, seed: int = 41
) -> QuickstartResult:
    """Fit ``I[p] = sum_k w[k] exp(-mu[k] integral_ray f(x) ds)``.

    A sphere with dimensionless fraction 0.75 generates independent Poisson
    observations on circular cone views. The unknown ``f`` starts at 0.05,
    remains in ``[0, 1]`` and is fitted using the Poisson objective with a
    quadratic spatial penalty. Lengths are in mm and ``mu`` is in mm^-1.
    The two coefficients and bin populations are synthetic numerical inputs.

    ``size`` selects both the cubic voxel grid and square detector, with
    shape ``(size, size, size)`` and ``(size, size)`` respectively. Voxels are
    1 mm apart and detector pixels are 2 mm apart. ``views`` selects equally
    spaced full-circle angles. View ``i`` uses ``ObservationIdentity(seed, i)``.
    Select ``device="cpu"`` or an available CUDA device explicitly. NumPy and
    Warp are required, and the first evaluation compiles uncached kernels.

    """
    size = integer(size, "size", minimum=8, maximum=128)
    views = integer(views, "views", minimum=2, maximum=128)
    iterations = integer(iterations, "iterations", minimum=1, maximum=1000)
    seed = integer(seed, "seed", maximum=2**64 - 1)
    np: Any = importlib.import_module("numpy")
    origin = -(size - 1) / 2
    grid = GridSpec((size, size, size), (1.0, 1.0, 1.0), (origin, origin, origin))
    truth = sample_ellipsoids(grid, (Ellipsoid((size * 0.3, size * 0.3, size * 0.3), 0.75),))
    geometries = circular_cone_beam(
        (2 * math.pi * i / views for i in range(views)),
        source_radius_mm=3 * size,
        detector_radius_mm=3 * size,
        shape_rows_columns=(size, size),
        spacing_column_row_mm=(2.0, 2.0),
    )
    spec = SpectralSpec(1, 2, output_unit="counts", precision="float64")
    coefficients = np.array([[0.015, 0.025]], dtype=np.float32)
    weights = np.array([[1500.0, 1000.0]], dtype=np.float32)
    response = np.ones((1, 2), dtype=np.float32)
    fitting: list[MaterialReconstructionView] = []
    counts: list[Any] = []
    # Prepare one geometry before importing Warp: prepare_context checks the
    # requested runtime and reports an ordinary package error if unavailable.
    projection = prepare_material_projection(
        grid,
        geometries[0],
        ProjectionSpec(2 * size, precision="float64", active_pose=False),
        materials=1,
        fields=_device_array(truth.reshape(-1), device, "float32"),
        out_paths=_device_empty(size * size, device, "float64"),
        device=device,
    )
    ctx, wp = projection.context, projection.context.wp
    fields, paths = projection.fields, projection.paths
    with ctx.scope():
        attenuation = wp.array(coefficients.reshape(-1), dtype=wp.float32, device=ctx.device)
        populations = wp.array(weights.reshape(-1), dtype=wp.float32, device=ctx.device)
        efficiency = wp.array(response.reshape(-1), dtype=wp.float32, device=ctx.device)
        mean = wp.empty(size * size, dtype=wp.float64, device=ctx.device)
        observed = wp.empty(size * size, dtype=wp.uint64, device=ctx.device)
        pose = wp.array(RigidTransform().packed(), dtype=wp.float64, device=ctx.device)
    spectral = prepare_spectral(spec, max_pixels=size * size, device=device, stream=ctx.stream)
    detector = prepare_detector(max_pixels=size * size, device=device, stream=ctx.stream)
    for i, geometry in enumerate(geometries):
        if i:
            projection = prepare_material_projection(
                grid,
                geometry,
                ProjectionSpec(2 * size, precision="float64", active_pose=False),
                materials=1,
                fields=fields,
                out_paths=paths,
                device=device,
                stream=ctx.stream,
            )
        project_material_paths(pose, workspace=projection)
        spectral_signal(
            paths, attenuation, populations, efficiency, out_mean=mean, workspace=spectral
        )
        sample_poisson_counts(
            mean, out_counts=observed, identity=ObservationIdentity(seed, i), workspace=detector
        )
        sampled = observed.numpy().reshape(1, size, size)
        if bool(np.any(sampled > 2**24)):
            raise ContractError("sampled counts exceed the exact binary32 observation range")
        image = np.ascontiguousarray(sampled, dtype=np.float32)
        counts.append(image)
        fitting.append(
            MaterialReconstructionView(geometry, RigidTransform(), image, weights, response)
        )
    solver = MaterialReconstruction(
        grid=grid,
        views=tuple(fitting),
        coefficients=coefficients,
        spectral_spec=spec,
        initial_fractions=np.full((1, *grid.shape), 0.05, dtype=np.float32),
        settings=MaterialReconstructionSettings(
            iterations=iterations,
            initial_step=1.0,
            maximum_step=1.0,
            step_selection="bb",
            regularisation_mm_inverse=0.01,
        ),
        samples_per_ray=2 * size,
        device=device,
    )
    initial_loss = solver.evaluate(gradient=False)
    result = solver.solve()
    recovered = np.ascontiguousarray(solver.fractions_numpy()[0])
    final_loss = solver.evaluate(gradient=False)
    summary: dict[str, object] = {
        "device": str(ctx.device),
        "grid_shape_zyx": list(grid.shape),
        "spacing_xyz_mm": list(grid.spacing_mm),
        "views": views,
        "iterations_requested": iterations,
        "seed": seed,
        "coefficients_mm_inverse": coefficients.tolist(),
        "open_beam_bin_counts": weights.tolist(),
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "initial_rmse": float(np.sqrt(np.mean((truth - 0.05) ** 2))),
        "final_rmse": float(np.sqrt(np.mean((truth - recovered) ** 2))),
        "solver": result,
    }
    return QuickstartResult(summary, truth, recovered, tuple(counts))


def _device_array(values: Any, device: str, dtype: str) -> Any:
    from dpt._runtime import prepare_context

    context = prepare_context(device=device)
    with context.scope():
        return context.wp.array(values, dtype=getattr(context.wp, dtype), device=context.device)


def _device_empty(length: int, device: str, dtype: str) -> Any:
    from dpt._runtime import prepare_context

    context = prepare_context(device=device)
    with context.scope():
        return context.wp.empty(length, dtype=getattr(context.wp, dtype), device=context.device)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--size", type=int, default=16)
    parser.add_argument("--views", type=int, default=6)
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument(
        "--output", type=Path, help="Create a new directory containing numeric results and hashes"
    )
    args = parser.parse_args()
    try:
        result = reconstruct_sphere(
            device=args.device,
            size=args.size,
            views=args.views,
            iterations=args.iterations,
            seed=args.seed,
        )
        if args.output is not None:
            result.write(args.output)
        print(json.dumps(result.summary, sort_keys=True, indent=2, allow_nan=False))
    except ContractError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()

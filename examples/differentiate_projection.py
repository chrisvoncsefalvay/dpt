"""Back-project a detector-region sensitivity into an actual input CT volume."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from dpt.imaging import standard_views
from dpt.rendering import figure_style, load_attenuation
from dpt.torch import Projector


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--units", choices=("HU", "mm^-1"))
    parser.add_argument("--series-uid")
    parser.add_argument("--energy-kev", type=float, default=60)
    parser.add_argument("--precision", choices=("fp32", "fp64"), default="fp32")
    parser.add_argument("--samples-per-ray", type=int, default=1024)
    parser.add_argument("--shape", nargs=2, type=int, default=(512, 512))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ct, mu = load_attenuation(
        args.input, units=args.units, series_uid=args.series_uid, energy_kev=args.energy_kev
    )
    geometry = standard_views(mu.grid, ("ap", "lateral"), shape=tuple(args.shape))
    field = torch.tensor(mu.values, device=args.device, dtype=torch.float32, requires_grad=True)
    projector = Projector(
        grid=mu.grid,
        geometry=geometry,
        precision=args.precision,
        samples_per_ray=args.samples_per_ray,
    )
    prediction = projector(field)
    rows, columns = args.shape
    region = (
        rows * 5 // 16,
        max(rows * 5 // 16 + 1, rows * 11 // 16),
        columns * 5 // 16,
        max(columns * 5 // 16 + 1, columns * 11 // 16),
    )
    objective = prediction[0, region[0] : region[1], region[2] : region[3]].mean()
    objective.backward()
    if field.grad is None:
        raise RuntimeError("The projector did not produce the requested volume derivative")
    # Explicit final output boundary; no host copies participate in differentiation.
    sensitivity = field.grad.detach().cpu().numpy()
    image = prediction.detach().cpu().numpy()
    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / "volume_sensitivity.npy", sensitivity, allow_pickle=False)
    np.save(args.output / "optical_depth.npy", image, allow_pickle=False)
    (args.output / "settings.json").write_text(
        json.dumps(
            {
                "input": str(args.input),
                "input_sha256": list(ct.metadata.source_sha256),
                "precision": args.precision,
                "samples_per_ray": args.samples_per_ray,
                "detector_region_row_start_stop_column_start_stop": region,
                "objective": "mean AP optical depth in the selected detector region",
                "objective_value": float(objective.detach().cpu()),
                "sensitivity_units": "mm",
                "shape_zyx": list(mu.grid.shape),
            },
            indent=2,
        )
        + "\n"
    )
    plt = figure_style()
    figure, axes = plt.subplots(1, 3, figsize=(13, 4.6), layout="constrained")
    nz, ny, nx = ct.grid.shape
    sx, sy, _ = ct.grid.spacing_mm
    extent = (-sx / 2, (nx - 0.5) * sx, -sy / 2, (ny - 0.5) * sy)
    plane = nz // 2
    axes[0].imshow(
        ct.values[plane],
        origin="lower",
        extent=extent,
        cmap="gray",
        vmin=-1000 if ct.metadata.value_units == "HU" else None,
        vmax=1000 if ct.metadata.value_units == "HU" else None,
    )
    axes[0].set(title=f"Input slice {plane}", xlabel="Grid x (mm)", ylabel="Grid y (mm)")
    panel = geometry.views[0]
    du, dv = panel.spacing_mm
    axes[1].imshow(
        image[0],
        origin="lower",
        cmap="gray",
        extent=(-columns * du / 2, columns * du / 2, -rows * dv / 2, rows * dv / 2),
    )
    from matplotlib.patches import Rectangle

    axes[1].add_patch(
        Rectangle(
            ((region[2] - columns / 2) * du, (region[0] - rows / 2) * dv),
            (region[3] - region[2]) * du,
            (region[1] - region[0]) * dv,
            edgecolor="#E69F00",
            fill=False,
            linewidth=1.5,
        )
    )
    axes[1].set(title="AP objective region", xlabel="Detector u (mm)", ylabel="Detector v (mm)")
    artist = axes[2].imshow(sensitivity[plane], origin="lower", extent=extent, cmap="viridis")
    axes[2].set(title=f"Sensitivity · slice {plane}", xlabel="Grid x (mm)", ylabel="Grid y (mm)")
    figure.colorbar(artist, ax=axes[2], label="d mean optical depth / d attenuation (mm)")
    for extension in ("png", "svg", "pdf"):
        figure.savefig(args.output / f"volume-sensitivity.{extension}", bbox_inches="tight")
    plt.close(figure)
    print(f"Actual-volume derivative outputs: {args.output.resolve()}")
    print("For a measured-image loss and calibrated differentiable pose, see examples/fit_pose.py.")


if __name__ == "__main__":
    main()

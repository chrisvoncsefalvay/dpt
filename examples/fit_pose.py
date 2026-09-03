"""Fit real measured images using explicit RAS calibration and a Torch optimiser.

This opt-in example requires user-supplied paired data. It makes no registration
claim until executed, and does not infer a DeepFluoro/DICOM projection convention.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from dpt.geometry import BatchedGeometry, DetectorGeometry, RigidTransform
from dpt.rendering import figure_style, load_attenuation


def load_calibration(path: Path) -> tuple[dict, BatchedGeometry, RigidTransform]:
    calibration = json.loads(path.read_text())
    if calibration.get("coordinate_frame") != "RAS" or calibration.get("length_units") != "mm":
        raise ValueError("Calibration must explicitly use RAS coordinates and millimetres")
    if calibration.get("pose_convention") != "object_to_world":
        raise ValueError("Calibration must declare pose_convention='object_to_world'")
    if calibration.get("image_quantity") not in ("optical_depth", "transmission"):
        raise ValueError(
            "Supply measured optical depth or calibrated transmission; "
            "raw display pixels are not calibrated"
        )
    if not calibration.get("geometry"):
        raise ValueError("Calibration needs one explicit geometry record per measured image")
    panels = tuple(
        DetectorGeometry(**{k: tuple(v) if isinstance(v, list) else v for k, v in panel.items()})
        for panel in calibration["geometry"]
    )
    anchor = RigidTransform(
        **{k: tuple(v) if isinstance(v, list) else v for k, v in calibration["anchor"].items()}
    )
    return calibration, BatchedGeometry(panels), anchor


def load_measured(path: Path, panel: DetectorGeometry, quantity: str) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        values = np.load(path, allow_pickle=False)
    elif path.suffix.lower() == ".dcm":
        import pydicom
        from pydicom.pixels import apply_modality_lut

        dataset = pydicom.dcmread(path)
        values = apply_modality_lut(dataset.pixel_array, dataset)
        # A detector-pitch tag must agree with the explicitly supplied panel.
        pitch = getattr(dataset, "ImagerPixelSpacing", None)
        if pitch is not None and not np.allclose(
            tuple(reversed(pitch)), panel.spacing_mm, rtol=1e-4, atol=1e-6
        ):
            raise ValueError(f"DICOM detector pitch disagrees with calibration: {path}")
    else:
        raise ValueError("Measured images must be numeric .npy or calibrated .dcm files")
    values = np.asarray(values, dtype=np.float32)
    if values.shape != panel.shape or not np.isfinite(values).all():
        raise ValueError(f"Measured image shape/values disagree with calibration: {path}")
    if np.any(values < 0) or (quantity == "transmission" and np.any(values > 1)):
        raise ValueError(
            "Measured quantities must be nonnegative; transmission must not exceed one"
        )
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("volume", type=Path)
    parser.add_argument("--units", choices=("HU", "mm^-1"))
    parser.add_argument("--series-uid")
    parser.add_argument("--measured", type=Path, nargs="+", required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--precision", choices=("fp32", "fp64"), default="fp32")
    parser.add_argument("--samples-per-ray", type=int, default=1024)
    parser.add_argument("--energy-kev", type=float, default=60)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    calibration, geometry, anchor = load_calibration(args.calibration)
    if len(args.measured) != len(geometry.views):
        parser.error("provide one measured image for each calibrated view")
    if args.iterations < 1 or args.learning_rate <= 0:
        parser.error("iterations and learning-rate must be positive")
    # Validate every measured input and print the physical convention before CUDA work.
    measured_host = np.stack(
        [
            load_measured(path, panel, calibration["image_quantity"])
            for path, panel in zip(args.measured, geometry.views, strict=True)
        ]
    )
    print(
        json.dumps(
            {
                "coordinate_frame": "RAS",
                "pose_convention": "object_to_world",
                "geometry": [asdict(v) for v in geometry.views],
                "anchor": asdict(anchor),
                "image_quantity": calibration["image_quantity"],
            },
            indent=2,
        ),
        flush=True,
    )
    _, attenuation = load_attenuation(
        args.volume, units=args.units, series_uid=args.series_uid, energy_kev=args.energy_kev
    )
    import torch

    from dpt.torch import Projector

    field = torch.tensor(attenuation.values, device=args.device, dtype=torch.float32)
    measured = torch.as_tensor(measured_host, device=args.device)
    projector = Projector(
        grid=attenuation.grid,
        geometry=geometry,
        precision=args.precision,
        samples_per_ray=args.samples_per_ray,
        anchor=anchor,
    )
    parameters = torch.nn.Parameter(torch.zeros(6, dtype=torch.float64, device=args.device))
    scales = torch.tensor([1, 1, 1, 0.01, 0.01, 0.01], dtype=torch.float64, device=args.device)
    optimiser = torch.optim.Adam([parameters], lr=args.learning_rate)

    def predict():
        depth = projector(field, pose=parameters * scales)
        return depth if calibration["image_quantity"] == "optical_depth" else torch.exp(-depth)

    with torch.no_grad():
        initial = predict().clone()
    history = []
    for iteration in range(args.iterations):
        optimiser.zero_grad(set_to_none=True)
        prediction = predict()
        loss = torch.mean((prediction - measured) ** 2)
        loss.backward()
        if parameters.grad is None or not torch.isfinite(parameters.grad).all():
            raise RuntimeError("Pose derivative is missing or nonfinite")
        optimiser.step()
        value = float(loss.detach())
        history.append({"iteration": iteration, "loss": value})
        print(f"iteration {iteration}: mean squared error {value:.8g}", flush=True)
    with torch.no_grad():
        final = predict()
        final_loss = float(torch.mean((final - measured) ** 2))
    args.output.mkdir(parents=True, exist_ok=True)
    initial_host, final_host = initial.cpu().numpy(), final.cpu().numpy()
    np.save(args.output / "initial_prediction.npy", initial_host, allow_pickle=False)
    np.save(args.output / "final_prediction.npy", final_host, allow_pickle=False)
    np.save(args.output / "measured.npy", measured_host, allow_pickle=False)
    torch.save(
        {"parameters": parameters.detach().cpu(), "optimiser": optimiser.state_dict()},
        args.output / "checkpoint.pt",
    )
    (args.output / "fit.json").write_text(
        json.dumps(
            {
                "calibration": calibration,
                "tangent_pose_mm_rad": (parameters.detach() * scales).cpu().tolist(),
                "termination": "requested iteration count reached",
                "history": history,
                "final_loss": final_loss,
                "loss_quantity": calibration["image_quantity"],
            },
            indent=2,
        )
        + "\n"
    )
    plt = figure_style()
    figure, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    panel = geometry.views[0]
    rows, columns = panel.shape
    du, dv = panel.spacing_mm
    extent = (-columns * du / 2, columns * du / 2, -rows * dv / 2, rows * dv / 2)
    for axis, predicted, title in zip(
        axes[:2],
        (initial_host[0], final_host[0]),
        ("Initial measured/predicted overlay", "Final measured/predicted overlay"),
        strict=True,
    ):
        axis.imshow(measured_host[0], cmap="gray", origin="lower", extent=extent)
        axis.contour(
            predicted, levels=6, colors="#E69F00", linewidths=0.6, origin="lower", extent=extent
        )
        axis.set(title=title, xlabel="Detector u (mm)", ylabel="Detector v (mm)")
    axes[2].plot(
        [row["iteration"] for row in history] + [args.iterations],
        [row["loss"] for row in history] + [final_loss],
        color="#0072B2",
    )
    axes[2].set(xlabel="Optimisation iteration", ylabel="Mean squared image error")
    for extension in ("png", "svg", "pdf"):
        figure.savefig(args.output / f"registration.{extension}", bbox_inches="tight")
    plt.close(figure)
    print(f"Measured registration outputs: {args.output.resolve()}")


if __name__ == "__main__":
    main()

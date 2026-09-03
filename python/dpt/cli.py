"""Project a real NIfTI or DICOM volume and save numerical images and PNGs."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from dpt.contracts import ContractError
from dpt.rendering import render_volume_file, save_projection_outputs


def project_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("input", type=Path, help="NIfTI file or DICOM series directory")
    parser.add_argument(
        "--units", choices=("HU", "mm^-1"), help="Declare unknown image-value units"
    )
    parser.add_argument(
        "--series-uid", help="Select a SeriesInstanceUID reported by the DICOM reader"
    )
    parser.add_argument(
        "--views", nargs="+", choices=("ap", "lateral", "oblique"), default=["ap", "lateral"]
    )
    parser.add_argument(
        "--shape", nargs=2, type=int, default=(512, 512), metavar=("ROWS", "COLUMNS")
    )
    parser.add_argument(
        "--energy-kev", type=float, default=60.0, help="Declared HU model energy; currently 60 keV"
    )
    parser.add_argument("--precision", choices=("fp32", "fp64"), default="fp32")
    parser.add_argument("--samples-per-ray", type=int, default=1024)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)


def run_projection(args: argparse.Namespace) -> tuple:
    result = render_volume_file(
        args.input,
        units=args.units,
        series_uid=args.series_uid,
        names=tuple(args.views),
        shape=tuple(args.shape),
        energy_kev=args.energy_kev,
        precision=args.precision,
        samples_per_ray=args.samples_per_ray,
        device=args.device,
    )
    images, volume, geometry, settings = result
    depth = save_projection_outputs(images, settings, args.output)
    print(json.dumps(settings, indent=2, allow_nan=False))
    print(f"Numerical optical depth: {(args.output / 'optical_depth.npy').resolve()}")
    print(f"Numerical transmission: {(args.output / 'transmission.npy').resolve()}")
    print(f"PNG images and settings: {args.output.resolve()}")
    return depth, volume, geometry, settings


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    project = commands.add_parser(
        "project", help="Project a real CT or calibrated attenuation volume"
    )
    project_parser(project)
    args = parser.parse_args(argv)
    try:
        run_projection(args)
    except (ContractError, FileNotFoundError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()

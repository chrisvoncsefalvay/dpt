"""Shared real-volume preparation and explicit image-file output for examples.

Projection and differentiation remain on the device. Only the save functions
detach/download tensors, after the caller has completed the calculation.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
import zipfile
import zlib
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.request import urlopen

from dpt.contracts import ContractError
from dpt.imaging import ct_to_attenuation, read_volume, standard_views


def load_attenuation(
    path: str | Path, *, units: str | None, series_uid: str | None = None, energy_kev: float = 60.0
) -> tuple[Any, Any]:
    """Load HU or calibrated mm^-1 data, retaining the original values separately."""
    volume = read_volume(path, units=units, series_uid=series_uid)
    if volume.metadata.value_units == "HU":
        return volume, ct_to_attenuation(volume, energy_kev=energy_kev)
    if volume.metadata.value_units != "mm^-1":
        raise ContractError(
            "declare --units HU or --units mm^-1; unknown units cannot be converted"
        )
    if (volume.values < 0).any():
        raise ContractError("calibrated attenuation must be nonnegative in mm^-1")
    return volume, volume


def render_volume_file(
    path: str | Path,
    *,
    units: str | None = None,
    series_uid: str | None = None,
    names: tuple[str, ...] = ("ap", "lateral"),
    shape: tuple[int, int] = (512, 512),
    energy_kev: float = 60.0,
    precision: str = "fp32",
    samples_per_ray: int = 1024,
    device: str = "cuda",
) -> tuple[Any, Any, Any, dict[str, Any]]:
    """Return device optical depth, original volume, geometry and resolved settings."""
    from dpt import project

    volume, attenuation = load_attenuation(
        path, units=units, series_uid=series_uid, energy_kev=energy_kev
    )
    geometry = standard_views(attenuation.grid, names, shape=shape)
    nz, ny, nx = volume.grid.shape
    centre = volume.grid.grid_to_object(((nx - 1) / 2, (ny - 1) / 2, (nz - 1) / 2))
    physical_settings = []
    for view in geometry.views:
        rows, columns = view.shape
        du, dv = view.spacing_mm
        detector_centre = tuple(
            view.origin_mm[i] + (columns - 1) * du * view.u[i] / 2 + (rows - 1) * dv * view.v[i] / 2
            for i in range(3)
        )
        physical_settings.append(
            {
                "source_to_volume_centre_mm": math.dist(view.source_mm, centre),
                "source_to_detector_centre_mm": math.dist(view.source_mm, detector_centre),
                "detector_pitch_columns_rows_mm": view.spacing_mm,
                "detector_shape_rows_columns": view.shape,
            }
        )
    images = project(
        attenuation, geometry, device=device, precision=precision, samples_per_ray=samples_per_ray
    )
    settings = {
        "input": str(Path(path).expanduser().resolve()),
        "input_sha256": list(volume.metadata.source_sha256),
        "input_units": volume.metadata.value_units,
        "grid": asdict(volume.grid),
        "coordinate_frame": "RAS",
        "output_quantity": "dimensionless optical depth",
        "view_names": list(names),
        "geometry": [asdict(view) for view in geometry.views],
        "physical_settings": physical_settings,
        "precision": precision,
        "field_storage_dtype": "float32",
        "output_dtype": str(images.dtype),
        "device": str(images.device),
        "sampling": {"rule": "clipped midpoint", "samples_per_ray": samples_per_ray},
        "attenuation_model": (
            {
                "name": "water-scaled monoenergetic HU",
                "energy_kev": energy_kev,
                "mu_water_mm_inverse": 0.02059,
                "water_density_g_cm3": 1.0,
            }
            if volume.metadata.value_units == "HU"
            else {"name": "supplied calibrated attenuation"}
        ),
    }
    return images, volume, geometry, settings


def save_gray_png(path: Path, values: Any, *, low: float, high: float) -> None:
    """Write a viewable 8-bit PNG with an explicit linear display window."""
    import numpy as np

    values = np.asarray(values)
    if values.ndim != 2 or not np.isfinite(values).all() or high <= low:
        raise ValueError("PNG output requires a finite 2-D array and an increasing display window")
    pixels = np.rint(np.clip((values - low) / (high - low), 0, 1) * 255).astype(np.uint8)
    rows, columns = pixels.shape

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
        )

    scanlines = b"".join(b"\0" + row.tobytes() for row in pixels)
    payload = b"\x89PNG\r\n\x1a\n"
    payload += chunk(b"IHDR", struct.pack(">IIBBBBB", columns, rows, 8, 0, 0, 0, 0))
    payload += chunk(b"IDAT", zlib.compress(scanlines)) + chunk(b"IEND", b"")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def save_projection_outputs(images: Any, settings: dict[str, Any], output: Path) -> Any:
    """Download final tensors and save exact arrays, PNGs and resolved settings."""
    import numpy as np

    depth = images.detach().cpu().numpy() if hasattr(images, "detach") else np.asarray(images)
    if (
        depth.ndim != 3
        or len(settings["view_names"]) != depth.shape[0]
        or not np.isfinite(depth).all()
    ):
        raise ValueError("projection output must be finite with shape (views,rows,columns)")
    output.mkdir(parents=True, exist_ok=True)
    transmission = np.exp(-depth)
    np.save(output / "optical_depth.npy", depth, allow_pickle=False)
    np.save(output / "transmission.npy", transmission, allow_pickle=False)
    display_high = max(float(depth.max()), float(np.finfo(depth.dtype).eps))
    for index, name in enumerate(settings["view_names"]):
        # Array row zero is inferior in the standard views; PNG top is superior.
        save_gray_png(
            output / f"{index:02d}-{name}-optical-depth.png",
            depth[index, ::-1],
            low=0,
            high=display_high,
        )
        save_gray_png(
            output / f"{index:02d}-{name}-transmission.png",
            transmission[index, ::-1],
            low=0,
            high=1,
        )
    saved = {
        **settings,
        "saved_at_utc": datetime.now(UTC).isoformat(),
        "png_display": {
            "depth_window": [0, display_high],
            "transmission_window": [0, 1],
            "row_order": "superior at top for standard views",
        },
    }
    (output / "settings.json").write_text(json.dumps(saved, indent=2, allow_nan=False) + "\n")
    return depth


def figure_style() -> Any:
    """Use black Geist Sans on white; cache the official checksum-pinned font."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.font_manager as fm
    import matplotlib.pyplot as plt

    directory = Path.home() / ".cache/dpt/fonts"
    directory.mkdir(parents=True, exist_ok=True)
    archive = directory / "geist-font-1.4.2.zip"
    expected = "e94caf733a6b019a0d4d97da9548c2ca8cbe4b2703ef1d07113a82d9e774cfe5"
    if not archive.exists():
        url = "https://github.com/vercel/geist-font/releases/download/1.4.2/geist-font-1.4.2.zip"
        with urlopen(url, timeout=120) as response:
            content = response.read()
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError("Official Geist font archive checksum mismatch")
        archive.write_bytes(content)
    if hashlib.sha256(archive.read_bytes()).hexdigest() != expected:
        raise ValueError("Cached Geist font archive checksum mismatch")
    with zipfile.ZipFile(archive) as fonts:
        for weight in ("Regular", "Bold"):
            filename = f"Geist-{weight}.ttf"
            path = directory / filename
            path.write_bytes(fonts.read("geist-font-1.4.2/fonts/Geist/ttf/" + filename))
            fm.fontManager.addfont(path)
        (directory / "OFL.txt").write_bytes(fonts.read("geist-font-1.4.2/OFL.txt"))
    family = fm.FontProperties(fname=directory / "Geist-Regular.ttf").get_name()
    plt.rcParams.update(
        {
            "font.family": family,
            "font.size": 10,
            "text.color": "black",
            "axes.labelcolor": "black",
            "axes.edgecolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "svg.fonttype": "path",
            "savefig.dpi": 300,
        }
    )
    return plt


def save_projection_figure(
    volume: Any, depth: Any, geometry: Any, names: tuple[str, ...], output: Path
) -> None:
    """Save the original CT plane and computed projections with physical axes."""
    import numpy as np

    plt = figure_style()
    figure, axes = plt.subplots(
        1, len(names) + 1, figsize=(4 * (len(names) + 1), 4.6), layout="constrained", squeeze=False
    )
    axes = axes[0]
    nz, ny, nx = volume.grid.shape
    sx, sy, _ = volume.grid.spacing_mm
    index = nz // 2
    hu = volume.metadata.value_units == "HU"
    image = axes[0].imshow(
        volume.values[index],
        origin="lower",
        extent=(-sx / 2, (nx - 0.5) * sx, -sy / 2, (ny - 0.5) * sy),
        cmap="gray",
        vmin=-1000 if hu else None,
        vmax=1000 if hu else None,
    )
    axes[0].set(title=f"Input slice {index}", xlabel="Grid x (mm)", ylabel="Grid y (mm)")
    figure.colorbar(image, ax=axes[0], location="bottom", label=volume.metadata.value_units)
    for i, name in enumerate(names):
        panel = geometry.views[i]
        rows, columns = panel.shape
        du, dv = panel.spacing_mm
        image = axes[i + 1].imshow(
            depth[i],
            origin="lower",
            cmap="gray",
            vmin=0,
            vmax=float(np.max(depth)),
            extent=(-columns * du / 2, columns * du / 2, -rows * dv / 2, rows * dv / 2),
        )
        axes[i + 1].set(title=name.upper(), xlabel="Detector u (mm)", ylabel="Detector v (mm)")
    figure.colorbar(
        image, ax=list(axes[1:]), location="bottom", label="Optical depth (dimensionless)"
    )
    for suffix in ("png", "svg", "pdf"):
        figure.savefig(output / f"ct-projections.{suffix}", bbox_inches="tight")
    plt.close(figure)

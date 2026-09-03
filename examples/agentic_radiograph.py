"""Match a CT's viewing pose to a radiograph using approximate camera geometry.

The reference radiograph can depict another patient. The calling agent chooses
camera settings and a bounded fit; each evaluated pose and image score is saved.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import math
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from dpt.contracts import ContractError, finite_scalar, integer
from dpt.geometry import BatchedGeometry, ConeBeamIntrinsics, RigidTransform, compose_pose
from dpt.rendering import load_attenuation
from dpt.volumes import GridSpec

_PITCH_SCALES = (1.0, 0.5, 0.65, 0.85, 1.2, 1.5)


def load_radiograph(path: Path) -> tuple[np.ndarray, dict]:
    """Read one raster without reducing a 16-bit image to an 8-bit display."""
    content = path.read_bytes()
    suffix = path.suffix.lower()
    conversion = "grayscale values"
    if suffix == ".npy":
        values = np.load(io.BytesIO(content), allow_pickle=False)
    elif suffix in (".dcm", ".dicom"):
        import pydicom
        from pydicom.pixels import apply_modality_lut

        dataset = pydicom.dcmread(io.BytesIO(content))
        if int(getattr(dataset, "NumberOfFrames", 1)) != 1:
            raise ContractError("radiograph DICOM must contain one frame")
        if int(getattr(dataset, "SamplesPerPixel", 1)) != 1:
            raise ContractError("radiograph DICOM must be monochrome")
        values = apply_modality_lut(dataset.pixel_array, dataset)
        conversion = "DICOM modality LUT; both brightness polarities scored"
    elif suffix in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
        from PIL import Image, ImageOps

        with Image.open(io.BytesIO(content)) as image:
            if getattr(image, "n_frames", 1) != 1:
                raise ContractError("radiograph image must contain one frame")
            image = ImageOps.exif_transpose(image)
            if image.mode in ("RGB", "RGBA", "P", "CMYK", "LA"):
                values = np.asarray(image.convert("RGB"), dtype=np.float64)
                values = values @ np.array([0.2126, 0.7152, 0.0722])
                conversion = "RGB luminance (0.2126 R + 0.7152 G + 0.0722 B)"
            else:
                values = np.asarray(image)
    else:
        raise ContractError("radiograph must be PNG, JPEG, TIFF, .npy or single-frame DICOM")
    if not np.issubdtype(values.dtype, np.number) or np.iscomplexobj(values):
        raise ContractError("radiograph must contain real numeric values")
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or min(values.shape) < 2:
        raise ContractError("radiograph must be a two-dimensional image with at least 2 x 2 pixels")
    if not np.isfinite(values).all():
        raise ContractError("radiograph values must be finite")
    low, high = float(values.min()), float(values.max())
    if low == high:
        raise ContractError("radiograph must contain intensity variation")
    # Divide before subtraction to keep finite, extreme float64 inputs finite.
    magnitude = max(abs(low), abs(high))
    scaled = values / magnitude
    normalised = (scaled - low / magnitude) / (high / magnitude - low / magnitude)
    return np.ascontiguousarray(normalised, dtype=np.float32), {
        "filename": path.name,
        "sha256": hashlib.sha256(content).hexdigest(),
        "shape": list(values.shape),
        "input_range": [low, high],
        "conversion": conversion,
        "row_order": "top to bottom",
    }


def resize_radiograph(values: np.ndarray, max_size: int) -> tuple[np.ndarray, float]:
    """Sample a centred square-pixel lattice, using one scale for both axes."""
    integer(max_size, "max_size", minimum=0, maximum=8192)
    if max_size == 1:
        raise ContractError("max_size must be zero or at least two")
    if max_size == 0 or max(values.shape) <= max_size:
        return values.copy(), 1.0
    scale = max(values.shape) / max_size
    shape = tuple(round(count / scale) for count in values.shape)
    if min(shape) < 2:
        raise ContractError("radiograph aspect ratio needs a larger --max-size")
    # Pixel centres remain symmetric about the original image centre. A common
    # sampling step preserves square pixels when rounded dimensions differ.
    yy, xx = [
        np.clip((np.arange(new) - (new - 1) / 2) * scale + (old - 1) / 2, 0, old - 1)
        for old, new in zip(values.shape, shape, strict=True)
    ]
    y0, x0 = np.floor(yy).astype(int), np.floor(xx).astype(int)
    y1, x1 = np.minimum(y0 + 1, values.shape[0] - 1), np.minimum(x0 + 1, values.shape[1] - 1)
    wy, wx = (yy - y0)[:, None], (xx - x0)[None, :]
    resized = (1 - wy) * ((1 - wx) * values[np.ix_(y0, x0)] + wx * values[np.ix_(y0, x1)]) + wy * (
        (1 - wx) * values[np.ix_(y1, x0)] + wx * values[np.ix_(y1, x1)]
    )
    resized = np.ascontiguousarray(resized, dtype=np.float32)
    if float(resized.max()) == float(resized.min()):
        raise ContractError("resized radiograph is constant; increase --max-size")
    return resized, scale


def physical_centre(grid: GridSpec) -> tuple[float, float, float]:
    nz, ny, nx = grid.shape
    return grid.grid_to_object(((nx - 1) / 2, (ny - 1) / 2, (nz - 1) / 2))


def approximate_geometry(
    grid: GridSpec,
    shape: tuple[int, int],
    *,
    raster_scale: float = 1.0,
    sod_mm: float | None = None,
    sdd_mm: float | None = None,
    pixel_spacing_mm: float | None = None,
) -> tuple[BatchedGeometry, dict]:
    """Resolve an AP camera centred on the native RAS CT support."""
    centre = physical_centre(grid)
    support = grid.support
    offsets = np.array(
        [
            np.asarray(grid.grid_to_object((x, y, z))) - centre
            for x in (support[0][0], support[1][0])
            for y in (support[0][1], support[1][1])
            for z in (support[0][2], support[1][2])
        ]
    )
    radius = float(np.linalg.norm(offsets, axis=1).max())
    radial_extent = float(np.linalg.norm(offsets[:, :2], axis=1).max())
    height = float(np.abs(offsets[:, 2]).max())
    sod = max(600.0, 3 * radius) if sod_mm is None else _positive(sod_mm, "sod_mm")
    sdd = 1.5 * sod if sdd_mm is None else _positive(sdd_mm, "sdd_mm")
    if sod <= radius:
        raise ContractError("sod_mm must exceed the CT support radius")
    if sdd <= sod:
        raise ContractError("sdd_mm must exceed sod_mm")
    scale = _positive(raster_scale, "raster_scale")
    if pixel_spacing_mm is None:
        pitch = (
            1.1
            * sdd
            / (sod - radial_extent)
            * max(2 * radial_extent / shape[1], 2 * height / shape[0])
        )
        pitch_source = "CT support with 10 percent image margin"
    else:
        pitch = _positive(pixel_spacing_mm, "pixel_spacing_mm") * scale
        pitch_source = "supplied original-raster detector pitch"
    intrinsics = ConeBeamIntrinsics(sod, sdd, shape, (pitch, pitch), centre_mm=centre)
    panel = intrinsics.view(math.pi)
    # ConeBeamIntrinsics rows increase superiorly. Reverse that basis and shift
    # the first-pixel centre so all arrays use ordinary top-left raster order.
    panel = replace(
        panel,
        origin_mm=tuple(panel.origin_mm[k] + (shape[0] - 1) * pitch * panel.v[k] for k in range(3)),
        v=tuple(-value for value in panel.v),
    )
    return BatchedGeometry((panel,)), {
        "kind": "approximate fixed cone-beam camera",
        "coordinate_frame": "RAS",
        "length_units": "mm",
        "physical_centre_mm": centre,
        "source_object_distance_mm": sod,
        "source_detector_distance_mm": sdd,
        "sod_source": "CT support; minimum 600 mm" if sod_mm is None else "supplied",
        "sdd_source": "1.5 times SOD" if sdd_mm is None else "supplied",
        "pixel_spacing_source": pitch_source,
        "pixel_spacing_original_raster_mm": pitch / scale,
        "pixel_spacing_fit_raster_mm": pitch,
        "original_pixels_per_fit_pixel": scale,
        "shape_rows_columns": shape,
        "row_order": "top to bottom; detector v points towards inferior RAS z",
        "geometry": asdict(panel),
    }


def _positive(value: float, name: str) -> float:
    value = finite_scalar(value, name)
    if value <= 0:
        raise ContractError(f"{name} must be positive")
    return value


def camera_hypotheses(geometry: BatchedGeometry, settings: dict, *, search_pitch: bool):
    """Vary a centred detector's field of view while retaining square pixels."""
    panel = geometry.views[0]
    rows, columns = panel.shape
    centre = tuple(
        panel.origin_mm[k]
        + (columns - 1) * panel.spacing_mm[0] * panel.u[k] / 2
        + (rows - 1) * panel.spacing_mm[1] * panel.v[k] / 2
        for k in range(3)
    )
    hypotheses = []
    for scale in _PITCH_SCALES if search_pitch else (1.0,):
        spacing = tuple(value * scale for value in panel.spacing_mm)
        scaled = replace(
            panel,
            spacing_mm=spacing,
            origin_mm=tuple(
                centre[k]
                - (columns - 1) * spacing[0] * panel.u[k] / 2
                - (rows - 1) * spacing[1] * panel.v[k] / 2
                for k in range(3)
            ),
        )
        # Keep supplied geometry byte-for-byte equivalent at unit scale.
        if scale == 1.0:
            scaled = panel
        hypotheses.append(
            (
                BatchedGeometry((scaled,)),
                {
                    **settings,
                    "id": f"pitch_{round(scale * 1000):04d}",
                    "camera_scale": scale,
                    "pixel_spacing_original_raster_mm": (
                        settings["pixel_spacing_original_raster_mm"] * scale
                    ),
                    "pixel_spacing_fit_raster_mm": settings["pixel_spacing_fit_raster_mm"] * scale,
                    "geometry": asdict(scaled),
                },
            )
        )
    return hypotheses


def seed_candidates(grid: GridSpec, angular_step_deg: int = 45) -> list[dict]:
    """Rotate the CT about its physical centre in a fixed AP camera."""
    integer(angular_step_deg, "angular_step_deg", minimum=15, maximum=180)
    if 360 % angular_step_deg:
        raise ContractError("angular_step_deg must divide 360")
    centre = np.asarray(physical_centre(grid))
    labels = {0: "AP", 90: "right lateral", 180: "PA", 270: "left lateral"}
    result = []
    for angle in range(0, 360, angular_step_deg):
        rotation = compose_pose(RigidTransform(), (0, 0, 0, 0, 0, math.radians(angle)))
        matrix = np.asarray(rotation.rotation).reshape(3, 3)
        pose = RigidTransform(rotation.rotation, tuple(centre - matrix @ centre))
        result.append(
            {
                "id": f"axial_{angle:03d}",
                "view": labels.get(angle, f"axial {angle} degrees"),
                "axial_rotation_degrees": angle,
                "seed_pose_object_to_world": asdict(pose),
                "seed_matrix_object_to_world": pose_matrix(pose),
                "source_direction_in_ct_ras": (matrix.T @ np.array([0, 1, 0])).tolist(),
            }
        )
    return result


def pose_camera_candidates(
    grid: GridSpec, cameras: list[dict], angular_step_deg: int
) -> list[dict]:
    """Evaluate the default AP camera first, then every bounded camera/view pair."""
    poses = seed_candidates(grid, angular_step_deg)
    return [
        {
            **pose,
            "id": pose["id"] if len(cameras) == 1 else f"{pose['id']}_{camera['id']}",
            "camera_id": camera["id"],
            "camera_scale": camera["camera_scale"],
            "pixel_spacing_original_raster_mm": camera["pixel_spacing_original_raster_mm"],
        }
        for camera in cameras
        for pose in poses
    ]


def pose_matrix(pose: RigidTransform) -> list[list[float]]:
    matrix = np.eye(4)
    matrix[:3, :3] = np.asarray(pose.rotation).reshape(3, 3)
    matrix[:3, 3] = pose.translation_mm
    return matrix.tolist()


def centred_torch_pose(seed_rotation, tangent, centre):
    """Return T(c) R_seed exp(xi) T(-c), with FP64 differentiable geometry."""
    import torch

    from dpt.torch.geometry import compose_pose as torch_compose_pose

    anchor = torch.cat((seed_rotation.reshape(9), torch.zeros_like(centre)))
    local = torch_compose_pose(anchor, tangent)
    rotation = local[:9].reshape(3, 3)
    return torch.cat((local[:9], centre + local[9:] - rotation @ centre)).contiguous()


def normalised_cross_correlation(prediction, target):
    """Differentiate signed NCC; a constant prediction receives score zero."""
    import torch

    first = prediction - prediction.mean()
    second = target - target.mean()
    # Clamp each squared norm before sqrt so the derivative is finite at zero.
    epsilon = torch.finfo(prediction.dtype).tiny
    denominator = torch.sqrt(torch.sum(first.square()).clamp_min(epsilon)) * torch.sqrt(
        torch.sum(second.square()).clamp_min(epsilon)
    )
    return (torch.sum(first * second) / denominator).clamp(-1.0, 1.0)


def best_evaluation(trajectory: list[dict]) -> dict:
    """Select the best scored pose, keeping the earliest result when scores tie."""
    if not trajectory:
        raise ContractError("at least one scored pose is required")
    return max(trajectory, key=lambda row: row["score"])


def best_pose_and_camera(trajectory: list[dict], cameras: list[dict]) -> tuple[dict, dict]:
    """Resolve the camera belonging to the winning pose evaluation."""
    best = best_evaluation(trajectory)
    matching = [camera for camera in cameras if camera["id"] == best["camera_id"]]
    if len(matching) != 1:
        raise ContractError("winning pose must identify exactly one camera hypothesis")
    return best, matching[0]


def fit_budget(starts: int, iterations: int, angular_step_deg: int, camera_count: int = 1) -> dict:
    integer(starts, "starts", minimum=1, maximum=24)
    integer(iterations, "iterations", minimum=1, maximum=10000)
    integer(angular_step_deg, "angular_step_deg", minimum=15, maximum=180)
    integer(camera_count, "camera_count", minimum=1, maximum=len(_PITCH_SCALES))
    if 360 % angular_step_deg:
        raise ContractError("angular_step_deg must divide 360")
    count = (360 // angular_step_deg) * camera_count
    if starts > count:
        raise ContractError("starts must not exceed the number of camera/view candidates")
    return {
        "camera_hypotheses": camera_count,
        "angular_candidates_per_camera": 360 // angular_step_deg,
        "seed_evaluations": count,
        "refined_starts": starts,
        "iterations_per_start": iterations,
        "maximum_projection_evaluations": count + starts * (iterations + 1),
        "maximum_backward_evaluations": starts * iterations,
    }


def _load_inputs(
    ct: Path,
    radiograph: Path,
    *,
    series_uid: str | None = None,
    max_size: int = 192,
    sod_mm: float | None = None,
    sdd_mm: float | None = None,
    pixel_spacing_mm: float | None = None,
):
    if not ct.is_dir() and not ct.name.lower().endswith((".nii", ".nii.gz")):
        raise ContractError("CT must be a NIfTI .nii/.nii.gz file or DICOM directory")
    # Check scalar settings and the radiograph before loading a potentially large CT.
    integer(max_size, "max_size", minimum=0, maximum=8192)
    for name, value in (
        ("sod_mm", sod_mm),
        ("sdd_mm", sdd_mm),
        ("pixel_spacing_mm", pixel_spacing_mm),
    ):
        if value is not None:
            _positive(value, name)
    original, image_metadata = load_radiograph(radiograph)
    image, raster_scale = resize_radiograph(original, max_size)
    volume, attenuation = load_attenuation(ct, units="HU", series_uid=series_uid)
    metadata = volume.metadata
    del volume
    geometry, settings = approximate_geometry(
        attenuation.grid,
        image.shape,
        raster_scale=raster_scale,
        sod_mm=sod_mm,
        sdd_mm=sdd_mm,
        pixel_spacing_mm=pixel_spacing_mm,
    )
    cameras = camera_hypotheses(geometry, settings, search_pitch=pixel_spacing_mm is None)
    inspection = {
        "status": "inputs_valid",
        "ct": {
            "filename": ct.name,
            "source_file_sha256": list(metadata.source_sha256),
            "series_uid": series_uid,
            "grid": asdict(attenuation.grid),
            "input_metadata": asdict(metadata),
            "resampled": False,
            "attenuation_model": "max(1 + HU/1000, 0) * 0.02059 mm^-1 at 60 keV",
        },
        "radiograph": {
            **image_metadata,
            "fit_shape": list(image.shape),
            "resampling": "centred bilinear; equal sampling step along both axes",
            "original_pixels_per_fit_pixel": raster_scale,
        },
        "camera": settings,
        "camera_hypotheses": [camera for _, camera in cameras],
        "camera_search": {
            "pitch_scales": [camera["camera_scale"] for _, camera in cameras],
            "rule": "fixed camera within each pose refinement",
            "initial_camera_id": cameras[0][1]["id"],
        },
        "objective": "1 - abs(normalised cross-correlation with projected optical depth)",
        "brightness_polarities": ["bright attenuation", "dark attenuation"],
        "pose_convention": "object_to_world; rotations about the CT physical centre",
    }
    return attenuation, image, geometry, inspection


def inspect_inputs(ct: Path, radiograph: Path, **options) -> dict:
    """Validate source bytes, intensities and physical coordinates on the host."""
    return _load_inputs(ct, radiograph, **options)[3]


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def fit_radiograph(
    ct: Path,
    radiograph: Path,
    output: Path,
    *,
    device: str = "cuda:0",
    starts: int = 3,
    iterations: int = 80,
    angular_step_deg: int = 45,
    samples_per_ray: int = 512,
    learning_rate: float = 0.8,
    translation_bound_mm: float | None = None,
    rotation_bound_deg: float = 45.0,
    **input_options,
) -> dict:
    """Compare coarse views, refine the best starts and save the best evaluated pose."""
    if output.exists():
        raise ContractError("output directory already exists; choose a new directory")
    search_pitch = input_options.get("pixel_spacing_mm") is None
    budget = fit_budget(
        starts, iterations, angular_step_deg, len(_PITCH_SCALES) if search_pitch else 1
    )
    integer(samples_per_ray, "samples_per_ray", minimum=1, maximum=65536)
    _positive(learning_rate, "learning_rate")
    _positive(rotation_bound_deg, "rotation_bound_deg")
    if rotation_bound_deg > 90:
        raise ContractError("rotation_bound_deg must not exceed 90")
    if translation_bound_mm is not None:
        _positive(translation_bound_mm, "translation_bound_mm")
    attenuation, image, geometry, inspection = _load_inputs(ct, radiograph, **input_options)
    import torch

    from dpt.torch import Projector

    target_device = torch.device(device)
    if target_device.type != "cuda" or not torch.cuda.is_available():
        raise ContractError("radiograph fitting requires a CUDA device")
    # Resolve invalid CUDA indices before creating an output directory.
    torch.cuda.get_device_properties(target_device)
    output.mkdir(parents=True, exist_ok=False)
    _write_json(output / "inputs.json", inspection)
    grid = attenuation.grid
    field = torch.tensor(attenuation.values, dtype=torch.float32, device=target_device)
    del attenuation
    target = torch.tensor(image, dtype=torch.float32, device=target_device)
    centre = torch.tensor(physical_centre(grid), dtype=torch.float64, device=target_device)
    diagonal = math.hypot(
        *(n * pitch for n, pitch in zip(reversed(grid.shape), grid.spacing_mm, strict=True))
    )
    translation_bound = 0.25 * diagonal if translation_bound_mm is None else translation_bound_mm
    scales = torch.tensor(
        [0.01 * diagonal] * 3 + [0.01] * 3, dtype=torch.float64, device=target_device
    )
    bounds = (
        torch.tensor(
            [translation_bound] * 3 + [math.radians(rotation_bound_deg)] * 3,
            dtype=torch.float64,
            device=target_device,
        )
        / scales
    )
    cameras = camera_hypotheses(geometry, inspection["camera"], search_pitch=search_pitch)
    camera_settings = [camera for _, camera in cameras]
    projectors = {
        camera["id"]: Projector(
            grid=grid,
            geometry=camera_geometry,
            precision="fp32",
            samples_per_ray=samples_per_ray,
        )
        for camera_geometry, camera in cameras
    }
    candidates = pose_camera_candidates(grid, camera_settings, angular_step_deg)
    trajectory = []
    best_depth, best_score = None, -1.0
    backwards = 0

    def evaluate(candidate, parameters, stage, iteration):
        nonlocal best_depth, best_score
        rotation = torch.tensor(
            candidate["seed_pose_object_to_world"]["rotation"],
            dtype=torch.float64,
            device=target_device,
        )
        tangent = parameters * scales
        packed = centred_torch_pose(rotation, tangent, centre)
        prediction = projectors[candidate["camera_id"]](field, packed_pose=packed)[0]
        signed = normalised_cross_correlation(prediction, target)
        score = signed.abs()
        if not bool(torch.isfinite(score)) or not bool(torch.isfinite(prediction).all()):
            raise RuntimeError("projection or image score became nonfinite")
        score_value, signed_value = float(score.detach()), float(signed.detach())
        packed_host = packed.detach().cpu().tolist()
        pose = RigidTransform(tuple(packed_host[:9]), tuple(packed_host[9:]))
        record = {
            "evaluation": len(trajectory),
            "candidate_id": candidate["id"],
            "camera_id": candidate["camera_id"],
            "camera_scale": candidate["camera_scale"],
            "stage": stage,
            "iteration": iteration,
            "score": score_value,
            "signed_ncc": signed_value,
            "polarity": "bright attenuation" if signed_value >= 0 else "dark attenuation",
            "centred_tangent_mm_rad": tangent.detach().cpu().tolist(),
            "pose_object_to_world": asdict(pose),
            "matrix_object_to_world": pose_matrix(pose),
        }
        trajectory.append(record)
        if score_value > best_score:
            best_score = score_value
            best_depth = prediction.detach().cpu().numpy().copy()
        return score, record

    with torch.no_grad():
        for candidate in candidates:
            _, record = evaluate(
                candidate, torch.zeros(6, dtype=torch.float64, device=target_device), "seed", 0
            )
            candidate.update(
                seed_score=record["score"],
                seed_signed_ncc=record["signed_ncc"],
                seed_polarity=record["polarity"],
                refined=False,
                best_evaluation=record["evaluation"],
                best_score=record["score"],
                final_score=record["score"],
                final_evaluation=record["evaluation"],
            )
    selected = sorted(candidates, key=lambda item: item["seed_score"], reverse=True)[:starts]
    for candidate in selected:
        parameters = torch.nn.Parameter(torch.zeros(6, dtype=torch.float64, device=target_device))
        optimiser = torch.optim.Adam([parameters], lr=learning_rate)
        candidate["refined"] = True
        candidate["termination"] = "iteration_budget"
        for iteration in range(iterations + 1):
            optimiser.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(iteration < iterations):
                score, record = evaluate(candidate, parameters, "refine", iteration)
            if record["score"] > candidate["best_score"]:
                candidate["best_score"] = record["score"]
                candidate["best_evaluation"] = record["evaluation"]
            candidate["final_score"] = record["score"]
            candidate["final_evaluation"] = record["evaluation"]
            candidate["iterations_completed"] = iteration
            if iteration == iterations:
                break
            (1.0 - score).backward()
            backwards += 1
            if parameters.grad is None or not bool(torch.isfinite(parameters.grad).all()):
                raise RuntimeError("pose derivative is missing or nonfinite")
            if float(parameters.grad.abs().max()) < 1e-10:
                candidate["termination"] = "zero_gradient"
                break
            optimiser.step()
            with torch.no_grad():
                parameters.clamp_(-bounds, bounds)
    # Bind results to the bytes that were actually fitted, including every DICOM slice.
    from dpt.imaging import read_volume

    checked = read_volume(ct, units="HU", series_uid=input_options.get("series_uid"))
    if list(checked.metadata.source_sha256) != inspection["ct"]["source_file_sha256"]:
        raise ContractError("CT input changed during fitting")
    if hashlib.sha256(radiograph.read_bytes()).hexdigest() != inspection["radiograph"]["sha256"]:
        raise ContractError("radiograph input changed during fitting")
    best, best_camera = best_pose_and_camera(trajectory, camera_settings)
    initial_score = trajectory[0]["score"]
    best_seed_score = max(candidate["seed_score"] for candidate in candidates)
    assert best_depth is not None
    settings = {
        "camera": best_camera,
        "camera_hypotheses": camera_settings,
        "budget": budget,
        "angular_step_degrees": angular_step_deg,
        "samples_per_ray": samples_per_ray,
        "precision": "fp32",
        "field_dtype": "float32",
        "pose_dtype": "float64",
        "device": str(target_device),
        "optimiser": "Adam",
        "learning_rate": learning_rate,
        "parameter_scales_mm_rad": scales.cpu().tolist(),
        "tangent_translation_component_bounds_mm": [-translation_bound, translation_bound],
        "tangent_rotation_component_bounds_degrees": [-rotation_bound_deg, rotation_bound_deg],
        "pose_chart": "T(c) R_seed exp(xi) T(-c); xi=(tx,ty,tz,rx,ry,rz)",
    }
    display = save_comparison(output, image, best_depth)
    result = {
        "status": "fit_complete",
        "outcome": "improved" if best["score"] - initial_score > 1e-7 else "no_improvement",
        "objective": inspection["objective"],
        "initial_score": initial_score,
        "initial_camera_id": trajectory[0]["camera_id"],
        "best_seed_score": best_seed_score,
        "initial_loss": 1.0 - initial_score,
        "best_loss": 1.0 - best["score"],
        "loss_reduction": best["score"] - initial_score,
        "refinement_loss_reduction": best["score"] - best_seed_score,
        "score": best["score"],
        "signed_ncc": best["signed_ncc"],
        "polarity": best["polarity"],
        "candidate_id": best["candidate_id"],
        "camera_id": best["camera_id"],
        "camera_scale": best["camera_scale"],
        "best_evaluation": best["evaluation"],
        "best_iteration": best["iteration"],
        "pose_object_to_world": best["pose_object_to_world"],
        "matrix_object_to_world": best["matrix_object_to_world"],
        "coordinate_frame": "RAS",
        "length_units": "mm",
        "camera": best_camera,
        "projection_evaluations": len(trajectory),
        "backward_evaluations": backwards,
        "budget": budget,
        "input_identity": {
            "ct_source_file_sha256": inspection["ct"]["source_file_sha256"],
            "radiograph_sha256": inspection["radiograph"]["sha256"],
        },
        "display_fit": display,
        "output": str(output.resolve()),
        "files": [
            "inputs.json",
            "settings.json",
            "candidates.json",
            "trajectory.json",
            "result.json",
            "radiograph.npy",
            "optical_depth.npy",
            "display_fit.npy",
            "projection.png",
            "comparison.png",
            "overlay.png",
        ],
    }
    _write_json(output / "settings.json", settings)
    _write_json(output / "candidates.json", candidates)
    _write_json(output / "trajectory.json", trajectory)
    _write_json(output / "result.json", result)
    return result


def save_comparison(output: Path, target: np.ndarray, depth: np.ndarray) -> dict:
    """Fit display gain and offset, then save the target, projection and overlay."""
    from PIL import Image

    target64, depth64 = target.astype(np.float64), depth.astype(np.float64)
    centred = depth64 - depth64.mean()
    variance = float(np.sum(centred * centred))
    gain = float(np.sum(centred * (target64 - target64.mean())) / variance) if variance else 0.0
    offset = float(target64.mean() - gain * depth64.mean())
    fitted = (gain * depth64 + offset).astype(np.float32)
    np.save(output / "radiograph.npy", target, allow_pickle=False)
    np.save(output / "optical_depth.npy", depth, allow_pickle=False)
    np.save(output / "display_fit.npy", fitted, allow_pickle=False)
    observed = np.rint(np.clip(target, 0, 1) * 255).astype(np.uint8)
    predicted = np.rint(np.clip(fitted, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(predicted).save(output / "projection.png")
    Image.fromarray(np.concatenate((observed, predicted), axis=1)).save(output / "comparison.png")
    Image.fromarray(np.stack((observed, predicted, observed), axis=-1)).save(output / "overlay.png")
    return {
        "gain": gain,
        "offset": offset,
        "formula": "display_fit = gain * optical_depth + offset",
        "comparison_order": ["input radiograph", "fitted projection"],
        "overlay_channels": {"red_blue": "input radiograph", "green": "fitted projection"},
        "row_order": "top to bottom",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect", help="inspect CT, radiograph and approximate camera")
    fit = commands.add_parser("fit", help="fit the CT pose to a supplied radiograph")
    for command in (inspect, fit):
        command.add_argument("ct", type=Path, help="NIfTI .nii/.nii.gz file or DICOM directory")
        command.add_argument("radiograph", type=Path, help="PNG, JPEG, TIFF, .npy or DICOM image")
        command.add_argument("--series-uid", help="select a DICOM CT SeriesInstanceUID")
        command.add_argument(
            "--max-size", type=int, default=192, help="longest fit edge; 0 keeps native"
        )
        command.add_argument(
            "--sod-mm", type=float, help="approximate source-to-CT-centre distance"
        )
        command.add_argument("--sdd-mm", type=float, help="approximate source-to-detector distance")
        command.add_argument(
            "--pixel-spacing-mm", type=float, help="square detector pitch of the original raster"
        )
    fit.add_argument("--output", type=Path, required=True)
    fit.add_argument("--device", default="cuda:0")
    fit.add_argument("--starts", type=int, default=3)
    fit.add_argument("--iterations", type=int, default=80)
    fit.add_argument("--angular-step-deg", type=int, default=45)
    fit.add_argument("--samples-per-ray", type=int, default=512)
    fit.add_argument("--learning-rate", type=float, default=0.8)
    fit.add_argument("--translation-bound-mm", type=float)
    fit.add_argument("--rotation-bound-deg", type=float, default=45.0)
    args = parser.parse_args()
    options = vars(args).copy()
    command = options.pop("command")
    try:
        # Keep stdout as one JSON response for a calling agent.
        with contextlib.redirect_stdout(sys.stderr):
            result = (
                inspect_inputs(**options) if command == "inspect" else fit_radiograph(**options)
            )
        print(json.dumps(result, indent=2, allow_nan=False))
    except (ContractError, OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()

"""Shared fixtures and independent oracles: inverse inputs."""

import importlib
import math
from typing import Any, cast

import numpy as np

from dpt.detector import BlurSpec
from dpt.geometry import DetectorGeometry, Matrix3, RigidTransform, Vector3
from dpt.materials import Provenance
from dpt.objectives import ObjectiveSpec
from dpt.recovery import PoseChart
from dpt.spectral import SpectralSpec
from dpt.spectral_recovery import (
    CalibrationBlock,
    SpectralPoseEvaluator,
    SpectralPoseProblem,
    SpectralView,
)
from dpt.validation.projection import integrate_sampled_field
from dpt.volumes import GridSpec

wp: Any = importlib.import_module("warp")


PROVENANCE = Provenance(
    "Independent analytic sampled-field/NumPy regression fixture",
    "0" * 64,
    "Original test fixture; Apache-2.0",
    "Numerical basis coefficients and deterministic expected signals; no physical-material claim",
)


def _reference_pose(parameters, scales, anchor):
    """NumPy Rodrigues exponential, independent of PoseChart/compose_pose."""
    velocity, omega = np.split(np.asarray(parameters) * scales, 2)
    x, y, z = omega
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    theta = np.linalg.norm(omega)
    if theta == 0:
        rotation = np.eye(3)
        displacement = velocity
    else:
        square = skew @ skew
        rotation = np.eye(3) + math.sin(theta) / theta * skew
        rotation += (1 - math.cos(theta)) / theta**2 * square
        left_jacobian = np.eye(3) + (1 - math.cos(theta)) / theta**2 * skew
        left_jacobian += (theta - math.sin(theta)) / theta**3 * square
        displacement = left_jacobian @ velocity
    base_rotation = np.asarray(anchor.rotation).reshape(3, 3)
    return RigidTransform(
        cast(Matrix3, tuple(float(value) for value in (base_rotation @ rotation).reshape(-1))),
        cast(
            Vector3,
            tuple(
                float(value)
                for value in np.asarray(anchor.translation_mm) + base_rotation @ displacement
            ),
        ),
    )


def _reference_paths(fields, grid, geometry, pose, samples):
    # NumPy's weak scalar promotion would otherwise round the reference
    # interpolator's Python-float arithmetic back to FP32 at every sample.
    fields = np.asarray(fields, dtype=np.float64)
    return np.asarray(
        [
            [
                integrate_sampled_field(
                    field.reshape(-1),
                    grid,
                    geometry,
                    pose,
                    row,
                    column,
                    midpoint_samples=samples,
                    validate=False,
                )
                for row in range(geometry.shape[0])
                for column in range(geometry.shape[1])
            ]
            for field in fields
        ]
    )


def _reference_blur(signal, shape, stencil):
    """Explicit zero-padded correlation, including unnormalised edge loss."""
    image = signal.reshape(shape)
    kernel = np.asarray(stencil, dtype=np.float32).reshape(3, 3).astype(float)
    extended = np.pad(image, 1)
    spread = np.zeros(shape)
    for row in range(3):
        for column in range(3):
            spread += (
                kernel[row, column] * extended[row : row + shape[0], column : column + shape[1]]
            )
    return spread.reshape(-1)


def _spectral_fixture(device):
    grid = GridSpec((5, 5, 5), (0.9, 1.1, 0.8), (-1.8, -2.2, -1.6))
    z, y, x = np.indices(grid.shape, dtype=float)
    x, y, z = (x - 2) * 0.9, (y - 2) * 1.1, (z - 2) * 0.8
    fields = np.asarray(
        [
            0.02 + 0.32 * np.exp(-((x + 0.55) ** 2 + 0.7 * (y - 0.4) ** 2 + z**2)),
            0.03 + 0.27 * np.exp(-((x - 0.65) ** 2 / 0.6 + (y + 0.7) ** 2 + (z - 0.3) ** 2 / 0.8)),
        ],
        dtype=np.float32,
    )
    angle = 0.19
    anchor = RigidTransform(
        (math.cos(angle), -math.sin(angle), 0, math.sin(angle), math.cos(angle), 0, 0, 0, 1),
        (0.09, -0.07, 0.04),
    )
    chart = PoseChart(anchor, scales=(0.3, 0.25, 0.2, 0.08, 0.07, 0.09))
    blocks = (
        CalibrationBlock(
            "shared",
            gain=1.1,
            exposure=0.9,
            offset=0.4,
            fit_scale="exposure",
            fit_offset=True,
            scale_step=0.25,
            offset_step=1.7,
            scale_prior_precision=0.4,
            offset_prior_precision=0.3,
        ),
        CalibrationBlock(
            "separate",
            gain=0.95,
            exposure=1.2,
            offset=-0.2,
            fit_scale="gain",
            fit_offset=True,
            scale_step=0.2,
            offset_step=1.3,
            scale_prior_precision=0.5,
            offset_prior_precision=0.2,
        ),
    )
    geometries = (
        DetectorGeometry(
            (-18, 0.13, 0.17), (18, -3.1, -2.7), (0, 1, 0), (0, 0, 1), (1.4, 1.6), (4, 5)
        ),
        DetectorGeometry(
            (0.17, -18, 0.13), (-3.1, 18, -2.7), (1, 0, 0), (0, 0, 1), (1.4, 1.6), (4, 5)
        ),
        DetectorGeometry(
            (0.13, 0.17, -18), (-3.1, -2.7, 18), (1, 0, 0), (0, 1, 0), (1.4, 1.6), (4, 5)
        ),
    )
    coefficients = np.asarray([[0.22, 0.13, 0.07], [0.08, 0.3, 0.2]], dtype=np.float32)
    rng = np.random.default_rng(8632)
    stencil = (0.01, 0.06, 0.02, 0.1, 0.55, 0.08, 0.02, 0.03, 0.01)
    records = []
    for index, geometry in enumerate(geometries):
        records.append(
            {
                "geometry": geometry,
                "weights": rng.uniform(30, 75, (3, geometry.pixels)).astype(np.float32),
                "response": rng.uniform(0.6, 1.4, (3, geometry.pixels)).astype(np.float32),
                "objective_weights": rng.uniform(0.3, 1.3, geometry.pixels).astype(np.float32),
                "view_weight": (0.7, 1.2, 0.9)[index],
                "group": 0 if index < 2 else 1,
                "stencil": stencil if index != 1 else None,
                "reduction": "mean" if index != 1 else "sum",
            }
        )
    samples = 48

    def reference_predictions(parameters):
        pose = _reference_pose(parameters[:6], chart.scales, anchor)
        predictions = []
        for record in records:
            paths = _reference_paths(fields, grid, record["geometry"], pose, samples)
            transmission = np.exp(-(coefficients.astype(float).T @ paths))
            signal = np.sum(
                record["weights"].astype(float) * record["response"].astype(float) * transmission,
                axis=0,
            )
            if record["stencil"] is not None:
                signal = _reference_blur(signal, record["geometry"].shape, record["stencil"])
            group = record["group"]
            block = blocks[group]
            scale_z, offset_z = parameters[6 + 2 * group : 8 + 2 * group]
            # Smooth physical chart reference; no production decode or VJP is called.
            multiplier = block.gain * block.exposure * math.exp(block.scale_step * scale_z)
            predictions.append(multiplier * signal + block.offset + block.offset_step * offset_z)
        return predictions

    truth = np.asarray((-0.22, 0.18, -0.16, -0.21, 0.17, -0.13, -0.12, 0.17, 0.11, -0.16))
    observations = [array.astype(np.float32) for array in reference_predictions(truth)]

    def reference_loss(parameters):
        total = 0.0
        for record, prediction, observation in zip(
            records, reference_predictions(parameters), observations, strict=True
        ):
            terms = 0.5 * record["objective_weights"] * (prediction - observation) ** 2
            value = terms.mean() if record["reduction"] == "mean" else terms.sum()
            total += record["view_weight"] * value
        for group, block in enumerate(blocks):
            scale_z, offset_z = parameters[6 + 2 * group : 8 + 2 * group]
            total += 0.5 * block.scale_prior_precision * scale_z**2
            total += 0.5 * block.offset_prior_precision * offset_z**2
        return float(total)

    def upload(values):
        return wp.array(np.asarray(values).reshape(-1), dtype=wp.float32, device=device)

    spectral = SpectralSpec(
        2,
        3,
        (PROVENANCE, PROVENANCE),
        PROVENANCE,
        PROVENANCE,
        "signal",
        "Analytic two-material three-energy basis with spatial input spectra and response",
        shared_weights=False,
        shared_response=False,
    )
    views = tuple(
        SpectralView(
            f"oracle-{index}",
            record["geometry"],
            spectral,
            upload(coefficients),
            upload(record["weights"]),
            upload(record["response"]),
            upload(observation),
            blocks[record["group"]].name,
            PROVENANCE,
            objective=ObjectiveSpec(reduction=record["reduction"], weighted=True),
            objective_weights=upload(record["objective_weights"]),
            objective_weight=record["view_weight"],
            spatial_response=BlurSpec(4, 5, 3, 3, stencil, PROVENANCE)
            if record["stencil"] is not None
            else None,
        )
        for index, (record, observation) in enumerate(zip(records, observations, strict=True))
    )
    problem = SpectralPoseProblem(grid, 2, upload(fields), PROVENANCE, views, blocks, samples)
    return SpectralPoseEvaluator(problem, chart, device=device), reference_loss


def _tomography_geometries():
    """Six distinct cone views of a small cube, including oblique directions."""
    geometries = []
    for normal in ((1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1)):
        n = np.asarray(normal, dtype=float)
        n /= np.linalg.norm(n)
        helper = np.eye(3)[np.argmin(np.abs(n))]
        u = np.cross(n, helper)
        u /= np.linalg.norm(u)
        v = np.cross(n, u)
        geometries.append(
            DetectorGeometry(
                tuple(-15 * n),
                tuple(15 * n - 2 * u - 2 * v),
                tuple(u),
                tuple(v),
                (1.0, 1.0),
                (5, 5),
            )
        )
    return tuple(geometries)

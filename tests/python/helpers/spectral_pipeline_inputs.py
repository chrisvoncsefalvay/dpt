"""Analytic inputs shared by CPU and CUDA spectral pipeline checks."""

import importlib
from dataclasses import replace
from typing import Any

import numpy as np

from dpt.detector import BlurSpec
from dpt.geometry import DetectorGeometry, RayBundleGeometry, RigidTransform
from dpt.objectives import ObjectiveSpec
from dpt.pipeline import SpectralPipeline
from dpt.projection import ProjectionSpec
from dpt.spectral import SpectralInputs, SpectralSpec
from dpt.spectral_recovery import CalibrationBlock, SpectralPoseProblem, SpectralView
from dpt.volumes import GridSpec

wp: Any = importlib.import_module("warp")


def make_pipeline(
    device: str,
    *,
    precision: str = "float64",
    shared: bool = True,
    ray_bundle: bool = False,
    domain: str = "linear",
    execution: str = "eager",
    channels: str = "serial",
) -> tuple[SpectralPipeline, Any, Any, tuple[Any, ...]]:
    grid = GridSpec((3, 3, 3), (1.0, 1.0, 1.0), (-1.0, -1.0, -1.0))
    geometry = (
        RayBundleGeometry(((-10.0, 0.0, 0.0),), ((10.0, 0.0, 0.0),))
        if ray_bundle
        else DetectorGeometry(
            (-10.0, 0.0, 0.0),
            (10.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 1.0),
            (1, 1),
        )
    )
    coefficients = wp.array([[0.1, 0.2], [0.4, 0.6]], dtype=wp.float32, device=device)
    weights = wp.array([800.0, 1000.0], dtype=wp.float32, device=device)
    response = wp.array([0.75, 0.5], dtype=wp.float32, device=device)
    if not shared:
        weights, response = weights.reshape((2, 1)), response.reshape((2, 1))
    spec = SpectralSpec(
        2,
        2,
        active_weights=True,
        active_response=True,
        active_coefficients=True,
        shared_weights=shared,
        shared_response=shared,
        precision=precision,  # type: ignore[arg-type]
        output_domain=domain,  # type: ignore[arg-type]
    )
    inputs = SpectralInputs.from_arrays(
        spec, coefficients, weights, response, pixels=1, device=device
    )
    pipeline = SpectralPipeline(
        grid,
        geometry,
        ProjectionSpec(48, active_volume=True, precision=precision),  # type: ignore[arg-type]
        inputs,
        execution=execution,  # type: ignore[arg-type]
        channel_execution=channels,  # type: ignore[arg-type]
    )
    host = np.empty((2, *grid.shape), dtype=np.float32)
    host[0], host[1] = 0.25, 0.5
    fields = wp.array(host, dtype=wp.float32, device=device)
    pose = wp.array(RigidTransform().packed(), dtype=wp.float64, device=device)
    return pipeline, fields, pose, (coefficients, weights, response)


def pose_problem(device: str, *, blurred: bool = False) -> SpectralPoseProblem:
    grid = GridSpec((4, 4, 4), (1.0, 1.0, 1.0), (-1.5, -1.5, -1.5))
    fields = wp.array(
        [0.125 + x / 64 + y / 32 + z / 128 for z in range(4) for y in range(4) for x in range(4)],
        dtype=wp.float32,
        device=device,
    )
    geometry = DetectorGeometry(
        (0.3, -0.7, -10.0),
        (-1.1, -0.8, 10.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.7, 0.8),
        (3, 4),
    )
    first = SpectralView(
        "first",
        geometry,
        SpectralSpec(1, 2, output_unit="signal"),
        wp.array((0.02, 0.04), dtype=wp.float32, device=device),
        wp.array((100.0, 200.0), dtype=wp.float32, device=device),
        wp.array((0.75, 0.5), dtype=wp.float32, device=device),
        wp.full(12, 40.0, dtype=wp.float32, device=device),
        "shared",
        objective=ObjectiveSpec(reduction="mean"),
        spatial_response=BlurSpec(3, 4, 1, 3, (0.125, 0.5, 0.375)) if blurred else None,
    )
    second = replace(
        first,
        name="second",
        geometry=replace(geometry, source_mm=(-1.0, 0.7, -8.0)),
        objective_weight=0.25,
    )
    return SpectralPoseProblem(
        grid,
        1,
        fields,
        None,
        (first, second),
        (CalibrationBlock("shared", fit_scale="exposure", fit_offset=True),),
        64,
    )

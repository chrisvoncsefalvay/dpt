"""Absorption layers with independent closed-form expected-score derivatives."""

import math
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Literal

import numpy as np

from dpt.geometry import DetectorGeometry
from dpt.interop.torch_transport import (
    ExpectedTransportEstimator,
    estimate_expected_transport,
    prepare_expected_transport,
)
from dpt.transport.forward import prepare_transport
from dpt.transport.model import MaterialGrid, PlanarDetector, TransportSpec
from dpt.transport.rng import HistoryBatch
from dpt.transport.source import ParallelBeam, PointConeSource, sample_point_cone


def layer_problem(
    torch,
    wp,
    device,
    *,
    estimator: Literal["analogue", "continuous-absorption"] = "continuous-absorption",
    scoring: Literal["photon-count", "energy-kev"] = "photon-count",
    amplitude=0.7,
    flat=False,
    active=("voxel", "material", "amplitude"),
    parameters=None,
    stream=None,
    materials=2,
    layers=4,
    slots=8,
    max_bytes=256 * 1024 * 1024,
    max_crossings=65536,
    detector_columns=1,
):
    ids = tuple(index % materials for index in range(layers))
    absorption = tuple(0.13 + 0.04 * index for index in range(materials))
    voxels = tuple(0.8 + 0.17 * index for index in range(layers))
    density = tuple(1.1 + 0.08 * index for index in range(materials))
    source = ParallelBeam((0.0, 0.0, -1.0), (0.0, 0.0), weight=1.3)
    spec = TransportSpec(
        MaterialGrid((-1.0, -1.0, 0.0), (2.0, 2.0, 0.6), (layers, 1, 1), materials),
        PlanarDetector(
            (-1.0, -1.0), (2.0 / detector_columns, 2.0), (detector_columns, 1), layers * 0.6 + 1.0
        ),
        80.0,
        "independent analytical absorption layers for Torch expected-score adapter",
        estimator=estimator,
        scoring=scoring,
        max_crossings=max_crossings,
    )
    with wp.ScopedStream(stream) if stream is not None else nullcontext():
        native = prepare_transport(
            spec,
            material_ids=wp.array(ids, dtype=wp.int32, device=device),
            absorption=wp.array(absorption, dtype=wp.float64, device=device),
            scattering=wp.zeros(materials, dtype=wp.float64, device=device),
            max_histories=8192,
            device=device,
            stream=stream,
        )
    workspace = prepare_expected_transport(
        native,
        source=source,
        parameters=parameters,
        slots_per_history=slots,
        max_snapshot_bytes=max_bytes,
    )
    voxel = torch.tensor([math.log(value) for value in voxels], dtype=torch.float64, device=device)
    if not flat:
        voxel = voxel.reshape(spec.grid.shape)
    voxel.requires_grad_("voxel" in active)
    material = torch.tensor(
        [math.log(value) for value in density], dtype=torch.float64, device=device
    )
    material.requires_grad_("material" in active)
    source_amplitude = torch.tensor(amplitude, dtype=torch.float64, device=device)
    source_amplitude.requires_grad_("amplitude" in active)
    batches = ExpectedTransportEstimator(HistoryBatch(7315, 0, 11), HistoryBatch(8427, 0, 17))
    return SimpleNamespace(
        workspace=workspace,
        native=native,
        source=source,
        spec=spec,
        ids=ids,
        absorption=absorption,
        log_voxel_density=voxel,
        log_material_density=material,
        source_amplitude=source_amplitude,
        estimator=batches,
    )


def evaluate(problem, *, estimator=None):
    return estimate_expected_transport(
        problem.log_voxel_density,
        problem.log_material_density,
        problem.source_amplitude,
        workspace=problem.workspace,
        estimator=problem.estimator if estimator is None else estimator,
    )


def coordinates(problem):
    return (problem.log_voxel_density, problem.log_material_density, problem.source_amplitude)


def analytic(problem, cotangent=1.0):
    """Only declared layer lengths, coefficients and exponential absorption enter."""
    voxel = problem.log_voxel_density.detach().cpu().numpy()
    material = problem.log_material_density.detach().cpu().numpy()
    amplitude = float(problem.source_amplitude.detach().cpu().item())
    depths = np.array(
        [
            problem.absorption[m] * math.exp(float(material[m])) * math.exp(float(eta)) * 0.6
            for m, eta in zip(problem.ids, voxel.reshape(-1), strict=True)
        ]
    )
    unit = problem.spec.energy_kev if problem.spec.scoring == "energy-kev" else 1.0
    base = problem.source.weight * unit * math.exp(-float(depths.sum()))
    mean = amplitude * base
    voxel_gradient = (-cotangent * depths * mean).reshape(voxel.shape)
    material_gradient = np.array(
        [
            -cotangent * mean * float(depths[np.array(problem.ids) == index].sum())
            for index in range(material.size)
        ]
    )
    parameters = problem.workspace.parameters
    if parameters.active_voxels is not None:
        keep = np.zeros(voxel.size, dtype=bool)
        keep[list(parameters.active_voxels)] = True
        voxel_gradient.reshape(-1)[~keep] = 0.0
    if parameters.active_materials is not None:
        keep = np.zeros(material.size, dtype=bool)
        keep[list(parameters.active_materials)] = True
        material_gradient[~keep] = 0.0
    return mean, (voxel_gradient, material_gradient, np.array(cotangent * base))


def point_source_problem(torch, wp, device):
    p = layer_problem(torch, wp, device)
    p.source = PointConeSource(
        DetectorGeometry(
            (0.0, 0.0, -1.0),
            (0.0, 0.0, p.spec.detector.z_mm),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.2, 0.2),
            (1, 1),
        ),
        weight=1.3,
    )
    p.workspace = prepare_expected_transport(p.native, source=p.source, slots_per_history=8)
    return p


def point_source_reference(problem, wp, cotangent):
    """Conditional closed-form chords for independently replayed source draws."""
    material = problem.log_material_density.detach().exp().cpu().numpy()
    voxel = problem.log_voxel_density.detach().exp().cpu().numpy().reshape(-1)
    tau = np.array(
        [
            problem.absorption[m] * material[m] * voxel[index] * 0.6
            for index, m in enumerate(problem.ids)
        ]
    )
    means = []
    gradient = None
    for batch in (problem.estimator.forward_batch, problem.estimator.derivative_batch):
        context = problem.native.context
        with context.scope():
            position = wp.empty(batch.count, dtype=wp.vec3d, device=context.device)
            direction = wp.empty(batch.count, dtype=wp.vec3d, device=context.device)
            weight = wp.empty(batch.count, dtype=wp.float64, device=context.device)
        sample_point_cone(
            problem.source,
            batch=batch,
            workspace=problem.native,
            out_position=position,
            out_direction=direction,
            out_weight=weight,
            stream=context.stream,
        )
        cosine = direction.numpy()[:, 2]
        depths = tau[:, None] / cosine[None, :]
        base = problem.source.weight * np.exp(-depths.sum(axis=0))
        score = float(problem.source_amplitude.item()) * base
        means.append(score.mean())
        if batch == problem.estimator.derivative_batch:
            per_voxel = (-cotangent * depths * score).mean(axis=1)
            per_material = np.array(
                [per_voxel[np.array(problem.ids) == index].sum() for index in range(material.size)]
            )
            gradient = (
                per_voxel.reshape(problem.log_voxel_density.shape),
                per_material,
                np.array(cotangent * base.mean()),
            )
    assert gradient is not None
    return means[0], gradient

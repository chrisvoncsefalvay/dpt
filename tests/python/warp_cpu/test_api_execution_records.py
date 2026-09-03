"""Actual device storage and numerical compatibility of separated API records."""

from __future__ import annotations

import importlib
from dataclasses import replace
from typing import Any, Literal

import numpy as np
import pytest

from dpt.contracts import ContractError
from dpt.geometry import DetectorGeometry
from dpt.precision import precision_contract
from dpt.projection import (
    ProjectionGradients,
    ProjectionSpec,
    prepare_projection,
    project_optical_depth,
    projection_vjp,
)
from dpt.transmission import TransmissionOutputs, TransmissionSpec, prepare_transmission, transmit
from dpt.transport import (
    HistoryBatch,
    IncompleteHistoryError,
    MaterialGrid,
    PlanarDetector,
    TransportExecution,
    TransportPhysics,
    TransportSpec,
)
from dpt.transport.forward import prepare_transport, trace_histories
from dpt.volumes import GridSpec

wp: Any = importlib.import_module("warp")


@pytest.mark.parametrize("storage", ["float32", "float16", "bfloat16"])
@pytest.mark.parametrize("sampling", ["float64", "mixed_float32"])
def test_projection_storage_preserves_constant_field_integral_and_transpose(
    storage: Literal["float32", "float16", "bfloat16"],
    sampling: Literal["float64", "mixed_float32"],
    warp_device: str,
) -> None:
    grid = GridSpec((2, 3, 4), (2.0, 3.0, 5.0))
    detector = DetectorGeometry(
        (-10.0, 0.0, 0.0), (10.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 1.0), (1, 1)
    )
    spec = ProjectionSpec(
        16,
        active_pose=False,
        active_volume=True,
        precision="float64",
        sampling_precision=sampling,
        storage_precision=storage,
    )
    modern = prepare_projection(grid, detector, spec, device=warp_device)
    field = wp.full(grid.voxels, 0.125, dtype=getattr(wp, storage), device=warp_device)
    pose = wp.array(
        (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
        dtype=wp.float64,
        device=warp_device,
    )
    depth = wp.empty(1, dtype=wp.float64, device=warp_device)
    gradient = wp.empty(grid.voxels, dtype=wp.float64, device=warp_device)
    cotangent = wp.ones(1, dtype=wp.float64, device=warp_device)
    project_optical_depth(field, pose, workspace=modern, out_L=depth)
    # x support is [-1,7] mm, so the constant 1/8 mm^-1 field has unit depth.
    assert depth.numpy()[0] == pytest.approx(1.0, rel=0.0, abs=2 * np.finfo(np.float64).eps)
    projection_vjp(
        field,
        pose,
        cotangent=cotangent,
        workspace=modern,
        outputs=ProjectionGradients(volume=gradient),
        volume_gradient_precision="float64",
    )
    assert sum(gradient.numpy()) == pytest.approx(8.0, rel=0.0, abs=32 * np.finfo(np.float64).eps)
    assert str(field.dtype.__name__) == precision_contract(spec).storage("field")
    with pytest.raises(ContractError, match="destinations"):
        projection_vjp(
            field,
            pose,
            cotangent=cotangent,
            workspace=modern,
            outputs=ProjectionGradients(volume=gradient),
            out_mu=gradient,
        )
    with pytest.raises(ContractError, match="shape"):
        project_optical_depth(
            field, pose, workspace=modern, out_L=wp.empty(2, dtype=wp.float64, device=warp_device)
        )


def test_transmission_matches_independent_exponential(warp_device: str) -> None:
    modern = prepare_transmission(TransmissionSpec("scalar"), max_pixels=4, device=warp_device)
    depth = wp.array([0.0, 0.1, 1.0, 10.0], dtype=wp.float32, device=warp_device)
    signal = wp.empty(4, dtype=wp.float32, device=warp_device)
    reference = wp.empty_like(signal)
    transmit(depth, 20.0, outputs=TransmissionOutputs(counts=signal), workspace=modern)
    expected = 20.0 * np.exp(-depth.numpy().astype(np.float64))
    np.testing.assert_allclose(signal.numpy(), expected, rtol=1e-7)
    with pytest.raises(ContractError, match="destinations"):
        transmit(
            depth,
            20.0,
            outputs=TransmissionOutputs(counts=signal),
            out_counts=reference,
            workspace=modern,
        )


def test_transport_selected_execution_matches_trace_and_preserves_fatal_guard(
    warp_device: str,
) -> None:
    grid = MaterialGrid((-10.0, -10.0, 0.0), (20.0, 20.0, 1.0), (1, 1, 1), 1)
    detector = PlanarDetector((-10.0, -10.0), (20.0, 20.0), (1, 1), 2.0)
    physics = TransportPhysics(grid, detector, 60.0, "supplied-analytic-slab")
    execution = TransportExecution(max_events=1, max_crossings=8, block_dim=64)
    material_ids = wp.zeros(1, dtype=wp.int32, device=warp_device)
    absorption = wp.zeros(1, dtype=wp.float64, device=warp_device)
    scattering = wp.zeros(1, dtype=wp.float64, device=warp_device)
    count = 4
    source = dict(
        positions=wp.full(count, wp.vec3d(0.0, 0.0, -1.0), dtype=wp.vec3d, device=warp_device),
        directions=wp.full(count, wp.vec3d(0.0, 0.0, 1.0), dtype=wp.vec3d, device=warp_device),
        weights=wp.ones(count, dtype=wp.float64, device=warp_device),
        density=wp.ones(1, dtype=wp.float64, device=warp_device),
    )
    model = dict(
        material_ids=material_ids,
        absorption=absorption,
        scattering=scattering,
        max_histories=count,
        device=warp_device,
    )
    modern = prepare_transport(physics, execution=execution, **model)
    legacy = prepare_transport(TransportSpec.from_physics(physics, execution), **model)

    def destinations() -> dict[str, Any]:
        return {
            f"out_{name}": wp.empty(count, dtype=dtype, device=warp_device)
            for name, dtype in (
                ("pixel", wp.int32),
                ("score", wp.float64),
                ("energy", wp.float64),
                ("events", wp.int32),
                ("status", wp.int32),
            )
        }

    output, reference = destinations(), destinations()
    batch = HistoryBatch(19, 0, count)
    trace_histories(workspace=modern, batch=batch, **source, **output)
    trace_histories(workspace=legacy, batch=batch, **source, **reference)
    for name in output:
        np.testing.assert_array_equal(output[name].numpy(), reference[name].numpy())
    np.testing.assert_array_equal(output["out_score"].numpy(), np.ones(count))
    assert modern.execution == execution
    scattering = wp.full(1, 1e9, dtype=wp.float64, device=warp_device)
    guarded = prepare_transport(
        physics,
        execution=replace(execution, max_crossings=64),
        **dict(model, scattering=scattering),
    )
    with pytest.raises(IncompleteHistoryError, match="budget"):
        trace_histories(workspace=guarded, batch=batch, **source, **output)
    with pytest.raises(IncompleteHistoryError, match="budget"):
        guarded.check_status()

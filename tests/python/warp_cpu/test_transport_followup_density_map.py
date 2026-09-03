"""Prepared heterogeneous density scales under the existing coefficient laws."""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false, reportUnknownParameterType=false
# pyright: reportMissingParameterType=false
import importlib
import math
from typing import Any

import numpy as np
import pytest

from dpt.transport.derivatives import TransportParameter, derivative_columns
from dpt.transport.forward import prepare_transport, trace_histories
from dpt.transport.model import MaterialGrid, PlanarDetector, TransportError, TransportSpec
from dpt.transport.rng import HistoryBatch

wp: Any = importlib.import_module("warp")


@pytest.mark.parametrize("estimator", ["analogue", "continuous-absorption"])
def test_voxel_density_map_and_global_scale_derivative_match_layer_integral(estimator, warp_device):
    count = 65536

    def arr(values, dtype=None):
        return wp.array(values, dtype=wp.float64 if dtype is None else dtype, device=warp_device)

    spec = TransportSpec(
        MaterialGrid((-1.0, -1.0, 0.0), (2.0, 2.0, 1.0), (4, 1, 1), 1),
        PlanarDetector((-1.0, -1.0), (2.0, 2.0), (1, 1), 5.0),
        80.0,
        "analytic heterogeneous density layers",
        estimator=estimator,
    )
    voxel_density = arr([0.0, 0.5, 1.0, 2.0])
    workspace = prepare_transport(
        spec,
        material_ids=arr([0] * 4, wp.int32),
        absorption=arr([0.2]),
        scattering=arr([0.0]),
        max_histories=count,
        voxel_density=voxel_density,
        device=warp_device,
    )
    inputs = [
        arr([[0.0, 0.0, -1.0]] * count, wp.vec3d),
        arr([[0.0, 0.0, 1.0]] * count, wp.vec3d),
        arr([1.0] * count),
        arr([1.3]),
    ]
    outputs = {
        f"out_{name}": wp.empty(count, dtype=dtype, device=warp_device)
        for name, dtype in (
            ("pixel", wp.int32),
            ("score", wp.float64),
            ("energy", wp.float64),
            ("events", wp.int32),
            ("status", wp.int32),
        )
    }
    batch = HistoryBatch(89245, 0, count)
    trace_histories(*inputs, batch=batch, workspace=workspace, **outputs)
    depth = 0.2 * 1.3 * 3.5
    transmission = math.exp(-depth)
    standard_error = math.sqrt(transmission * (1 - transmission) / count)
    assert abs(outputs["out_score"].numpy().mean() - transmission) < 7 * standard_error + 1e-13
    derivative = wp.empty(count, dtype=wp.float64, device=warp_device)
    derivative_columns(
        *inputs,
        parameters=(TransportParameter("log-material-density", 0),),
        batch=batch,
        workspace=workspace,
        out_pixel=outputs["out_pixel"],
        out_derivatives=derivative,
        out_status=outputs["out_status"],
    )
    assert (
        abs(derivative.numpy().mean() + depth * transmission) < 7 * depth * standard_error + 1e-13
    )
    if estimator == "continuous-absorption":
        np.testing.assert_allclose(derivative.numpy(), -depth * transmission, rtol=1e-14)
    assert workspace.voxel_density is voxel_density


@pytest.mark.parametrize("bad", [-1.0, math.nan, math.inf])
def test_prepared_density_map_rejects_nonphysical_values(bad, warp_device):
    def arr(values, dtype=None):
        return wp.array(values, dtype=wp.float64 if dtype is None else dtype, device=warp_device)

    spec = TransportSpec(
        MaterialGrid((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (1, 1, 1), 1),
        PlanarDetector((0.0, 0.0), (1.0, 1.0), (1, 1), 2.0),
        80.0,
        "density map guard",
    )
    with pytest.raises(TransportError, match="invalid device"):
        prepare_transport(
            spec,
            material_ids=arr([0], wp.int32),
            absorption=arr([0.1]),
            scattering=arr([0.0]),
            max_histories=1,
            voxel_density=arr([bad]),
            device=warp_device,
        )

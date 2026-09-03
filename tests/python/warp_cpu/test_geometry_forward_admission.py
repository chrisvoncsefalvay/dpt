"""A full-disc miss must still reject an inadmissible regular proposal plan."""

# Warp annotations use runtime DSL expressions.
# pyright: reportInvalidTypeForm=false, reportUnknownMemberType=false
# pyright: reportUntypedFunctionDecorator=false
import math
from dataclasses import replace

import numpy as np
import pytest

from dpt.transport.geometry_forward_proposal_arguments import forward_proposal_plan_arguments
from dpt.transport.geometry_forward_proposals import prepare_forward_area_mixture
from dpt.transport.model import TransportError
from helpers.geometry_forward_native import _pack, _run, _samples, _scene, regular_native, wp


def test_zero_grid_plane_separation_is_fatal_before_full_disc_null(warp_device):
    incoming, spec = _scene()
    incoming = replace(incoming, origin_mm=(0.0, 0.6, 0.5))
    with pytest.raises(TransportError):
        prepare_forward_area_mixture(incoming, spec)
    output, status = _run(incoming, spec, warp_device, [0.75], [0.25], [(0.9, 0.25)])
    assert status[0] == 4
    np.testing.assert_array_equal(output[0], 0.0)


@pytest.mark.parametrize("malformation", ["axis", "position", "coefficient", "zero-slope"])
def test_invalid_plane_row_is_fatal_before_full_disc_null(warp_device, malformation):
    incoming, spec = _scene()
    _, geometry = _pack(spec, warp_device)
    records, _ = forward_proposal_plan_arguments(spec)
    record = records[0]
    plane = regular_native.DetectorProposalPlane()
    plane.coefficients = wp.vec3d(*record.coefficients)
    plane.normal_separation_mm = record.normal_separation_mm
    plane.axis = record.axis
    plane.position_mm = record.position_mm
    if malformation == "axis":
        plane.axis = 3
    elif malformation == "position":
        plane.position_mm = math.nan
    elif malformation == "coefficient":
        plane.coefficients = wp.vec3d(math.nan, 1.0, 0.0)
    else:
        plane.coefficients = wp.vec3d(0.0, 0.0, 1.0)
    output = wp.zeros((1, 4), dtype=wp.float64, device=warp_device)
    status = wp.zeros(1, dtype=wp.int32, device=warp_device)
    wp.launch(
        _samples,
        dim=1,
        inputs=[
            wp.array([plane], dtype=regular_native.DetectorProposalPlane, device=warp_device),
            0,
            1,
            geometry,
            wp.array([incoming.origin_mm], dtype=wp.vec3d, device=warp_device),
            wp.array([incoming.direction], dtype=wp.vec3d, device=warp_device),
            wp.array([0.75], dtype=wp.float64, device=warp_device),
            wp.array([0.25], dtype=wp.float64, device=warp_device),
            wp.array([(0.9, 0.25)], dtype=wp.vec2d, device=warp_device),
            output,
            status,
        ],
        device=warp_device,
    )
    assert status.numpy()[0] == 4
    np.testing.assert_array_equal(output.numpy()[0], 0.0)

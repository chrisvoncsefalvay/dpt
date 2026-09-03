"""Declared local grazing rays under proper poses and complete physical futures."""

# Warp records and the independent addressed reference are explicit providers.
# pyright: reportPrivateUsage=false, reportUnknownMemberType=false
import importlib
import math
from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from dpt.geometry import DetectorGeometry, RigidTransform
from dpt.transport.forward import prepare_transport
from dpt.transport.geometry_histories import GeometryPhysicalMedium, complete_geometry_history
from dpt.transport.geometry_source_jumps import (
    material_source_ray_boundaries,
    sample_material_source_ray_jump,
)
from dpt.transport.geometry_tangents import RigidMotion
from dpt.transport.model import MaterialGrid, OrientedDetector, TransportSpec
from dpt.transport.source import PointConeSource
from dpt.transport.uncollided_geometry import AcquisitionGeometryTangent, SourceGeometryTangent
from helpers.geometry_reference_native import _Addressed, _launch
from helpers.physical_geometry_native import actual_tables as actual_tables

wp: Any = importlib.import_module("warp")


def _prepared_rotated(device, tables, angle, law):
    nodes, absorption, scattering, physical, shells = tables
    if law == "bound-electron-compton":
        physical = replace(physical, bound_compton=(shells,))
    c, s = math.cos(angle), math.sin(angle)
    pose = RigidTransform((c, 0, s, 0, 1, 0, -s, 0, c))
    source_geometry = DetectorGeometry(
        pose.point((0.5, -1, 0)),
        pose.point((0.5, 3, 0)),
        pose.direction((1, 0, 0)),
        pose.direction((0, 0, 1)),
        (0.8, 2),
        (1, 1),
    )
    detector_geometry = replace(
        source_geometry, origin_mm=pose.point((-1, 3, -1)), spacing_mm=(1, 1), shape=(4, 4)
    )
    source = PointConeSource(source_geometry)
    spec = TransportSpec(
        MaterialGrid((0, 0, 0), (1, 1, 1), (1, 1, 1), 1),
        OrientedDetector(detector_geometry),
        80,
        "Pinned actual aluminium on a rotated grazing source chart",
        nodes,
        object_pose=pose,
        scattering_law=law,
        max_events=4,
        max_angle_trials=256,
        scoring="energy-kev",
    )
    workspace = prepare_transport(
        spec,
        material_ids=wp.array([0], dtype=wp.int32, device=device),
        absorption=wp.array(absorption, dtype=wp.float64, device=device),
        scattering=wp.array(scattering, dtype=wp.float64, device=device),
        physical=physical,
        max_histories=8,
        device=device,
    )
    tangent = AcquisitionGeometryTangent(
        source=SourceGeometryTangent(target_motion=RigidMotion(pose.direction((0, 0, 1))))
    )
    record = next(
        row
        for row in material_source_ray_boundaries(source, spec, tangent)
        if row.axis == 2 and row.face == 0
    )
    ray = sample_material_source_ray_jump(source, record, 0.37, spec=spec).ray
    medium = GeometryPhysicalMedium((0,), absorption, scattering, (1,), (1,), physical)
    return workspace, medium, wp.array([1], dtype=wp.float64, device=device), ray


@pytest.mark.parametrize("angle", [0.37, 0.73, 1.2])
@pytest.mark.parametrize("sign", [-1, 1])
def test_rotated_local_prefix_matches_independent_exact_chord(device, actual_tables, angle, sign):
    workspace, medium, density, ray = _prepared_rotated(
        device, actual_tables, angle, "free-electron-compton"
    )
    output, metadata, _, _ = _launch(
        workspace, density, [3], energy=10, side=(2, 0, sign), ray=ray, prefix=True, offset=0
    )
    assert metadata[0, 0] == 0
    if sign < 0:
        assert output[0, 4] == 0
    else:
        # Incident E=10 is an exact table node. The path through local y[0,1]
        # has length hypot(4, .8*(u-.5))/4, independently of the rigid pose.
        expected = medium.extinction(10, workspace.spec)[0] * math.hypot(4, 0.8 * (0.37 - 0.5)) / 4
        assert output[0, 4] == pytest.approx(expected, rel=1e-13, abs=1e-13)


@pytest.mark.parametrize("angle", [0.37, 1.2])
@pytest.mark.parametrize(
    "law", ["isotropic-elastic", "free-electron-compton", "bound-electron-compton"]
)
def test_rotated_attached_complete_physical_continuations_match_addressed_reference(
    device, actual_tables, angle, law
):
    workspace, medium, density, ray = _prepared_rotated(device, actual_tables, angle, law)
    histories = list(range(8))
    output, metadata, _, seed = _launch(
        workspace, density, histories, energy=10, side=(2, 0, 1), ray=ray, offset=0
    )
    assert np.all(metadata[:, 0] == 0)
    for index, history in enumerate(histories):
        expected = complete_geometry_history(
            ray,
            10,
            workspace.spec,
            medium,
            _Addressed(workspace.spec, history, 2, 0),
            lineage=(history,),
            cotangent=seed,
            initial_material_side=(2, 0, 1),
        )
        np.testing.assert_allclose(
            output[index], (expected.response, expected.energy_kev), rtol=3e-12, atol=3e-12
        )
        assert metadata[index, 1] == len(expected.prefixes)
    assert np.any(metadata[:, 1] > 0)


@pytest.mark.parametrize(
    "origin,direction", [((0.5, -1, 0.01), (0, 1, 0)), ((0.5, -1, 0), (0, math.sqrt(0.99), 0.1))]
)
def test_native_attached_side_rejects_wrong_nominal_constraint(
    device, actual_tables, origin, direction
):
    from dpt.transport.geometry_tangents import RayTangent

    workspace, _, density, _ = _prepared_rotated(
        device, actual_tables, 0.37, "free-electron-compton"
    )
    pose = workspace.spec.object_pose
    ray = RayTangent(pose.point(origin), pose.direction(direction))
    _, metadata, _, _ = _launch(
        workspace, density, [3], side=(2, 0, 1), ray=ray, prefix=True, offset=0
    )
    assert metadata[0, 0] == 4

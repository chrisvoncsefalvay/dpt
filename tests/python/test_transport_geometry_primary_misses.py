"""Ordinary backward/parallel primary misses retain complete physical histories."""

from dataclasses import replace
from typing import Literal

import pytest

from dpt.transport.geometry_composed import composed_geometry_observation
from dpt.transport.geometry_histories import GeometryPhysicalMedium, PhysicalGeometryPrefix
from dpt.transport.geometry_tangents import GridGeometryTangent, RayTangent, RigidMotion
from dpt.transport.geometry_terminal import terminal_geometry_observation
from dpt.transport.model import MaterialGrid, PlanarDetector, TransportSpec
from dpt.transport.source import ParallelBeam
from dpt.transport.uncollided_geometry import (
    AcquisitionGeometryTangent,
    _detector_boundary,
    _ray_measure,
)


class FixedCoordinates:
    identity = "explicit independent source/generator coordinates"
    layout: Literal["legacy-v1", "packed-compton-v2", "bound-compton-v3"] = "legacy-v1"

    def uniform(self, lineage, event, domain, lane):
        if domain == 0 and lane == 0:
            return 0.7046880897187134
        return 0.37


def _scene(direction):
    spec = TransportSpec(
        MaterialGrid((0, 0, 0), (1, 1, 1), (1, 1, 1), 1),
        PlanarDetector((-2, -2), (5, 5), (1, 1), 3),
        80,
        "primary miss",
        max_events=4,
        max_angle_trials=2,
    )
    medium = GeometryPhysicalMedium((0,), (0.7,), (0,), (1,), (1,))
    source = ParallelBeam((0.4, 0.5, 2), (0, 0), direction=direction)
    return source, spec, medium


@pytest.mark.parametrize("direction", [(0, 0, -1), (1, 0, 0)])
def test_primary_missed_detector_is_successful_zero(direction):
    source, spec, medium = _scene(direction)
    tangent = AcquisitionGeometryTangent(object=GridGeometryTangent(motion=RigidMotion((1, 0, 0))))
    assert _ray_measure(source, spec, medium.extinction(80, spec), (1,), tangent, (0.37, 0.37)) == (
        0,
        0,
    )
    assert all(
        _detector_boundary(source, spec, tangent, axis, face) is None
        for axis in (0, 1)
        for face in (0, 1)
    )
    result = composed_geometry_observation(
        source,
        (0.37, 0.37),
        80,
        spec,
        medium,
        FixedCoordinates(),
        original_history=0,
        cotangent=(1,),
        tangent=tangent,
    )
    assert result.mean == result.derivative == 0
    assert result.physical_prefixes == int(direction == (0, 0, -1))


def test_absorbed_whole_face_prefix_does_not_prepare_absent_radiative_proposal():
    _source, spec, medium = _scene((0, 0, 1))
    spec = replace(spec, grid=MaterialGrid((0, 0, 0), (1, 1, 1), (1, 1, 2), 2))
    medium = GeometryPhysicalMedium((0, 1), (0.7, 0.7), (0, 0), (1, 1), (1, 1))
    prefix = PhysicalGeometryPrefix(RayTangent((1, 0.5, 0.4), (0, 0, 1)), 80, 1, 0, 0)
    value = terminal_geometry_observation(
        prefix,
        spec,
        medium,
        AcquisitionGeometryTangent(),
        RigidMotion(),
        (1,),
        detector_component=0.37,
        detector_coordinate=(0.37, 0.37),
    )
    assert value.value == value.derivative == value.cutoff_boundary == 0
    assert value.outgoing_channels == 0

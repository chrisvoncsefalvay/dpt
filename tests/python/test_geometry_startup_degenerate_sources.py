"""Classical startup kinks and half-open ownership for collapsed source laws."""

import math
from dataclasses import replace
from decimal import Decimal, localcontext

import pytest

from dpt.transport.geometry_histories import GeometryPhysicalMedium
from dpt.transport.geometry_startup_admission import admit_source_startup_chart
from dpt.transport.geometry_tangents import GridGeometryTangent, RigidMotion
from dpt.transport.model import (
    MaterialGrid,
    PlanarDetector,
    TransportError,
    TransportSpec,
)
from dpt.transport.source import ParallelBeam
from dpt.transport.uncollided_geometry import (
    AcquisitionGeometryTangent,
    SourceGeometryTangent,
)

EXTENTS = ((0.0, 0.0), (0.25, 0.0), (0.0, 0.25))
ACTIVE = AcquisitionGeometryTangent(
    source=SourceGeometryTangent(motion=RigidMotion((0.0, 0.0, 1.0)))
)


def _scene(
    extent: tuple[float, float],
    *,
    lower: tuple[float, float, float] = (0.25, 0.25, 1.0),
    shape: tuple[int, int, int] = (2, 1, 1),
    material_ids: tuple[int, ...] = (0, 1),
    absorption: tuple[float, ...] = (0.7, 0.9),
) -> tuple[ParallelBeam, TransportSpec, GeometryPhysicalMedium]:
    spec = TransportSpec(
        MaterialGrid((0, 0, 0), (1, 1, 1), shape, len(absorption)),
        PlanarDetector((-5, -5), (10, 10), (1, 1), 4),
        80,
        "declared collapsed-source startup law",
        (80,),
        max_events=4,
        max_angle_trials=2,
    )
    medium = GeometryPhysicalMedium(
        material_ids,
        absorption,
        (0.0,) * len(absorption),
        (1.0,) * len(absorption),
        (1.0,) * math.prod(shape),
    )
    return ParallelBeam(lower, extent), spec, medium


def _admit(
    source: ParallelBeam,
    spec: TransportSpec,
    medium: GeometryPhysicalMedium,
    tangent: AcquisitionGeometryTangent = ACTIVE,
) -> None:
    medium.validate(spec)
    admit_source_startup_chart(source, spec, medium, tangent)


@pytest.mark.parametrize("extent", EXTENTS)
def test_collapsed_source_lower_face_has_two_analytic_transmission_slopes(
    extent: tuple[float, float],
) -> None:
    source, spec, medium = _scene(
        extent,
        lower=(0.25, 0.25, 0.0),
        shape=(1, 1, 1),
        material_ids=(0,),
        absorption=(0.7,),
    )
    # Each original ray has T(t)=exp(-mu) on the left and exp[-mu(1-t)]
    # on the right. Collapsing source coordinates does not remove this kink.
    with localcontext() as context:
        context.prec = 90
        mu = Decimal.from_float(0.7)
        left = Decimal(0)
        right = mu * (-mu).exp()
        assert left != right
        assert float(right) == pytest.approx(0.7 * math.exp(-0.7), abs=1e-16)
    with pytest.raises(TransportError, match="discontinuous transverse"):
        _admit(source, spec, medium)


@pytest.mark.parametrize("extent", EXTENTS)
def test_collapsed_source_internal_face_has_distinct_complete_slopes(
    extent: tuple[float, float],
) -> None:
    source, spec, medium = _scene(extent)
    # At z=1, T(t)=exp[-mu_right+mu_left*t] for t<0 and
    # exp[-mu_right+mu_right*t] for t>0, without truncating the future.
    with localcontext() as context:
        context.prec = 90
        first, second = map(Decimal.from_float, (0.7, 0.9))
        assert first * (-second).exp() != second * (-second).exp()
    with pytest.raises(TransportError, match="discontinuous transverse"):
        _admit(source, spec, medium)


@pytest.mark.parametrize("extent", EXTENTS)
def test_collapsed_source_upper_face_has_distinct_escape_slopes(
    extent: tuple[float, float],
) -> None:
    source, spec, medium = _scene(
        extent,
        lower=(0.25, 0.25, 1.0),
        shape=(1, 1, 1),
        material_ids=(0,),
        absorption=(0.7,),
    )
    # T(t)=exp(mu*t) on the left of the upper face and 1 on the right.
    assert Decimal.from_float(0.7) != Decimal(0)
    with pytest.raises(TransportError, match="discontinuous transverse"):
        _admit(source, spec, medium)


@pytest.mark.parametrize("extent", EXTENTS)
@pytest.mark.parametrize("lower", ((0.25, 0.25, 0.5), (2.25, 0.25, 1.0)))
def test_collapsed_interior_and_invisible_launches_remain_admitted(
    extent: tuple[float, float], lower: tuple[float, float, float]
) -> None:
    source, spec, medium = _scene(extent, lower=lower)
    _admit(source, spec, medium)


@pytest.mark.parametrize("extent", EXTENTS)
@pytest.mark.parametrize("transparent", ("zero_rates", "zero_voxel", "equal_materials"))
def test_collapsed_transparent_or_identical_local_generators_are_admitted(
    extent: tuple[float, float], transparent: str
) -> None:
    source, spec, medium = _scene(extent)
    medium = (
        replace(medium, absorption_mm_inverse=(0.0, 0.0))
        if transparent == "zero_rates"
        else replace(medium, voxel_density=(0.0, 0.0))
        if transparent == "zero_voxel"
        else replace(medium, absorption_mm_inverse=(0.7, 0.7))
    )
    _admit(source, spec, medium)


@pytest.mark.parametrize("extent", EXTENTS)
@pytest.mark.parametrize(
    "motion", ("common_translation", "common_rotation", "tangential", "face_spacing")
)
def test_collapsed_regular_relative_or_common_motion_remains_admitted(
    extent: tuple[float, float], motion: str
) -> None:
    source, spec, medium = _scene(extent)
    if motion == "common_translation":
        tangent = replace(ACTIVE, object=GridGeometryTangent(motion=ACTIVE.source.motion))
    elif motion == "common_rotation":
        common = RigidMotion(angular=(1.0, 0.0, 0.0), pivot_mm=(0.0, 0.0, 0.0))
        tangent = AcquisitionGeometryTangent(
            object=GridGeometryTangent(motion=common),
            source=SourceGeometryTangent(motion=common),
        )
    elif motion == "tangential":
        tangent = AcquisitionGeometryTangent(
            source=SourceGeometryTangent(motion=RigidMotion((1.0, 0.0, 0.0)))
        )
    else:
        tangent = replace(ACTIVE, object=GridGeometryTangent(spacing_dot=(0.0, 0.0, 1.0)))
    _admit(source, spec, medium, tangent)


@pytest.mark.parametrize("constant_axis", (0, 1))
@pytest.mark.parametrize("line", (False, True))
@pytest.mark.parametrize("coordinate,should_reject", ((0.5, True), (1.0, False), (2.0, False)))
def test_constant_tangential_coordinate_uses_the_half_open_cell_only(
    constant_axis: int, line: bool, coordinate: float, should_reject: bool
) -> None:
    # Only the lower-left bottom voxel differs. A constant coordinate 1 belongs
    # to the next cell, and the outer upper coordinate 2 belongs to no cell.
    lower = [0.25, 0.25, 1.0]
    lower[constant_axis] = coordinate
    extent = [0.0, 0.0]
    if line:
        extent[1 - constant_axis] = 0.25
    source, spec, medium = _scene(
        (extent[0], extent[1]),
        lower=(lower[0], lower[1], lower[2]),
        shape=(2, 2, 2),
        material_ids=(1, 0, 0, 0, 0, 0, 0, 0),
    )
    if should_reject:
        with pytest.raises(TransportError, match="discontinuous transverse"):
            _admit(source, spec, medium)
    else:
        _admit(source, spec, medium)


@pytest.mark.parametrize(
    "lower,should_reject",
    (((0.0, 0.0, 1.0), True), ((2.0, 0.0, 1.0), False), ((0.0, 2.0, 1.0), False)),
)
def test_pencil_corner_ownership_retains_lower_edges_and_excludes_upper_edges(
    lower: tuple[float, float, float], should_reject: bool
) -> None:
    source, spec, medium = _scene(
        (0.0, 0.0), lower=lower, shape=(2, 2, 2), material_ids=(1, 0, 0, 0, 0, 0, 0, 0)
    )
    if should_reject:
        with pytest.raises(TransportError, match="discontinuous transverse"):
            _admit(source, spec, medium)
    else:
        _admit(source, spec, medium)

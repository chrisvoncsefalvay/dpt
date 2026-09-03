"""Independent geometric recovery metrics; objective decrease is a separate fact.

These CPU reports consume explicit small landmark sets and rigid transforms.
They never estimate a reference pose from the image used by the optimiser.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from dpt.contracts import ContractError, NumericalError
from dpt.geometry import DetectorGeometry, RigidTransform, Vector3, vector3


@dataclass(frozen=True, slots=True)
class PoseError:
    translation_mm: float
    rotation_radians: float
    landmark_rms_mm: float
    landmark_max_mm: float
    landmarks: int


def pose_error(
    recovered: RigidTransform,
    reference: RigidTransform,
    landmarks_object_mm: Sequence[Vector3],
) -> PoseError:
    """e_t=||t_hat-t_ref|| and e_r=angle(R_refᵀ R_hat).

    Report these origin and rotation errors together with RMS and maximum
    landmark displacement ||T_hat x_i-T_ref x_i||. Each transform maps object
    coordinates into world coordinates, with a row-major length-9 rotation and
    length-3 translation in mm. ``landmarks_object_mm`` is a non-empty (N,3)
    sequence of held-out object points in mm. The scalar ``PoseError`` fields
    contain translation, landmark RMS and maximum in mm, rotation in radians
    and the integer landmark count.

    Python binary64 geometry and stable norm reductions are used without
    differentiation. Landmark error includes rotation about the object origin,
    whereas the translation norm measures displacement of that origin alone.

        >>> from dpt.geometry import RigidTransform
        >>> from dpt.validation.recovery import pose_error
        >>> error = pose_error(RigidTransform(translation_mm=(3., 4., 0.)),
        ...                    RigidTransform(), ((0., 0., 0.), (1., 0., 0.)))
        >>> (error.translation_mm, error.rotation_radians, error.landmark_rms_mm)
        (5.0, 0.0, 5.0)

    """
    if not landmarks_object_mm:
        raise ContractError("recovery metrics need at least one declared landmark")
    # Relative rotation R_reference.T @ R_recovered, independent scalar indexing.
    relative = tuple(
        math.fsum(reference.rotation[3 * k + i] * recovered.rotation[3 * k + j] for k in range(3))
        for i in range(3)
        for j in range(3)
    )
    cosine = max(-1.0, min(1.0, (relative[0] + relative[4] + relative[8] - 1.0) / 2.0))
    sine = 0.5 * math.hypot(
        relative[7] - relative[5], relative[2] - relative[6], relative[3] - relative[1]
    )
    distances: list[float] = []
    for landmark in landmarks_object_mm:
        point = vector3(landmark, "landmark")
        a, b = recovered.point(point), reference.point(point)
        distances.append(math.hypot(*(x - y for x, y in zip(a, b, strict=True))))
    translation = math.hypot(
        *(a - b for a, b in zip(recovered.translation_mm, reference.translation_mm, strict=True))
    )
    rms = math.hypot(*(value / math.sqrt(len(distances)) for value in distances))
    if not math.isfinite(translation) or not math.isfinite(rms):
        raise NumericalError("recovery metric exceeds the finite coordinate range")
    return PoseError(translation, math.atan2(sine, cosine), rms, max(distances), len(distances))


def detector_coordinate(geometry: DetectorGeometry, point_world_mm: Vector3) -> tuple[float, float]:
    """(c,r)=(u·(h-o)/s_c, v·(h-o)/s_r) at the ray-plane intersection h.

    ``point_world_mm`` is a length-3 world point and ``geometry`` supplies the
    source s, first pixel centre o and orthonormal column/row directions u and v.
    All positions and pixel spacings use mm. Intersect s+alpha*(point-s) with
    the detector plane and return continuous (column,row) coordinates as two
    Python binary64 values in pixels. The detector origin therefore maps to
    (0,0), rather than a pixel corner.

    Off-detector coordinates retain their residual meaning. A source-parallel
    ray or a non-positive intersection scale is rejected. Geometry and point
    are fixed inputs to this forward reference and no gradient is returned.

        >>> from dpt.geometry import DetectorGeometry
        >>> from dpt.validation.recovery import detector_coordinate
        >>> geometry = DetectorGeometry((0., 0., -10.), (0., 0., 10.),
        ...     (1., 0., 0.), (0., 1., 0.), (1., 1.), (8, 8))
        >>> detector_coordinate(geometry, (1., 2., 0.))
        (2.0, 4.0)

    """
    point = vector3(point_world_mm, "world landmark")
    u, v = geometry.u, geometry.v
    normal = (u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0])
    offset = tuple(o - s for o, s in zip(geometry.origin_mm, geometry.source_mm, strict=True))
    ray = tuple(p - s for p, s in zip(point, geometry.source_mm, strict=True))
    plane_distance = math.fsum(a * b for a, b in zip(offset, normal, strict=True))
    direction = math.fsum(a * b for a, b in zip(ray, normal, strict=True))
    if direction == 0 or not math.isfinite(direction):
        raise ContractError("landmark projection is undefined on a source-parallel plane")
    scale = plane_distance / direction
    if not math.isfinite(scale) or scale <= 0:
        raise ContractError("landmark must project onto the forward detector plane")
    hit = tuple(scale * r - o for r, o in zip(ray, offset, strict=True))
    coordinate = (
        math.fsum(a * b for a, b in zip(hit, u, strict=True)) / geometry.spacing_mm[0],
        math.fsum(a * b for a, b in zip(hit, v, strict=True)) / geometry.spacing_mm[1],
    )
    if not all(map(math.isfinite, coordinate)):
        raise NumericalError("projected landmark exceeds the finite pixel range")
    return coordinate


def reprojection_rms_pixels(
    recovered: RigidTransform,
    reference: RigidTransform,
    geometry: DetectorGeometry,
    landmarks_object_mm: Sequence[Vector3],
) -> float:
    """e_pix = sqrt(sum_i ||pi(T_hat x_i)-pi(T_ref x_i)||²/N).

    ``landmarks_object_mm`` is a non-empty (N,3) sequence in object mm, and both
    rigid transforms map those points into the world frame of ``geometry``.
    Perspective coordinates use (column,row) pixel units, with (0,0) at the
    first detector pixel centre. The result is one Python binary64 RMS Euclidean
    displacement in pixels per landmark, so the denominator is N rather than
    the number of scalar coordinate residuals.

    This forward recovery metric has no active parameters. Landmarks whose
    perspective projection is undefined are rejected by ``detector_coordinate``.

        >>> from dpt.geometry import DetectorGeometry, RigidTransform
        >>> from dpt.validation.recovery import reprojection_rms_pixels
        >>> geometry = DetectorGeometry((0., 0., -10.), (0., 0., 10.),
        ...     (1., 0., 0.), (0., 1., 0.), (1., 1.), (8, 8))
        >>> reprojection_rms_pixels(RigidTransform(translation_mm=(1., 0., 0.)),
        ...     RigidTransform(), geometry, ((0., 0., 0.),))
        2.0

    """
    if not landmarks_object_mm:
        raise ContractError("reprojection metrics need at least one declared landmark")
    differences: list[float] = []
    for landmark in landmarks_object_mm:
        point = vector3(landmark, "landmark")
        a = detector_coordinate(geometry, recovered.point(point))
        b = detector_coordinate(geometry, reference.point(point))
        differences.extend(x - y for x, y in zip(a, b, strict=True))
    result = math.hypot(*(x / math.sqrt(len(landmarks_object_mm)) for x in differences))
    if not math.isfinite(result):
        raise NumericalError("reprojection residual exceeds the finite pixel range")
    return result

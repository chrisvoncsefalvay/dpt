"""Immutable detector plans for native forward-disc importance sampling."""

from __future__ import annotations

from dataclasses import dataclass

from dpt.geometry import Vector3

from .geometry_detector_proposal_arguments import (
    DetectorProposalGeometryArguments,
    DetectorProposalPlaneArguments,
    detector_proposal_plan_arguments,
)
from .model import TransportSpec


@dataclass(frozen=True, slots=True)
class ForwardProposalGeometryArguments:
    regular: DetectorProposalGeometryArguments
    detector_normal: Vector3


def forward_proposal_plan_arguments(
    spec: TransportSpec,
    *,
    redundant_planes: frozenset[tuple[int, int]] = frozenset(),
) -> tuple[tuple[DetectorProposalPlaneArguments, ...], ForwardProposalGeometryArguments]:
    """Prepare fixed detector/grid fields; each collision supplies its own ray.

    The normal retains the detector's incident side, including oriented
    detectors whose normal is the negative of the column/row cross product.
    """
    planes, regular = detector_proposal_plan_arguments(spec, redundant_planes=redundant_planes)
    return planes, ForwardProposalGeometryArguments(regular, spec.detector.normal)


__all__ = ["ForwardProposalGeometryArguments", "forward_proposal_plan_arguments"]

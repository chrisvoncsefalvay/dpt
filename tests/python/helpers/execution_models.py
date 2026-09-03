"""Separated transport model fixture shared by host execution-policy tests."""

from __future__ import annotations

from dpt.transport import MaterialGrid, PlanarDetector, TransportPhysics


def transport_physics() -> TransportPhysics:
    return TransportPhysics(
        MaterialGrid((0.0, 0.0, 0.0), (1.0, 2.0, 3.0), (2, 3, 4), 1),
        PlanarDetector((-3.0, -4.0), (2.0, 3.0), (5, 7), 20.0),
        60.0,
        "supplied-linear-coefficients",
    )

"""Small declared fixed-cell scene for eager/graph replay comparisons."""

import importlib
from typing import Any, Literal

from dpt.geometry import DetectorGeometry
from dpt.transport.forward import prepare_transport
from dpt.transport.model import MaterialGrid, OrientedDetector, TransportSpec
from dpt.transport.simulation import SimulationWorkspace, prepare_simulation
from dpt.transport.source import ParallelBeam, PointConeSource

wp: Any = importlib.import_module("warp")


def graph_scene(
    device: str,
    *,
    count: int = 257,
    estimator: Literal["analogue", "continuous-absorption"] = "analogue",
    strategy: Literal["plain", "grouped"] = "plain",
    cone: bool = False,
    stream: Any = None,
    max_crossings: int = 65536,
) -> tuple[SimulationWorkspace, Any, ParallelBeam | PointConeSource]:
    geometry = DetectorGeometry(
        (0.0, 0.0, -1.0),
        (-1.0, -1.0, 3.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (2.0, 2.0),
        (2, 2),
    )
    spec = TransportSpec(
        MaterialGrid((-2.0, -2.0, 0.0), (4.0, 4.0, 0.5), (2, 1, 1), 1),
        OrientedDetector(geometry),
        80.0,
        "analytic two-cell slab: absorption0.3/mm, scattering0.2/mm; Apache-2.0",
        estimator=estimator,
        max_crossings=max_crossings,
    )
    scope = wp.ScopedStream(stream) if stream is not None else wp.ScopedDevice(device)
    with scope:
        transport = prepare_transport(
            spec,
            material_ids=wp.zeros(2, dtype=wp.int32, device=device),
            absorption=wp.array([0.3], dtype=wp.float64, device=device),
            scattering=wp.array([0.2], dtype=wp.float64, device=device),
            max_histories=count,
            device=device,
            stream=stream,
        )
        density = wp.ones(1, dtype=wp.float64, device=device)
        workspace = prepare_simulation(transport, tally_strategy=strategy)
    source = (
        PointConeSource(geometry, weight=2.0)
        if cone
        else ParallelBeam((-1.5, -1.5, -1.0), (3.0, 3.0), weight=2.0)
    )
    return workspace, density, source

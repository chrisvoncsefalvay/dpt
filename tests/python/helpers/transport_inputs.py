"""Shared fixtures and independent oracles: transport inputs."""

import importlib
import math
from typing import Any, Literal

from dpt.transport.derivatives import TransportParameter
from dpt.transport.forward import prepare_transport
from dpt.transport.inverse import prepare_transport_inverse
from dpt.transport.model import MaterialGrid, PlanarDetector, TransportSpec
from dpt.transport.source import ParallelBeam

wp: Any = importlib.import_module("warp")


def absorbing_oracle(
    estimator: Literal["analogue", "continuous-absorption"],
    capacity: int,
    device: str,
    *,
    local_model: bool = False,
) -> Any:
    spec = TransportSpec(
        MaterialGrid((-1.0, -1.0, 0.0), (2.0, 2.0, 1.0), (1, 1, 1), 1),
        PlanarDetector((-1.0, -1.0), (2.0, 2.0), (1, 1), 2.0),
        80.0,
        "analytic density recovery fixture",
        estimator=estimator,
    )
    w = prepare_transport(
        spec,
        material_ids=wp.array([0], dtype=wp.int32, device=device),
        absorption=wp.array([1.0], dtype=wp.float64, device=device),
        scattering=wp.zeros(1, dtype=wp.float64, device=device),
        max_histories=capacity,
        device=device,
    )
    return prepare_transport_inverse(
        w,
        source=ParallelBeam((0.0, 0.0, -1.0), (0.0, 0.0)),
        parameters=(TransportParameter("log-material-density", 0),),
        observation=wp.array([math.exp(-1.5)], dtype=wp.float64, device=device),
        pixel_weights=wp.ones(1, dtype=wp.float64, device=device),
        base_density=(1.0,),
        local_model=local_model,
    )

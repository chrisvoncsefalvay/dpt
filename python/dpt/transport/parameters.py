"""First-order coefficient-table expected-score derivatives.

A log table node changes its interpolated partial coefficient by the node value
multiplied by its interpolation basis. Analogue paths use the selected partial
collision density and total survival integral. Continuous absorption separates
scattering likelihood from the explicit absorption weight. Coefficient zeros
stay fixed, conditional scattering laws and geometry stay fixed, and no sampled
collision location is differentiated.
"""

from __future__ import annotations

# pyright: reportPrivateUsage=false
from dataclasses import dataclass
from typing import Any, Literal

from dpt._runtime import load_kernels

from .forward import TransportWorkspace, _validate_call
from .model import TransportError
from .rng import HistoryBatch


@dataclass(frozen=True, slots=True)
class CoefficientParameter:
    """Select ``eta = log(mu[material, energy_node])`` for one partial.

    The physical table contains nonnegative coefficients in mm⁻¹ and uses
    material-major storage with energy fastest. ``partial`` selects absorption
    or aggregate scattering, whilst both indices are zero-based. A zero node
    remains fixed under its multiplicative coordinate. Geometry, the source
    law and conditional angular laws remain fixed. With an independently fixed
    coherent partial, an active scattering node must leave a positive
    incoherent channel whenever its coefficient is positive.

    ``CoefficientParameter("absorption", 0, 0)`` selects the first material's
    first absorption node. The derivative of transmission through a unit slab
    at that node is ``-mu * exp(-mu)`` at unit material density.

    """

    partial: Literal["absorption", "scattering"]
    material: int
    energy_node: int

    def __post_init__(self) -> None:
        if self.partial not in ("absorption", "scattering"):
            raise TransportError("coefficient partial must be absorption or scattering")
        for value in (self.material, self.energy_node):
            if type(value) is not int or value < 0:
                raise TransportError("coefficient parameter indices must be nonnegative integers")


def coefficient_derivative_histories(
    positions: Any,
    directions: Any,
    weights: Any,
    density: Any,
    *,
    parameter: CoefficientParameter,
    batch: HistoryBatch,
    workspace: TransportWorkspace,
    out_pixel: Any,
    out_score: Any,
    out_likelihood: Any,
    out_pathwise: Any,
    out_status: Any,
    source_amplitude: float = 1.0,
    stream: Any = None,
    validate: bool = True,
) -> None:
    """Write ``dE[X]/deta = E[X * dlog(p)/deta + dX/deta]`` per history.

    For ``eta = log(mu[m,k])``, the coefficient derivative is the selected
    node value times its linear energy-interpolation basis. Analogue transport
    differentiates the selected collision partial and total survival depth.
    Continuous absorption differentiates scattering likelihood and its explicit
    absorption weight separately. ``out_likelihood`` and ``out_pathwise`` are
    already weighted contributions and their sum estimates the derivative.
    Sampled collision positions are retained by replay.

    Let ``N = batch.count`` and ``M = workspace.spec.grid.materials``.
    ``positions`` and ``directions`` are contiguous Warp ``vec3d`` arrays of
    shape ``(N,)``, with positions in mm and unit directions. ``weights`` is a
    nonnegative FP64 array of shape ``(N,)`` and ``density`` is a positive,
    dimensionless FP64 array of shape ``(M,)``. ``out_pixel`` and ``out_status``
    are writable ``int32`` arrays of shape ``(N,)``. ``out_score``,
    ``out_likelihood`` and ``out_pathwise`` are writable FP64 arrays of shape
    ``(N,)``. Score and derivative units follow the workspace's photon-count
    or keV scoring convention, per unit dimensionless log coefficient.
    All arrays belong to the prepared device and stream and writable storage
    must be disjoint from inputs and other outputs. The call allocates no
    device storage and runs outside a Warp tape. Every original history,
    including misses and absorption, remains in the mean denominator.
    ``validate=False`` requires a subsequent ``workspace.check_status()``.

    A continuous-absorption slab gives an exact explicit-score derivative::

        import math
        import warp as wp
        from dpt.transport import (
            CoefficientParameter, HistoryBatch, MaterialGrid, PlanarDetector,
            TransportSpec, coefficient_derivative_histories,
        )
        from dpt.transport.forward import prepare_transport

        device, n = "cuda:0", 8
        spec = TransportSpec(
            MaterialGrid((-1., -1., 0.), (2., 2., 1.), (1, 1, 1), 1),
            PlanarDetector((-1., -1.), (2., 2.), (1, 1), 2.),
            80., "analytic absorption 0.3/mm", estimator="continuous-absorption",
        )
        transport = prepare_transport(
            spec, material_ids=wp.array([0], dtype=wp.int32, device=device),
            absorption=wp.array([0.3], dtype=wp.float64, device=device),
            scattering=wp.array([0.], dtype=wp.float64, device=device),
            max_histories=n, device=device,
        )
        pixel = wp.empty(n, dtype=wp.int32, device=device)
        status = wp.empty_like(pixel)
        score = wp.empty(n, dtype=wp.float64, device=device)
        likelihood = wp.empty_like(score)
        pathwise = wp.empty_like(score)
        coefficient_derivative_histories(
            wp.array([(0., 0., -1.)] * n, dtype=wp.vec3d, device=device),
            wp.array([(0., 0., 1.)] * n, dtype=wp.vec3d, device=device),
            wp.array([1.] * n, dtype=wp.float64, device=device),
            wp.array([1.], dtype=wp.float64, device=device),
            parameter=CoefficientParameter("absorption", 0, 0),
            batch=HistoryBatch(92761, 0, n), workspace=transport,
            out_pixel=pixel, out_score=score, out_likelihood=likelihood,
            out_pathwise=pathwise, out_status=status,
        )
        assert all(x == 0. for x in likelihood.numpy())
        assert math.isclose(float(pathwise.numpy().mean()),
                            -0.3 * math.exp(-0.3), rel_tol=1e-12)

    """
    if parameter.material >= workspace.spec.grid.materials or parameter.energy_node >= len(
        workspace.spec.energy_nodes
    ):
        raise TransportError("coefficient parameter is outside the prepared table")
    wp = workspace.context.wp
    _validate_call(
        workspace,
        batch,
        positions,
        directions,
        weights,
        density,
        [
            ("out_pixel", out_pixel, wp.int32),
            ("out_score", out_score, wp.float64),
            ("out_likelihood", out_likelihood, wp.float64),
            ("out_pathwise", out_pathwise, wp.float64),
            ("out_status", out_status, wp.int32),
        ],
        source_amplitude,
        stream,
        validate,
    )
    workspace._launch(
        workspace._kernels.validate_coefficient_parameter,
        1,
        [
            workspace.scattering,
            workspace._parameters,
            parameter.material,
            parameter.energy_node,
            int(parameter.partial == "scattering"),
            workspace._status,
        ],
    )
    if validate:
        workspace.check_status()
    kernels = load_kernels("dpt.transport.parameter_kernels")
    active = workspace._kernels.CoefficientRecords()
    active.enabled = 1
    active.material = parameter.material
    active.node = parameter.energy_node
    active.partial = int(parameter.partial == "scattering")
    workspace._launch(
        kernels.coefficient_derivative_histories,
        batch.count,
        [
            positions,
            directions,
            weights,
            density,
            *workspace._model_inputs(),
            wp.uint64(batch.seed),
            wp.uint64(batch.first_history),
            source_amplitude,
            active,
            out_pixel,
            out_score,
            out_likelihood,
            out_pathwise,
            out_status,
            workspace._status,
        ],
    )
    if validate:
        workspace.check_status()

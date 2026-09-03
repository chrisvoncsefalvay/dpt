"""First-order expected-score derivatives with a reviewed parameter boundary.

For analogue sampling and log density eta_m, the path-density derivative is
N_m - integral_path rho_m (mu_a,m(E) + mu_s,m(E)) ds. Conditional interaction-type
and angular/energy distributions do not depend on rho_m, so their *combined*
collision density contributes one per collision. The integral includes every
survival segment, including the final escape. Ignoring that final term biases
transmission derivatives even when no collision occurs.

Continuous absorption instead samples only scattering events. For a fixed
scattering path its likelihood derivative is N_s,m - integral rho_m mu_s,m(E) ds;
the explicit absorption weight contributes -integral rho_m mu_a,m(E) ds.
The resulting expected-score derivative is X times their sum. Conditional
Compton angles and the energy sequence are independent of density at a fixed
path, so energy-dependent coefficients enter each segment's integral without an
extra angular derivative. This assumes fixed coefficient zeros, positive active
densities, fixed source/geometry and sufficient integrability to differentiate
the expectation. No derivative of a realised sampled collision location is used.

This is a likelihood-ratio estimator, not automatic differentiation through a
realised random trace. The same counter addresses permit deterministic replay;
pathwise support/boundary derivatives are not implied by that replay facility.
"""

from __future__ import annotations

# Workspace internals are shared only within this package.
# pyright: reportPrivateUsage=false
from dataclasses import dataclass, field
from typing import Any, Literal

from dpt._runtime import load_kernels, require_no_tape

from ._constants import LOG_SOURCE_AMPLITUDE, SOURCE_AMPLITUDE
from .forward import TransportWorkspace, _validate_call
from .model import TransportError
from .rng import HistoryBatch, require_independent


@dataclass(frozen=True, slots=True)
class DerivativeCapability:
    parameter: str
    supported: bool
    density_term: str
    pathwise_term: str
    support_assumptions: str


CAPABILITIES = (
    DerivativeCapability(
        "log-material-density",
        True,
        "analogue: collisions minus total depth; "
        "continuous absorption: scatterings minus scattering depth",
        "analogue: zero; continuous absorption: minus absorption depth times weighted score",
        "fixed geometry; positive density; fixed partial-coefficient ratios and conditional laws",
    ),
    DerivativeCapability(
        "source-amplitude",
        True,
        "zero",
        "base detector score, including at zero amplitude",
        "fixed source sampling and importance weights; amplitude multiplies measurement only",
    ),
    DerivativeCapability(
        "log-source-amplitude",
        True,
        "zero",
        "amplitude times base detector score",
        "strictly positive amplitude; fixed source distribution and importance weights",
    ),
    DerivativeCapability(
        "moving-geometry",
        False,
        "not implemented",
        "not implemented",
        "requires a boundary-aware estimator for material visibility and detector edges",
    ),
    DerivativeCapability(
        "energy-or-angular-law",
        False,
        "not implemented",
        "not implemented",
        "requires derivatives of the conditional law, coefficient interpolation and energy score",
    ),
    DerivativeCapability(
        "source-distribution",
        False,
        "not implemented",
        "not implemented",
        "sampling-density and source-support terms must be derived before activation",
    ),
)


@dataclass(frozen=True, slots=True)
class TransportParameter:
    kind: Literal["log-material-density", "source-amplitude", "log-source-amplitude"]
    material: int | None = None

    def __post_init__(self) -> None:
        if self.kind == "log-material-density":
            if type(self.material) is not int or self.material < 0:
                raise TransportError("a log-density derivative requires a nonnegative material ID")
        elif self.kind in ("source-amplitude", "log-source-amplitude"):
            if self.material is not None:
                raise TransportError("source amplitude is global and has no material ID")
        else:
            raise TransportError(
                f"unsupported transport derivative: {self.kind!r}; see CAPABILITIES"
            )


FUSED_PARAMETER_LIMIT = 4
# Public preparation accounting is CPU-safe. The private kernel derives its
# 700 signed 32-bit digit bins from IEEE factor/history/attenuation bounds.
TRANSPORT_VJP_BYTES_PER_PARAMETER = 5600


@dataclass(slots=True)
class TransportVJPWorkspace:
    """Bounded first-order expected-signal contraction with caller cotangents."""

    transport: TransportWorkspace
    parameters: tuple[TransportParameter, ...]
    _arrays: dict[str, Any] = field(repr=False)
    _kernels: Any = field(repr=False)

    def check_status(self) -> None:
        self.transport.check_status()

    @property
    def scratch_bytes(self) -> int:
        wp = self.transport.context.wp
        return sum(
            int(value.size) * wp.types.type_size_in_bytes(value.dtype)
            for value in self._arrays.values()
        )


def prepare_transport_vjp(
    workspace: TransportWorkspace,
    *,
    parameters: tuple[TransportParameter, ...],
    max_bytes: int = 256 * 1024 * 1024,
) -> TransportVJPWorkspace:
    """Prepare a resident cotangent contraction for one to four supported parameters."""
    require_no_tape()
    parameters = tuple(parameters)
    if not 1 <= len(parameters) <= FUSED_PARAMETER_LIMIT or len(set(parameters)) != len(parameters):
        raise TransportError("a transport VJP supports one to four distinct parameters")
    if any(
        parameter.material is not None and parameter.material >= workspace.spec.grid.materials
        for parameter in parameters
    ):
        raise TransportError("active material is outside this workspace")
    if type(max_bytes) is not int or max_bytes <= 0:
        raise TransportError("VJP max_bytes must be a positive integer")
    required = TRANSPORT_VJP_BYTES_PER_PARAMETER * len(parameters)
    if required > max_bytes:
        raise TransportError(
            f"transport VJP needs at most {required} bytes, exceeding preparation budget"
        )
    kernels = load_kernels("dpt.transport.contracted_kernels")
    if 8 * kernels.BIN_COUNT != TRANSPORT_VJP_BYTES_PER_PARAMETER:
        raise RuntimeError("transport contraction layout disagrees with preparation accounting")
    context = workspace.context
    wp = context.wp
    with context.scope():
        arrays = {
            "bins": wp.empty(
                kernels.BIN_COUNT * len(parameters), dtype=wp.int64, device=context.device
            )
        }
    return TransportVJPWorkspace(workspace, parameters, arrays, kernels)


def transport_vjp(
    positions: Any,
    directions: Any,
    weights: Any,
    density: Any,
    detector_cotangent: Any,
    *,
    batch: HistoryBatch,
    workspace: TransportVJPWorkspace,
    out_gradient: Any,
    cotangent_batch: HistoryBatch | None = None,
    source_amplitude: float = 1.0,
    stream: Any = None,
    validate: bool = True,
) -> None:
    """Grad[d] = sum_i detector_cotangent[p_i]*Z_i,d/N contracts expected-score derivatives.

    Positions/directions are vec3d (N,) in mm/unit direction, weights is FP64
    (N,), density is positive dimensionless FP64 (materials,) and detector_cotangent
    is signed FP64 (pixels,). The prepared reverse selects D <= 4 material
    log-density or linear/log-source-amplitude coordinates. Out_gradient is FP64
    (D,), with loss units per selected coordinate. Every original history stays
    in N. A fixed detector covector may omit cotangent_batch, while a stochastic
    covector must name an independent complete batch to avoid product bias.
    Original source, score and covector factors combine before attenuation and
    bounded integer bins retain binary exponents through cancellation before
    final FP64 rounding. Geometry and conditional laws stay fixed. This is an
    explicit first-order expected-score estimator, independent of an ambient tape.

    """
    require_no_tape()
    if cotangent_batch is not None:
        require_independent(batch, cotangent_batch)
    transport = workspace.transport
    context = transport.context
    context.assert_stream(stream)
    context.array(
        detector_cotangent,
        "detector_cotangent",
        dtype=context.wp.float64,
        shape=(transport.spec.detector.pixels,),
    )
    context.array(
        out_gradient, "out_gradient", dtype=context.wp.float64, shape=(len(workspace.parameters),)
    )
    _validate_call(
        transport,
        batch,
        positions,
        directions,
        weights,
        density,
        [],
        source_amplitude,
        stream,
        validate,
    )
    context.disjoint(
        [
            *transport._reads(),
            ("positions", positions),
            ("directions", directions),
            ("weights", weights),
            ("density", density),
            ("detector_cotangent", detector_cotangent),
        ],
        [("out_gradient", out_gradient), *workspace._arrays.items()],
    )
    materials = _parameter_materials(transport, workspace.parameters, source_amplitude)
    wp = context.wp
    with context.scope():
        workspace._arrays["bins"].zero_()
    transport._launch(
        workspace._kernels.validate_cotangent,
        transport.spec.detector.pixels,
        [detector_cotangent, transport._status],
    )
    transport._launch(
        workspace._kernels.replay_contracted,
        batch.count,
        [
            positions,
            directions,
            weights,
            density,
            *transport._model_inputs(),
            wp.uint64(batch.seed),
            wp.uint64(batch.first_history),
            wp.vec4i(*materials),
            len(workspace.parameters),
            source_amplitude,
            detector_cotangent,
            workspace._arrays["bins"],
            transport._status,
        ],
    )
    transport._launch(
        workspace._kernels.finish_contracted,
        len(workspace.parameters),
        [workspace._arrays["bins"], batch.count, out_gradient, transport._status],
    )
    if validate:
        workspace.check_status()


def _parameter_materials(
    workspace: TransportWorkspace,
    parameters: tuple[TransportParameter, ...],
    source_amplitude: float,
) -> list[int]:
    """Pack the shared four-coordinate chart without changing derivative semantics."""
    materials: list[int] = []
    for parameter in parameters:
        material = SOURCE_AMPLITUDE if parameter.material is None else parameter.material
        if parameter.kind == "log-source-amplitude":
            if source_amplitude <= 0:
                raise TransportError(
                    "log-amplitude derivatives require strictly positive amplitude"
                )
            material = LOG_SOURCE_AMPLITUDE
        if material >= workspace.spec.grid.materials:
            raise TransportError("active material is outside this workspace")
        materials.append(material)
    materials.extend([SOURCE_AMPLITUDE] * (FUSED_PARAMETER_LIMIT - len(materials)))
    return materials


def derivative_columns(
    positions: Any,
    directions: Any,
    weights: Any,
    density: Any,
    *,
    parameters: tuple[TransportParameter, ...],
    batch: HistoryBatch,
    workspace: TransportWorkspace,
    out_pixel: Any,
    out_derivatives: Any,
    out_status: Any,
    source_amplitude: float = 1.0,
    stream: Any = None,
    validate: bool = True,
) -> None:
    """Out_derivatives[d*N+i] = Z_i,d writes up to four complete-history derivative scores.

    Parameters is a tuple of one to four distinct material log-density or
    linear/log-source-amplitude selectors. Positions/directions are vec3d (N,)
    in mm/unit direction, weights is FP64 (N,) and density is positive FP64
    (materials,) dimensionless scale. Out_pixel/out_status are int32 (N,) and
    out_derivatives is contiguous FP64 (D*N,), column-major by parameter then
    history. Each signed score has output-score units per selected coordinate.
    Material columns include all interaction and survival/absorption terms on
    the complete canonical path and amplitude columns retain the direct zero
    boundary derivative. Conditional laws/geometry remain fixed. The bounded
    vector walk shares Philox addresses across columns without an event tape.
    Reduce each column using all N original histories and use independent complete
    replicates for product-gradient uncertainty.

    """
    parameters = tuple(parameters)
    if not 1 <= len(parameters) <= FUSED_PARAMETER_LIMIT:
        raise TransportError("a fused replay supports one to four parameters")
    if len(set(parameters)) != len(parameters):
        raise TransportError("fused replay parameters must be distinct")
    materials = _parameter_materials(workspace, parameters, source_amplitude)
    wp = workspace.context.wp
    _validate_call(
        workspace,
        batch,
        positions,
        directions,
        weights,
        density,
        [("out_pixel", out_pixel, wp.int32), ("out_status", out_status, wp.int32)],
        source_amplitude,
        stream,
        validate,
    )
    workspace.context.array(
        out_derivatives,
        "out_derivatives",
        dtype=wp.float64,
        shape=(len(parameters) * batch.count,),
    )
    workspace.context.disjoint(
        [
            *workspace._reads(),
            ("positions", positions),
            ("directions", directions),
            ("weights", weights),
            ("density", density),
            ("out_pixel", out_pixel),
            ("out_status", out_status),
        ],
        [("out_derivatives", out_derivatives)],
    )
    workspace._launch(
        workspace._kernels.derivative_columns,
        batch.count,
        [
            positions,
            directions,
            weights,
            density,
            *workspace._model_inputs(),
            wp.uint64(batch.seed),
            wp.uint64(batch.first_history),
            wp.vec4i(*materials),
            len(parameters),
            source_amplitude,
            out_pixel,
            out_derivatives,
            out_status,
            workspace._status,
        ],
    )
    if validate:
        workspace.check_status()


def derivative_histories(
    positions: Any,
    directions: Any,
    weights: Any,
    density: Any,
    *,
    parameter: TransportParameter,
    batch: HistoryBatch,
    workspace: TransportWorkspace,
    out_pixel: Any,
    out_derivative: Any,
    out_status: Any,
    source_amplitude: float = 1.0,
    stream: Any = None,
    validate: bool = True,
) -> None:
    """Z_i=X_i*(K_i-tau_i) estimates dE[X]/d(log density) for the selected material.

    Analogue K counts its interactions and tau is its total extinction depth
    over the complete realised path. Continuous absorption uses scatterings
    minus scattering depth for the likelihood term and adds the explicit
    minus-absorption-depth weight derivative. Source-amplitude parameters use
    the direct weighted score, with the one-sided derivative retained at zero.
    Positions/directions are vec3d (N,) in mm/unit direction, weights is FP64
    (N,) and density is positive dimensionless FP64 (materials,). Out_pixel and
    out_status are int32 (N,), while out_derivative is signed FP64 (N,) in score
    per parameter-coordinate units. Misses/absorption contribute zero and all
    original histories remain in the later mean denominator. Parameter selects
    one log-density or linear/log amplitude coordinate. Model geometry and
    conditional scattering law are fixed. Reusing a batch reproduces its paths
    without requiring a forward call or event tape. Nonlinear product factors
    instead require independent batches. Combine factors before attenuation
    to retain represented derivative tails.

    """
    material = SOURCE_AMPLITUDE if parameter.material is None else parameter.material
    if parameter.kind == "log-source-amplitude":
        if source_amplitude <= 0:
            raise TransportError("log-amplitude derivatives require strictly positive amplitude")
        material = LOG_SOURCE_AMPLITUDE
    if material >= workspace.spec.grid.materials:
        raise TransportError("active material is outside this workspace")
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
            ("out_derivative", out_derivative, wp.float64),
            ("out_status", out_status, wp.int32),
        ],
        source_amplitude,
        stream,
        validate,
    )
    workspace._launch(
        workspace._kernels.derivative_histories,
        batch.count,
        [
            positions,
            directions,
            weights,
            density,
            *workspace._model_inputs(),
            wp.uint64(batch.seed),
            wp.uint64(batch.first_history),
            material,
            source_amplitude,
            out_pixel,
            out_derivative,
            out_status,
            workspace._status,
        ],
    )
    if validate:
        workspace.check_status()

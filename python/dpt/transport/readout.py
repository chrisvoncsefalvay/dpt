"""Scored transport readouts with original-history branch aggregation."""

from __future__ import annotations

# pyright: reportPrivateUsage=false
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from dpt._runtime import load_kernels, require_no_tape

from .forward import TransportWorkspace, _validate_call
from .model import TransportError, _physical
from .parameters import CoefficientParameter
from .rng import HistoryBatch
from .source_spectrum import SpectrumParameter


@dataclass(frozen=True, slots=True)
class TransportReadout:
    """Declare the complete original-history score ``Y[i]`` before reduction.

    ``mode`` selects total, primary or scattered detector contributions.
    Primary means zero real interactions, so Woodcock null candidates do not
    change this classification. ``energy_edges_kev`` defines increasing,
    half-open incident-energy bins in keV. An empty tuple scores all energies
    in one channel. A nonempty tuple excludes energies outside its support.

    ``next_event`` scores uncollided transmission explicitly and samples one
    uniform detector-area connection at each real scattering or fluorescent
    emission. Each connection includes the selected conditional phase density,
    projected area over squared distance and intervening total attenuation.
    Direct escape scores are replaced by these contributions.
    ``split_branches=K`` gives each source descendant weight ``weights[i]/K``
    and distinct collision counters. They share the original source sample and
    are combined within ``Y[i]`` before any mean or variance is estimated.
    Roulette applies to the live continuous-absorption weight at interactions,
    using the declared survival probability and reciprocal survivor weight.

    ``tally_strategy="plain"`` uses FP64 atomics. ``"deterministic"`` uses a
    compensated reduction in increasing original-history order, reproducing
    bits for the same device, input buffers and batch identity. ``precision``
    selects reference FP64 rates or ``"mixed-rates"`` FP32 table interpolation,
    rate sums and density products, retaining FP64 geometry, optical depths,
    attenuation and scoring. Coefficient derivatives require reference rates.

    ``TransportReadout(mode="scatter", next_event=True,
    energy_edges_kev=(20., 50., 100.), tally_strategy="deterministic")`` scores
    two incident-energy channels of the scattered signal.

    Estimator: [Russian roulette and splitting](https://pbr-book.org/3ed-2018/Monte_Carlo_Integration/Russian_Roulette_and_Splitting).
    """

    mode: Literal["total", "primary", "scatter"] = "total"
    energy_edges_kev: tuple[float, ...] = ()
    next_event: bool = False
    split_branches: int = 1
    roulette_threshold: float = 0.0
    roulette_survival: float = 0.5
    tally_strategy: Literal["plain", "deterministic"] = "plain"
    precision: Literal["reference", "mixed-rates"] = "reference"

    def __post_init__(self) -> None:
        object.__setattr__(self, "energy_edges_kev", tuple(self.energy_edges_kev))
        if self.mode not in ("total", "primary", "scatter") or self.tally_strategy not in (
            "plain",
            "deterministic",
        ):
            raise TransportError("unsupported transport readout or tally mode")
        if self.precision not in ("reference", "mixed-rates") or type(self.next_event) is not bool:
            raise TransportError("unsupported transport precision or next-event policy")
        if type(self.split_branches) is not int or not 1 <= self.split_branches <= 4096:
            raise TransportError("source splitting supports one to 4096 branches")
        _physical(self.roulette_threshold, "roulette threshold", nonnegative=True)
        _physical(self.roulette_survival, "roulette survival", positive=True)
        if self.roulette_survival > 1:
            raise TransportError("roulette survival must not exceed one")
        if self.energy_edges_kev:
            if len(self.energy_edges_kev) < 2:
                raise TransportError("energy-binned scoring requires at least two edges")
            for edge in self.energy_edges_kev:
                _physical(edge, "readout energy edge", nonnegative=True)
            if any(
                b <= a
                for a, b in zip(self.energy_edges_kev, self.energy_edges_kev[1:], strict=False)
            ):
                raise TransportError("readout energy edges must increase strictly")

    @property
    def energy_bins(self) -> int:
        """Return one all-energy channel or ``len(energy_edges_kev) - 1`` bins."""
        return max(1, len(self.energy_edges_kev) - 1)


@dataclass(slots=True)
class TransportReadoutWorkspace:
    """Own resident ``(max_histories, energy_bins * pixels)`` score scratch.

    Scratch uses flattened original-history-major, energy-major and then
    detector-pixel storage. The workspace binds one optional coefficient-node
    or spectral-logit parameter. Caller outputs remain separate from this
    mutable scratch and concurrent calls require separate workspaces.
    """

    transport: TransportWorkspace
    spec: TransportReadout
    parameter: CoefficientParameter | SpectrumParameter | None
    _scores: Any = field(repr=False)
    _derivatives: Any = field(repr=False)
    _history_status: Any = field(repr=False)
    _edges: Any = field(repr=False)
    _kernels: Any = field(repr=False)

    @property
    def columns(self) -> int:
        """Return ``energy_bins * pixels``, the length of each output image."""
        return self.spec.energy_bins * self.transport.spec.detector.pixels

    @property
    def scratch_bytes(self) -> int:
        """Return resident FP64 score/derivative/edge and int32 status bytes."""
        return (
            8 * (int(self._scores.size) + int(self._derivatives.size))
            + 4 * int(self._history_status.size)
            + 8 * int(self._edges.size)
        )

    def check_status(self) -> None:
        """Verify persistent completion status before accepting unchecked results."""
        self.transport.check_status()


def prepare_transport_readout(
    workspace: TransportWorkspace,
    *,
    spec: TransportReadout,
    max_bytes: int = 256 * 1024 * 1024,
    parameter: CoefficientParameter | SpectrumParameter | None = None,
) -> TransportReadoutWorkspace:
    """Allocate ``H * (8*C*(1+D) + 4) + 8*E`` resident scratch bytes.

    Here ``H = workspace.max_histories``, ``C = spec.energy_bins * pixels``,
    ``D`` is one when a derivative parameter is supplied and zero otherwise,
    and ``E = len(spec.energy_edges_kev)``. ``max_bytes`` is a positive byte
    budget checked before allocation. Scores and derivatives are FP64 with
    flat shape ``(H*C,)`` and per-history status is ``int32`` with shape
    ``(H,)``. Edge values are FP64 in keV. All storage resides on the transport's
    device and stream and the resulting workspace supports sequential calls
    outside a Warp tape. Model buffers remain immutable for its lifetime.

    ``parameter`` binds a ``CoefficientParameter`` or ``SpectrumParameter``.
    The coefficient's log-node coordinate fixes the other physical partials
    and conditional scattering laws. The spectral logit requires a bound
    ``SourceSpectrum``. A bound parameter requires ``out_derivative_mean`` in
    subsequent ``trace_readout`` calls. Complete descendant contributions are
    combined before estimating original-history moments.

    Prepare a deterministic primary readout for a unit absorbing slab::

        import warp as wp
        from dpt.transport import (
            MaterialGrid, PlanarDetector, TransportReadout, TransportSpec,
            prepare_transport_readout,
        )
        from dpt.transport.forward import prepare_transport

        device, n = "cuda:0", 32
        transport = prepare_transport(
            TransportSpec(
                MaterialGrid((-1., -1., 0.), (2., 2., 1.), (1, 1, 1), 1),
                PlanarDetector((-1., -1.), (2., 2.), (1, 1), 2.),
                80., "analytic absorption 0.3/mm", estimator="continuous-absorption",
            ),
            material_ids=wp.array([0], dtype=wp.int32, device=device),
            absorption=wp.array([0.3], dtype=wp.float64, device=device),
            scattering=wp.array([0.], dtype=wp.float64, device=device),
            max_histories=n, device=device,
        )
        readout = prepare_transport_readout(
            transport,
            spec=TransportReadout(mode="primary", next_event=True,
                                  tally_strategy="deterministic"),
            max_bytes=1024,
        )
        assert readout.columns == 1 and readout.scratch_bytes == 32 * 12

    """
    require_no_tape()
    if spec.split_branches > 1 and workspace.spec.max_angle_trials >= 65536:
        raise TransportError("split counter domains require fewer than 65536 angular trials")
    if isinstance(parameter, CoefficientParameter) and (
        parameter.material >= workspace.spec.grid.materials
        or parameter.energy_node >= len(workspace.spec.energy_nodes)
    ):
        raise TransportError("coefficient parameter is outside the prepared table")
    if isinstance(parameter, CoefficientParameter) and spec.precision != "reference":
        raise TransportError("coefficient derivatives require reference transport rates")
    if isinstance(parameter, SpectrumParameter):
        spectrum = None if workspace.physical is None else workspace.physical.spectrum
        if spectrum is None or parameter.energy_bin >= len(spectrum.energy_kev):
            raise TransportError(
                "spectrum parameter requires a prepared source spectrum and valid bin"
            )
    if isinstance(parameter, CoefficientParameter):
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
        workspace.check_status()
    columns = spec.energy_bins * workspace.spec.detector.pixels
    required = workspace.max_histories * (
        8 * columns * (1 + int(parameter is not None)) + 4
    ) + 8 * len(spec.energy_edges_kev)
    if (
        type(max_bytes) is not int
        or max_bytes <= 0
        or required > max_bytes
        or workspace.max_histories * columns >= 2**31
    ):
        raise TransportError(
            f"transport readout requires {required} bytes within "
            "signed indexing and the preparation budget"
        )
    context = workspace.context
    with context.scope():
        scores = context.wp.empty(
            workspace.max_histories * columns, dtype=context.wp.float64, device=context.device
        )
        derivatives = context.wp.empty(
            workspace.max_histories * columns if parameter is not None else 0,
            dtype=context.wp.float64,
            device=context.device,
        )
        status = context.wp.empty(
            workspace.max_histories, dtype=context.wp.int32, device=context.device
        )
        edges = context.wp.array(
            list(spec.energy_edges_kev), dtype=context.wp.float64, device=context.device
        )
    return TransportReadoutWorkspace(
        workspace,
        spec,
        parameter,
        scores,
        derivatives,
        status,
        edges,
        load_kernels("dpt.transport.readout_kernels"),
    )


def trace_readout(
    positions: Any,
    directions: Any,
    weights: Any,
    density: Any,
    *,
    batch: HistoryBatch,
    workspace: TransportReadoutWorkspace,
    out_mean: Any,
    out_variance_of_mean: Any = None,
    out_derivative_mean: Any = None,
    source_amplitude: float = 1.0,
    stream: Any = None,
    validate: bool = True,
) -> None:
    """Write ``mean[c] = sum_i Y[i,c]/N`` for complete original-history scores.

    ``Y[i,c]`` combines all source descendants and detector connections of
    original history ``i``. For IID original histories, optional marginal
    variance is ``sum_i (Y[i,c] - mean[c])**2 / (N*(N-1))``. It includes zero
    contributions from misses and absorption. This variance requires ``N>=2``.
    Stratified and scrambled-Sobol designs require independent complete-batch
    replicates. Pixel/energy correlations require joint replicate statistics
    for combined objectives.

    Let ``N = batch.count``, ``M = workspace.transport.spec.grid.materials``
    and ``C = workspace.columns``. ``positions`` and ``directions`` are
    contiguous Warp ``vec3d`` arrays of shape ``(N,)`` with mm positions and
    unit directions. ``weights`` is nonnegative FP64 with shape ``(N,)`` and
    ``density`` is positive dimensionless FP64 with shape ``(M,)``. Writable
    ``out_mean`` and optional ``out_variance_of_mean`` and
    ``out_derivative_mean`` are FP64 arrays of shape ``(C,)``. Channels are
    energy-major, with row-major detector pixels within each channel. Mean
    units are photon-count or keV score per original history and variance has
    squared mean units. Derivatives use the same score units per unit bound
    log coefficient or spectral logit.

    Every array belongs to the prepared device and stream and output storage
    must be disjoint from inputs, scratch and other outputs. This call reuses
    resident scratch without allocation and runs outside a Warp tape.
    Deterministic reduction reproduces bits for fixed device, input buffers and
    original identities. ``validate=False`` requires
    ``workspace.check_status()`` before outputs are accepted. Event, crossing,
    angular or arithmetic failures invalidate the entire estimate.

    Forced primary detection yields exact transmission through a unit slab::

        import math
        import warp as wp
        from dpt.transport import (
            HistoryBatch, MaterialGrid, PlanarDetector, TransportReadout,
            TransportSpec, prepare_transport_readout, trace_readout,
        )
        from dpt.transport.forward import prepare_transport

        device, n = "cuda:0", 32
        transport = prepare_transport(
            TransportSpec(
                MaterialGrid((-1., -1., 0.), (2., 2., 1.), (1, 1, 1), 1),
                PlanarDetector((-1., -1.), (2., 2.), (1, 1), 2.),
                80., "analytic absorption 0.3/mm", estimator="analogue",
            ),
            material_ids=wp.array([0], dtype=wp.int32, device=device),
            absorption=wp.array([0.3], dtype=wp.float64, device=device),
            scattering=wp.array([0.], dtype=wp.float64, device=device),
            max_histories=n, device=device,
        )
        readout = prepare_transport_readout(
            transport, spec=TransportReadout(next_event=True,
                                            tally_strategy="deterministic"),
        )
        mean = wp.empty(1, dtype=wp.float64, device=device)
        variance = wp.empty_like(mean)
        trace_readout(
            wp.array([(0., 0., -1.)] * n, dtype=wp.vec3d, device=device),
            wp.array([(0., 0., 1.)] * n, dtype=wp.vec3d, device=device),
            wp.array([1.] * n, dtype=wp.float64, device=device),
            wp.array([1.], dtype=wp.float64, device=device),
            batch=HistoryBatch(92761, 0, n), workspace=readout,
            out_mean=mean, out_variance_of_mean=variance,
        )
        assert math.isclose(float(mean.numpy()[0]), math.exp(-0.3), rel_tol=1e-12)
        assert float(variance.numpy()[0]) < 1e-28

    Connection estimator: [Volume scattering integrators](https://pbr-book.org/4ed/Light_Transport_II_Volume_Rendering/Volume_Scattering_Integrators).
    """
    transport = workspace.transport
    context = transport.context
    wp = context.wp
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
    context.array(out_mean, "out_mean", dtype=wp.float64, shape=(workspace.columns,))
    writes = [
        ("out_mean", out_mean),
        ("history_scores", workspace._scores),
        ("history_status", workspace._history_status),
    ]
    if workspace.parameter is not None:
        if out_derivative_mean is None:
            raise TransportError("prepared parameter readout requires out_derivative_mean")
        context.array(
            out_derivative_mean, "out_derivative_mean", dtype=wp.float64, shape=(workspace.columns,)
        )
        writes.extend(
            [
                ("out_derivative_mean", out_derivative_mean),
                ("history_derivatives", workspace._derivatives),
            ]
        )
    elif out_derivative_mean is not None:
        raise TransportError(
            "prepare a coefficient or spectrum parameter before requesting readout derivatives"
        )
    if (
        batch.sampling is not None
        and workspace.spec.split_branches > batch.sampling.design.max_branches
    ):
        raise TransportError("splitting exceeds the declared full-path sampling branch budget")
    if out_variance_of_mean is not None:
        if batch.sampling is not None:
            raise TransportError(
                "dependent path designs require independent complete-batch replicate variance"
            )
        if batch.count < 2:
            raise TransportError("readout variance requires at least two original histories")
        if batch.source_identity is not None and any(
            term in batch.source_identity.sampler for term in ("stratified", "sobol")
        ):
            raise TransportError(
                "dependent source designs require independent complete-batch replicate variance"
            )
        context.array(
            out_variance_of_mean,
            "out_variance_of_mean",
            dtype=wp.float64,
            shape=(workspace.columns,),
        )
        writes.append(("out_variance_of_mean", out_variance_of_mean))
    context.disjoint(
        [
            *transport._reads(),
            ("positions", positions),
            ("directions", directions),
            ("weights", weights),
            ("density", density),
            ("energy_edges", workspace._edges),
        ],
        writes,
    )
    with context.scope():
        workspace._scores.zero_()
        out_mean.zero_()
        if workspace.parameter is not None:
            workspace._derivatives.zero_()
            cast(Any, out_derivative_mean).zero_()
    records = transport._kernels.ReadoutRecords()
    records.enabled = 1
    records.pixels = transport.spec.detector.pixels
    records.bins = workspace.spec.energy_bins
    records.mode = ("total", "primary", "scatter").index(workspace.spec.mode)
    records.next_event = int(workspace.spec.next_event)
    records.mixed_rates = int(workspace.spec.precision == "mixed-rates")
    records.roulette_threshold = workspace.spec.roulette_threshold
    records.roulette_survival = workspace.spec.roulette_survival
    records.energy_edges = workspace._edges
    records.scores = workspace._scores
    records.derivatives = workspace._derivatives
    if isinstance(workspace.parameter, CoefficientParameter):
        records.derivative_enabled = 1
        records.derivative_kind = 1
        records.coefficient_material = workspace.parameter.material
        records.coefficient_node = workspace.parameter.energy_node
        records.coefficient_partial = int(workspace.parameter.partial == "scattering")
    if isinstance(workspace.parameter, SpectrumParameter):
        assert transport.physical is not None and transport.physical.spectrum is not None
        records.derivative_enabled = 1
        records.derivative_kind = 2
        records.spectrum_bin = workspace.parameter.energy_bin
        records.spectrum_probability = transport.physical.spectrum.probabilities[
            workspace.parameter.energy_bin
        ]
    transport._launch(
        workspace._kernels.trace_readout,
        batch.count,
        [
            positions,
            directions,
            weights,
            density,
            *transport._model_inputs(),
            wp.uint64(batch.seed),
            wp.uint64(batch.first_history),
            source_amplitude,
            workspace.spec.split_branches,
            records,
            workspace._history_status,
            transport._status,
        ],
    )
    if workspace.spec.tally_strategy == "plain":
        transport._launch(
            workspace._kernels.readout_plain,
            batch.count * workspace.columns,
            [workspace._scores, batch.count, workspace.columns, out_mean, transport._status],
        )
    if workspace.spec.tally_strategy == "deterministic" or out_variance_of_mean is not None:
        transport._launch(
            workspace._kernels.readout_deterministic,
            workspace.columns,
            [
                workspace._scores,
                batch.count,
                workspace.columns,
                out_mean,
                out_mean if out_variance_of_mean is None else out_variance_of_mean,
                int(out_variance_of_mean is not None),
                int(workspace.spec.tally_strategy == "deterministic"),
                transport._status,
            ],
        )
    if workspace.parameter is not None:
        if workspace.spec.tally_strategy == "plain":
            transport._launch(
                workspace._kernels.readout_plain,
                batch.count * workspace.columns,
                [
                    workspace._derivatives,
                    batch.count,
                    workspace.columns,
                    out_derivative_mean,
                    transport._status,
                ],
            )
        else:
            transport._launch(
                workspace._kernels.readout_deterministic,
                workspace.columns,
                [
                    workspace._derivatives,
                    batch.count,
                    workspace.columns,
                    out_derivative_mean,
                    out_derivative_mean,
                    0,
                    1,
                    transport._status,
                ],
            )
    if validate:
        workspace.check_status()

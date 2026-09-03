"""Discrete source spectra and expected-score derivatives of spectral logits."""

# Shared preparation and physical-value checks are transport package internals.
# pyright: reportPrivateUsage=false

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from dpt._runtime import load_kernels, require_no_tape
from dpt.materials import Provenance

from .model import TransportError, _physical
from .rng import HistoryBatch

if TYPE_CHECKING:
    from .forward import TransportWorkspace


@dataclass(frozen=True, slots=True)
class SourceSpectrum:
    """Define a discrete emitted-photon energy law and its provenance.

    ``P(E=E_k)=p_k`` with ``p_k>0`` and ``sum(p_k)=1``. A history samples exactly
    one initial support energy using its reserved source counter. The expected
    pixel score is ``sum_k p_k*E[score|E_k]`` and source amplitude carries the
    total emitted measurement. Final photon energy can change during transport,
    while the original sampled index remains available for replay.

    ``energy_kev`` and ``probabilities`` are immutable host tuples of shape
    ``(K,)``. Energies are positive keV values and increase strictly. Probabilities
    are dimensionless, normalised to binary64 accuracy, and remain distinguishable
    in the cumulative distribution. ``provenance`` is either a ``Provenance``
    record or an explicit nonempty source identifier. Preparation uploads the
    support and CDF to the workspace's device without inferring an energy grid,
    emission rate or detector response. Support must lie within coefficient data.

    With fixed energies and logits ``a_k``, ``p_k=exp(a_k)/sum_j exp(a_j)`` and
    ``d log p_selected/d a_k = 1[selected=k]-p_k``. Spectrum replay uses this
    likelihood term with zero explicit-score derivative for the fixed support.
    The class is host data and is not a Warp tape input. Density differentiation
    holds this source law fixed and differentiating support energies requires
    a separately declared physical model.

    Example::

        from dpt.transport import SourceSpectrum
        spectrum = SourceSpectrum(
            (40.0, 80.0), (0.25, 0.75), "calibrated two-line emitted spectrum"
        )
        assert spectrum.cdf == (0.25, 1.0)

    Pass it as ``PhysicalTransport(..., spectrum=spectrum)``. The weights above
    are probabilities, while the independent source amplitude sets exposure.

    """

    energy_kev: tuple[float, ...]
    probabilities: tuple[float, ...]
    provenance: Provenance | str

    def __post_init__(self) -> None:
        object.__setattr__(self, "energy_kev", tuple(self.energy_kev))
        object.__setattr__(self, "probabilities", tuple(self.probabilities))
        if not self.energy_kev or len(self.energy_kev) != len(self.probabilities):
            raise TransportError("source spectrum needs matching nonempty energies and masses")
        for energy in self.energy_kev:
            _physical(energy, "spectrum energy", positive=True)
        if any(b <= a for a, b in zip(self.energy_kev, self.energy_kev[1:], strict=False)):
            raise TransportError("source spectrum energies must increase strictly")
        for probability in self.probabilities:
            _physical(probability, "spectrum probability", positive=True)
        if abs(math.fsum(self.probabilities) - 1.0) > 2e-15:
            raise TransportError("source spectrum probabilities must sum to one")
        cdf = self.cdf
        if any(b <= a for a, b in zip((0.0, *cdf[:-1]), cdf, strict=True)):
            raise TransportError(
                "source spectrum masses must remain distinguishable in binary64 CDF"
            )
        if not isinstance(self.provenance, Provenance) and (
            type(self.provenance) is not str or not self.provenance.strip()
        ):
            raise TransportError("source spectrum requires provenance")

    @property
    def cdf(self) -> tuple[float, ...]:
        """Return ``C_k=sum_{j<=k} p_j`` with the final value exactly one.

        The dimensionless host tuple has shape ``(K,)`` and is copied to the
        workspace device during preparation. Source sampling selects the first
        index for which ``draw<C_k``. The cumulative values are derived from
        fixed masses rather than being separate active gradient parameters.

        """
        return tuple(
            math.fsum(self.probabilities[: i + 1]) if i + 1 < len(self.probabilities) else 1.0
            for i in range(len(self.probabilities))
        )


@dataclass(frozen=True, slots=True)
class SpectrumParameter:
    """Select a dimensionless logit ``eta[k]`` with ``p = softmax(eta)``.

    ``energy_bin`` is a zero-based index into the prepared ``SourceSpectrum``.
    For emitted index ``J``, ``dlog(p[J])/deta[k] = 1[J=k] - p[k]``. Support
    energies, total source amplitude, coefficients and geometry remain fixed.
    For ``SourceSpectrum((40., 80.), (0.25, 0.75), "discrete source")``,
    ``SpectrumParameter(0)`` gives likelihood factors ``0.75`` and ``-0.25``.

    """

    energy_bin: int

    def __post_init__(self) -> None:
        if type(self.energy_bin) is not int or self.energy_bin < 0:
            raise TransportError("spectrum parameter requires a nonnegative energy bin")


def spectrum_derivative_histories(
    positions: Any,
    directions: Any,
    weights: Any,
    density: Any,
    *,
    parameter: SpectrumParameter,
    batch: HistoryBatch,
    workspace: TransportWorkspace,
    out_pixel: Any,
    out_score: Any,
    out_derivative: Any,
    out_status: Any,
    source_amplitude: float = 1.0,
    stream: Any = None,
    validate: bool = True,
) -> None:
    """Write ``dE[X]/deta[k] = E[X * (1[J=k] - p[k])]`` by source replay.

    ``J`` is the emitted source-spectrum index and remains attached to the
    original history through energy-changing scattering. The conditional
    physical path and explicit score have zero logit derivative. The spectrum
    must be bound to ``workspace.physical.spectrum`` before preparation.

    With ``N = batch.count`` and ``M = workspace.spec.grid.materials``, inputs
    ``positions`` and ``directions`` are contiguous Warp ``vec3d`` arrays of
    shape ``(N,)`` containing mm positions and unit directions. ``weights`` is
    nonnegative FP64 with shape ``(N,)`` and ``density`` is positive FP64 with
    shape ``(M,)`` and dimensionless entries. Outputs ``out_pixel`` and
    ``out_status`` are ``int32`` arrays of shape ``(N,)``, whilst ``out_score``
    and ``out_derivative`` are FP64 arrays of shape ``(N,)``. Derivative units
    are photon-count or keV score per unit dimensionless logit. Every original
    history stays in the denominator, including zero contributions. Arrays
    must be contiguous on the prepared device and stream, with disjoint
    writable storage. Execution allocates no device memory and runs outside
    a Warp tape. An unchecked call requires ``workspace.check_status()``
    before either score or derivative is accepted.

    A unit absorbing slab has the analytical logit derivative shown below::

        import math
        import warp as wp
        from dpt.transport import (
            HistoryBatch, MaterialGrid, PhysicalTransport, PlanarDetector,
            SourceSpectrum, SpectrumParameter, TransportSpec,
            spectrum_derivative_histories, xraylib_atomic_scattering,
        )
        from dpt.transport.forward import prepare_transport

        device, n = "cuda:0", 4096
        spectrum = SourceSpectrum((40., 80.), (0.25, 0.75), "discrete source")
        physics = PhysicalTransport(
            (xraylib_atomic_scattering(13, (0., 0.1, 1., 10.)),),
            (0., 0.), spectrum=spectrum,
        )
        spec = TransportSpec(
            MaterialGrid((-1., -1., 0.), (2., 2., 1.), (1, 1, 1), 1),
            PlanarDetector((-1., -1.), (2., 2.), (1, 1), 2.),
            80., "analytic two-energy absorption slab",
            coefficient_energies_kev=(40., 80.), estimator="analogue",
        )
        transport = prepare_transport(
            spec, material_ids=wp.array([0], dtype=wp.int32, device=device),
            absorption=wp.array([0.1, 0.5], dtype=wp.float64, device=device),
            scattering=wp.array([0., 0.], dtype=wp.float64, device=device),
            physical=physics, max_histories=n, device=device,
        )
        score = wp.empty(n, dtype=wp.float64, device=device)
        derivative = wp.empty_like(score)
        spectrum_derivative_histories(
            wp.array([(0., 0., -1.)] * n, dtype=wp.vec3d, device=device),
            wp.array([(0., 0., 1.)] * n, dtype=wp.vec3d, device=device),
            wp.array([1.] * n, dtype=wp.float64, device=device),
            wp.array([1.], dtype=wp.float64, device=device),
            parameter=SpectrumParameter(0), batch=HistoryBatch(519103, 0, n),
            workspace=transport, out_score=score, out_derivative=derivative,
            out_pixel=wp.empty(n, dtype=wp.int32, device=device),
            out_status=wp.empty(n, dtype=wp.int32, device=device),
        )
        expected = 0.25 * math.exp(-0.1) - 0.25 * (
            0.25 * math.exp(-0.1) + 0.75 * math.exp(-0.5)
        )
        samples = derivative.numpy()
        standard_error = float(samples.std(ddof=1)) / math.sqrt(n)
        assert abs(float(samples.mean()) - expected) < 6 * standard_error

    """
    from .forward import _validate_call

    spectrum = None if workspace.physical is None else workspace.physical.spectrum
    if spectrum is None or parameter.energy_bin >= len(spectrum.energy_kev):
        raise TransportError(
            "spectrum derivative requires a prepared sampled spectrum and valid bin"
        )
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
            ("out_derivative", out_derivative, wp.float64),
            ("out_status", out_status, wp.int32),
        ],
        source_amplitude,
        stream,
        validate,
    )
    kernels = load_kernels("dpt.transport.parameter_kernels")
    workspace._launch(
        kernels.spectrum_derivative_histories,
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
            parameter.energy_bin,
            spectrum.probabilities[parameter.energy_bin],
            out_pixel,
            out_score,
            out_derivative,
            out_status,
            workspace._status,
        ],
    )
    if validate:
        workspace.check_status()


def spectrum_logit_derivative(
    pixel: Any,
    score: Any,
    selected_energy: Any,
    *,
    spectrum: SourceSpectrum,
    energy_bin: int,
    batch: HistoryBatch,
    workspace: TransportWorkspace,
    out_derivative: Any,
    stream: Any = None,
    validate: bool = True,
) -> None:
    """Write ``Z[i] = score[i] * (1[selected_energy[i]=k] - p[k])``.

    ``Z`` estimates the derivative of an expected score with respect to one
    dimensionless spectral logit. ``selected_energy`` contains emitted source
    indices, whilst ``pixel`` identifies the scored detector pixel or ``-1``
    for a miss. A zero score contributes zero. Spectral masses, physical tables
    and the original-history identity must match the supplied histories and
    the spectrum bound to ``workspace.physical.spectrum``.

    All four arrays are contiguous Warp arrays of shape ``(batch.count,)`` on
    the workspace's device and stream. ``pixel`` and ``selected_energy`` use
    ``int32`` and ``score`` and ``out_derivative`` use FP64. ``out_derivative``
    is writable and disjoint from inputs and prepared model storage. Score
    units are the workspace's photon-count or keV units, and ``Z`` has the same
    units per unit logit. The call allocates no device storage and runs outside
    a Warp tape. An unchecked launch needs ``workspace.check_status()``.

    For a prepared transparent two-energy source, supplied emitted indices
    and unit scores give zero summed logit derivative at the declared masses::

        import warp as wp
        from dpt.transport import (
            HistoryBatch, MaterialGrid, PhysicalTransport, PlanarDetector,
            SourceSpectrum, TransportSpec, spectrum_logit_derivative,
            xraylib_atomic_scattering,
        )
        from dpt.transport.forward import prepare_transport

        device = "cuda:0"
        spectrum = SourceSpectrum((40., 80.), (0.25, 0.75), "discrete source")
        transport = prepare_transport(
            TransportSpec(
                MaterialGrid((-1., -1., 0.), (2., 2., 1.), (1, 1, 1), 1),
                PlanarDetector((-1., -1.), (2., 2.), (1, 1), 2.),
                80., "transparent two-energy slab", coefficient_energies_kev=(40., 80.),
            ),
            material_ids=wp.array([0], dtype=wp.int32, device=device),
            absorption=wp.array([0., 0.], dtype=wp.float64, device=device),
            scattering=wp.array([0., 0.], dtype=wp.float64, device=device),
            physical=PhysicalTransport(
                (xraylib_atomic_scattering(13, (0., 0.1, 1., 10.)),),
                (0., 0.), spectrum=spectrum,
            ),
            max_histories=4, device=device,
        )
        derivative = wp.empty(4, dtype=wp.float64, device=device)
        spectrum_logit_derivative(
            wp.array([0] * 4, dtype=wp.int32, device=device),
            wp.array([1.] * 4, dtype=wp.float64, device=device),
            wp.array([0, 1, 1, 1], dtype=wp.int32, device=device),
            spectrum=spectrum, energy_bin=0, batch=HistoryBatch(121, 0, 4),
            workspace=transport, out_derivative=derivative,
        )
        assert list(derivative.numpy()) == [0.75, -0.25, -0.25, -0.25]

    """
    require_no_tape()
    if workspace.physical is None or workspace.physical.spectrum != spectrum:
        raise TransportError("spectrum likelihood must match the prepared source law")
    if type(energy_bin) is not int or not 0 <= energy_bin < len(spectrum.energy_kev):
        raise TransportError("active spectrum bin is outside its support")
    context = workspace.context
    context.assert_stream(stream)
    for name, value, dtype in (
        ("pixel", pixel, context.wp.int32),
        ("score", score, context.wp.float64),
        ("selected_energy", selected_energy, context.wp.int32),
        ("out_derivative", out_derivative, context.wp.float64),
    ):
        context.array(value, name, dtype=dtype, shape=(batch.count,))
    context.disjoint(
        [
            *workspace._reads(),
            ("pixel", pixel),
            ("score", score),
            ("selected_energy", selected_energy),
        ],
        [("out_derivative", out_derivative)],
    )
    if validate:
        workspace.check_status()
    kernel = load_kernels("dpt.transport.spectrum_kernels")
    workspace._launch(
        kernel.spectrum_logit_derivative,
        batch.count,
        [
            pixel,
            score,
            selected_energy,
            len(spectrum.energy_kev),
            energy_bin,
            spectrum.probabilities[energy_bin],
            out_derivative,
            workspace._status,
        ],
    )
    if validate:
        workspace.check_status()

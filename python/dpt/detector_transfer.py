"""Sampled-lattice MTF, noise power and DQE for a declared linear detector model.

An independent compound-Poisson signal exists at each incident lattice pixel.
Each event has supplied first/second signal moments, including missed photons.
A fixed passive PSF spreads that signal; independent white electronic noise is
added afterwards. Frequencies are cycles/mm inside the sampled Nyquist band.
"""

from __future__ import annotations

# Small immutable calibration inputs and diagnostics are workspace-owned.
# pyright: reportPrivateUsage=false
import math
from dataclasses import dataclass, field
from typing import Any, cast

from dpt._runtime import (
    DeviceContext,
    StatusReadback,
    load_kernels,
    prepare_context,
    require_no_tape,
)
from dpt.contracts import ContractError, finite_scalar, finite_tuple, integer, raise_device_status
from dpt.detector_physics import DetectorResponse, _response_value
from dpt.materials import Provenance


@dataclass(frozen=True, slots=True)
class DetectorTransferSpec:
    """Poisson populations per pixel, per-photon moments and a supplied PSF.

    The moments include rejection/detection probability: a missed photon scores
    zero. ``second_moments`` are E[X²], rather than E[X]². Populations are integrated
    photons per energy bin per incident pixel. PSF indices are (row,column), centred
    on the unique centre of its odd footprint; spacing is (column,row) in mm.

    This is the infinite sampled lattice defined by the supplied discrete PSF.
    Its transfer is the interior/interior limit of a finite zero-extended blur;
    finite image boundaries are not assumed periodic by any image operator.
    The input signal modulates every energy-bin population by the same fractional
    amplitude, preserving the supplied spectral composition. PSF spreading is a
    deterministic linear operation on event signal, with correlated shot noise.
    """

    incident_bin_photons: tuple[float, ...]
    first_moments: tuple[float, ...]
    second_moments: tuple[float, ...]
    signal_unit: str
    psf_shape: tuple[int, int]
    psf_weights: tuple[float, ...]
    spacing_mm: tuple[float, float]
    white_read_noise_variance: float = 0.0
    moments_provenance: Provenance | None = None
    psf_provenance: Provenance | None = None
    response_recipe: DetectorResponse | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        finite_tuple(self.incident_bin_photons, "incident bin photons")
        finite_tuple(self.first_moments, "first moments", length=len(self.incident_bin_photons))
        finite_tuple(self.second_moments, "second moments", length=len(self.incident_bin_photons))
        if any(
            mean > math.sqrt(second)
            for mean, second in zip(self.first_moments, self.second_moments, strict=True)
        ):
            raise ContractError("second moments must be at least the squared first moments")
        if not isinstance(cast(object, self.signal_unit), str) or not self.signal_unit.strip():
            raise ContractError("signal_unit must name the declared signal units")
        if not isinstance(cast(object, self.psf_shape), tuple) or len(self.psf_shape) != 2:
            raise ContractError("psf_shape must be (rows,columns)")
        for value in self.psf_shape:
            integer(value, "PSF dimension", minimum=1)
            if value % 2 != 1:
                raise ContractError("PSF dimensions must be odd with a unique centre")
        finite_tuple(self.psf_weights, "PSF weights", length=self.psf_shape[0] * self.psf_shape[1])
        mass = math.fsum(self.psf_weights)
        if not 0 < mass <= 1:
            raise ContractError("a passive PSF must have total mass in (0,1]")
        finite_tuple(self.spacing_mm, "pixel spacing in mm", positive=True, length=2)
        finite_scalar(self.white_read_noise_variance, "white read-noise variance", minimum=0)
        if not math.isfinite(self.spacing_mm[0] * self.spacing_mm[1]):
            raise ContractError("pixel area overflows binary64")
        for name in ("moments_provenance", "psf_provenance"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, Provenance):
                raise ContractError(f"{name} must be a Provenance record when supplied")
        try:
            population = math.fsum(self.incident_bin_photons)
            mean_products = tuple(
                phi * moment
                for phi, moment in zip(self.incident_bin_photons, self.first_moments, strict=True)
            )
            variance_products = tuple(
                phi * moment
                for phi, moment in zip(self.incident_bin_photons, self.second_moments, strict=True)
            )
            if any(
                phi > 0 and moment > 0 and product == 0
                for moments, products in (
                    (self.first_moments, mean_products),
                    (self.second_moments, variance_products),
                )
                for phi, moment, product in zip(
                    self.incident_bin_photons, moments, products, strict=True
                )
            ):
                raise ContractError("population-weighted moment underflows binary64")
            mean = math.fsum(mean_products)
            variance = math.fsum(variance_products)
        except OverflowError as error:
            raise ContractError("population or signal moments overflow binary64") from error
        if population <= 0 or not all(math.isfinite(v) for v in (population, mean, variance)):
            raise ContractError(
                "model requires finite positive incident population and finite moments"
            )

    @classmethod
    def from_response(
        cls,
        response: DetectorResponse,
        incident_bin_photons: tuple[float, ...],
        *,
        psf_shape: tuple[int, int],
        psf_weights: tuple[float, ...],
        spacing_mm: tuple[float, float],
        conditional_second_moments: tuple[float, ...] | None = None,
        second_moments_provenance: Provenance | None = None,
        psf_provenance: Provenance | None = None,
        white_read_noise_variance: float = 0.0,
    ) -> DetectorTransferSpec:
        """Include first-interaction thinning in both signal moments.

        Counting and explicitly ideal full-energy models have deterministic conditional
        scores. Supplied deposition needs supplied conditional E[X²] in keV² and its
        own provenance; a conditional mean does not determine quantum noise.
        """
        if not isinstance(cast(object, response), DetectorResponse):
            raise ContractError("response must be a DetectorResponse recipe")
        scores = response.model.scores(response.energies_kev)
        if response.model.photoelectric_table is not None:
            if conditional_second_moments is not None or second_moments_provenance is not None:
                raise ContractError(
                    "photoelectric branch model derives its conditional second moments"
                )
            conditional = tuple(
                energy * score for energy, score in zip(response.energies_kev, scores, strict=True)
            )
            second_moments_provenance = response.model.provenance
        elif response.model.kind == "supplied-deposition":
            finite_tuple(
                conditional_second_moments, "conditional second moments", length=len(scores)
            )
            if not isinstance(second_moments_provenance, Provenance):
                raise ContractError("supplied conditional second moments need their provenance")
            conditional = cast(tuple[float, ...], conditional_second_moments)
            if any(
                mean > math.sqrt(second) or second > energy * mean
                for mean, second, energy in zip(
                    scores, conditional, response.energies_kev, strict=True
                )
            ):
                raise ContractError(
                    "deposition second moments must obey mean² <= E[X²] <= energy*mean"
                )
        else:
            if conditional_second_moments is not None or second_moments_provenance is not None:
                raise ContractError(
                    "deterministic conditional scores derive their own second moments"
                )
            conditional = tuple(score * score for score in scores)
            if any(not math.isfinite(value) for value in conditional):
                raise ContractError("conditional second moments overflow binary64")
        coefficients = response.material.at_energies(
            response.energies_kev, edge_side=response.edge_side
        )
        second = tuple(
            _response_value(mu, response.thickness_mm, response.incidence_cosine, moment)
            for mu, moment in zip(coefficients, conditional, strict=True)
        )
        result = cls(
            incident_bin_photons,
            response.values,
            second,
            response.model.signal_unit,
            psf_shape,
            psf_weights,
            spacing_mm,
            white_read_noise_variance,
            second_moments_provenance,
            psf_provenance,
        )
        object.__setattr__(result, "response_recipe", response)
        return result

    @property
    def incident_photons_per_pixel(self) -> float:
        return math.fsum(self.incident_bin_photons)

    @property
    def mean_signal_before_spread(self) -> float:
        return math.fsum(
            phi * moment
            for phi, moment in zip(self.incident_bin_photons, self.first_moments, strict=True)
        )

    @property
    def quantum_variance_before_spread(self) -> float:
        return math.fsum(
            phi * moment
            for phi, moment in zip(self.incident_bin_photons, self.second_moments, strict=True)
        )

    @property
    def nps_unit(self) -> str:
        return f"({self.signal_unit})^2 mm^2"


@dataclass(frozen=True, slots=True)
class DetectorTransferWorkspace:
    """Own one small supplied PSF and diagnostics; output spectra are caller-owned."""

    spec: DetectorTransferSpec
    max_frequencies: int
    context: DeviceContext
    _weights: Any = field(repr=False)
    _kernels: Any = field(repr=False)
    _status: Any = field(repr=False)
    _readback: StatusReadback = field(repr=False)

    @property
    def scratch_bytes(self) -> int:
        return 8 * len(self.spec.psf_weights) + 4

    @property
    def host_bytes(self) -> int:
        return self._readback.host_bytes

    def check_status(self) -> None:
        raise_device_status(self._readback.read(), "detector transfer")

    def clear_status(self) -> None:
        with self.context.scope():
            self._status.zero_()


def prepare_detector_transfer(
    spec: DetectorTransferSpec,
    *,
    max_frequencies: int,
    device: str = "cuda:0",
    stream: Any = None,
    max_owned_bytes: int = 1 << 20,
) -> DetectorTransferWorkspace:
    require_no_tape()
    if not isinstance(cast(object, spec), DetectorTransferSpec):
        raise ContractError("spec must be DetectorTransferSpec")
    integer(max_frequencies, "max_frequencies", minimum=1)
    integer(max_owned_bytes, "max_owned_bytes", minimum=1, maximum=2**63 - 1)
    if 8 * len(spec.psf_weights) + 8 > max_owned_bytes:
        raise ContractError("PSF factors exceed max_owned_bytes")
    ctx = prepare_context(device=device, stream=stream)
    kernels = load_kernels("dpt.kernels.detector_transfer")
    with ctx.scope():
        weights = ctx.wp.array(spec.psf_weights, dtype=ctx.wp.float64, device=ctx.device)
        status = ctx.wp.zeros(1, dtype=ctx.wp.int32, device=ctx.device)
    ctx.wp.synchronize_stream(ctx.stream)
    return DetectorTransferWorkspace(
        spec, max_frequencies, ctx, weights, kernels, status, StatusReadback(ctx, status)
    )


def detector_transfer(
    frequencies_cycles_mm: Any,
    *,
    out_mtf: Any,
    out_nps: Any,
    out_dqe: Any,
    workspace: DetectorTransferWorkspace,
    stream: Any = None,
    validate: bool = True,
) -> None:
    """Write ``MTF=|H|/H(0)``, ``NPS=A(q|H|^2+v)`` and ``DQE=m^2|H|^2/[Phi(q|H|^2+v)]``.

    ``H(f)`` is the supplied discrete PSF's finite Fourier sum, ``A`` is pixel
    area in mm^2, ``m=sum_k Phi[k] E[X[k]]`` and
    ``q=sum_k Phi[k] E[X[k]^2]``. ``Phi=sum_k Phi[k]`` is the incident photon
    population per pixel and ``v`` is independent white read-noise variance in
    squared signal units. Per-photon moments include missed photons as zero
    scores, and the modulated populations retain their supplied spectral mixture.

    ``frequencies_cycles_mm`` is contiguous binary64 ``(F,2)`` with columns
    ``(column,row)`` in cycles/mm and ``1 <= F <= workspace.max_frequencies``.
    Both finite frequency components must lie within their axis's inclusive
    Nyquist band ``|f| <= 1/(2*spacing_mm)``. The three disjoint caller-owned
    destinations are binary64 ``(F,)``. MTF and DQE are dimensionless, while
    NPS has units ``(spec.signal_unit)^2 mm^2``. Zero transfer or null signal
    gives DQE zero.

    This is the infinite sampled lattice defined by the supplied passive PSF,
    using the large-area gain/MTF/NPS convention of Tang et al., Medical Physics
    39 (2012), equation 36 (https://pmc.ncbi.nlm.nih.gov/articles/PMC3396709/).
    Finite image operators retain their actual cropped boundaries. Each
    frequency uses a direct finite-stencil sum with binary64 arithmetic and
    range-preserving moment products, without an image FFT, dense matrix or
    host spectrum. Calibration inputs remain fixed and this analysis has no
    active VJP. All arrays belong to the workspace device, inputs remain
    unchanged and outputs are overwritten on the owning stream. The return
    value is ``None``. ``validate=False`` requires
    ``workspace.check_status()`` before output use.

    Example::

        import math
        import warp as wp
        from dpt.detector_transfer import (
            DetectorTransferSpec, detector_transfer, prepare_detector_transfer,
        )

        spec = DetectorTransferSpec((10.0,), (1.0,), (1.0,), "counts",
                                    (1, 1), (1.0,), (0.5, 0.5), 2.0)
        workspace = prepare_detector_transfer(spec, max_frequencies=1, device="cpu")
        frequency = wp.array([[0.0, 0.0]], dtype=wp.float64, device="cpu")
        mtf, nps, dqe = (wp.empty(1, dtype=wp.float64, device="cpu") for _ in range(3))
        detector_transfer(frequency, out_mtf=mtf, out_nps=nps, out_dqe=dqe, workspace=workspace)
        assert float(mtf.numpy()[0]) == 1.0
        assert math.isclose(float(nps.numpy()[0]), 3.0, rel_tol=1e-12)
        assert math.isclose(float(dqe.numpy()[0]), 5.0 / 6.0, rel_tol=1e-12)

    """
    require_no_tape()
    if type(validate) is not bool:
        raise ContractError("validate must be a boolean")
    ctx = workspace.context
    ctx.assert_stream(stream)
    ctx.array(frequencies_cycles_mm, "frequencies_cycles_mm", dtype=ctx.wp.float64, ndim=2)
    count = frequencies_cycles_mm.shape[0]
    if (
        tuple(frequencies_cycles_mm.shape) != (count, 2)
        or not 1 <= count <= workspace.max_frequencies
    ):
        raise ContractError("frequencies must have shape (1..max_frequencies,2)")
    for name, array in (("out_mtf", out_mtf), ("out_nps", out_nps), ("out_dqe", out_dqe)):
        ctx.array(array, name, dtype=ctx.wp.float64, shape=(count,))
    ctx.disjoint(
        [("frequencies_cycles_mm", frequencies_cycles_mm)],
        [("out_mtf", out_mtf), ("out_nps", out_nps), ("out_dqe", out_dqe)],
    )
    if validate:
        workspace.check_status()
    spec = workspace.spec
    ctx.wp.launch(
        workspace._kernels.transfer,
        dim=count,
        inputs=[
            frequencies_cycles_mm,
            workspace._weights,
            *spec.psf_shape,
            *spec.spacing_mm,
            math.fsum(spec.psf_weights),
            spec.incident_photons_per_pixel,
            spec.mean_signal_before_spread,
            spec.quantum_variance_before_spread,
            spec.white_read_noise_variance,
        ],
        outputs=[out_mtf, out_nps, out_dqe, workspace._status],
        device=ctx.device,
        stream=ctx.stream,
        record_tape=False,
    )
    if validate:
        workspace.check_status()

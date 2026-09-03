"""GPU-resident polychromatic primary transmission from material paths.

Importing this module is CPU-safe. Explicit host data preparation precedes CUDA
uploads; paths and signals have an explicit retained precision. Reverse products
recompute energy depths and preserve original inputs, with no P-by-K tape.
The API intentionally requires explicit VJP composition and rejects ambient tape
recording. It does not differentiate material interpolation or energy nodes.
"""

from __future__ import annotations

# Public Python boundaries validate runtime inputs; workspace scratch stays module-owned.
# pyright: reportPrivateUsage=false, reportUnnecessaryIsInstance=false
import hashlib
import json
import math
from dataclasses import asdict, dataclass, field, replace
from typing import TYPE_CHECKING, Any, Literal, cast

from dpt._runtime import (
    DeviceContext,
    StatusReadback,
    load_kernels,
    prepare_context,
    require_no_tape,
)
from dpt.contracts import (
    ContractError,
    NumericalError,
    finite_scalar,
    finite_tuple,
    integer,
    raise_device_status,
)
from dpt.materials import MassFractionMixture, MaterialTable, Provenance

if TYPE_CHECKING:
    import warp as wp


def _metadata_hash(record: dict[str, object]) -> str:
    encoded = json.dumps(
        record, sort_keys=True, separators=(",", ":"), allow_nan=False, default=float
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class SpectrumFilter:
    """Beer-Lambert recipe ``phi_out(E) = phi_in(E) exp(-mu(E) thickness_mm)``.

    ``material`` retains its supplied table units, density and source provenance.
    ``input_metadata_sha256`` and ``recipe_sha256`` identify canonical JSON
    metadata; original source-byte hashes remain in their Provenance records.
    Incident values and quadrature weights retain the thickness derivative's
    exponent range when filtered populations round to zero.
    """

    material: MaterialTable | MassFractionMixture
    thickness_mm: float
    edge_side: Literal["below", "above"]
    input_metadata_sha256: str
    incident_values: tuple[float, ...]
    quadrature_weights_kev: tuple[float, ...] | None
    recipe_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.material, (MaterialTable, MassFractionMixture)):
            raise ContractError("a spectrum filter requires a MaterialTable or MassFractionMixture")
        object.__setattr__(
            self, "thickness_mm", finite_scalar(self.thickness_mm, "thickness_mm", minimum=0)
        )
        if self.edge_side not in ("below", "above"):
            raise ContractError("filter edge_side must be below or above")
        digest = self.input_metadata_sha256
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ContractError("input_metadata_sha256 must be a lowercase SHA256 digest")
        finite_tuple(self.incident_values, "incident spectrum values")
        if self.quadrature_weights_kev is not None:
            finite_tuple(
                self.quadrature_weights_kev,
                "incident quadrature weights",
                positive=True,
                length=len(self.incident_values),
            )
        object.__setattr__(
            self,
            "recipe_sha256",
            _metadata_hash(
                {
                    "schema_version": 1,
                    "operation": "beer-lambert",
                    "material": asdict(self.material),
                    "thickness_mm": self.thickness_mm,
                    "edge_side": self.edge_side,
                    "input_metadata_sha256": digest,
                    "incident_values": self.incident_values,
                    "quadrature_weights_kev": self.quadrature_weights_kev,
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class Spectrum:
    """A supplied open-beam spectrum at the detector, before response losses.

    Density values are photons/keV and require positive quadrature weights in
    keV. Bin values are already integrated expected photon populations and must
    not receive a second energy weight. No normalisation or inverse-square
    correction is guessed. Discrete lines may be included as integrated bins.
    """

    energies_kev: tuple[float, ...]
    values: tuple[float, ...]
    representation: Literal["density", "bin-fluence"]
    provenance: Provenance | None = None
    quadrature_weights_kev: tuple[float, ...] | None = None
    filters: tuple[SpectrumFilter, ...] = ()

    def __post_init__(self) -> None:
        finite_tuple(self.energies_kev, "energies_kev", positive=True)
        finite_tuple(self.values, "spectrum")
        if len(self.values) != len(self.energies_kev):
            raise ContractError("spectrum values and energies must have equal length")
        if any(a >= b for a, b in zip(self.energies_kev, self.energies_kev[1:], strict=False)):
            raise ContractError("spectrum energy nodes must be strictly increasing")
        if self.provenance is not None and not isinstance(self.provenance, Provenance):
            raise ContractError("spectrum provenance must be a Provenance record when supplied")
        if not isinstance(self.filters, tuple) or any(
            not isinstance(recipe, SpectrumFilter) for recipe in self.filters
        ):
            raise ContractError(
                "spectrum filters must be an immutable tuple of SpectrumFilter records"
            )
        if self.representation == "density":
            if self.quadrature_weights_kev is None:
                raise ContractError("a density needs explicit energy quadrature weights")
            finite_tuple(self.quadrature_weights_kev, "quadrature weights", positive=True)
            if len(self.quadrature_weights_kev) != len(self.values):
                raise ContractError("one quadrature weight is required for each density value")
        elif self.representation == "bin-fluence":
            if self.quadrature_weights_kev is not None:
                raise ContractError("bin-integrated fluence must not be weighted twice")
        else:
            raise ContractError("spectrum representation must be density or bin-fluence")

    @property
    def bin_fluence(self) -> tuple[float, ...]:
        """Materialise energy-integrated host values at the explicit ingestion boundary."""
        if self.quadrature_weights_kev is None:
            return self.values
        result = tuple(
            value * weight
            for value, weight in zip(self.values, self.quadrature_weights_kev, strict=True)
        )
        if any(not math.isfinite(value) for value in result):
            raise ContractError("energy integration overflowed")
        if any(
            source > 0 and target == 0 for source, target in zip(self.values, result, strict=True)
        ):
            raise ContractError("energy integration underflowed a positive bin population")
        return result

    def filtered(
        self,
        table: MaterialTable | MassFractionMixture,
        thickness_mm: float,
        *,
        edge_side: Literal["below", "above"] | None = None,
    ) -> Spectrum:
        """Return ``phi_out[k] = phi_in[k] exp(-mu(E[k]) thickness_mm)``.

        Energies remain in keV, thickness is in mm, and the table converts its
        coefficients to mm⁻¹ before interpolation. Density/bin representation,
        quadrature weights and original spectrum provenance are retained. Each
        call appends its immutable filter recipe and metadata hash.

        Supply ``edge_side="below"`` or ``"above"`` at exact absorption-edge
        nodes. Between nodes the material's declared interpolant applies, within
        its supplied energy support.

        Example::

            hardened = spectrum.filtered(aluminium, thickness_mm=2.0)
        """
        if not isinstance(table, (MaterialTable, MassFractionMixture)):
            raise ContractError("a spectrum filter requires a MaterialTable or MassFractionMixture")
        thickness = finite_scalar(thickness_mm, "thickness_mm", minimum=0)
        if edge_side is None:
            tables = (
                (table,)
                if isinstance(table, MaterialTable)
                else tuple(
                    material
                    for material, fraction in zip(table.tables, table.mass_fractions, strict=True)
                    if fraction > 0
                )
            )
            if any(
                left == right and left in self.energies_kev
                for material in tables
                for left, right in zip(
                    material.energies_kev, material.energies_kev[1:], strict=False
                )
            ):
                raise ContractError("supply filter edge_side at an exact absorption-edge energy")
            edge_side = "above"
        coefficients = table.at_energies(self.energies_kev, edge_side=edge_side)
        filtered_values: list[float] = []
        for population, coefficient in zip(self.values, coefficients, strict=True):
            depth = coefficient * thickness
            if population == 0:
                filtered_values.append(0.0)
            elif depth < 700:
                filtered_values.append(population * math.exp(-depth))
            else:
                filtered_values.append(math.exp(math.log(population) - depth))
        recipe = SpectrumFilter(
            table,
            thickness,
            edge_side,
            _metadata_hash(
                {
                    "schema_version": 1,
                    "energies_kev": self.energies_kev,
                    "values": self.values,
                    "representation": self.representation,
                    "quadrature_weights_kev": self.quadrature_weights_kev,
                    "provenance": asdict(self.provenance) if self.provenance is not None else None,
                    "filter_recipes": tuple(item.recipe_sha256 for item in self.filters),
                }
            ),
            self.values,
            self.quadrature_weights_kev,
        )
        return replace(self, values=tuple(filtered_values), filters=(*self.filters, recipe))

    def filter_thickness_vjp(self, grad_bin_fluence: tuple[float, ...]) -> float:
        """Return ``-sum_k grad_phi[k] mu[k] phi_out[k]`` for the latest filter.

        ``grad_bin_fluence[K]`` is the cotangent of energy-integrated photon
        populations, including any density quadrature weights. This host product
        differentiates the filter thickness in mm at fixed interpolated tables.
        It recomputes attenuated factors from the incident values to retain deep
        derivative tails before their final binary64 rounding.

        Example::

            d_photons_d_mm = hardened.filter_thickness_vjp((1.0,) * len(hardened.values))
        """
        finite_tuple(grad_bin_fluence, "grad_bin_fluence", minimum=None, length=len(self.values))
        if not self.filters:
            raise ContractError("prepare a filtered spectrum before its thickness VJP")
        recipe = self.filters[-1]
        coefficients = recipe.material.at_energies(self.energies_kev, edge_side=recipe.edge_side)
        quadrature = recipe.quadrature_weights_kev or (1.0,) * len(self.values)
        terms: list[float] = []
        for seed, population, weight, coefficient in zip(
            grad_bin_fluence, recipe.incident_values, quadrature, coefficients, strict=True
        ):
            if seed == 0 or population == 0 or coefficient == 0:
                terms.append(0.0)
                continue
            exponent = (
                math.log(abs(seed))
                + math.log(population)
                + math.log(weight)
                + math.log(coefficient)
                - coefficient * recipe.thickness_mm
            )
            try:
                terms.append(-math.copysign(math.exp(exponent), seed))
            except OverflowError as error:
                raise NumericalError(
                    "spectrum filter thickness cotangent overflows binary64"
                ) from error
        try:
            return math.fsum(terms)
        except OverflowError as error:
            raise NumericalError("spectrum filter thickness VJP overflows binary64") from error

    def to_host(self, *, dtype: Literal["float32", "float64"] = "float32") -> Any:
        """Explicitly convert integrated bin populations to a contiguous host array."""
        from dpt.materials import _host_values

        return _host_values(self.bin_fluence, dtype=dtype)

    def to_device(
        self, context: DeviceContext, *, dtype: Literal["float32", "float64"] = "float32"
    ) -> Any:
        """Upload once on the supplied context; synchronise before releasing host storage."""
        values = self.to_host(dtype=dtype)
        scalar = context.wp.float64 if dtype == "float64" else context.wp.float32
        with context.scope():
            result = context.wp.array(values, dtype=scalar, device=context.device)
        context.wp.synchronize_stream(context.stream)
        return result


@dataclass(frozen=True, slots=True)
class SpectralSpec:
    """Physical array layout and parameter capabilities, fixed at preparation.

    The coefficients are total primary attenuation in mm⁻¹, with material paths
    in mm. ``input_description`` records the basis mixing/density assumption and
    detector acceptance used to turn source output into per-pixel fluence.
    Active weights are bin populations, not unconstrained normalised fractions.
    ``precision`` selects paths, means, image seeds and path gradients. Supplied
    coefficients, weights and responses remain binary32. Shared parameter
    gradients may be binary32 or binary64; per-pixel nuisance gradients remain
    binary32.
    Retained binary64 combines attenuation and factor exponents before final
    rounding, preserving weighted tails even when exp(-depth) underflows.
    Optional provenance/description fields preserve supplied metadata; numerical
    preparation does not require governance records. ``arithmetic="float32"``
    selects binary32 depth/exp with FP64 products and accumulators, and loses
    sufficiently deep tails. Its VJP is the smooth derivative at represented
    arithmetic values, as with the retained binary32 storage boundary.
    ``output_domain="log"`` uses a strict natural-log mean and an independent
    normalizer for its VJP; it allocates O(P) rather than O(P*K) extra scratch.
    """

    materials: int
    energies: int
    coefficients_provenance: tuple[Provenance, ...] | None = None
    spectrum_provenance: Provenance | None = None
    response_provenance: Provenance | None = None
    output_unit: str | None = None
    input_description: str | None = None
    shared_weights: bool = True
    shared_response: bool = True
    active_paths: bool = True
    active_weights: bool = False
    active_response: bool = False
    active_coefficients: bool = False
    reduction_groups: int = 64
    precision: Literal["float32", "float64"] = "float32"
    arithmetic: Literal["strict", "float32"] = "strict"
    output_domain: Literal["linear", "log"] = "linear"

    def __post_init__(self) -> None:
        # Reverse keeps two M-element FP64 vectors per pixel. This bound is an
        # explicit implementation envelope, not an empirical performance claim.
        if self.precision not in ("float32", "float64"):
            raise ContractError("spectral precision must be float32 or float64")
        if self.arithmetic not in ("strict", "float32"):
            raise ContractError("spectral arithmetic must be strict or float32")
        if self.arithmetic == "float32" and self.precision != "float32":
            raise ContractError("float32 arithmetic requires float32 retained precision")
        if self.output_domain not in ("linear", "log"):
            raise ContractError("spectral output_domain must be linear or log")
        if self.output_domain == "log" and self.arithmetic != "strict":
            raise ContractError("log signal requires strict arithmetic")
        integer(self.materials, "materials", minimum=1, maximum=32)
        integer(self.energies, "energies", minimum=1, maximum=65536)
        integer(self.reduction_groups, "reduction_groups", minimum=1, maximum=256)
        if self.coefficients_provenance is not None and (
            not isinstance(self.coefficients_provenance, tuple)
            or len(self.coefficients_provenance) != self.materials
            or any(not isinstance(p, Provenance) for p in self.coefficients_provenance)
        ):
            raise ContractError("one immutable provenance record is required per material")
        for name in ("spectrum_provenance", "response_provenance"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, Provenance):
                raise ContractError(f"{name} must be a Provenance record when supplied")
        for name in ("output_unit", "input_description"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ContractError(f"spectral {name} must be non-empty when supplied")
        for name in (
            "shared_weights",
            "shared_response",
            "active_paths",
            "active_weights",
            "active_response",
            "active_coefficients",
        ):
            if type(getattr(self, name)) is not bool:
                raise ContractError(f"{name} must be a boolean")


@dataclass(frozen=True, slots=True, init=False)
class SpectralInputs:
    """Validated spectral factors for ``I[p] = sum_k w[k,p] r[k,p] exp(-mu[:,k]·A[:,p])``.

    Coefficients have shape ``(M,K)`` and units mm⁻¹. Shared weights and responses
    have shape ``(K,)``; pixel-specific factors have shape ``(K,P)``. Weights are
    integrated photon populations. All factors use binary32 storage.

    ``from_arrays`` checks values once and owns independent device copies. Keep
    its arrays unchanged and prepare another record when the factors change.
    Mutating the arrays used to construct the record leaves these copies intact.

    Example::

        inputs = SpectralInputs.from_arrays(
            spec,
            coefficients,
            weights,
            response,
            pixels=geometry.pixels,
            device="cuda:0",
        )
    """

    spec: SpectralSpec
    pixels: int
    context: DeviceContext
    coefficients: wp.array[wp.float32]
    weights: wp.array[wp.float32]
    response: wp.array[wp.float32]

    def __init__(self) -> None:
        raise TypeError("prepare spectral inputs with SpectralInputs.from_arrays(...)")

    @classmethod
    def from_arrays(
        cls,
        spec: SpectralSpec,
        coefficients: wp.array[wp.float32],
        weights: wp.array[wp.float32],
        response: wp.array[wp.float32],
        *,
        pixels: int,
        device: str = "cuda:0",
        stream: wp.Stream | None = None,
    ) -> SpectralInputs:
        """Copy and validate ``mu[M,K]``, ``w[K or K,P]`` and ``r[K or K,P]`` once.

        The explicit device and stream become the record's execution context.
        Flat contiguous arrays also accept the same material/energy-major layout.
        """
        require_no_tape()
        validation_spec = replace(
            spec,
            active_paths=False,
            active_weights=False,
            active_response=False,
            active_coefficients=False,
            output_domain="linear",
        )
        workspace = prepare_spectral(
            validation_spec, max_pixels=pixels, device=device, stream=stream
        )
        fixed = _fixed_inputs(coefficients, weights, response, pixels, workspace)
        ctx = workspace.context
        with ctx.scope():
            owned = tuple(ctx.wp.clone(values) for _, values in fixed)
        workspace.validate_inputs(*owned, pixels=pixels, stream=ctx.stream)
        shaped = (
            owned[0].reshape((spec.materials, spec.energies)),
            owned[1].reshape((spec.energies,) if spec.shared_weights else (spec.energies, pixels)),
            owned[2].reshape((spec.energies,) if spec.shared_response else (spec.energies, pixels)),
        )
        record = object.__new__(cls)
        for name, value in (
            ("spec", spec),
            ("pixels", pixels),
            ("context", ctx),
            ("coefficients", cast("wp.array[wp.float32]", shaped[0])),
            ("weights", cast("wp.array[wp.float32]", shaped[1])),
            ("response", cast("wp.array[wp.float32]", shaped[2])),
        ):
            object.__setattr__(record, name, value)
        return record

    @property
    def storage_bytes(self) -> int:
        """Bytes retained by the three owned binary32 input buffers."""
        return 4 * (self.coefficients.size + self.weights.size + self.response.size)


@dataclass(slots=True)
class SpectralWorkspace:
    """One stream, persistent diagnostics and bounded shared-parameter reduction."""

    spec: SpectralSpec
    max_pixels: int
    context: DeviceContext
    _kernels: Any = field(repr=False)
    _status: Any = field(repr=False)
    _empty: Any = field(repr=False)
    _empty_signal: Any = field(repr=False)
    _partials: Any = field(repr=False)
    _log_mean: Any = field(repr=False)
    _status_readback: StatusReadback = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._status_readback = StatusReadback(self.context, self._status)

    @property
    def dtype(self) -> Any:
        return (
            self.context.wp.float64 if self.spec.precision == "float64" else self.context.wp.float32
        )

    @property
    def scratch_bytes(self) -> int:
        return 4 + 8 * (int(self._partials.size) + int(self._log_mean.size))

    @property
    def host_bytes(self) -> int:
        return self._status_readback.host_bytes

    def clear_status(self) -> None:
        with self.context.scope():
            self._status.zero_()

    def check_status(self) -> None:
        """Explicit completion point; output/gradient overflow invalidates the pass."""
        code = self._status_readback.read()
        raise_device_status(code, "spectral")

    def consume_status(self) -> None:
        """Deliver and clear one checkpoint, permitting reuse after a checked failure."""
        raise_device_status(self._status_readback.read(reset=True), "spectral")

    def _check_values(self, arrays: list[tuple[Any, bool]]) -> None:
        self.check_status()
        for values, nonnegative in arrays:
            if values.size:
                self.context.wp.launch(
                    self._kernels.get_value_check(
                        values.dtype == self.context.wp.float64, nonnegative
                    ),
                    dim=values.size,
                    inputs=[values],
                    outputs=[self._status],
                    device=self.context.device,
                    stream=self.context.stream,
                    record_tape=False,
                )
        self.consume_status()

    def validate_cotangent(
        self,
        grad_image: wp.array[wp.float32] | wp.array[wp.float64],
        *,
        stream: wp.Stream | None = None,
    ) -> None:
        """Check finite image cotangents ``grad_image[P]`` before a prepared VJP.

        Cotangents use the workspace's retained precision. This scans the
        current cotangent without repeating validation of fixed spectral inputs.
        """
        require_no_tape()
        self.context.assert_stream(stream)
        size = self.context.array(grad_image, "grad_image", dtype=self.dtype)
        if size > self.max_pixels:
            raise ContractError("image cotangent exceeds workspace pixel capacity")
        with self.context.scope():
            self._check_values([(grad_image, False)])

    def validate_inputs(
        self,
        coefficients: Any,
        weights: Any,
        response: Any,
        *,
        pixels: int,
        paths: Any = None,
        probability_response: bool = False,
        stream: wp.Stream | None = None,
    ) -> None:
        """Validate current supplied arrays without evaluating a spectral image.

        A count observation requires probabilities; a general first-moment
        response only requires nonnegative values and its declared output unit.
        Omit paths when preparing fixed inputs before the material projection.
        """
        require_no_tape()
        ctx, spec = self.context, self.spec
        ctx.assert_stream(stream)
        integer(pixels, "pixels", maximum=self.max_pixels)
        if type(probability_response) is not bool:
            raise ContractError("probability_response must be a boolean")
        fixed = _fixed_inputs(coefficients, weights, response, pixels, self)
        arrays = [(values, True) for _, values in fixed]
        response = fixed[-1][1]
        if paths is not None:
            paths = _flat_array(
                paths, "paths", self, dtype=self.dtype, shape=(spec.materials, pixels)
            )
            arrays.append((paths, True))
        with ctx.scope():
            self._check_values(arrays)
            if probability_response and response.size:
                ctx.wp.launch(
                    self._kernels.check_probability,
                    dim=response.size,
                    inputs=[response],
                    outputs=[self._status],
                    device=ctx.device,
                    stream=ctx.stream,
                    record_tape=False,
                )
                self.consume_status()


def prepare_spectral(
    spec: SpectralSpec,
    *,
    max_pixels: int,
    device: str | wp.Device = "cuda:0",
    stream: wp.Stream | None = None,
) -> SpectralWorkspace:
    """Allocate scratch only; callers own source coefficients, weights and destinations."""
    integer(
        max_pixels,
        "max_pixels",
        maximum=(2**31 - 1)
        // max(spec.materials, spec.energies, 2 if spec.output_domain == "log" else 1),
    )
    context = prepare_context(device=device, stream=stream)
    kernels = load_kernels("dpt.kernels.spectral")
    parameters = 0
    if spec.active_weights and spec.shared_weights:
        parameters += spec.energies
    if spec.active_response and spec.shared_response:
        parameters += spec.energies
    if spec.active_coefficients:
        parameters += spec.materials * spec.energies
    groups = min(spec.reduction_groups, max(1, (max_pixels + 255) // 256))
    with context.scope():
        status = context.wp.zeros(1, dtype=context.wp.int32, device=context.device)
        empty = context.wp.empty(0, dtype=context.wp.float32, device=context.device)
        empty_signal = context.wp.empty(
            0,
            dtype=context.wp.float64 if spec.precision == "float64" else context.wp.float32,
            device=context.device,
        )
        partials = context.wp.empty(
            parameters * groups,
            dtype=context.wp.float64,
            device=context.device,
        )
        log_mean = context.wp.empty(
            2 * max_pixels if spec.output_domain == "log" else 0,
            dtype=context.wp.float64,
            device=context.device,
        )
    return SpectralWorkspace(
        spec, max_pixels, context, kernels, status, empty, empty_signal, partials, log_mean
    )


def _inputs(
    paths: Any,
    coefficients: Any,
    weights: Any,
    response: Any,
    workspace: SpectralWorkspace,
) -> tuple[int, list[tuple[str, Any]]]:
    ctx, spec = workspace.context, workspace.spec
    if getattr(paths, "ndim", 1) == 2:
        shape = tuple(paths.shape)
        if shape[0] != spec.materials:
            raise ContractError("shaped paths must have shape (M,P)")
        paths = _flat_array(paths, "paths", workspace, dtype=workspace.dtype, shape=shape)
    size = ctx.array(paths, "paths", dtype=workspace.dtype)
    if size % spec.materials:
        raise ContractError("material-major paths must contain M*P values")
    pixels = size // spec.materials
    if pixels > workspace.max_pixels:
        raise ContractError("material paths exceed workspace pixel capacity")
    return pixels, [
        ("paths", paths),
        *_fixed_inputs(coefficients, weights, response, pixels, workspace),
    ]


def _fixed_inputs(
    coefficients: Any,
    weights: Any,
    response: Any,
    pixels: int,
    workspace: SpectralWorkspace,
) -> list[tuple[str, Any]]:
    ctx, spec = workspace.context, workspace.spec
    coefficients = _flat_array(
        coefficients,
        "coefficients",
        workspace,
        dtype=ctx.wp.float32,
        shape=(spec.materials, spec.energies),
    )
    weights = _flat_array(
        weights,
        "weights",
        workspace,
        dtype=ctx.wp.float32,
        shape=(spec.energies,) if spec.shared_weights else (spec.energies, pixels),
    )
    response = _flat_array(
        response,
        "response",
        workspace,
        dtype=ctx.wp.float32,
        shape=(spec.energies,) if spec.shared_response else (spec.energies, pixels),
    )
    return [
        ("coefficients", coefficients),
        ("weights", weights),
        ("response", response),
    ]


def _flat_array(
    values: Any, name: str, workspace: SpectralWorkspace, *, dtype: Any, shape: tuple[int, ...]
) -> Any:
    """Validate either the declared axes or their contiguous flat view without copying."""
    ctx = workspace.context
    if getattr(values, "ndim", 1) == len(shape) and len(shape) > 1:
        ctx.array(values, name, dtype=dtype, shape=shape, ndim=len(shape))
        return values.flatten()
    ctx.array(values, name, dtype=dtype, shape=(math.prod(shape),))
    return values


def spectral_signal(
    paths: wp.array[wp.float32] | wp.array[wp.float64],
    coefficients: wp.array[wp.float32],
    weights: wp.array[wp.float32],
    response: wp.array[wp.float32],
    *,
    out_mean: wp.array[wp.float32] | wp.array[wp.float64],
    workspace: SpectralWorkspace,
    stream: wp.Stream | None = None,
    validate: bool = True,
) -> None:
    """I[p] = sum_k w[k,p]*r[k,p]*exp(-sum_m mu[m,k]*A[m,p]).

    Paths A have shape (M,P) in mm, coefficients mu have FP32 (M,K) storage
    in mm^-1, and weights/response have FP32 (K,) shared storage or (K,P)
    pixel-specific storage according to SpectralSpec. Weights are energy-integrated
    photon populations and response declares signal per photon. Paths and out_mean
    (P,) use the selected FP32/FP64 retained precision. Flat material/energy-major
    contiguous arrays accept the same layouts without copies. The output is the
    linear mean, or its natural logarithm under output_domain="log". Log zero is
    -inf and a log VJP requires a positive mean. FP64 retained arithmetic combines
    factor exponents before rounding, while arithmetic="float32" selects FP32
    depth/exp with FP64 products/sums. Parameter activity is declared at preparation.
    Inputs and destinations are disjoint, and checked calls scan current values
    and deliver completion diagnostics.

    """
    # Runtime contracts validate the dynamically loaded Warp boundary.
    _api_paths: Any = cast(Any, paths)
    _api_coefficients: Any = cast(Any, coefficients)
    _api_weights: Any = cast(Any, weights)
    _api_response: Any = cast(Any, response)
    _api_out_mean: Any = cast(Any, out_mean)
    require_no_tape()
    ctx, spec = workspace.context, workspace.spec
    ctx.assert_stream(stream)
    pixels, reads = _inputs(_api_paths, _api_coefficients, _api_weights, _api_response, workspace)
    _api_paths, _api_coefficients, _api_weights, _api_response = (value for _, value in reads)
    ctx.array(_api_out_mean, "out_mean", dtype=workspace.dtype, shape=(pixels,))
    ctx.disjoint(reads, [("out_mean", _api_out_mean)])
    with ctx.scope():
        if validate:
            workspace.validate_inputs(
                _api_coefficients,
                _api_weights,
                _api_response,
                pixels=pixels,
                paths=_api_paths,
                stream=ctx.stream,
            )
        if pixels:
            ctx.wp.launch(
                (
                    workspace._kernels.get_log_forward
                    if spec.output_domain == "log"
                    else workspace._kernels.get_forward
                )(
                    spec.materials,
                    spec.energies,
                    spec.shared_weights,
                    spec.shared_response,
                    spec.precision == "float64",
                    spec.arithmetic == "float32",
                ),
                dim=pixels,
                # Smaller blocks pack the register-heavy spectral work more evenly.
                # The reduction kernels retain their separate 256-lane layout.
                block_dim=128,
                inputs=[_api_paths, _api_coefficients, _api_weights, _api_response, pixels],
                outputs=[_api_out_mean, workspace._status],
                device=ctx.device,
                stream=ctx.stream,
                record_tape=False,
            )
        if validate:
            workspace.consume_status()


def spectral_vjp(
    paths: wp.array[wp.float32] | wp.array[wp.float64],
    coefficients: wp.array[wp.float32],
    weights: wp.array[wp.float32],
    response: wp.array[wp.float32],
    *,
    cotangent: wp.array[wp.float32] | wp.array[wp.float64] | None,
    out_grad_paths: wp.array[wp.float32] | wp.array[wp.float64] | None = None,
    out_grad_weights: wp.array[wp.float32] | wp.array[wp.float64] | None = None,
    out_grad_response: wp.array[wp.float32] | wp.array[wp.float64] | None = None,
    out_grad_coefficients: wp.array[wp.float32] | wp.array[wp.float64] | None = None,
    workspace: SpectralWorkspace,
    stream: wp.Stream | None = None,
    validate: bool = True,
) -> None:
    """Grad_A[m,p] = -c[p]*sum_k mu[m,k]*w[k,p]*r[k,p]*exp(-mu[:,k].A[:,p]).

    The weight, response and coefficient products differentiate those same factors,
    with a sum over pixels for shared parameters. In log mode each image derivative
    is divided by the physical mean using a recomputed FP64 log normaliser. Paths
    are (M,P) in mm, coefficients are FP32 (M,K) in mm^-1, and FP32 weights/response
    are (K,) shared or (K,P) per pixel. Cotangent is (P,) in image precision, with
    Grad_paths is (M,P) in path precision, grad_coefficients
    is (M,K), and grad_weights/grad_response match their inputs. Shared factor
    destinations may be FP32/FP64 and pixel-specific destinations are FP32.
    Request only declared active parameters. Outputs overwrite disjoint caller-owned
    buffers and preserve the covector. Zero factors retain algebraic one-sided
    derivatives and shared reductions use a fixed FP64 tree without atomics.
    Higher-order products and ambient tapes are rejected.

    """
    # Runtime contracts validate the dynamically loaded Warp boundary.
    _api_paths: Any = cast(Any, paths)
    _api_coefficients: Any = cast(Any, coefficients)
    _api_weights: Any = cast(Any, weights)
    _api_response: Any = cast(Any, response)
    _api_cotangent: Any = cast(Any, cotangent)
    _api_out_grad_paths: Any = cast(Any, out_grad_paths)
    _api_out_grad_weights: Any = cast(Any, out_grad_weights)
    _api_out_grad_response: Any = cast(Any, out_grad_response)
    _api_out_grad_coefficients: Any = cast(Any, out_grad_coefficients)
    require_no_tape()
    ctx, spec = workspace.context, workspace.spec
    ctx.assert_stream(stream)
    pixels, reads = _inputs(_api_paths, _api_coefficients, _api_weights, _api_response, workspace)
    _api_paths, _api_coefficients, _api_weights, _api_response = (value for _, value in reads)
    ctx.array(_api_cotangent, "cotangent", dtype=workspace.dtype, shape=(pixels,))
    destinations = (
        ("paths", _api_out_grad_paths, spec.active_paths, spec.materials * pixels),
        (
            "weights",
            _api_out_grad_weights,
            spec.active_weights,
            spec.energies * (1 if spec.shared_weights else pixels),
        ),
        (
            "response",
            _api_out_grad_response,
            spec.active_response,
            spec.energies * (1 if spec.shared_response else pixels),
        ),
        (
            "coefficients",
            _api_out_grad_coefficients,
            spec.active_coefficients,
            spec.materials * spec.energies,
        ),
    )
    writes: list[tuple[str, Any]] = []
    normalized: dict[str, Any] = {}
    for name, output, active, length in destinations:
        if output is not None:
            if not active:
                raise ContractError(f"{name} was declared fixed")
            dtype = workspace.dtype if name == "paths" else ctx.wp.float32
            shared_parameter = (
                name == "coefficients"
                or (name == "weights" and spec.shared_weights)
                or (name == "response" and spec.shared_response)
            )
            if shared_parameter:
                dtype = getattr(output, "dtype", None)
                if dtype not in (ctx.wp.float32, ctx.wp.float64):
                    raise ContractError("shared spectral gradients must be binary32 or binary64")
            rows = spec.materials if name in ("paths", "coefficients") else spec.energies
            shape = (
                (length,) if shared_parameter and name != "coefficients" else (rows, length // rows)
            )
            output = _flat_array(output, f"out_grad_{name}", workspace, dtype=dtype, shape=shape)
            normalized[name] = output
            writes.append((f"out_grad_{name}", output))
    if not writes:
        raise ContractError("request at least one active spectral gradient")
    ctx.disjoint([*reads, ("cotangent", _api_cotangent)], writes)
    _api_out_grad_paths = normalized.get("paths")
    _api_out_grad_weights = normalized.get("weights")
    _api_out_grad_response = normalized.get("response")
    _api_out_grad_coefficients = normalized.get("coefficients")
    with ctx.scope():
        if validate:
            workspace._check_values(
                [*((value, True) for _, value in reads), (_api_cotangent, False)]
            )
        logarithmic = spec.output_domain == "log"
        log_inputs = [workspace._log_mean] if logarithmic else []
        if pixels and logarithmic:
            ctx.wp.launch(
                workspace._kernels.get_log_forward(
                    spec.materials,
                    spec.energies,
                    spec.shared_weights,
                    spec.shared_response,
                    spec.precision == "float64",
                    output64=True,
                    retain_normalizer=True,
                ),
                dim=pixels,
                block_dim=128,
                inputs=[_api_paths, _api_coefficients, _api_weights, _api_response, pixels],
                outputs=[workspace._log_mean, workspace._status],
                device=ctx.device,
                stream=ctx.stream,
                record_tape=False,
            )
        pixel_weights = _api_out_grad_weights is not None and not spec.shared_weights
        pixel_response = _api_out_grad_response is not None and not spec.shared_response
        if pixels and (_api_out_grad_paths is not None or pixel_weights or pixel_response):
            ctx.wp.launch(
                (
                    workspace._kernels.get_log_pixel_vjp
                    if logarithmic
                    else workspace._kernels.get_pixel_vjp
                )(
                    spec.materials,
                    spec.energies,
                    spec.shared_weights,
                    spec.shared_response,
                    _api_out_grad_paths is not None,
                    pixel_weights,
                    pixel_response,
                    spec.precision == "float64",
                    spec.arithmetic == "float32",
                ),
                dim=pixels,
                block_dim=128,
                inputs=[
                    _api_paths,
                    _api_coefficients,
                    _api_weights,
                    _api_response,
                    _api_cotangent,
                    *log_inputs,
                    pixels,
                ],
                outputs=[
                    _api_out_grad_paths
                    if _api_out_grad_paths is not None
                    else workspace._empty_signal,
                    _api_out_grad_weights if pixel_weights else workspace._empty,
                    _api_out_grad_response if pixel_response else workspace._empty,
                    workspace._status,
                ],
                device=ctx.device,
                stream=ctx.stream,
                record_tape=False,
            )
        groups = min(spec.reduction_groups, max(1, (pixels + 255) // 256))
        requested = [
            (kind, output, parameters)
            for kind, output, shared, parameters in (
                (0, _api_out_grad_weights, spec.shared_weights, spec.energies),
                (1, _api_out_grad_response, spec.shared_response, spec.energies),
                (2, _api_out_grad_coefficients, True, spec.materials * spec.energies),
            )
            if output is not None and shared
        ]
        fused = len(requested) > 1 or logarithmic
        if pixels and fused:
            ctx.wp.launch_tiled(
                (
                    workspace._kernels.get_log_nuisance_partials
                    if logarithmic
                    else workspace._kernels.get_nuisance_partials
                )(
                    spec.materials,
                    spec.energies,
                    spec.shared_weights,
                    spec.shared_response,
                    any(kind == 0 for kind, _, _ in requested),
                    any(kind == 1 for kind, _, _ in requested),
                    any(kind == 2 for kind, _, _ in requested),
                    spec.precision == "float64",
                    spec.arithmetic == "float32",
                ),
                dim=(groups, spec.energies),
                block_dim=256,
                inputs=[
                    _api_paths,
                    _api_coefficients,
                    _api_weights,
                    _api_response,
                    _api_cotangent,
                    *log_inputs,
                    pixels,
                    groups,
                ],
                outputs=[workspace._partials, workspace._status],
                device=ctx.device,
                stream=ctx.stream,
                record_tape=False,
            )
        offset = 0
        for kind, output, parameters in requested:
            if pixels:
                partial_kernel = (
                    workspace._kernels.get_coefficient_partials(
                        spec.materials,
                        spec.energies,
                        spec.shared_weights,
                        spec.shared_response,
                        spec.precision == "float64",
                        spec.arithmetic == "float32",
                    )
                    if kind == 2
                    else workspace._kernels.get_shared_partials(
                        spec.materials,
                        spec.energies,
                        spec.shared_weights,
                        spec.shared_response,
                        kind,
                        spec.precision == "float64",
                        spec.arithmetic == "float32",
                    )
                )
                partials = workspace._partials[offset * groups : (offset + parameters) * groups]
                if not fused:
                    ctx.wp.launch_tiled(
                        partial_kernel,
                        dim=(groups, spec.energies if kind == 2 else parameters),
                        block_dim=256,
                        inputs=[
                            _api_paths,
                            _api_coefficients,
                            _api_weights,
                            _api_response,
                            _api_cotangent,
                            pixels,
                            groups,
                        ],
                        outputs=[partials, workspace._status],
                        device=ctx.device,
                        stream=ctx.stream,
                        record_tape=False,
                    )
                ctx.wp.launch(
                    workspace._kernels.get_finish_shared(output.dtype == ctx.wp.float64),
                    dim=parameters,
                    inputs=[partials, groups],
                    outputs=[output, workspace._status],
                    device=ctx.device,
                    stream=ctx.stream,
                    record_tape=False,
                )
            else:
                output.zero_()
            offset += parameters
        if validate:
            workspace.consume_status()


def spectral_bin_counts(
    paths: wp.array[wp.float32] | wp.array[wp.float64],
    coefficients: wp.array[wp.float32],
    weights: wp.array[wp.float32],
    detection_probability: Any,
    *,
    out_bin_counts: Any,
    workspace: SpectralWorkspace,
    stream: wp.Stream | None = None,
    validate: bool = True,
) -> None:
    """Lambda[k,p] = w[k,p]*q[k,p]*exp(-sum_m mu[m,k]*A[m,p]) gives detected bin means.

    Paths are (M,P) in mm and path precision, coefficients are FP32 (M,K) in
    mm^-1, weights are FP32 (K,) or (K,P) integrated photon populations, and
    detection_probability has the corresponding response layout with values in
    [0,1]. Out_bin_counts is caller-owned (K,P) in spectral precision and holds
    expected detected photon counts. Flat energy-major arrays are accepted.
    Arithmetic follows the spectral workspace and each output rounds once to
    its retained storage. These are independent Poisson rates to pair with
    per-photon scores in compound-Poisson sampling. The helper provides no tape
    or derivative rule and leaves differentiation of the mean to spectral_vjp.

    """
    # Runtime contracts validate the dynamically loaded Warp boundary.
    _api_paths: Any = cast(Any, paths)
    _api_coefficients: Any = cast(Any, coefficients)
    _api_weights: Any = cast(Any, weights)
    require_no_tape()
    ctx, spec = workspace.context, workspace.spec
    ctx.assert_stream(stream)
    pixels, reads = _inputs(
        _api_paths, _api_coefficients, _api_weights, detection_probability, workspace
    )
    _api_paths, _api_coefficients, _api_weights, detection_probability = (
        value for _, value in reads
    )
    out_bin_counts = _flat_array(
        out_bin_counts,
        "out_bin_counts",
        workspace,
        dtype=workspace.dtype,
        shape=(spec.energies, pixels),
    )
    ctx.disjoint(reads, [("out_bin_counts", out_bin_counts)])
    with ctx.scope():
        if validate:
            workspace.validate_inputs(
                _api_coefficients,
                _api_weights,
                detection_probability,
                pixels=pixels,
                paths=_api_paths,
                probability_response=True,
                stream=ctx.stream,
            )
        if pixels:
            ctx.wp.launch(
                workspace._kernels.get_bin_counts(
                    spec.materials,
                    spec.energies,
                    spec.shared_weights,
                    spec.shared_response,
                    spec.precision == "float64",
                    spec.arithmetic == "float32",
                ),
                dim=(spec.energies, pixels),
                inputs=[_api_paths, _api_coefficients, _api_weights, detection_probability, pixels],
                outputs=[out_bin_counts, workspace._status],
                device=ctx.device,
                stream=ctx.stream,
                record_tape=False,
            )
        if validate:
            workspace.consume_status()

"""Pixel acceptance and first-interaction detector response.

Panel coordinates and thicknesses are millimetres, energies are keV and solid
angles are steradians. Compact panel metadata crosses the host boundary once;
pixel acceptance and spatially varying slab response run on the owning device.
"""

from __future__ import annotations

# Owned factors and diagnostic words stay private to each workspace.
# pyright: reportPrivateUsage=false
import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
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
from dpt.geometry import (
    BatchedGeometry,
    CurvedDetectorGeometry,
    DetectorGeometry,
    ParallelGeometry,
    cross,
    dot,
)
from dpt.materials import MaterialTable, Provenance, _host_values, _upload_values

if TYPE_CHECKING:
    import warp as wp

    from dpt.physical_data import PhotonCoefficientTable


@dataclass(frozen=True, slots=True)
class FirstInteractionModel:
    """Declare what one interaction contributes to the detector signal.

    Interaction counting scores every first interaction as one count. Supplied
    deposition scores it by a caller-supplied conditional mean deposited energy in
    keV, retaining that input's provenance. Ideal full-energy scores it by the
    incident energy. Total attenuation governs the first-interaction probability;
    the deposited-energy model separately governs its conditional signal.
    """

    kind: Literal["interaction-counting", "supplied-deposition", "ideal-full-energy"]
    mean_deposited_energy_kev: tuple[float, ...] | None = None
    provenance: Provenance | None = None
    photoelectric_table: PhotonCoefficientTable | None = field(default=None, init=False)
    photoelectric_total: MaterialTable | None = field(default=None, init=False)
    photoelectric_energies_kev: tuple[float, ...] | None = field(default=None, init=False)
    photoelectric_edge_side: Literal["below", "above"] | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.kind not in ("interaction-counting", "supplied-deposition", "ideal-full-energy"):
            raise ContractError("select a declared first-interaction response model")
        if self.kind == "supplied-deposition":
            finite_tuple(self.mean_deposited_energy_kev, "mean deposited energies in keV")
            if not isinstance(self.provenance, Provenance):
                raise ContractError("supplied deposition needs its own Provenance record")
        elif self.mean_deposited_energy_kev is not None or self.provenance is not None:
            raise ContractError("only supplied-deposition accepts deposited energies/provenance")

    @classmethod
    def ideal_full_energy(cls) -> FirstInteractionModel:
        """Score every first interaction by its incident energy in keV."""
        return cls("ideal-full-energy")

    @classmethod
    def photoelectric_only(
        cls,
        total: MaterialTable,
        photoelectric: PhotonCoefficientTable,
        energies_kev: tuple[float, ...],
        *,
        edge_side: Literal["below", "above"],
    ) -> FirstInteractionModel:
        """Score a photoelectric first interaction by E, other branches by zero.

        The conditional mean score is E*μ_photoelectric/μ_total. Total primary
        attenuation still sets the interaction probability. The branch table's
        kind, composition, energy support and reference density must match the
        total table. This is a full-energy photoelectric branch model; it retains
        the original coefficients/provenance and has conditional E[X²]=E*E[X].
        """
        from dpt.materials import validate_energy_grid
        from dpt.physical_data import PhotonCoefficientTable

        if not isinstance(cast(object, total), MaterialTable) or not isinstance(
            cast(object, photoelectric), PhotonCoefficientTable
        ):
            raise ContractError("photoelectric response requires typed total and branch tables")
        if photoelectric.kind != "photoelectric":
            raise ContractError("photoelectric-only response requires a photoelectric table")
        if total.units != "cm^2/g" or total.density_g_cm3 != photoelectric.density_g_cm3:
            raise ContractError(
                "total and photoelectric tables must share a declared mass-table density"
            )
        if total.name != photoelectric.name or total.energies_kev != photoelectric.energies_kev:
            raise ContractError("total and photoelectric composition/support must match")
        validate_energy_grid(energies_kev)
        coefficients = total.at_energies(energies_kev, edge_side=edge_side)
        branches = photoelectric.at_energies(energies_kev, edge_side=edge_side)
        # Check the supplied knots too: a malformed unused knot is not licensed
        # by selecting a smaller query subset.
        if any(
            branch > coefficient
            for coefficient, branch in zip(
                total.linear_mm_inverse, photoelectric.linear_mm_inverse, strict=True
            )
        ) or any(
            branch > coefficient for coefficient, branch in zip(coefficients, branches, strict=True)
        ):
            raise ContractError("photoelectric branch must not exceed total attenuation")
        scores = tuple(
            _energy_weighted_branch(energy, branch, coefficient)
            for energy, branch, coefficient in zip(
                energies_kev, branches, coefficients, strict=True
            )
        )
        model = cls("supplied-deposition", scores, photoelectric.provenance)
        object.__setattr__(model, "photoelectric_table", photoelectric)
        object.__setattr__(model, "photoelectric_total", total)
        object.__setattr__(model, "photoelectric_energies_kev", energies_kev)
        object.__setattr__(model, "photoelectric_edge_side", edge_side)
        return model

    def scores(self, energies_kev: tuple[float, ...]) -> tuple[float, ...]:
        if self.photoelectric_table is not None and energies_kev != self.photoelectric_energies_kev:
            raise ContractError("photoelectric scores retain their declared energy nodes")
        if self.kind == "interaction-counting":
            return (1.0,) * len(energies_kev)
        if self.kind == "ideal-full-energy":
            return energies_kev
        assert self.mean_deposited_energy_kev is not None
        if len(self.mean_deposited_energy_kev) != len(energies_kev):
            raise ContractError("one conditional deposited energy is required per energy node")
        if any(
            deposit > energy
            for deposit, energy in zip(self.mean_deposited_energy_kev, energies_kev, strict=True)
        ):
            raise ContractError("conditional deposited energy must not exceed incident energy")
        return self.mean_deposited_energy_kev

    @property
    def response_unit(self) -> str:
        return (
            "interactions/incident photon"
            if self.kind == "interaction-counting"
            else "keV/incident photon"
        )

    @property
    def signal_unit(self) -> str:
        return "expected interactions" if self.kind == "interaction-counting" else "keV"


def _depth(coefficient: float, thickness: float, cosine: float) -> tuple[float, float]:
    if coefficient == 0.0 or thickness == 0.0:
        return 0.0, -math.inf
    product = coefficient * thickness
    normal = 2.2250738585072014e-308
    if math.isfinite(product) and product >= normal:
        direct = product / cosine
        return direct, math.log(direct)
    ratio = coefficient / cosine
    if math.isfinite(ratio) and ratio >= normal:
        direct = ratio * thickness
        if math.isfinite(direct) and direct >= normal:
            return direct, math.log(direct)
    logarithm = math.log(coefficient) + math.log(thickness) - math.log(cosine)
    try:
        depth = math.exp(logarithm)
    except OverflowError:
        depth = math.inf
    return depth, logarithm


def _response_value(coefficient: float, thickness: float, cosine: float, score: float) -> float:
    if coefficient == 0.0 or thickness == 0.0 or score == 0.0:
        return 0.0
    depth, logarithm = _depth(coefficient, thickness, cosine)
    if depth < 2.2250738585072014e-308:
        # The relative correction is below binary64 epsilon here. Combining
        # exponents retains a scored tail even if the probability underflows.
        return math.exp(math.log(score) + logarithm)
    return score * -math.expm1(-depth)


def _energy_weighted_branch(energy: float, branch: float, total: float) -> float:
    if branch == 0.0 or total == 0.0:
        return 0.0
    fraction = branch / total
    if fraction >= 2.2250738585072014e-308:
        return energy * fraction
    product = energy * branch
    if math.isfinite(product) and product >= 2.2250738585072014e-308:
        return product / total
    return math.exp(math.log(energy) + math.log(branch) - math.log(total))


@dataclass(frozen=True, slots=True)
class DetectorResponse:
    """A thickness-derived response on explicit energy nodes.

    For total primary attenuation μ(E) in mm⁻¹ and slab-normal thickness t in mm,
    the first-interaction probability is ``-expm1(-μ t / cosine)``. The mandatory
    model selects the conditional score. Supply a deposition model to account for
    partial energy deposition; the ideal full-energy model scores each interaction
    by E. Edge-side and the original material/deposition provenance are retained.
    """

    material: MaterialTable
    energies_kev: tuple[float, ...]
    thickness_mm: float
    model: FirstInteractionModel
    edge_side: Literal["below", "above"]
    incidence_cosine: float = 1.0
    values: tuple[float, ...] = field(init=False)
    recipe_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        from dpt.materials import validate_energy_grid

        if not isinstance(cast(object, self.material), MaterialTable):
            raise ContractError("scintillator material must be a total-attenuation MaterialTable")
        if not isinstance(cast(object, self.model), FirstInteractionModel):
            raise ContractError("response requires a FirstInteractionModel")
        if self.model.photoelectric_table is not None and (
            self.model.photoelectric_total != self.material
            or self.model.photoelectric_energies_kev != self.energies_kev
            or self.model.photoelectric_edge_side != self.edge_side
        ):
            raise ContractError(
                "photoelectric score recipe must retain its total table, energies and edge side"
            )
        validate_energy_grid(self.energies_kev)
        thickness = finite_scalar(self.thickness_mm, "normal thickness in mm", minimum=0)
        cosine = finite_scalar(self.incidence_cosine, "incidence cosine", minimum=0)
        if not 0.0 < cosine <= 1.0:
            raise ContractError("incidence cosine must be in (0,1]")
        object.__setattr__(self, "thickness_mm", thickness)
        object.__setattr__(self, "incidence_cosine", cosine)
        coefficients = self.material.at_energies(self.energies_kev, edge_side=self.edge_side)
        scores = self.model.scores(self.energies_kev)
        values = tuple(
            _response_value(mu, thickness, cosine, score)
            for mu, score in zip(coefficients, scores, strict=True)
        )
        if any(not math.isfinite(value) for value in values):
            raise NumericalError("detector response cannot be represented in binary64")
        object.__setattr__(self, "values", values)
        recipe = {
            "schema_version": 1,
            "operation": "first-interaction-slab-response",
            "material": asdict(self.material),
            "energies_kev": self.energies_kev,
            "thickness_mm": thickness,
            "incidence_cosine": cosine,
            "model": asdict(self.model),
            "edge_side": self.edge_side,
        }
        encoded = json.dumps(recipe, sort_keys=True, separators=(",", ":"), allow_nan=False)
        object.__setattr__(self, "recipe_sha256", hashlib.sha256(encoded.encode()).hexdigest())

    @property
    def response_unit(self) -> str:
        return self.model.response_unit

    @property
    def source_provenance(self) -> tuple[Provenance | None, Provenance | None]:
        return self.material.provenance, self.model.provenance

    def thickness_vjp(self, cotangent: tuple[float, ...]) -> float:
        """Return ``sum_k g[k] score[k] (mu[k]/c) exp(-mu[k] t/c)``.

        ``t`` is the recipe's normal thickness in mm and ``c`` its dimensionless
        incidence cosine. ``mu[k]`` is total attenuation in mm^-1 at the fixed
        energy nodes, and ``score[k]`` is the declared conditional first-interaction
        score. ``cotangent`` is an immutable tuple ``(K,)`` of finite signed weights,
        with one entry per response value.

        The result is a binary64 Python scalar with units of the seeded scalar per
        mm. It uses the current immutable recipe and stable attenuated products
        with an accurate host sum. Material interpolation, energy nodes, incidence
        cosine and the conditional score remain fixed, and caller inputs are unchanged.

        Example::

            import math
            from dpt.detector_physics import DetectorResponse, FirstInteractionModel
            from dpt.materials import MaterialTable

            material = MaterialTable("constant attenuation example", (40.0, 80.0),
                                     (0.1, 0.1), "mm^-1")
            response = DetectorResponse(material, (40.0, 80.0), 2.0,
                                        FirstInteractionModel.ideal_full_energy(), "above", 0.5)
            gradient = response.thickness_vjp(cotangent=(1.0, -0.25))
            assert math.isclose(gradient, 4.0 * math.exp(-0.4), rel_tol=1e-12)

        """
        finite_tuple(cotangent, "response cotangent", minimum=None, length=len(self.values))
        coefficients = self.material.at_energies(self.energies_kev, edge_side=self.edge_side)
        terms: list[float] = []
        for mu, score, weight in zip(
            coefficients, self.model.scores(self.energies_kev), cotangent, strict=True
        ):
            if mu == 0.0 or score == 0.0 or weight == 0.0:
                terms.append(0.0)
                continue
            depth, _ = _depth(mu, self.thickness_mm, self.incidence_cosine)
            exponent = (
                math.log(abs(weight))
                + math.log(score)
                + math.log(mu)
                - math.log(self.incidence_cosine)
                - depth
            )
            weighted = mu * score
            numerator = weighted * weight
            direct = numerator / self.incidence_cosine
            try:
                value = (
                    direct * math.exp(-depth)
                    if (
                        math.isfinite(direct)
                        and direct != 0.0
                        and abs(weighted) >= 2.2250738585072014e-308
                        and abs(numerator) >= 2.2250738585072014e-308
                        and depth < 600
                    )
                    else math.copysign(math.exp(exponent), weight)
                )
                terms.append(value)
            except OverflowError as error:
                raise NumericalError("response thickness cotangent overflows binary64") from error
        try:
            return math.fsum(terms)
        except OverflowError as error:
            raise NumericalError("response thickness VJP overflows binary64") from error

    def to_host(self, *, dtype: Literal["float32", "float64"] = "float32") -> Any:
        """Convert the shared K-vector once at the explicit ingestion boundary."""
        return _host_values(self.values, dtype=dtype)

    def to_device(
        self, context: DeviceContext, *, dtype: Literal["float32", "float64"] = "float32"
    ) -> Any:
        return _upload_values(self.to_host(dtype=dtype), context, dtype=dtype)


@dataclass(frozen=True, slots=True)
class SolidAngleWorkspace:
    """Own O(views) compact panel metadata and a sticky diagnostic word."""

    geometry: BatchedGeometry
    context: DeviceContext
    _metadata: Any = field(repr=False)
    _modes: Any = field(repr=False)
    _kernels: Any = field(repr=False)
    _status: Any = field(repr=False)
    _readback: StatusReadback = field(repr=False)

    @property
    def scratch_bytes(self) -> int:
        return len(self.geometry.views) * (6 * 8 + 4) + 4

    @property
    def host_bytes(self) -> int:
        return self._readback.host_bytes

    def check_status(self) -> None:
        raise_device_status(self._readback.read(), "pixel solid angle")

    def clear_status(self) -> None:
        with self.context.scope():
            self._status.zero_()


def prepare_pixel_solid_angle(
    geometry: BatchedGeometry | DetectorGeometry | CurvedDetectorGeometry,
    *,
    device: str | wp.Device = "cuda:0",
    stream: wp.Stream | None = None,
    max_owned_bytes: int = 1 << 20,
) -> SolidAngleWorkspace:
    """Prepare exact planar or source-centred cylindrical pixel solid angles.

    The cylindrical formula integrates the curved surface itself, including finite
    row and arc-width boundaries. Pixel widths are the geometry's physical spacing;
    the first pixel is centred at origin_mm. A full-circle cylindrical pixel is
    rejected. Parallel rays have no shared point-source solid angle.
    """
    require_no_tape()
    if isinstance(cast(object, geometry), (DetectorGeometry, CurvedDetectorGeometry)):
        geometry = BatchedGeometry((cast(DetectorGeometry | CurvedDetectorGeometry, geometry),))
    if not isinstance(cast(object, geometry), BatchedGeometry):
        raise ContractError("pixel solid angle requires compact point-source panel geometry")
    geometry = cast(BatchedGeometry, geometry)
    integer(max_owned_bytes, "max_owned_bytes", minimum=1, maximum=2**63 - 1)
    if 52 * len(geometry.views) + 8 > max_owned_bytes:
        raise ContractError("solid-angle metadata exceeds max_owned_bytes")
    metadata: list[tuple[float, ...]] = []
    modes: list[int] = []
    for panel in geometry.views:
        if isinstance(panel, ParallelGeometry):
            raise ContractError("parallel rays have no shared point-source solid angle")
        delta = tuple(panel.origin_mm[i] - panel.source_mm[i] for i in range(3))
        x = dot(cast(Any, delta), panel.u)
        y = dot(cast(Any, delta), panel.v)
        distance = abs(dot(cast(Any, delta), cross(panel.u, panel.v)))
        du, dv = panel.spacing_mm
        curved = isinstance(panel, CurvedDetectorGeometry)
        if curved:
            distance = panel.radius_mm
            if du * panel.shape[1] >= 2 * math.pi * distance:
                raise ContractError("curved pixel boundaries must cover less than one revolution")
        values = (x, y, distance, du, dv, distance if curved else 0.0)
        if any(not math.isfinite(value) for value in values):
            raise ContractError("relative panel coordinates overflow binary64")
        if not all(
            math.isfinite(start + spacing * count)
            for start, spacing, count in ((x, du, panel.shape[1]), (y, dv, panel.shape[0]))
        ):
            raise ContractError("panel extent overflows binary64")
        metadata.append(values)
        modes.append(int(curved))
    ctx = prepare_context(device=device, stream=stream)
    kernels = load_kernels("dpt.kernels.detector_physics")
    with ctx.scope():
        packed = ctx.wp.array(metadata, dtype=ctx.wp.float64, device=ctx.device)
        mode_array = ctx.wp.array(modes, dtype=ctx.wp.int32, device=ctx.device)
        status = ctx.wp.zeros(1, dtype=ctx.wp.int32, device=ctx.device)
    ctx.wp.synchronize_stream(ctx.stream)
    return SolidAngleWorkspace(
        geometry, ctx, packed, mode_array, kernels, status, StatusReadback(ctx, status)
    )


def pixel_solid_angle(
    *,
    out_solid_angle: wp.array[wp.float64],
    out_incidence_cosine: wp.array[wp.float64],
    workspace: SolidAngleWorkspace,
    stream: wp.Stream | None = None,
    validate: bool = True,
) -> None:
    """Write ``Omega[p] = integral_pixel |n dot r|/|r|^3 dA`` and centre incidence cosine.

    Both caller-owned outputs are contiguous binary64
    ``(views, rows, columns)`` arrays in view/row/column order.
    ``out_solid_angle`` is in steradians and ``out_incidence_cosine`` is
    dimensionless. The workspace retains fixed point-source flat/cylindrical
    panel geometry in mm. Parallel rays do not define a shared point-source
    solid angle and are rejected at preparation.

    Flat rectangles use robust triangle solid angles after partitioning at the
    source-projection axes (van Oosterom and Strackee, 1983,
    doi:10.1109/TBME.1983.325207). Cylindrical panels integrate
    ``R^2/(R^2+z^2)^(3/2)`` over arc angle and axial height. These are finite-pixel
    acceptances, without a centre-area/distance-squared approximation.

    Outputs must be disjoint on the workspace device. The call overwrites them,
    returns ``None`` and runs on the owning stream with no per-pixel allocation.
    Geometry remains fixed, and acquisition derivatives belong to the ray
    operators. ``validate=False`` requires ``workspace.check_status()`` before use.

    Example::

        import math
        import warp as wp
        from dpt.detector_physics import pixel_solid_angle, prepare_pixel_solid_angle
        from dpt.geometry import BatchedGeometry, DetectorGeometry

        panel = DetectorGeometry((0.0, 0.0, -10.0), (0.0, 0.0, 0.0),
                                 (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (2.0, 2.0), (1, 1))
        workspace = prepare_pixel_solid_angle(BatchedGeometry((panel,)), device="cpu")
        solid_angle = wp.empty((1, 1, 1), dtype=wp.float64, device="cpu")
        cosine = wp.empty_like(solid_angle)
        pixel_solid_angle(out_solid_angle=solid_angle, out_incidence_cosine=cosine,
                          workspace=workspace)
        expected = 4.0 * math.atan(1.0 / (10.0 * math.sqrt(102.0)))
        assert math.isclose(float(solid_angle.numpy()[0, 0, 0]), expected, rel_tol=1e-12)
        assert float(cosine.numpy()[0, 0, 0]) == 1.0

    """
    # Runtime contracts validate the dynamically loaded Warp boundary.
    _api_out_solid_angle: Any = cast(Any, out_solid_angle)
    _api_out_incidence_cosine: Any = cast(Any, out_incidence_cosine)
    require_no_tape()
    if type(validate) is not bool:
        raise ContractError("validate must be a boolean")
    ctx = workspace.context
    ctx.assert_stream(stream)
    for name, values in (
        ("out_solid_angle", _api_out_solid_angle),
        ("out_incidence_cosine", _api_out_incidence_cosine),
    ):
        ctx.array(values, name, dtype=ctx.wp.float64, ndim=3, shape=workspace.geometry.shape)
    ctx.disjoint(
        [],
        [
            ("out_solid_angle", _api_out_solid_angle),
            ("out_incidence_cosine", _api_out_incidence_cosine),
        ],
    )
    if validate:
        workspace.check_status()
    ctx.wp.launch(
        workspace._kernels.pixel_acceptance,
        dim=workspace.geometry.shape,
        inputs=[workspace._metadata, workspace._modes, *workspace.geometry.shape_rows_columns],
        outputs=[
            _api_out_solid_angle.flatten(),
            _api_out_incidence_cosine.flatten(),
            workspace._status,
        ],
        device=ctx.device,
        stream=ctx.stream,
        record_tape=False,
    )
    if validate:
        workspace.check_status()


@dataclass(frozen=True, slots=True)
class ScintillatorWorkspace:
    """Own fixed K factors; response images and active parameters remain caller-owned."""

    material: MaterialTable
    energies_kev: tuple[float, ...]
    model: FirstInteractionModel
    edge_side: Literal["below", "above"]
    max_pixels: int
    precision: Literal["float32", "float64"]
    context: DeviceContext
    _coefficients: Any = field(repr=False)
    _scores: Any = field(repr=False)
    _kernels: Any = field(repr=False)
    _status: Any = field(repr=False)
    _readback: StatusReadback = field(repr=False)

    @property
    def scratch_bytes(self) -> int:
        return 16 * len(self.energies_kev) + 4

    @property
    def host_bytes(self) -> int:
        return self._readback.host_bytes

    @property
    def dtype(self) -> Any:
        return self.context.wp.float64 if self.precision == "float64" else self.context.wp.float32

    def check_status(self) -> None:
        raise_device_status(self._readback.read(), "scintillator response")

    def clear_status(self) -> None:
        with self.context.scope():
            self._status.zero_()


def prepare_scintillator_response(
    material: MaterialTable,
    energies_kev: tuple[float, ...],
    *,
    model: FirstInteractionModel,
    edge_side: Literal["below", "above"],
    max_pixels: int,
    precision: Literal["float32", "float64"] = "float32",
    device: str | wp.Device = "cuda:0",
    stream: wp.Stream | None = None,
    max_owned_bytes: int = 1 << 20,
) -> ScintillatorWorkspace:
    """Retain total μ[K] in mm⁻¹ and conditional score[K] at supplied energies.

    The first-interaction model is mandatory. FP64 thickness [P] and incidence
    cosine [P] can change each call. Response [K,P] uses the selected precision;
    its explicit VJP returns FP64 cotangents per pixel in fixed energy order.
    """
    require_no_tape()
    recipe = DetectorResponse(material, energies_kev, 0.0, model, edge_side)
    integer(max_pixels, "max_pixels", minimum=1)
    integer(max_owned_bytes, "max_owned_bytes", minimum=1, maximum=2**63 - 1)
    if precision not in ("float32", "float64"):
        raise ContractError("response precision must be float32 or float64")
    if len(energies_kev) * max_pixels > 2**31 - 1:
        raise ContractError("response exceeds signed 32-bit indexing")
    if 16 * len(energies_kev) + 8 > max_owned_bytes:
        raise ContractError("scintillator factors exceed max_owned_bytes")
    ctx = prepare_context(device=device, stream=stream)
    kernels = load_kernels("dpt.kernels.detector_physics")
    with ctx.scope():
        coefficients = ctx.wp.array(
            material.at_energies(energies_kev, edge_side=edge_side),
            dtype=ctx.wp.float64,
            device=ctx.device,
        )
        scores = ctx.wp.array(
            recipe.model.scores(energies_kev), dtype=ctx.wp.float64, device=ctx.device
        )
        status = ctx.wp.zeros(1, dtype=ctx.wp.int32, device=ctx.device)
    ctx.wp.synchronize_stream(ctx.stream)
    return ScintillatorWorkspace(
        material,
        energies_kev,
        model,
        edge_side,
        max_pixels,
        precision,
        ctx,
        coefficients,
        scores,
        kernels,
        status,
        StatusReadback(ctx, status),
    )


def _slab_inputs(
    thickness_mm: Any,
    incidence_cosine: Any,
    workspace: ScintillatorWorkspace,
    stream: wp.Stream | None,
    validate: bool,
) -> int:
    require_no_tape()
    if type(validate) is not bool:
        raise ContractError("validate must be a boolean")
    ctx = workspace.context
    ctx.assert_stream(stream)
    pixels = ctx.array(thickness_mm, "thickness_mm", dtype=ctx.wp.float64)
    if not 1 <= pixels <= workspace.max_pixels:
        raise ContractError("thickness_mm must contain 1..max_pixels values")
    ctx.array(incidence_cosine, "incidence_cosine", dtype=ctx.wp.float64, shape=(pixels,))
    if validate:
        workspace.check_status()
    return pixels


def scintillator_response(
    thickness_mm: wp.array[wp.float64],
    incidence_cosine: wp.array[wp.float64],
    *,
    out_response: wp.array[wp.float32] | wp.array[wp.float64],
    workspace: ScintillatorWorkspace,
    stream: wp.Stream | None = None,
    validate: bool = True,
) -> None:
    """Overwrite ``response[k,p] = score[k] (1-exp(-mu[k] thickness[p]/cosine[p]))``.

    ``thickness_mm`` and ``incidence_cosine`` are contiguous binary64 ``(P,)``
    arrays, with ``1 <= P <= workspace.max_pixels``. Normal thickness is finite
    and nonnegative in mm, and cosine is finite in ``(0,1]``. The fixed ``K``
    energy nodes are in keV and their total attenuation coefficients in mm^-1.
    The declared model supplies a conditional first-interaction score.

    The caller-owned destination is energy-major ``(K,P)`` in the workspace's
    binary32/binary64 precision. Its units are interactions/incident photon for
    interaction counting or keV/incident photon for deposited-energy models.
    Stable binary64 probability/product arithmetic precedes final output
    rounding. Inputs and output must be disjoint on the workspace device.
    The call overwrites the response, leaves inputs unchanged and returns
    ``None`` on the owning stream without response-sized scratch.
    ``validate=False`` requires ``workspace.check_status()`` before response use.

    Example::

        import math
        import warp as wp
        from dpt.detector_physics import FirstInteractionModel, prepare_scintillator_response
        from dpt.materials import MaterialTable

        material = MaterialTable("constant attenuation example", (40.0, 80.0),
                                 (0.1, 0.1), "mm^-1")
        workspace = prepare_scintillator_response(
            material, (40.0, 80.0), model=FirstInteractionModel.ideal_full_energy(),
            edge_side="above", max_pixels=1, precision="float64", device="cpu",
        )
        thickness = wp.array([2.0], dtype=wp.float64, device="cpu")
        cosine = wp.array([0.5], dtype=wp.float64, device="cpu")

        from dpt.detector_physics import scintillator_response
        response = wp.empty((2, 1), dtype=wp.float64, device="cpu")
        scintillator_response(thickness, cosine, out_response=response, workspace=workspace)
        expected = 40.0 * -math.expm1(-0.4)
        assert math.isclose(float(response.numpy()[0, 0]), expected, rel_tol=1e-12)

    """
    # Runtime contracts validate the dynamically loaded Warp boundary.
    _api_thickness_mm: Any = cast(Any, thickness_mm)
    _api_incidence_cosine: Any = cast(Any, incidence_cosine)
    _api_out_response: Any = cast(Any, out_response)
    pixels = _slab_inputs(_api_thickness_mm, _api_incidence_cosine, workspace, stream, validate)
    ctx = workspace.context
    ctx.array(
        _api_out_response,
        "out_response",
        dtype=workspace.dtype,
        ndim=2,
        shape=(len(workspace.energies_kev), pixels),
    )
    ctx.disjoint(
        [("thickness_mm", _api_thickness_mm), ("incidence_cosine", _api_incidence_cosine)],
        [("out_response", _api_out_response)],
    )
    ctx.wp.launch(
        workspace._kernels.response_forward(workspace.precision == "float64"),
        dim=(len(workspace.energies_kev), pixels),
        inputs=[
            _api_thickness_mm,
            _api_incidence_cosine,
            workspace._coefficients,
            workspace._scores,
        ],
        outputs=[_api_out_response, workspace._status],
        device=ctx.device,
        stream=ctx.stream,
        record_tape=False,
    )
    if validate:
        workspace.check_status()


def scintillator_response_vjp(
    thickness_mm: wp.array[wp.float64],
    incidence_cosine: wp.array[wp.float64],
    *,
    cotangent: wp.array[wp.float32] | wp.array[wp.float64] | None,
    out_thickness: wp.array[wp.float64],
    out_incidence_cosine: wp.array[wp.float64],
    workspace: ScintillatorWorkspace,
    stream: wp.Stream | None = None,
    validate: bool = True,
) -> None:
    """Write ``bar_t[p]=sum_k g[k,p] score[k] (mu[k]/c[p]) exp(-mu[k] t[p]/c[p])``.

    The cosine derivative is
    ``bar_c[p]=-sum_k g[k,p] score[k] mu[k] t[p]/c[p]^2 exp(-mu[k] t[p]/c[p])``.
    ``thickness_mm=t`` and ``incidence_cosine=c`` are binary64 ``(P,)``
    arrays with finite nonnegative thickness in mm and cosine in ``(0,1]``.
    The signed ``cotangent=g`` is energy-major ``(K,P)`` in the workspace's
    response precision, binary32 or binary64. Material interpolation, energies and the
    conditional interaction score remain fixed.

    Both required caller-owned destinations are binary64 ``(P,)``.
    ``out_thickness`` is the seeded scalar's derivative per mm and
    ``out_incidence_cosine`` its derivative per unit cosine. Each pixel sums
    energies in ascending order with stable attenuated products, without
    atomics or a ``(K,P)`` derivative tape. All buffers are contiguous,
    disjoint and on the workspace device. The call recomputes the derivatives,
    overwrites both destinations, leaves inputs unchanged and returns ``None``
    on the owning stream. Unchecked calls still diagnose invalid current inputs,
    and require ``workspace.check_status()`` before gradient use.

    Example::

        import math
        import warp as wp
        from dpt.detector_physics import FirstInteractionModel, prepare_scintillator_response
        from dpt.materials import MaterialTable

        material = MaterialTable("constant attenuation example", (40.0, 80.0),
                                 (0.1, 0.1), "mm^-1")
        workspace = prepare_scintillator_response(
            material, (40.0, 80.0), model=FirstInteractionModel.ideal_full_energy(),
            edge_side="above", max_pixels=1, precision="float64", device="cpu",
        )
        thickness = wp.array([2.0], dtype=wp.float64, device="cpu")
        cosine = wp.array([0.5], dtype=wp.float64, device="cpu")

        from dpt.detector_physics import scintillator_response_vjp
        cotangent = wp.array([[1.0], [1.0]], dtype=wp.float64, device="cpu")
        thickness_gradient = wp.empty(1, dtype=wp.float64, device="cpu")
        cosine_gradient = wp.empty(1, dtype=wp.float64, device="cpu")
        scintillator_response_vjp(
            thickness, cosine, cotangent=cotangent, out_thickness=thickness_gradient,
            out_incidence_cosine=cosine_gradient, workspace=workspace,
        )
        assert math.isclose(float(thickness_gradient.numpy()[0]), 24.0 * math.exp(-0.4),
                            rel_tol=1e-12)
        assert math.isclose(float(cosine_gradient.numpy()[0]), -96.0 * math.exp(-0.4),
                            rel_tol=1e-12)

    """
    # Runtime contracts validate the dynamically loaded Warp boundary.
    _api_thickness_mm: Any = cast(Any, thickness_mm)
    _api_incidence_cosine: Any = cast(Any, incidence_cosine)
    _api_cotangent: Any = cast(Any, cotangent)
    _api_out_thickness: Any = cast(Any, out_thickness)
    _api_out_incidence_cosine: Any = cast(Any, out_incidence_cosine)
    pixels = _slab_inputs(_api_thickness_mm, _api_incidence_cosine, workspace, stream, validate)
    ctx = workspace.context
    ctx.array(
        _api_cotangent,
        "cotangent",
        dtype=workspace.dtype,
        ndim=2,
        shape=(len(workspace.energies_kev), pixels),
    )
    for name, values in (
        ("out_thickness", _api_out_thickness),
        ("out_incidence_cosine", _api_out_incidence_cosine),
    ):
        ctx.array(values, name, dtype=ctx.wp.float64, shape=(pixels,))
    ctx.disjoint(
        [
            ("thickness_mm", _api_thickness_mm),
            ("incidence_cosine", _api_incidence_cosine),
            ("cotangent", _api_cotangent),
        ],
        [
            ("out_thickness", _api_out_thickness),
            ("out_incidence_cosine", _api_out_incidence_cosine),
        ],
    )
    ctx.wp.launch(
        workspace._kernels.response_reverse(workspace.precision == "float64"),
        dim=pixels,
        inputs=[
            _api_thickness_mm,
            _api_incidence_cosine,
            workspace._coefficients,
            workspace._scores,
            _api_cotangent,
            len(workspace.energies_kev),
        ],
        outputs=[_api_out_thickness, _api_out_incidence_cosine, workspace._status],
        device=ctx.device,
        stream=ctx.stream,
        record_tape=False,
    )
    if validate:
        workspace.check_status()

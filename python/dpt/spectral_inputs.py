"""Explicit supplied-data ingestion shared by spectral experiment drivers.

JSON records frames, units, array roles and provenance; NPZ stores only numeric
arrays. Object arrays/pickles, implicit precision changes and omitted physical
metadata are rejected. This preparation boundary uploads once and synchronises
before releasing the source host arrays. No physical data are bundled here.
"""

from __future__ import annotations

# JSON input still requires runtime type checks despite annotated helper calls.
# pyright: reportUnnecessaryIsInstance=false, reportPrivateUsage=false
import importlib
from pathlib import Path
from typing import Any

from dpt._runtime import prepare_context
from dpt.contracts import ContractError, integer
from dpt.detector import BlurSpec
from dpt.geometry import DetectorGeometry, RigidTransform
from dpt.materials import (
    MassFractionMixture,
    MaterialBasis,
    MaterialTable,
    Provenance,
    _host_values,
    validate_energy_grid,
)
from dpt.objectives import ObjectiveSpec
from dpt.recovery import PoseChart
from dpt.spectral import SpectralSpec
from dpt.spectral_recovery import CalibrationBlock, SpectralPoseProblem, SpectralView
from dpt.volumes import GridSpec


def _keys(value: dict[str, Any], allowed: set[str], required: set[str], where: str) -> None:
    unknown, missing = set(value) - allowed, required - set(value)
    if unknown or missing:
        raise ContractError(
            f"{where}: unknown keys {sorted(unknown)}; missing keys {sorted(missing)}"
        )


def _provenance(value: dict[str, Any] | None) -> Provenance | None:
    return None if value is None else Provenance(**value)


def _supplied_material(value: dict[str, Any]) -> MaterialTable | MassFractionMixture:
    values = dict(value)
    kind = values.pop("kind", "table")
    values["provenance"] = _provenance(values.get("provenance"))
    if kind == "mass-fraction-mixture":
        components: list[MaterialTable] = []
        for item in values["tables"]:
            component = _supplied_material(item)
            if not isinstance(component, MaterialTable):
                raise ContractError(
                    "supply component mass tables directly, without nested mixtures"
                )
            components.append(component)
        values["tables"] = tuple(components)
        values["mass_fractions"] = tuple(values["mass_fractions"])
        return MassFractionMixture(**values)
    if kind != "table":
        raise ContractError("supplied material kind must be table or mass-fraction-mixture")
    values["energies_kev"] = tuple(values["energies_kev"])
    values["coefficients"] = tuple(values["coefficients"])
    return MaterialTable(**values)


def read_spectral_inputs(
    archive: Path,
    metadata: dict[str, Any],
    *,
    samples_per_ray: int,
    device: str = "cuda:0",
    stream: Any = None,
) -> tuple[SpectralPoseProblem, PoseChart]:
    """S[p]=sum_k weights[k,p]*response[k,p]*exp(-sum_m mu[m,k]*A[m,p]).

    Assemble the fixed inputs for this spectral model and its pose/calibration
    recovery problem from a numeric NPZ ``archive`` and schema-version-1
    ``metadata``. With M materials, K energy nodes, V voxels and P pixels per
    view, material fractions accept (M,nz,ny,nx), (M,V) or (M*V,), coefficients
    accept (M,K) or (M*K,), and observations accept (height,width) or (P,).
    Columns/x vary fastest and material/energy axes are outer axes. Spectrum
    and response accept (K,) when shared or (K,height,width), (K,P) or (K*P,)
    when pixel-specific. Objective weights use observation layouts.

    NPZ arrays must already have native contiguous binary32 storage and are
    uploaded once as flat Warp arrays to the explicit device, defaulting to
    cuda:0. ``device="cpu"`` selects CPU allocations and accepts no CUDA stream.
    Fractions are dimensionless, coefficients are total attenuation in mm^-1,
    geometry uses mm and energy nodes use keV. Spectrum weights contain
    integrated photon populations and response converts those populations to
    the declared output unit. Observations share that unit. Supplied material
    tables require an explicit point/edge-side or uniform-bin averaging rule.

    Return (``SpectralPoseProblem``, ``PoseChart``), with owning device arrays,
    geometry, objectives and named calibration groups. In subsequent recovery
    the pose and declared group gain/exposure/offset coordinates are active,
    while fractions, coefficients, spectrum and response remain fixed. Optional
    provenance describes the original data and no pickle/object arrays or
    implicit precision conversions are accepted.

    """
    _keys(
        metadata,
        {
            "schema_version",
            "grid",
            "materials",
            "fields",
            "fields_provenance",
            "views",
            "calibration",
            "pose_chart",
        },
        {
            "schema_version",
            "grid",
            "materials",
            "fields",
            "views",
            "calibration",
        },
        "model metadata",
    )
    if metadata["schema_version"] != 1:
        raise ContractError("spectral input metadata requires schema_version=1")
    raw_grid = dict(metadata["grid"])
    for name in ("shape", "spacing_mm", "origin_mm", "orientation"):
        if name in raw_grid:
            raw_grid[name] = tuple(raw_grid[name])
    grid = GridSpec(**raw_grid)
    materials = integer(metadata["materials"], "materials", minimum=1, maximum=32)
    blocks = tuple(CalibrationBlock(**value) for value in metadata["calibration"])
    chart_values = dict(metadata.get("pose_chart", {}))
    if "anchor" in chart_values:
        anchor = dict(chart_values["anchor"])
        for name in ("rotation", "translation_mm"):
            if name in anchor:
                anchor[name] = tuple(anchor[name])
        chart_values["anchor"] = RigidTransform(**anchor)
    if "scales" in chart_values:
        chart_values["scales"] = tuple(chart_values["scales"])
    chart = PoseChart(**chart_values)
    ctx = prepare_context(device=device, stream=stream)
    np: Any = importlib.import_module("numpy")
    uploads: dict[str, tuple[tuple[int, ...], Any]] = {}
    source_arrays: list[Any] = []
    with np.load(archive, allow_pickle=False) as arrays, ctx.scope():

        def upload(name: str, shapes: tuple[tuple[int, ...], ...]) -> Any:
            if not isinstance(name, str) or not name:
                raise ContractError("array roles must name a non-empty NPZ member")
            if name in uploads:
                actual_shape, result = uploads[name]
                if actual_shape not in shapes:
                    raise ContractError(
                        f"NPZ array {name} was bound with an incompatible role shape"
                    )
                return result
            if name not in arrays:
                raise ContractError(f"NPZ archive lacks required array {name}")
            value = arrays[name]
            if value.dtype != np.dtype("float32") or not value.flags.c_contiguous:
                raise ContractError(
                    f"{name} must already be contiguous native binary32; convert explicitly"
                )
            if tuple(value.shape) not in shapes:
                raise ContractError(f"{name} has shape {value.shape}; expected one of {shapes}")
            source_arrays.append(value)
            result = ctx.wp.array(value.reshape(-1), dtype=ctx.wp.float32, device=ctx.device)
            uploads[name] = (tuple(value.shape), result)
            return result

        try:
            fields = upload(
                metadata["fields"],
                ((materials, *grid.shape), (materials, grid.voxels), (materials * grid.voxels,)),
            )
            views: list[SpectralView] = []
            for raw in metadata["views"]:
                _keys(
                    raw,
                    {
                        "name",
                        "geometry",
                        "energies_kev",
                        "coefficients",
                        "coefficients_unit",
                        "material_tables",
                        "coefficient_sampling",
                        "coefficients_provenance",
                        "weights",
                        "spectrum_provenance",
                        "response",
                        "response_provenance",
                        "shared_weights",
                        "shared_response",
                        "output_unit",
                        "input_description",
                        "observation",
                        "observation_provenance",
                        "calibration_group",
                        "objective",
                        "objective_weights",
                        "objective_weight",
                        "spatial_response",
                    },
                    {
                        "name",
                        "geometry",
                        "energies_kev",
                        "weights",
                        "response",
                        "output_unit",
                        "input_description",
                        "observation",
                        "calibration_group",
                        "objective",
                    },
                    f"view {raw.get('name', '')}",
                )
                supplied_arrays = raw.get("coefficients") is not None
                supplied_tables = raw.get("material_tables") is not None
                if supplied_arrays == supplied_tables:
                    raise ContractError(
                        "supply exactly one coefficient NPZ role or material_tables"
                    )
                if supplied_arrays and raw.get("coefficients_unit") != "mm^-1":
                    raise ContractError(
                        "NPZ coefficients must already be total attenuation in mm^-1"
                    )
                raw_geometry = {name: tuple(value) for name, value in raw["geometry"].items()}
                geometry = DetectorGeometry(**raw_geometry)
                energies = tuple(raw["energies_kev"])
                validate_energy_grid(energies)
                basis = None
                if supplied_tables:
                    tables = [_supplied_material(item) for item in raw["material_tables"]]
                    if len(tables) != materials:
                        raise ContractError(
                            "material_tables must contain one supplied table per material"
                        )
                    basis = MaterialBasis(tuple(tables), raw["input_description"])
                provenance = raw.get("coefficients_provenance")
                records = (
                    tuple(Provenance(**p) for p in provenance) if provenance is not None else None
                )
                if (
                    records is None
                    and basis is not None
                    and all(t.provenance is not None for t in basis.tables)
                ):
                    records = tuple(t.provenance for t in basis.tables if t.provenance is not None)
                spec = SpectralSpec(
                    materials,
                    len(energies),
                    records,
                    _provenance(raw.get("spectrum_provenance")),
                    _provenance(raw.get("response_provenance")),
                    raw["output_unit"],
                    raw["input_description"],
                    shared_weights=raw.get("shared_weights", True),
                    shared_response=raw.get("shared_response", True),
                )
                count, channels = geometry.pixels, spec.energies
                field_shapes = ((channels, *geometry.shape), (channels, count), (channels * count,))
                if basis is None:
                    if raw.get("coefficient_sampling") is not None:
                        raise ContractError(
                            "coefficient_sampling applies only to supplied material_tables"
                        )
                    coefficients = upload(
                        raw["coefficients"], ((materials, channels), (materials * channels,))
                    )
                else:
                    sampling = dict(raw.get("coefficient_sampling", {}))
                    if sampling.get("kind") == "point":
                        _keys(
                            sampling,
                            {"kind", "edge_side"},
                            {"kind", "edge_side"},
                            "coefficient_sampling",
                        )
                        host_coefficients = basis.to_host(energies, edge_side=sampling["edge_side"])
                    elif sampling.get("kind") == "uniform-bin-average":
                        _keys(
                            sampling,
                            {"kind", "bin_edges_kev", "weighting", "quadrature_order"},
                            {"kind", "bin_edges_kev", "weighting"},
                            "coefficient_sampling",
                        )
                        bin_edges = tuple(sampling["bin_edges_kev"])
                        if len(bin_edges) != channels + 1 or any(
                            not low <= e <= high
                            for e, low, high in zip(
                                energies, bin_edges, bin_edges[1:], strict=False
                            )
                        ):
                            raise ContractError(
                                "coefficient bins must contain their declared energy nodes"
                            )
                        host_coefficients = _host_values(
                            basis.bin_averages(
                                bin_edges,
                                weighting=sampling["weighting"],
                                quadrature_order=sampling.get("quadrature_order", 16),
                            ),
                            dtype="float32",
                        )
                    else:
                        raise ContractError(
                            "material_tables require an explicit coefficient_sampling rule"
                        )
                    source_arrays.append(host_coefficients)
                    coefficients = ctx.wp.array(
                        host_coefficients.reshape(-1), dtype=ctx.wp.float32, device=ctx.device
                    )
                weights = upload(
                    raw["weights"], ((channels,),) if spec.shared_weights else field_shapes
                )
                response = upload(
                    raw["response"], ((channels,),) if spec.shared_response else field_shapes
                )
                observation = upload(raw["observation"], (geometry.shape, (count,)))
                objective = ObjectiveSpec(**raw["objective"])
                objective_weights = (
                    upload(raw["objective_weights"], (geometry.shape, (count,)))
                    if raw.get("objective_weights") is not None
                    else None
                )
                spatial = None
                if raw.get("spatial_response") is not None:
                    values = dict(raw["spatial_response"])
                    values["weights"] = tuple(values["weights"])
                    values["provenance"] = _provenance(values.get("provenance"))
                    spatial = BlurSpec(geometry.shape[0], geometry.shape[1], **values)
                views.append(
                    SpectralView(
                        raw["name"],
                        geometry,
                        spec,
                        coefficients,
                        weights,
                        response,
                        observation,
                        raw["calibration_group"],
                        _provenance(raw.get("observation_provenance")),
                        objective,
                        objective_weights,
                        raw.get("objective_weight", 1.0),
                        spatial,
                    )
                )
            return SpectralPoseProblem(
                grid,
                materials,
                fields,
                _provenance(metadata.get("fields_provenance")),
                tuple(views),
                blocks,
                samples_per_ray,
            ), chart
        finally:
            # Source NumPy allocations must outlive every asynchronous upload.
            ctx.wp.synchronize_stream(ctx.stream)

"""Storage, arithmetic and reduction contracts for prepared scientific operators.

Each record lists buffer storage, local arithmetic, accumulated values and
reductions separately. Query these numerical policies before preparing a device
operator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast

from dpt.contracts import ContractError

if TYPE_CHECKING:
    from dpt.objectives import ObjectiveSpec
    from dpt.projection import ProjectionSpec
    from dpt.regularization import RegularizationSpec
    from dpt.spectral import SpectralSpec
    from dpt.transmission import TransmissionSpec
    from dpt.transport.model import TransportPhysics, TransportSpec

type StorageDtype = Literal["float16", "bfloat16", "float32", "float64", "int32", "uint64"]
type ArithmeticDtype = Literal["float32", "float64", "mixed_float32", "fast_float32"]


@dataclass(frozen=True, slots=True)
class ArrayPrecision:
    """One named buffer's storage dtype and numerical role."""

    name: str
    dtype: StorageDtype
    role: Literal["input", "output", "cotangent", "scratch"]


@dataclass(frozen=True, slots=True)
class PrecisionContract:
    """Effective buffer dtypes and arithmetic stages of one operator family.

    ``arithmetic`` names local field/depth arithmetic, ``accumulation`` names
    serial sums/products and ``reduction`` names the tree or atomic update.
    ``coordinates`` names geometric arithmetic, or None for a non-geometric
    operator. Nuisance cotangents may permit both floating storage dtypes.
    """

    operator: str
    arrays: tuple[ArrayPrecision, ...]
    arithmetic: ArithmeticDtype
    accumulation: Literal["float32", "float64"]
    reduction: Literal["float64", "float32-or-float64-atomic", "none"]
    coordinates: Literal["float32", "float64"] | None = None
    notes: tuple[str, ...] = ()

    def storage(self, name: str) -> StorageDtype:
        """Return a declared named dtype, failing for an unknown buffer role."""
        for array in self.arrays:
            if array.name == name:
                return array.dtype
        raise ContractError(f"{self.operator} has no precision role {name!r}")


def precision_contract(
    spec: ProjectionSpec
    | TransmissionSpec
    | SpectralSpec
    | ObjectiveSpec
    | RegularizationSpec
    | TransportSpec
    | TransportPhysics,
) -> PrecisionContract:
    """Describe an existing specification without importing its device runtime.

    Separated physics records use their execution defaults. For another selected
    execution policy, resolve it with the corresponding specification adapter
    before querying this contract.
    """
    from dpt.objectives import ObjectiveSpec
    from dpt.projection import ProjectionSpec
    from dpt.regularization import RegularizationSpec
    from dpt.spectral import SpectralSpec
    from dpt.transmission import TransmissionSpec
    from dpt.transport.model import TransportPhysics, TransportSpec

    if isinstance(spec, TransportPhysics):
        spec = TransportSpec.from_physics(spec)
    if isinstance(spec, ProjectionSpec):
        return PrecisionContract(
            "projection",
            (
                ArrayPrecision("field", spec.storage_precision, "input"),
                ArrayPrecision("pose", "float64", "input"),
                ArrayPrecision("depth", spec.precision, "output"),
                ArrayPrecision("depth_cotangent", spec.precision, "cotangent"),
                ArrayPrecision("pose_cotangent", "float64", "cotangent"),
            ),
            spec.sampling_precision,
            "float32" if spec.sampling_precision == "float32" else "float64",
            "float32-or-float64-atomic",
            "float32" if spec.sampling_precision == "float32" else "float64",
            (
                "Local FP32 arithmetic uses FP64 ray setup and final pose chain/reduction.",
                "Pose reduction uses a fixed binary64 tree.",
                "Volume cotangent storage is selected separately by volume_gradient_precision.",
                "mixed_float32 rounds local marching and interpolation weights only.",
                "fast_float32 also rounds local field products and partial sums.",
            ),
        )
    if isinstance(spec, TransmissionSpec):
        return PrecisionContract(
            "transmission",
            (
                ArrayPrecision("depth", "float32", "input"),
                ArrayPrecision("beam", "float32", "input"),
                ArrayPrecision("signal", "float32", "output"),
                ArrayPrecision("signal_cotangent", "float32", "cotangent"),
                ArrayPrecision("input_cotangent", "float32", "cotangent"),
            ),
            "float64",
            "float64",
            "float64",
            notes=(
                "The Python scalar beam is validated as a binary32 scalar.",
                "Only an active device-scalar beam requires the binary64 reduction tree.",
            ),
        )
    if isinstance(spec, SpectralSpec):
        return PrecisionContract(
            "spectral",
            (
                ArrayPrecision("paths", spec.precision, "input"),
                ArrayPrecision("coefficients", "float32", "input"),
                ArrayPrecision("weights", "float32", "input"),
                ArrayPrecision("response", "float32", "input"),
                ArrayPrecision("signal", spec.precision, "output"),
                ArrayPrecision("signal_cotangent", spec.precision, "cotangent"),
                ArrayPrecision("path_cotangent", spec.precision, "cotangent"),
            ),
            "float64" if spec.arithmetic == "strict" else "float32",
            "float64",
            "float64",
            notes=(
                "Shared coefficient/weight/response cotangents permit binary32 or binary64.",
                "Per-pixel nuisance cotangents remain binary32.",
                "float32 arithmetic rounds depth/exp, retaining binary64 products and sums.",
            ),
        )
    if isinstance(spec, ObjectiveSpec):
        return PrecisionContract(
            "objective",
            (
                ArrayPrecision("prediction", spec.precision, "input"),
                ArrayPrecision("observation", "float32", "input"),
                ArrayPrecision("weight", "float32", "input"),
                ArrayPrecision("prediction_cotangent", spec.precision, "cotangent"),
                ArrayPrecision("loss", "float64", "output"),
            ),
            "float64",
            "float64",
            "float64",
        )
    if isinstance(spec, RegularizationSpec):
        return PrecisionContract(
            "regularization",
            (
                ArrayPrecision("field", spec.precision, "input"),
                ArrayPrecision("gradient", spec.precision, "cotangent"),
                ArrayPrecision("proximal_field", spec.precision, "output"),
                ArrayPrecision("primal_dual", "float64", "scratch"),
                ArrayPrecision("value", "float64", "output"),
            ),
            "float64",
            "float64",
            "float64",
            "float64",
        )
    if isinstance(cast(object, spec), TransportSpec):
        return PrecisionContract(
            "transport",
            (
                ArrayPrecision("material_ids", "int32", "input"),
                ArrayPrecision("coefficients", "float64", "input"),
                ArrayPrecision("density", "float64", "input"),
                ArrayPrecision("positions", "float64", "input"),
                ArrayPrecision("directions", "float64", "input"),
                ArrayPrecision("weights", "float64", "input"),
                ArrayPrecision("score", "float64", "output"),
                ArrayPrecision("mean", "float64", "output"),
                ArrayPrecision("parameter_cotangent", "float64", "cotangent"),
            ),
            "float64",
            "float64",
            "float64",
            "float64",
            ("Positions and directions use three-component binary64 Warp vectors.",),
        )
    raise ContractError("precision_contract requires a supported scientific specification")

"""Recover three volume fractions from two independent energy bands.

This homogeneous-cell example uses two free coordinates and the complement
fraction. The declared path is known, and the three materials fill the cell.
Its default spectra are synthetic; supplied coefficients retain their provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from threading import Lock

from dpt._runtime import HostReadback
from dpt.contracts import (
    ContractError,
    NumericalError,
    TrialDomainError,
    binary32_scalar,
    finite_scalar,
    finite_tuple,
)
from dpt.materials import Provenance
from dpt.registration import Evaluation, RecoveryPolicy, RecoveryResult, Vector, recover_parameters
from dpt.spectral import SpectralSpec, prepare_spectral, spectral_signal, spectral_vjp

# Public input boundaries also reject invalid runtime types.
# pyright: reportUnnecessaryIsInstance=false


def fraction_complement(coordinates: Vector) -> tuple[float, float, float]:
    """Decode two fractions; the third is their sum-to-one complement."""
    if len(coordinates) != 2:
        raise TrialDomainError("supply two free material fractions")
    try:
        first, second = (finite_scalar(value, "fraction", minimum=0) for value in coordinates)
    except ContractError as error:
        raise TrialDomainError(str(error)) from error
    third = math.fsum((1.0, -first, -second))
    if third < 0:
        raise TrialDomainError("the first two material fractions must sum to at most one")
    return first, second, third


def _condition_2(matrix: tuple[tuple[float, float], tuple[float, float]]) -> float:
    """Two-by-two spectral condition, scaled before forming squared quantities."""
    scale = max(abs(value) for row in matrix for value in row)
    if scale == 0 or not math.isfinite(scale):
        return math.inf
    a, b, c, d = (value / scale for row in matrix for value in row)
    determinant = a * d - b * c
    if determinant == 0:
        return math.inf
    norm_squared = math.fsum((a * a, b * b, c * c, d * d))
    discriminant = max(0.0, norm_squared * norm_squared - 4 * determinant * determinant)
    largest_squared = (norm_squared + math.sqrt(discriminant)) / 2
    return largest_squared / abs(determinant)


def _represented_rows(
    rows: tuple[tuple[float, float], ...], name: str, count: int
) -> tuple[tuple[float, float], ...]:
    if not isinstance(rows, tuple) or len(rows) != count:
        raise ContractError(f"{name} requires {count} immutable rows")
    result: list[tuple[float, float]] = []
    for row in rows:
        supplied = finite_tuple(row, name, length=2)
        rounded = tuple(binary32_scalar(value, name, minimum=0) for value in supplied)
        if any(value > 0 and stored == 0 for value, stored in zip(supplied, rounded, strict=True)):
            raise ContractError(f"{name} loses a positive value in binary32 storage")
        result.append((rounded[0], rounded[1]))
    return tuple(result)


@dataclass(frozen=True, slots=True)
class ThreeMaterialModel:
    """Known homogeneous path and two calibrated, independent energy bands.

    Coefficients are total linear attenuation in mm^-1 at the two keV nodes.
    Rows of band_weights are integrated expected photon populations per band.
    Response is a shared nonnegative first moment. The two free volume fractions
    determine the third by complement. The local Jacobian and its global bound
    concern this cell's composition; spatial reconstruction needs its own coverage.
    """

    coefficients_mm_inverse: tuple[tuple[float, float], ...]
    energies_kev: tuple[float, float]
    path_mm: float
    band_weights: tuple[tuple[float, float], ...]
    response: tuple[float, float]
    coefficients_provenance: tuple[Provenance, ...]
    spectrum_provenance: Provenance
    response_provenance: Provenance
    maximum_condition: float = 10.0

    def __post_init__(self) -> None:
        coefficients = _represented_rows(self.coefficients_mm_inverse, "coefficients", 3)
        bands = _represented_rows(self.band_weights, "band_weights", 2)
        energies = finite_tuple(self.energies_kev, "energies_kev", positive=True, length=2)
        if energies[0] >= energies[1]:
            raise ContractError("energy nodes must be increasing")
        path = finite_scalar(self.path_mm, "path_mm", minimum=0)
        if path == 0:
            raise ContractError("path_mm must be positive and known")
        response_row = _represented_rows((self.response,), "response", 1)[0]
        if any(value <= 0 for value in response_row):
            raise ContractError("both energy nodes need positive response")
        ceiling = finite_scalar(self.maximum_condition, "maximum_condition", minimum=1)
        if (
            not isinstance(self.coefficients_provenance, tuple)
            or len(self.coefficients_provenance) != 3
            or any(not isinstance(value, Provenance) for value in self.coefficients_provenance)
            or not isinstance(self.spectrum_provenance, Provenance)
            or not isinstance(self.response_provenance, Provenance)
        ):
            raise ContractError("supply three material, spectrum and response provenance records")
        object.__setattr__(self, "coefficients_mm_inverse", coefficients)
        object.__setattr__(self, "band_weights", bands)
        object.__setattr__(self, "response", response_row)
        object.__setattr__(self, "energies_kev", (energies[0], energies[1]))
        object.__setattr__(self, "path_mm", path)
        object.__setattr__(self, "maximum_condition", ceiling)
        if any(value <= 0 or not math.isfinite(value) for value in self.open_beam):
            raise ContractError("each band needs a finite positive open beam")
        if not math.isfinite(self.contrast_condition):
            raise ContractError("three-material coefficient contrasts have rank less than two")
        if not math.isfinite(_condition_2(self.normalised_bands)):
            raise ContractError("the two energy bands are linearly dependent")
        if self.global_condition_bound > ceiling:
            raise ContractError("composition conditioning exceeds maximum_condition")

    @property
    def open_beam(self) -> tuple[float, float]:
        values = tuple(
            math.fsum(
                weight * response for weight, response in zip(row, self.response, strict=True)
            )
            for row in self.band_weights
        )
        return values[0], values[1]

    @property
    def normalised_bands(self) -> tuple[tuple[float, float], tuple[float, float]]:
        rows = tuple(
            tuple(
                weight * response / beam
                for weight, response in zip(row, self.response, strict=True)
            )
            for row, beam in zip(self.band_weights, self.open_beam, strict=True)
        )
        return (rows[0][0], rows[0][1]), (rows[1][0], rows[1][1])

    @property
    def contrast_matrix(self) -> tuple[tuple[float, float], tuple[float, float]]:
        coefficients, length = self.coefficients_mm_inverse, self.path_mm
        rows = tuple(
            tuple(length * (coefficients[m][e] - coefficients[2][e]) for m in range(2))
            for e in range(2)
        )
        return (rows[0][0], rows[0][1]), (rows[1][0], rows[1][1])

    @property
    def contrast_condition(self) -> float:
        return _condition_2(self.contrast_matrix)

    @property
    def global_condition_bound(self) -> float:
        depths = tuple(
            self.path_mm * value for row in self.coefficients_mm_inverse for value in row
        )
        try:
            bound = (
                _condition_2(self.normalised_bands)
                * math.exp(max(depths) - min(depths))
                * self.contrast_condition
            )
        except OverflowError as error:
            raise ContractError("the declared model exceeds the conditioning range") from error
        if not math.isfinite(bound):
            raise ContractError("the declared model has nonfinite conditioning")
        return bound

    def response_jacobian(
        self, coordinates: Vector
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        """Jacobian of open-beam-normalised means at the supplied fractions."""
        fractions = fraction_complement(coordinates)
        transmissions = tuple(
            math.exp(
                -self.path_mm
                * math.fsum(
                    fraction * self.coefficients_mm_inverse[m][e]
                    for m, fraction in enumerate(fractions)
                )
            )
            for e in range(2)
        )
        bands, contrasts = self.normalised_bands, self.contrast_matrix
        rows = tuple(
            tuple(
                -math.fsum(bands[b][e] * transmissions[e] * contrasts[e][j] for e in range(2))
                for j in range(2)
            )
            for b in range(2)
        )
        return (rows[0][0], rows[0][1]), (rows[1][0], rows[1][1])

    def response_condition(self, coordinates: Vector) -> float:
        return _condition_2(self.response_jacobian(coordinates))


def synthetic_model() -> ThreeMaterialModel:
    """Return the declared dyadic numerical fixture, with explicit provenance."""
    definition = (
        b"synthetic A/B/C; keV=50,100; mu=(1/8,1/32;1/32,5/32;1/16,1/16) mm^-1; "
        b"L=8 mm; bands=4096*I2; response=(1,1)"
    )
    provenance = Provenance(
        source="analytic:dpt.examples.three_material.synthetic_model",
        sha256=hashlib.sha256(definition).hexdigest(),
        rights="Original numerical fixture; Apache-2.0",
        description="Synthetic homogeneous-cell attenuation and monochromatic exposures",
    )
    return ThreeMaterialModel(
        ((0.125, 0.03125), (0.03125, 0.15625), (0.0625, 0.0625)),
        (50.0, 100.0),
        8.0,
        ((4096.0, 0.0), (0.0, 4096.0)),
        (1.0, 1.0),
        (provenance, provenance, provenance),
        provenance,
        provenance,
    )


class ThreeMaterialEvaluator:
    """Prepared FP64 spectral mean/VJP with two free fraction coordinates."""

    def __init__(
        self, model: ThreeMaterialModel, observations: tuple[float, float], *, device: str = "cpu"
    ) -> None:
        values = finite_tuple(observations, "observations", positive=True, length=2)
        observations = values[0], values[1]
        if any(value > beam for value, beam in zip(observations, model.open_beam, strict=True)):
            raise ContractError("observed means exceed the declared open beam")
        self.model, self.observations = model, observations
        spec = SpectralSpec(
            3,
            2,
            coefficients_provenance=model.coefficients_provenance,
            spectrum_provenance=model.spectrum_provenance,
            response_provenance=model.response_provenance,
            output_unit="expected photons",
            input_description="Three filling volume fractions in a known homogeneous path",
            precision="float64",
            shared_weights=False,
        )
        self.workspace = prepare_spectral(spec, max_pixels=2, device=device)
        self.context = self.workspace.context
        ctx, wp = self.context, self.context.wp
        with ctx.scope():
            self._paths = wp.empty(6, dtype=wp.float64, device=ctx.device)
            self._mean = wp.empty(2, dtype=wp.float64, device=ctx.device)
            self._seed = wp.empty(2, dtype=wp.float64, device=ctx.device)
            self._path_gradient = wp.empty(6, dtype=wp.float64, device=ctx.device)
            self._coefficients = wp.array(
                [value for row in model.coefficients_mm_inverse for value in row],
                dtype=wp.float32,
                device=ctx.device,
            )
            self._weights = wp.array(
                [model.band_weights[b][e] for e in range(2) for b in range(2)],
                dtype=wp.float32,
                device=ctx.device,
            )
            self._response = wp.array(model.response, dtype=wp.float32, device=ctx.device)
            self._paths_host = wp.empty(
                6, dtype=wp.float64, device="cpu", pinned=ctx.device.is_cuda
            )
            self._seed_host = wp.empty(2, dtype=wp.float64, device="cpu", pinned=ctx.device.is_cuda)
        self._paths_view, self._seed_view = self._paths_host.numpy(), self._seed_host.numpy()
        self._mean_readback = HostReadback(ctx, self._mean)
        self._gradient_readback = HostReadback(ctx, self._path_gradient)
        self._evaluation_lock = Lock()

    def _predict(self, coordinates: Vector) -> tuple[float, float]:
        fractions = fraction_complement(coordinates)
        if self.model.response_condition(coordinates) > self.model.maximum_condition:
            raise TrialDomainError("trial composition exceeds the conditioning limit")
        ctx = self.context
        self._paths_view[:] = [
            self.model.path_mm * fraction for fraction in fractions for _ in range(2)
        ]
        with ctx.scope():
            ctx.wp.copy(self._paths, self._paths_host, stream=ctx.stream)
            spectral_signal(
                self._paths,
                self._coefficients,
                self._weights,
                self._response,
                out_mean=self._mean,
                workspace=self.workspace,
            )
            values = self._mean_readback.read()
        prediction = float(values[0]), float(values[1])
        if any(not math.isfinite(value) or value <= 0 for value in prediction):
            raise NumericalError("spectral means must remain finite and positive")
        return prediction

    def predict(self, coordinates: Vector) -> tuple[float, float]:
        if not self._evaluation_lock.acquire(blocking=False):
            raise ContractError("a three-material evaluator cannot be used concurrently")
        try:
            return self._predict(coordinates)
        finally:
            self._evaluation_lock.release()

    def __call__(self, coordinates: Vector) -> Evaluation:
        if not self._evaluation_lock.acquire(blocking=False):
            raise ContractError("a three-material evaluator cannot be used concurrently")
        try:
            prediction = self._predict(coordinates)
            beam = self.model.open_beam
            residual = tuple(
                (value - observed) / scale
                for value, observed, scale in zip(prediction, self.observations, beam, strict=True)
            )
            loss = math.fsum(value * value / 2 for value in residual)
            self._seed_view[:] = [
                value / scale for value, scale in zip(residual, beam, strict=True)
            ]
            ctx = self.context
            with ctx.scope():
                ctx.wp.copy(self._seed, self._seed_host, stream=ctx.stream)
                spectral_vjp(
                    self._paths,
                    self._coefficients,
                    self._weights,
                    self._response,
                    cotangent=self._seed,
                    out_grad_paths=self._path_gradient,
                    workspace=self.workspace,
                )
                values = self._gradient_readback.read()
            physical = tuple(
                math.fsum(float(values[2 * m + b]) for b in range(2)) for m in range(3)
            )
            gradient = tuple(self.model.path_mm * (physical[m] - physical[2]) for m in range(2))
            if not all(math.isfinite(value) for value in (loss, *gradient)):
                raise NumericalError("the composition objective or gradient is nonfinite")
            return Evaluation(loss, gradient)
        finally:
            self._evaluation_lock.release()


@dataclass(frozen=True, slots=True)
class ThreeMaterialResult:
    fractions: tuple[float, float, float]
    contrast_condition: float
    response_condition: float
    global_condition_bound: float
    optimisation: RecoveryResult


def recover_three_material(
    observations: tuple[float, float],
    initial: Vector = (0.125, 0.125),
    *,
    model: ThreeMaterialModel | None = None,
    device: str = "cpu",
    policy: RecoveryPolicy | None = None,
) -> ThreeMaterialResult:
    """Fit a known homogeneous composition using the existing chart optimiser."""
    model = synthetic_model() if model is None else model
    fraction_complement(initial)
    evaluator = ThreeMaterialEvaluator(model, observations, device=device)
    result = recover_parameters(
        evaluator,
        initial,
        policy=policy
        or RecoveryPolicy(
            max_iterations=80,
            max_evaluations=240,
            gradient_tolerance=1e-10,
            relative_loss_tolerance=0,
            step_tolerance=0,
        ),
    )
    return ThreeMaterialResult(
        fraction_complement(result.parameters),
        model.contrast_condition,
        model.response_condition(result.parameters),
        model.global_condition_bound,
        result,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    model = synthetic_model()
    truth = (0.25, 0.375, 0.375)
    # Scalar Beer-Lambert observations are independent of the native forward.
    observations = tuple(
        4096.0
        * math.exp(
            -8.0 * math.fsum(f * model.coefficients_mm_inverse[m][e] for m, f in enumerate(truth))
        )
        for e in range(2)
    )
    result = recover_three_material(
        (observations[0], observations[1]), model=model, device=args.device
    )
    print(
        json.dumps(
            {
                "fixture": "synthetic homogeneous cell",
                "device": args.device,
                "energies_kev": model.energies_kev,
                "coefficients_mm_inverse": model.coefficients_mm_inverse,
                "path_mm": model.path_mm,
                "truth_fractions": truth,
                "observed_means": observations,
                **asdict(result),
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()

"""Matrix-free material reconstruction of declared count or mean-signal data.

This is a fixed-geometry, fixed-spectrum primary-transmission model. It receives
only fitting measurements and calibration, never reference material anatomy.
Inputs are explicit FP32 host arrays; preparation uploads private copies once.
Repeated evaluations retain CUDA fields, observations, channel buffers and
cached Warp views. Eager execution makes scalar acceptance decisions on the
host. Graph execution captures each complete bounded projected Armijo/GLL
iteration and transfers one compact control ledger before publishing an
accepted update at the callback boundary. Conditional graphs require CUDA
toolkit and driver 12.4 or newer. Legacy defaults retain independent Poisson
counts and fraction-simplex
fields. Explicit signal WLS accepts fixed masks and deterministic objective
weights; an equivalent-basis domain permits nonnegative fields without a sum
cap. Basis units/scales and the observation law belong to supplied provenance.
The module imports without optional GPU/numpy dependencies.
"""

from __future__ import annotations

# Public runtime contracts deliberately validate callers beyond static typing.
# pyright: reportUnnecessaryIsInstance=false
import copy
import importlib
import json
import math
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast

from dpt._reductions import prepare_reduction
from dpt._runtime import (
    CompletionReadback,
    HostReadback,
    StatusCheckpoint,
    StatusReadback,
    load_kernels,
    prepare_context,
    require_no_tape,
    scoped_graph_capture,
)
from dpt.contracts import ContractError, NumericalError, finite_scalar, integer
from dpt.geometry import DetectorGeometry, RigidTransform
from dpt.material_projection import (
    material_projection_vjp,
    prepare_material_projection,
    project_material_paths,
)
from dpt.objectives import ObjectiveSpec, evaluate_objective, prepare_objective
from dpt.projection import ProjectionSpec
from dpt.solver_lifecycle import LifecycleObserver, LifecycleRecorder
from dpt.spectral import SpectralSpec, prepare_spectral, spectral_signal, spectral_vjp
from dpt.volumes import GridSpec

if TYPE_CHECKING:
    import numpy as np
    import warp as wp
    from numpy.typing import NDArray


class ResidentPenalty(Protocol):
    """Prepared additive objective term with a canonical resident derivative.

    Fields are borrowed read-only FP32 material-major values. The implementation
    writes one FP64 loss and optionally adds its Euclidean derivative into the
    FP32 gradient. It uses the supplied context stream, preserves field storage
    and raises or flags nonfinite arithmetic in the supplied int32 status. All
    shape-specific scratch must be prepared before a solve. Convexity is not
    assumed by the line search; no preconditioner bound is inferred for a plugin.
    """

    def __call__(
        self, fields: Any, /, *, out_loss: Any, gradient: Any, status: Any, context: Any
    ) -> None: ...


class GraphResidentPenalty(ResidentPenalty, Protocol):
    """Resident penalty that explicitly supports CUDA graph capture.

    ``graph_safe`` must be exactly True. Each invocation enqueues only work on
    the supplied stream: no allocation, host array export, synchronisation or
    data-dependent Python branching. Prepared buffers and launch bindings stay
    fixed and alive for the solver's lifetime. The callable writes its loss on
    every invocation and reports arithmetic failures in the supplied status.
    """

    @property
    def graph_safe(self) -> bool: ...


class FisherResidentPenalty(ResidentPenalty, Protocol):
    """Resident penalty with a declared smooth Hessian row majorant.

    ``add_fisher_curvature`` receives an initially zero FP64 array of V voxel
    scalars. It adds a finite nonnegative bound k[v] such that, in material-major
    Euclidean coordinates, the penalty Hessian is bounded above by
    diag(k[v] * minimum_metric_eigenvalue), repeated for each material. Hence
    blockdiag(k[v] * H) majorises the Hessian for the fixed positive metric H.
    The bound includes all spatial and material couplings. Fields are borrowed
    read-only; gradients and fitting inputs remain unchanged. Only resident
    launches and the supplied status/stream are used.

    A nonsmooth subgradient or a proximal map does not supply this capability.
    """

    def add_fisher_curvature(
        self,
        fields: Any,
        /,
        *,
        out_curvature: Any,
        minimum_metric_eigenvalue: float,
        status: Any,
        context: Any,
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class MaterialReconstructionView:
    """Independent channels at one fixed geometry; no observations are generated.

    ``counts`` has shape (C,H,W), integer detected counts represented in FP32.
    ``weights`` and ``response`` have shape (C,E); weights are already integrated
    expected open-beam photon populations per pixel, response is a detection
    probability in [0,1]. Independent exposures or disjoint Poisson-thinned
    channels must be justified by the caller. Overlapping cumulative thresholds
    are not independent channels. No normalisation or inverse-square adjustment
    is inferred. Inputs must be finite, native-endian, contiguous FP32 arrays.
    """

    geometry: DetectorGeometry
    pose: RigidTransform
    counts: Any
    weights: Any
    response: Any


@dataclass(frozen=True, slots=True)
class MaterialReconstructionSignalView:
    """Fixed-weight mean-signal observations; no independent-count noise claim.

    Observations and deterministic objective weights have shape (C,H,W),
    native contiguous FP32. ``valid`` is an immutable uint8 0/1 mask of that
    shape; masked targets may be nonfinite. Spectral weights have shape (C,E)
    or (C,E,H,W) according to shared_weights. They already include any spatial
    open-beam scale exactly once. Response is a finite nonnegative first
    moment in the declared signal unit, shared (C,E); it need not be bounded
    by one. Fixed objective
    weights are not statistical precision unless independently justified.
    """

    geometry: DetectorGeometry
    pose: RigidTransform
    observations: Any
    weights: Any
    response: Any
    objective_weights: Any
    valid: Any


@dataclass(frozen=True, slots=True)
class MaterialReconstructionSettings:
    """Dimensionless projected steps; fraction-gradient penalty beta is mm^-1.

    The stopping test is ||(f-P(f-alpha*g))/alpha||_infinity at the fixed
    ``mapping_step``. The relative threshold is fixed by its initial value.
    Trial steps grow from the last accepted step and are checked by Armijo;
    neither objective stagnation nor an iteration budget means stationarity.
    ``step_selection="bb"`` optionally proposes the metric Barzilai--Borwein
    step (s^T H s)/(s^T y), using consecutive accepted fields and gradients.
    Nonpositive or nonfinite curvature falls back to the geometric proposal.
    Its finite positive proposal is clipped to ``secant_minimum_step`` and
    ``maximum_step`` before the same constrained Armijo search.
    ``acceleration="inertial"`` adds a feasible momentum proposal using the
    preceding accepted field. A failed momentum proposal restarts the ordinary
    projected search at the same step. Every accepted state meets Armijo against
    the maximum objective in the declared nonmonotone window, using the true
    gradient and rounded displacement. With the default window of one, the
    objective decreases on every accepted step.

    ``gradient_mapping_policy="relative"`` is an opt-in scale-invariant test.
    It divides the diagnostic step by the initial gradient infinity norm and
    tests the resulting update-metric mapping divided by that same norm. The
    configured tolerances then refer to a dimensionless diagnostic. Scaling the
    complete objective by a positive constant leaves this diagnostic unchanged
    at corresponding fields, without changing the objective or search policy.
    """

    iterations: int = 100
    initial_step: float = 1.0
    maximum_step: float = 1.0
    regularisation_mm_inverse: float = 0.0
    gradient_mapping_tolerance: float = 1.0e-5
    relative_gradient_mapping_tolerance: float = 1.0e-4
    mapping_step: float = 1.0
    backtracking_factor: float = 0.5
    armijo: float = 1.0e-4
    maximum_backtracks: int = 40
    step_selection: Literal["geometric", "bb"] = "geometric"
    secant_minimum_step: float = 1.0e-16
    acceleration: Literal["none", "inertial"] = "none"
    # GLL uses the maximum of this many consecutive accepted objectives.
    # No convergence-rate promise is made for the nonlinear forward model.
    gradient_mapping_policy: Literal["fixed", "relative"] = "fixed"
    execution: Literal["eager", "graph"] = "eager"
    cache_accepted_forward: bool = True
    nonmonotone_window: int = 1
    regularisation_per_material: tuple[float, ...] | None = None
    penalty: Literal["quadratic", "huber"] = "quadratic"
    huber_gradient_delta: float = 1.0

    def __post_init__(self) -> None:
        integer(self.iterations, "iterations", minimum=1)
        if self.execution not in ("eager", "graph"):
            raise ContractError("execution must be eager or graph")
        if type(self.cache_accepted_forward) is not bool:
            raise ContractError("cache_accepted_forward must be a boolean")
        integer(self.nonmonotone_window, "nonmonotone_window", minimum=1)
        if self.nonmonotone_window > 1 and self.acceleration != "none":
            raise ContractError("nonmonotone search requires acceleration=none")
        if self.penalty not in ("quadratic", "huber"):
            raise ContractError("penalty must be quadratic or huber")
        if finite_scalar(self.huber_gradient_delta, "huber_gradient_delta", minimum=0) <= 0:
            raise ContractError("huber_gradient_delta must be positive")
        if self.regularisation_per_material is not None:
            if (
                not isinstance(self.regularisation_per_material, tuple)
                or not self.regularisation_per_material
            ):
                raise ContractError("regularisation_per_material must be a nonempty tuple")
            for beta in self.regularisation_per_material:
                finite_scalar(beta, "material regularisation", minimum=0)
        integer(self.maximum_backtracks, "maximum_backtracks", minimum=1)
        for name in ("initial_step", "maximum_step", "mapping_step"):
            if finite_scalar(getattr(self, name), name, minimum=0) <= 0:
                raise ContractError(f"{name} must be positive")
        if self.maximum_step < self.initial_step:
            raise ContractError("maximum_step must be at least initial_step")
        if self.step_selection not in ("geometric", "bb"):
            raise ContractError("step_selection must be geometric or bb")
        if self.acceleration not in ("none", "inertial"):
            raise ContractError("acceleration must be none or inertial")
        if self.gradient_mapping_policy not in ("fixed", "relative"):
            raise ContractError("gradient_mapping_policy must be fixed or relative")
        if finite_scalar(self.secant_minimum_step, "secant_minimum_step", minimum=0) <= 0:
            raise ContractError("secant_minimum_step must be positive")
        if self.step_selection == "bb" and self.secant_minimum_step > self.maximum_step:
            raise ContractError("secant_minimum_step must not exceed maximum_step")
        for name in (
            "regularisation_mm_inverse",
            "gradient_mapping_tolerance",
            "relative_gradient_mapping_tolerance",
        ):
            finite_scalar(getattr(self, name), name, minimum=0)
        for name in ("backtracking_factor", "armijo"):
            if not 0 < finite_scalar(getattr(self, name), name) < 1:
                raise ContractError(f"{name} must lie strictly between zero and one")


@dataclass(frozen=True, slots=True)
class MaterialReconstructionCheckpoint:
    """Explicit host snapshot of the accepted field and algorithm state.

    Fitting inputs and the physical model stay with the caller, identified by
    objective_id. Resume verifies the accepted loss before taking another step.
    Array exports occur only when a checkpoint is explicitly requested; no
    detector image or prediction cache is saved. NPZ storage uses no pickle.
    """

    objective_id: str
    settings: MaterialReconstructionSettings
    signature: dict[str, Any]
    fields: Any
    secant_fields: Any
    secant_gradient: Any
    inertial_previous: Any
    voxel_scale: Any
    state: dict[str, Any]

    def save(self, path: Any) -> None:
        np = importlib.import_module("numpy")
        metadata = json.dumps(
            {
                "schema_version": 1,
                "objective_id": self.objective_id,
                "settings": asdict(self.settings),
                "signature": self.signature,
                "state": self.state,
            },
            allow_nan=False,
        )
        arrays = {
            name: getattr(self, name)
            if getattr(self, name) is not None
            else np.empty(0, dtype=np.float32)
            for name in (
                "fields",
                "secant_fields",
                "secant_gradient",
                "inertial_previous",
                "voxel_scale",
            )
        }
        with open(path, "wb") as destination:
            np.savez(destination, metadata=np.asarray(metadata), **arrays)

    @classmethod
    def read(cls, path: Any) -> MaterialReconstructionCheckpoint:
        np = importlib.import_module("numpy")
        try:
            with np.load(path, allow_pickle=False) as source:
                metadata = json.loads(str(source["metadata"]))
                if metadata["schema_version"] != 1:
                    raise ContractError("unsupported material checkpoint schema")
                settings = metadata["settings"]
                if settings.get("regularisation_per_material") is not None:
                    settings["regularisation_per_material"] = tuple(
                        settings["regularisation_per_material"]
                    )
                arrays: list[Any] = []
                for name in (
                    "fields",
                    "secant_fields",
                    "secant_gradient",
                    "inertial_previous",
                    "voxel_scale",
                ):
                    values = source[name].copy()
                    if (
                        values.dtype != np.float32
                        or not values.flags.c_contiguous
                        or not np.isfinite(values).all()
                    ):
                        raise ContractError("checkpoint arrays must be finite contiguous FP32")
                    values.setflags(write=False)
                    arrays.append(values if values.size else None)
                return cls(
                    metadata["objective_id"],
                    MaterialReconstructionSettings(**settings),
                    metadata["signature"],
                    arrays[0],
                    arrays[1],
                    arrays[2],
                    arrays[3],
                    arrays[4],
                    metadata["state"],
                )
        except (KeyError, TypeError, ValueError) as error:
            raise ContractError("invalid material checkpoint") from error


@dataclass(slots=True)
class ChannelState:
    counts: Any
    weights: Any
    response: Any
    mean: Any
    loss: Any
    objective_weights: Any = None
    valid: Any = None


@dataclass(slots=True)
class ViewState:
    pose: Any
    pixels: int
    shape: tuple[int, ...]
    paths: Any
    trial_paths: Any
    path_seed: Any
    path_total: Any
    channel_seed: Any
    image_seed: Any
    channels: list[ChannelState] = field(default_factory=list[ChannelState])
    accepted: Any = None
    trial: Any = None

    def projection(self, trial: bool) -> Any:
        return self.trial if trial else self.accepted


@dataclass(frozen=True, slots=True)
class StepPolicy:
    """Host acceptance: monotone by default, optional finite GLL reference."""

    settings: MaterialReconstructionSettings

    def accepts(
        self, incumbent: float, trial: float, slope: float, *, reference: float | None = None
    ) -> bool:
        reference = incumbent if reference is None else reference
        return (
            all(map(math.isfinite, (incumbent, trial, slope)))
            and slope < 0
            and trial < reference
            and trial <= reference + self.settings.armijo * slope
        )

    def backtrack(self, step: float) -> float:
        return step * self.settings.backtracking_factor

    def grow(self, step: float) -> float:
        return min(step / self.settings.backtracking_factor, self.settings.maximum_step)


class _MaterialSolveState:
    """Accepted scalar and report state for one material solve or checkpoint resume."""

    policy: MaterialReconstructionSettings
    checkpoint_relative_tolerance: float
    resumed: dict[str, Any]
    start_iteration: int
    completed_iterations: int
    started: float
    step_policy: StepPolicy
    history: list[dict[str, Any]]
    rejected: list[dict[str, Any]]
    reason: str
    step: float
    initial_mapping: float
    mapping_normaliser: float
    mapping_step: float
    relative_mapping: bool
    tolerance: float
    accepted: int
    reference_losses: list[float]
    use_secant: bool
    use_inertia: bool
    inertial_t: float
    gradient_current: bool
    loss: float
    mapping: float
    base_losses: dict[str, float]
    lifecycle_started: bool
    verified_checkpoint: bool


class MaterialReconstruction:
    """Persistent CUDA composition, supporting one to eight fraction fields.

    ``fractions`` and ``gradient`` expose owned flat Warp arrays for explicit
    inspection. Do not mutate them during evaluation, solve or callbacks. Use
    ``set_fractions`` between solves for independently prepared initial fields.
    ``solve`` never accesses any external reference or evaluation data.

    An optional ``material_metric=(h00,h01,h11)`` supplies a fixed symmetric
    positive-definite two-material update metric, with condition at most 1e6.
    It changes the update and its matching metric projection, never the
    objective, canonical Euclidean derivative or default stopping diagnostic. It is an
    algorithmic metric; no full voxel-Hessian or uncertainty claim is implied.

    ``field_domain="nonnegative"`` treats the dimensionless arrays as equivalent
    basis coefficients with no artificial sum cap. Any density normalisation
    must be declared in the supplied spectral specification. The legacy names
    ``initial_fractions``, ``fractions`` and ``fractions_numpy`` retain their API
    meaning as the actual stored field values in either explicit domain.
    ``observation_model="signal_wls"`` requires signal views; objective weights
    are fixed numerical scale factors, not an inferred statistical noise law.

    Data-gradient scatter terms retain their FP32 rounding and range checks,
    accumulate privately in FP64 across all fitting views, then narrow once
    into ``gradient`` before the FP32 prior and plugin derivatives are added.
    Scatter atomics do not guarantee bitwise deterministic evaluation.
    """

    def __init__(
        self,
        *,
        grid: GridSpec,
        views: tuple[MaterialReconstructionView | MaterialReconstructionSignalView, ...],
        coefficients: Any,
        spectral_spec: SpectralSpec,
        initial_fractions: Any,
        settings: MaterialReconstructionSettings,
        samples_per_ray: int,
        device: str = "cuda:0",
        material_metric: tuple[float, float, float] | None = None,
        material_diagonal_metric: tuple[float, ...] | None = None,
        voxel_metric_scale: Any = None,
        field_domain: Literal["fractions", "nonnegative"] = "fractions",
        observation_model: Literal["poisson", "signal_wls"] = "poisson",
        penalty_operator: ResidentPenalty | None = None,
    ) -> None:
        if field_domain not in ("fractions", "nonnegative"):
            raise ContractError("field_domain must be fractions or nonnegative")
        if observation_model not in ("poisson", "signal_wls"):
            raise ContractError("observation_model must be poisson or signal_wls")
        if penalty_operator is not None and not callable(penalty_operator):
            raise ContractError("penalty_operator must implement the resident penalty protocol")
        self.penalty_operator = penalty_operator
        self.field_domain, self.observation_model = field_domain, observation_model
        signal_wls = observation_model == "signal_wls"
        if not isinstance(grid, GridSpec) or not isinstance(
            settings, MaterialReconstructionSettings
        ):
            raise ContractError("supply GridSpec and MaterialReconstructionSettings")
        if not isinstance(spectral_spec, SpectralSpec):
            raise ContractError("supply the spectral specification with input provenance")
        if spectral_spec.output_domain != "linear":
            raise ContractError("material reconstruction requires linear spectral signal")
        integer(spectral_spec.materials, "materials", minimum=1, maximum=8)
        if (
            not spectral_spec.active_paths
            or (not signal_wls and not spectral_spec.shared_weights)
            or not spectral_spec.shared_response
            or spectral_spec.active_weights
            or spectral_spec.active_response
            or spectral_spec.active_coefficients
            or (not signal_wls and spectral_spec.output_unit != "counts")
        ):
            raise ContractError(
                "fixed spectra/coefficients and shared response required; "
                "Poisson needs shared count spectra"
            )
        if not isinstance(views, tuple) or not views:
            raise ContractError("supply a nonempty tuple of fitting views")
        samples_per_ray = integer(samples_per_ray, "samples_per_ray", minimum=1)
        self.grid, self.settings, self.spec = grid, settings, spectral_spec
        self.precision: Literal["float32", "float64"] = spectral_spec.precision
        self.materials, self.energies = spectral_spec.materials, spectral_spec.energies
        self.material_metric, self._metric_inverse, self._metric_condition = _prepare_metric(
            material_metric, self.materials
        )
        if material_diagonal_metric is not None:
            if material_metric is not None:
                raise ContractError("choose a coupled or diagonal material metric")
            if (
                not isinstance(material_diagonal_metric, tuple)
                or len(material_diagonal_metric) != self.materials
            ):
                raise ContractError(
                    "material_diagonal_metric requires one positive entry per material"
                )
            diagonal = tuple(
                finite_scalar(value, "material diagonal metric", minimum=0)
                for value in material_diagonal_metric
            )
            if min(diagonal) <= 0 or max(diagonal) / min(diagonal) > 1e6:
                raise ContractError(
                    "material diagonal metric must be positive with range at most 1e6"
                )
            if any(not math.isfinite(1 / value) for value in diagonal):
                raise ContractError("material diagonal metric inverse must be representable")
        else:
            diagonal = (1.0,) * self.materials
        self.material_diagonal_metric = material_diagonal_metric
        self._material_diagonal_values = diagonal
        self.np: Any = importlib.import_module("numpy")
        self._shape = (self.materials, *grid.shape)
        self.size = integer(self.materials * grid.voxels, "fraction field size", minimum=1)
        initial = self._fraction_input(initial_fractions)
        coefficients = self._array(coefficients, "coefficients", (self.materials, self.energies))
        if (coefficients < 0).any():
            raise ContractError("coefficients must be nonnegative linear attenuation in mm^-1")
        prepared = self._prepare_observation_views(views, spectral_spec, signal_wls)
        self.context = prepare_context(device=device)
        self.wp = self.context.wp
        self.stream = self.context.stream
        self.device = self.context.device
        require_no_tape()
        self.kernels = load_kernels("dpt.kernels.material_reconstruction")
        self._project_kernel = self.kernels.projected_step_tiled(
            self.materials, field_domain == "nonnegative"
        )
        self._metric_arguments: tuple[Any, ...] = ()
        if self.material_metric is not None:
            assert self._metric_inverse is not None
            self._metric_arguments = (
                self.wp.vec3d(*self.material_metric),
                self.wp.vec3d(*self._metric_inverse),
            )
        maximum_pixels = max(view.geometry.pixels for view in views)
        channels_total = sum(int(counts.shape[0]) for _, counts, _, _, _, _ in prepared)
        betas = (
            settings.regularisation_per_material
            or (settings.regularisation_mm_inverse,) * self.materials
        )
        if len(betas) != self.materials:
            raise ContractError(
                "regularisation_per_material must have one coefficient per material"
            )
        self._penalty_by_material = tuple(
            tuple(beta * math.prod(grid.spacing_mm) / h**2 for h in grid.spacing_mm)
            for beta in betas
        )
        self._penalty_coefficients = tuple(
            max(row[axis] for row in self._penalty_by_material) for axis in range(3)
        )
        if any(not math.isfinite(value) for value in self._penalty_coefficients):
            raise ContractError("regularisation or grid spacing overflows its physical scale")
        if settings.execution == "graph":
            if not self.device.is_cuda:
                raise ContractError("graph execution requires CUDA")
            if (
                penalty_operator is not None
                and getattr(penalty_operator, "graph_safe", None) is not True
            ):
                raise ContractError("a custom graph penalty must declare graph_safe=True")
        self._graph_penalty = penalty_operator
        self._evaluation_graphs: dict[tuple[bool, bool, bool], Any] = {}
        self.views: list[ViewState] = []
        with self.context.scope():
            self._prepare_resident_buffers(
                initial,
                coefficients,
                maximum_pixels,
                channels_total,
                signal_wls,
                penalty_operator,
                device,
                grid,
                settings,
                spectral_spec,
            )
            self._prepare_resident_views(
                prepared,
                signal_wls,
                field_domain,
                samples_per_ray,
                device,
                grid,
            )
        self._prepared_data_gradient_binding = self._data_gradient_binding()
        self._evaluation_checkpoints = {
            trial: StatusCheckpoint(
                self.context,
                (
                    self,
                    self._spectral,
                    self._objective,
                    *(view.projection(trial) for view in self.views),
                ),
            )
            for trial in (False, True)
        }
        self._completion_readbacks = {
            trial: CompletionReadback(checkpoint, (self._summary,))
            for trial, checkpoint in self._evaluation_checkpoints.items()
        }
        self._checkpoint_last_state: dict[str, Any] | None = None
        self._checkpoint_objective_id: str | None = None
        self._last_loss: dict[str, float] = {}
        self.evaluations = 0
        self._prediction_state = "uninitialised"
        self._accepted_forward_available = False
        self._status_readback = StatusReadback(self.context, self._status)
        self._trial_readback = HostReadback(self.context, self._trial_controls)
        self._mapping_readback = HostReadback(
            self.context, self._max_buffers[-1] if self._max_buffers else self._mappings
        )
        # Optional secant storage is allocated once, outside the solve loop.
        self._secant_fields: Any = None
        self._secant_gradient: Any = None
        self._secant_products: Any = None
        self._secant_totals: Any = None
        self._secant_numerator: Any = None
        self._secant_denominator: Any = None
        self._secant_metric: Any = None
        self._secant_host: Any = None
        self._secant_host_status: Any = None
        self._secant_host_view: Any = None
        self._secant_host_status_view: Any = None
        self._inertial_previous: Any = None
        self.voxel_metric_scale: Any = None
        self._voxel_metric_summary: dict[str, float] | None = None
        self._voxel_initial_step: float | None = None
        if voxel_metric_scale is not None:
            self.set_voxel_metric_scale(voxel_metric_scale)

    def _prepare_resident_buffers(
        self,
        initial: Any,
        coefficients: Any,
        maximum_pixels: int,
        channels_total: int,
        signal_wls: bool,
        penalty_operator: ResidentPenalty | None,
        device: str,
        grid: GridSpec,
        settings: MaterialReconstructionSettings,
        spectral_spec: SpectralSpec,
    ) -> None:
        wp = self.wp
        value_dtype = wp.float64 if self.precision == "float64" else wp.float32
        self._diagonal_device = wp.array(
            self._material_diagonal_values, dtype=wp.float64, device=self.device
        )
        self._penalty_device = wp.array(
            tuple(value for row in self._penalty_by_material for value in row),
            dtype=wp.float64,
            device=self.device,
        )
        self._huber_delta = wp.vec3d(*(settings.huber_gradient_delta * h for h in grid.spacing_mm))
        self.fractions = wp.array(initial.reshape(-1), dtype=wp.float32, device=self.device)
        self.trial = wp.empty_like(self.fractions)
        self.gradient = wp.empty_like(self.fractions)
        self._data_gradient = wp.empty(self.size, dtype=wp.float64, device=self.device)
        self._scalar_tiles = (self.size + 255) // 256
        self._voxel_tiles = (grid.voxels + 255) // 256
        self._components = wp.empty(self._scalar_tiles, dtype=wp.float64, device=self.device)
        self._mappings = wp.empty(self._voxel_tiles, dtype=wp.float64, device=self.device)
        self._status = wp.zeros(1, dtype=wp.int32, device=self.device)
        self._summary = wp.zeros(3, dtype=wp.float64, device=self.device)
        self._data_loss, self._prior_loss, self._total_loss = (
            self._summary[i : i + 1] for i in range(3)
        )
        self._slope = wp.zeros(1, dtype=wp.float64, device=self.device)
        self._trial_controls = wp.empty(3, dtype=wp.float64, device=self.device)
        self._plugin_loss = (
            wp.empty(1, dtype=wp.float64, device=self.device)
            if penalty_operator is not None
            else None
        )
        self._losses = wp.empty(channels_total, dtype=wp.float64, device=self.device)
        self._coefficients = wp.array(
            coefficients.reshape(-1), dtype=wp.float32, device=self.device
        )
        self._reduction = prepare_reduction(self.context, max(self._scalar_tiles, channels_total))
        self._max_buffers: list[Any] = []
        count = self._voxel_tiles
        while count > 1:
            count = (count + 255) // 256
            self._max_buffers.append(wp.empty(count, dtype=wp.float64, device=self.device))
        self._trial_paths = wp.empty(
            self.materials * maximum_pixels, dtype=value_dtype, device=self.device
        )
        self._path_total = wp.empty(
            self.materials * maximum_pixels, dtype=wp.float64, device=self.device
        )
        self._path_seed = (
            self._path_total
            if self.precision == "float64"
            else wp.empty(self.materials * maximum_pixels, dtype=value_dtype, device=self.device)
        )
        self._channel_seed = wp.empty(
            self.materials * maximum_pixels, dtype=value_dtype, device=self.device
        )
        self._image_seed = wp.empty(maximum_pixels, dtype=value_dtype, device=self.device)
        self._add_paths = self.kernels.add_paths(self.precision)
        self._spectral = prepare_spectral(
            spectral_spec, max_pixels=maximum_pixels, device=device, stream=self.stream
        )
        self._objective = prepare_objective(
            ObjectiveSpec(
                kind="squared_error",
                domain="signal",
                weighted=True,
                masked=True,
                precision=self.precision,
            )
            if signal_wls
            else ObjectiveSpec(kind="poisson", domain="counts", precision=self.precision),
            max_pixels=maximum_pixels,
            device=device,
            stream=self.stream,
        )

    def _prepare_resident_views(
        self,
        prepared: list[Any],
        signal_wls: bool,
        field_domain: Literal["fractions", "nonnegative"],
        samples_per_ray: int,
        device: str,
        grid: GridSpec,
    ) -> None:
        wp = self.wp
        value_dtype = wp.float64 if self.precision == "float64" else wp.float32
        loss_index = 0
        fixed_uploads: dict[tuple[int, int], Any] = {}
        for view, counts, weights, response, objective_weights, valid in prepared:
            pixels = view.geometry.pixels
            paths = wp.empty(self.materials * pixels, dtype=value_dtype, device=self.device)
            trial_paths = wp.empty_like(paths)
            path_seed = self._path_seed[: self.materials * pixels]
            item = ViewState(
                pose=wp.array(view.pose.packed(), dtype=wp.float64, device=self.device),
                pixels=pixels,
                shape=tuple(counts.shape),
                paths=paths,
                trial_paths=trial_paths,
                path_seed=path_seed,
                path_total=self._path_total[: self.materials * pixels],
                channel_seed=self._channel_seed[: self.materials * pixels],
                image_seed=self._image_seed[:pixels],
                channels=[],
            )
            for name, fields in (("accepted", self.fractions), ("trial", self.trial)):
                prepared_projection = prepare_material_projection(
                    grid,
                    view.geometry,
                    ProjectionSpec(
                        samples_per_ray,
                        active_pose=False,
                        active_volume=True,
                        precision=self.precision,
                    ),
                    materials=self.materials,
                    fields=fields,
                    field_domain=field_domain,
                    out_paths=trial_paths if name == "trial" else paths,
                    cotangent_paths=path_seed,
                    out_fields=self._data_gradient,
                    volume_gradient_precision="float64",
                    _float32_volume_increments=True,
                    device=device,
                    stream=self.stream,
                )
                setattr(item, name, prepared_projection)
            for channel in range(counts.shape[0]):
                row = ChannelState(
                    counts=wp.array(
                        counts[channel].reshape(-1), dtype=wp.float32, device=self.device
                    ),
                    weights=self._fixed_upload(weights, channel, fixed_uploads),
                    response=self._fixed_upload(response, channel, fixed_uploads),
                    mean=wp.empty(pixels, dtype=value_dtype, device=self.device),
                    loss=self._losses[loss_index : loss_index + 1],
                )
                if signal_wls:
                    row.objective_weights = wp.array(
                        objective_weights[channel].reshape(-1),
                        dtype=wp.float32,
                        device=self.device,
                    )
                    row.valid = wp.array(
                        valid[channel].reshape(-1), dtype=wp.uint8, device=self.device
                    )
                    self._objective.validate_observation(
                        row.counts,
                        weights=row.objective_weights,
                        valid=row.valid,
                        stream=self.stream,
                    )
                self._spectral.validate_inputs(
                    self._coefficients,
                    row.weights,
                    row.response,
                    pixels=pixels,
                    probability_response=not signal_wls,
                    stream=self.stream,
                )
                item.channels.append(row)
                loss_index += 1
            self.views.append(item)

    def set_voxel_metric_scale(
        self,
        values: NDArray[np.float32] | wp.array[wp.float32],
        *,
        curvature_normaliser: float | None = None,
    ) -> None:
        """Copy a fixed positive voxel scale D for update blocks D_i H.

        Supply a contiguous native FP32 host grid or a flat FP32 array on this
        workspace's CUDA device. Preparation checks values on device and reads
        only scalar bounds. Use this between solves; it never changes the
        objective, gradient or default Euclidean stationarity diagnostic.
        An explicit curvature_normaliser retains the scale removed by mean
        normalisation: initial_step/normaliser is the first proposal, still
        safeguarded by the configured search.
        """
        # Runtime contracts inspect metadata from dynamically loaded array providers.
        _api_values: Any = cast(Any, values)
        require_no_tape()
        # Deliver pending arithmetic before starting a checked metric update.
        self._check_code(self._status_readback.read(reset=True))
        initial_step = None
        if curvature_normaliser is not None:
            normaliser = finite_scalar(curvature_normaliser, "curvature_normaliser", minimum=0)
            if normaliser <= 0 or not math.isfinite(1.0 / normaliser):
                raise ContractError("curvature_normaliser must have a finite positive reciprocal")
            initial_step = 1.0 / normaliser
        if isinstance(_api_values, self.np.ndarray):
            _api_values = self._array(_api_values, "voxel metric scale", self.grid.shape)
            if (_api_values <= 0).any():
                raise ContractError("voxel metric scale must be strictly positive")
            with self.context.scope():
                candidate = self.wp.array(
                    _api_values.reshape(-1), dtype=self.wp.float32, device=self.device
                )
        else:
            if (
                not isinstance(_api_values, self.wp.array)
                or _api_values.dtype != self.wp.float32
                or _api_values.device != self.device
                or _api_values.shape != (self.grid.voxels,)
                or not _api_values.is_contiguous
            ):
                raise ContractError(
                    "supply a flat contiguous FP32 voxel scale on the solver device"
                )
            with self.context.scope():
                candidate = self.wp.empty_like(_api_values)
                self.wp.copy(candidate, _api_values, stream=self.stream)
        with self.context.scope():
            self._status.zero_()
            self.wp.launch_tiled(
                self.kernels.check_voxel_metric_scale_tiled,
                dim=self._voxel_tiles,
                inputs=[
                    self.grid.voxels,
                    candidate,
                    self._components,
                    self._mappings,
                    self._status,
                ],
                stream=self.stream,
                block_dim=256,
                record_tape=False,
            )
            try:
                self._check_code(self._status_readback.read(reset=True))
            except NumericalError as error:
                raise ContractError(
                    "voxel metric scale must be finite and strictly positive"
                ) from error
            bounds: list[float] = []
            for source in (self._components, self._mappings):
                previous, count = source, self._voxel_tiles
                for following in self._max_buffers:
                    self.wp.launch_tiled(
                        self.kernels.max_partials,
                        dim=(count + 255) // 256,
                        inputs=[previous, count, following],
                        block_dim=256,
                        stream=self.stream,
                        record_tape=False,
                    )
                    previous, count = following, (count + 255) // 256
                self.wp.synchronize_stream(self.stream)
                bounds.append(float(previous.numpy()[0]))
        maximum, minimum = bounds[0], -bounds[1]
        if maximum / minimum > 1e6 * (1 + 8 * 2**-24):
            raise ContractError("voxel metric scale range must not exceed 1e6")
        self._checkpoint_last_state = None
        self.voxel_metric_scale = candidate
        self._voxel_initial_step = initial_step
        self._voxel_metric_summary = {
            "minimum": minimum,
            "maximum": maximum,
            "maximum_over_minimum": maximum / minimum,
        }

    def _prepare_observation_views(
        self,
        views: tuple[MaterialReconstructionView | MaterialReconstructionSignalView, ...],
        spectral_spec: SpectralSpec,
        signal_wls: bool,
    ) -> list[tuple[Any, Any, Any, Any, Any, Any]]:
        """Validate fixed observations before preparing resident workspaces."""
        prepared: list[tuple[Any, Any, Any, Any, Any, Any]] = []
        effective_observations = 0
        expected_view = (
            MaterialReconstructionSignalView if signal_wls else MaterialReconstructionView
        )
        for view in views:
            if not isinstance(view, expected_view):
                raise ContractError("view type must match the declared observation model")
            if not isinstance(view.geometry, DetectorGeometry) or not isinstance(
                view.pose, RigidTransform
            ):
                raise ContractError("view geometry and pose must use canonical contracts")
            source = (
                view.observations
                if isinstance(view, MaterialReconstructionSignalView)
                else view.counts
            )
            if not hasattr(source, "shape") or len(source.shape) != 3:
                raise ContractError("observations must have shape (channels,height,width)")
            channels = integer(int(source.shape[0]), "channels", minimum=1)
            shape = (channels, *view.geometry.shape)
            counts = self._array(source, "observations", shape, finite=not signal_wls)
            weight_shape = (channels, self.energies)
            if not spectral_spec.shared_weights:
                weight_shape = (*weight_shape, *view.geometry.shape)
            weights = self._array(view.weights, "weights", weight_shape)
            response = self._array(view.response, "response", (channels, self.energies))
            objective_weights, valid = None, None
            if isinstance(view, MaterialReconstructionSignalView):
                objective_weights = self._array(view.objective_weights, "objective_weights", shape)
                if (objective_weights < 0).any():
                    raise ContractError("objective weights must be nonnegative")
                valid = view.valid
                if (
                    not isinstance(valid, self.np.ndarray)
                    or valid.dtype != self.np.dtype("uint8")
                    or tuple(valid.shape) != shape
                    or not valid.flags.c_contiguous
                    or (valid > 1).any()
                ):
                    raise ContractError(
                        "valid must be a contiguous uint8 0/1 array matching observations"
                    )
                if not self.np.isfinite(counts[valid == 1]).all():
                    raise ContractError("unmasked observations must be finite")
                effective_observations += int(((valid == 1) & (objective_weights > 0)).sum())
            elif (counts < 0).any() or (counts > 2**24).any() or (counts % 1 != 0).any():
                raise ContractError("counts must be integers in [0,2^24]")
            if (weights < 0).any() or (response < 0).any():
                raise ContractError("weights and response moments must be nonnegative")
            if not signal_wls and (response > 1).any():
                raise ContractError("Poisson response probabilities must lie in [0,1]")
            product = weights.astype(self.np.float64) * (
                response if spectral_spec.shared_weights else response[:, :, None, None]
            )
            open_beam = self.np.sum(product, axis=1)
            required = (
                open_beam
                if valid is None or spectral_spec.shared_weights
                else open_beam[valid == 1]
            )
            if (required <= 0).any():
                raise ContractError("each observed channel/ray must have a positive open-beam mean")
            prepared.append((view, counts, weights, response, objective_weights, valid))
        if signal_wls and effective_observations == 0:
            raise ContractError(
                "a signal fit requires at least one valid, positive-weight observation"
            )
        return prepared

    def _restore_solve_checkpoint(
        self,
        checkpoint: MaterialReconstructionCheckpoint,
        objective_id: str | None,
        policy: MaterialReconstructionSettings,
    ) -> None:
        """Validate and restore a supplied accepted checkpoint."""
        if objective_id is None or objective_id != checkpoint.objective_id:
            raise ContractError("checkpoint fitting-model identity differs")
        if replace(checkpoint.settings, iterations=policy.iterations) != policy:
            raise ContractError("checkpoint numerical policy differs; only iterations may change")
        if json.dumps(checkpoint.signature, sort_keys=True) != json.dumps(
            self._checkpoint_signature(), sort_keys=True
        ):
            raise ContractError("checkpoint physical configuration or material metric differs")
        state = checkpoint.state
        for name in ("completed_iterations", "accepted", "evaluations"):
            value = state.get(name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ContractError("checkpoint counters must be nonnegative integers")
        if state["accepted"] > state["completed_iterations"]:
            raise ContractError("checkpoint accepted count exceeds completed iterations")
        for name in ("next_step", "mapping_normaliser", "mapping_step", "inertial_t"):
            value = finite_scalar(state.get(name), f"checkpoint {name}", minimum=0)
            if value <= 0:
                raise ContractError("checkpoint algorithm scales must be positive")
        for name in ("initial_mapping", "tolerance", "wall_seconds"):
            finite_scalar(state.get(name), f"checkpoint {name}", minimum=0)
        finite_scalar(state.get("accepted_loss"), "checkpoint accepted_loss")
        if not isinstance(state.get("history"), list) or not isinstance(
            state.get("rejected"), list
        ):
            raise ContractError("checkpoint history and rejected trials must be lists")
        references = state.get("reference_losses")
        if not isinstance(references, list):
            raise ContractError("checkpoint nonmonotone history must be a list")
        reference_values = cast(list[Any], references)
        if len(reference_values) > policy.nonmonotone_window:
            raise ContractError("checkpoint nonmonotone history has the wrong capacity")
        for value in reference_values:
            finite_scalar(value, "checkpoint reference loss")
        required_secant = policy.step_selection == "bb" and state["accepted"] > 0
        for values, required in (
            (checkpoint.secant_fields, required_secant),
            (checkpoint.secant_gradient, required_secant),
            (checkpoint.inertial_previous, policy.acceleration == "inertial"),
        ):
            if values is None:
                if required:
                    raise ContractError("checkpoint is missing required algorithm arrays")
            elif (
                not isinstance(values, self.np.ndarray)
                or values.dtype != self.np.float32
                or values.shape != (self.size,)
                or not values.flags.c_contiguous
                or not self.np.isfinite(values).all()
            ):
                raise ContractError("checkpoint algorithm arrays must be finite contiguous FP32")
        self.set_fractions(checkpoint.fields)
        if checkpoint.voxel_scale is not None:
            hint = checkpoint.state["voxel_initial_step"]
            self.set_voxel_metric_scale(
                checkpoint.voxel_scale.reshape(self.grid.shape),
                curvature_normaliser=None if hint is None else 1.0 / hint,
            )
        elif self.voxel_metric_scale is not None:
            raise ContractError("checkpoint has no voxel scale but the solver does")

    def _fixed_upload(self, source: Any, channel: int, cache: dict[tuple[int, int], Any]) -> Any:
        key = (id(source), channel)
        if key not in cache:
            cache[key] = self.wp.array(
                source[channel].reshape(-1), dtype=self.wp.float32, device=self.device
            )
        return cache[key]

    def _array(self, value: Any, name: str, shape: tuple[int, ...], *, finite: bool = True) -> Any:
        if not isinstance(value, self.np.ndarray) or value.dtype != self.np.dtype("float32"):
            raise ContractError(f"{name} must be a native FP32 numpy array; no implicit casts")
        if tuple(value.shape) != shape or not value.flags.c_contiguous:
            raise ContractError(f"{name} must be contiguous with shape {shape}")
        if finite and not self.np.isfinite(value).all():
            raise ContractError(f"{name} must contain finite values")
        return value

    def _fraction_input(self, value: Any) -> Any:
        value = self._array(value, "fractions", self._shape)
        if (value < 0).any() or (
            self.field_domain == "fractions"
            and (self.np.sum(value, axis=0, dtype=self.np.float64) > 1 + 2**-24).any()
        ):
            raise ContractError("fields must satisfy the declared nonnegative or fraction domain")
        return value

    def set_fractions(self, values: NDArray[np.float32]) -> None:
        """Upload a new supplied initial state; no reference inputs are inferred."""
        values = self._fraction_input(values)
        with self.context.scope():
            self.fractions.assign(values.reshape(-1))
        self.invalidate_predictions()

    def set_fractions_resident(self, values: wp.array[wp.float32]) -> None:
        """Copy flat FP32 Warp fields on this device into the owned field buffer.

        Call between solves, with the source ready on the owning stream. Use
        ``context.wait_for`` for another producer stream. Validation reads only
        status words back to the host and preserves the existing field/graph
        bindings. Rejected inputs leave fields and prediction caches intact.
        """
        # Runtime contracts inspect metadata from dynamically loaded array providers.
        _api_values: Any = cast(Any, values)
        require_no_tape()
        ctx = self.context
        ctx.array(_api_values, "resident fractions", dtype=self.wp.float32, shape=(self.size,))
        identical = int(_api_values.ptr) == int(self.fractions.ptr)
        if not identical:
            ctx.disjoint([("resident fractions", _api_values)], [("fractions", self.fractions)])
        ctx.disjoint([("resident fractions", _api_values)], [("status", self._status)])
        with ctx.scope():
            # Pending accepted or trial failures must survive a refused update.
            for checkpoint in self._evaluation_checkpoints.values():
                checkpoint.check()
            material = self.views[0].accepted
            validator = (
                material.kernels.validate_fractions
                if self.field_domain == "fractions"
                else material.kernels.validate_nonnegative
            )
            self.wp.launch(
                validator,
                dim=self.grid.voxels,
                inputs=[_api_values, self.materials, self.grid.voxels],
                outputs=[self._status],
                device=self.device,
                stream=self.stream,
                record_tape=False,
            )
            if self._status_readback.read(reset=True):
                raise ContractError(
                    "fields must satisfy the declared finite nonnegative or fraction domain"
                )
            if not identical:
                self.wp.copy(self.fractions, _api_values, stream=self.stream)
        self.invalidate_predictions()

    def clear_status(self) -> None:
        """Clear the solver-owned scalar arithmetic diagnostic explicitly."""
        with self.context.scope():
            self._status.zero_()

    def check_status(self) -> None:
        """Inspect sticky solver arithmetic status without resetting it."""
        self._check()

    def _check(self) -> None:
        self._check_code(self._status_readback.read())

    @staticmethod
    def _check_code(code: int) -> None:
        if code & 4:
            raise NumericalError("no active metric-projection face could be certified")
        if code:
            raise NumericalError("material reconstruction arithmetic is not representable")

    @property
    def spectral_workspace(self) -> Any:
        return self._spectral

    @property
    def coefficients(self) -> Any:
        return self._coefficients

    @property
    def penalty_coefficients(self) -> tuple[float, ...]:
        return self._penalty_coefficients

    def invalidate_predictions(self) -> None:
        self._prediction_state = "uninitialised"
        self._accepted_forward_available = False
        self._checkpoint_last_state = None

    def evaluate(self, *, gradient: bool = True, trial: bool = False) -> float:
        """F(f)=sum_channels ell(spectral(project(f)),observation)+R(f).

        The solver retains FP32 fractions and gradient in flat material-major
        (M*voxels,) storage, equivalent to (M,nz,ny,nx). Fractions are dimensionless,
        paths are mm and coefficients are mm^-1. The configured Poisson or signal-WLS
        model and penalty define the scalar returned binary64 loss. Path/signal/seed
        storage follows spectral precision and local loss/reductions retain FP64.
        Gradient=True overwrites the ambient field derivative. Trial=True evaluates
        the resident trial fractions while preserving accepted fractions/gradient.
        Acquisition poses and fixed spectral/observation data stay unchanged.
        Variable fractions are validated once per evaluation and all composed
        diagnostics complete before return. Resident prediction scratch belongs
        to the most recent evaluation, so request it after evaluating accepted fields.

        """
        try:
            return self._evaluate(gradient=gradient, trial=trial)
        except NumericalError as error:
            # Validation can fail before the final aggregate checkpoint. Finish
            # and consume every delivered flag now, so a repaired evaluation
            # cannot erase a new unchecked failure on its next entry.
            try:
                self._evaluation_checkpoints[trial].consume()
            except (ContractError, NumericalError) as completed:
                if completed is not error:
                    error.add_note(str(completed))
                    error.__dict__["status_diagnostics"] = completed.__dict__.get(
                        "status_diagnostics", ()
                    )
            raise

    def _evaluate(self, *, gradient: bool, trial: bool) -> float:
        if trial and gradient:
            raise ContractError("trial evaluations must preserve the accepted gradient")
        require_no_tape()
        fields = self.trial if trial else self.fractions
        reuse_forward = not trial and self._accepted_forward_available
        self._accepted_forward_available = False
        self._prediction_state = "uninitialised"
        with self.context.scope():
            # Pending unchecked work is reported before any new input launch.
            self._evaluation_checkpoints[trial].consume()
            self.views[0].projection(trial).validate_inputs(stream=self.stream)
            self._run_evaluation(
                gradient=gradient, trial=trial, reuse_forward=reuse_forward, fields=fields
            )
            (summary,) = self._completion_readbacks[trial].consume()
        loss = float(summary[2])
        if not math.isfinite(loss):
            raise NumericalError("material reconstruction objective is nonfinite")
        self._last_loss = {
            "data_objective": float(summary[0]),
            "regularisation_objective": float(summary[1]),
        }
        self.evaluations += 1
        self._prediction_state = "trial" if trial else "accepted"
        return loss

    def _data_gradient_binding(self) -> tuple[Any, ...]:
        """Bind the private accumulation dtype and every prepared VJP view."""
        self.context.array(
            self._data_gradient,
            "data gradient",
            dtype=self.wp.float64,
            shape=(self.size,),
        )
        self.context.array(self.gradient, "gradient", dtype=self.wp.float32, shape=(self.size,))
        self.context.disjoint(
            [("data gradient", self._data_gradient)], [("gradient", self.gradient)]
        )

        def array_binding(values: Any) -> tuple[Any, ...]:
            return (
                int(values.ptr),
                values.dtype,
                values.shape,
                values.device,
                values.strides,
                values.is_contiguous,
            )

        for view in self.views:
            for projection in (view.accepted, view.trial):
                precision = projection.volume_gradient_precision
                if (
                    precision not in ("float32", "float64")
                    or type(projection._float32_volume_increments) is not bool
                ):
                    raise ContractError(
                        "the prepared data-gradient bindings must remain fixed: "
                        "precision policy is invalid"
                    )
                if projection._float32_volume_increments and precision != "float64":
                    raise ContractError(
                        "the prepared data-gradient bindings must remain fixed: "
                        "FP32 increments require an FP64 gradient destination"
                    )
                dtype = getattr(self.wp, precision)
                self.context.array(
                    projection.grad_fields,
                    "data gradient destination",
                    dtype=dtype,
                    shape=(self.size,),
                )
                if len(projection.gradient_views) != self.materials:
                    raise ContractError(
                        "the prepared data-gradient bindings must remain fixed: "
                        "material gradient views must retain their prepared destination"
                    )
                for material, values in enumerate(projection.gradient_views):
                    self.context.array(
                        values,
                        "data gradient view",
                        dtype=dtype,
                        shape=(self.grid.voxels,),
                    )
                    expected_pointer = int(
                        projection.grad_fields.ptr
                    ) + material * self.grid.voxels * (8 if precision == "float64" else 4)
                    if int(values.ptr) != expected_pointer:
                        raise ContractError(
                            "the prepared data-gradient bindings must remain fixed: "
                            "material gradient views must retain their prepared destination"
                        )

        projections = tuple(
            (
                array_binding(projection.grad_fields),
                projection.volume_gradient_precision,
                projection._float32_volume_increments,
                tuple(array_binding(values) for values in projection.gradient_views),
            )
            for view in self.views
            for projection in (view.accepted, view.trial)
        )
        return (array_binding(self._data_gradient), array_binding(self.gradient), projections)

    def _check_data_gradient_binding(self) -> None:
        if self._data_gradient_binding() != self._prepared_data_gradient_binding:
            raise ContractError("the prepared data-gradient bindings must remain fixed")

    def _run_evaluation(
        self, *, gradient: bool, trial: bool, reuse_forward: bool, fields: Any
    ) -> None:
        """Warm/capture only resident launches; validation and reads stay outside."""
        self._check_data_gradient_binding()
        if self.settings.execution == "eager":
            self._launch_evaluation(
                gradient=gradient, trial=trial, reuse_forward=reuse_forward, fields=fields
            )
            return
        if self.penalty_operator is not self._graph_penalty:
            raise ContractError("the prepared graph penalty binding must remain fixed")
        if (
            self.penalty_operator is not None
            and getattr(self.penalty_operator, "graph_safe", None) is not True
        ):
            raise ContractError("a custom graph penalty must retain graph_safe=True")
        key = (gradient, trial, reuse_forward)
        graph = self._evaluation_graphs.get(key)
        if graph is None:
            self._launch_evaluation(
                gradient=gradient, trial=trial, reuse_forward=reuse_forward, fields=fields
            )
            self._evaluation_checkpoints[trial].consume()
            with scoped_graph_capture(self.context) as capture:
                self._launch_evaluation(
                    gradient=gradient, trial=trial, reuse_forward=reuse_forward, fields=fields
                )
            graph = capture.graph
            self._evaluation_graphs[key] = graph
        self.wp.capture_launch(graph, stream=self.stream)

    def _launch_evaluation(
        self, *, gradient: bool, trial: bool, reuse_forward: bool, fields: Any
    ) -> None:
        """Fixed-pointer core: no allocation, host transfer or stream wait."""
        self._check_data_gradient_binding()
        if gradient:
            self.gradient.zero_()
            self._data_gradient.zero_()
        for view in self.views:
            projection = view.projection(trial)
            paths = view.trial_paths if trial else view.paths
            if not reuse_forward:
                project_material_paths(
                    view.pose, workspace=projection, stream=self.stream, validate=False
                )
            if gradient:
                view.path_total.zero_()
            for channel in view.channels:
                if not reuse_forward:
                    spectral_signal(
                        paths,
                        self._coefficients,
                        channel.weights,
                        channel.response,
                        out_mean=channel.mean,
                        workspace=self._spectral,
                        stream=self.stream,
                        validate=False,
                    )
                evaluate_objective(
                    channel.mean,
                    channel.counts,
                    out_loss=channel.loss,
                    out_cotangent=view.image_seed if gradient else None,
                    weights=channel.objective_weights,
                    valid=channel.valid,
                    workspace=self._objective,
                    stream=self.stream,
                    validate=False,
                )
                if gradient:
                    spectral_vjp(
                        paths,
                        self._coefficients,
                        channel.weights,
                        channel.response,
                        cotangent=view.image_seed,
                        out_grad_paths=view.channel_seed,
                        workspace=self._spectral,
                        stream=self.stream,
                        validate=False,
                    )
                    self.wp.launch(
                        self._add_paths,
                        dim=self.materials * view.pixels,
                        inputs=[view.channel_seed, view.path_total, self._status],
                        stream=self.stream,
                        record_tape=False,
                    )
            if gradient:
                if self.precision == "float32":
                    self.wp.launch(
                        self.kernels.narrow_paths,
                        dim=self.materials * view.pixels,
                        inputs=[view.path_total, view.path_seed, self._status],
                        stream=self.stream,
                        record_tape=False,
                    )
                material_projection_vjp(
                    view.pose,
                    workspace=projection,
                    accumulate=True,
                    stream=self.stream,
                    validate=False,
                    check_volume_gradient=False,
                )
            if trial and self.settings.cache_accepted_forward:
                # One trial scratch is reused across views. Each view retains
                # the current candidate paths; rejected candidates invalidate
                # reuse and an accepted candidate becomes the new cache.
                self.wp.copy(view.paths, paths, stream=self.stream)
        self._reduction.finish(
            int(self._losses.size), self._data_loss, self._status, source=self._losses
        )
        if gradient:
            # All views share one FP64 accumulation epoch. Narrow its data
            # term once before the existing FP32 prior/plugin/update chain.
            self.wp.launch(
                self.kernels.narrow_data_gradient,
                dim=self.size,
                inputs=[self._data_gradient, self.gradient, self._status],
                stream=self.stream,
                record_tape=False,
            )
        # The prior pass checks every assembled FP32 entry, including when
        # regularisation is zero. Trial value-only calls retain both gradients.
        self.wp.launch_tiled(
            self.kernels.regularise_tiled,
            dim=self._scalar_tiles,
            inputs=[
                self.size,
                fields,
                self.grid.shape[2],
                self.grid.shape[1],
                self.grid.shape[0],
                self._penalty_device,
                self.settings.penalty == "huber",
                self._huber_delta,
                gradient,
                self.gradient,
                self._components,
                self._status,
            ],
            stream=self.stream,
            block_dim=256,
            record_tape=False,
        )
        self._reduction.finish(
            self._scalar_tiles, self._prior_loss, self._status, source=self._components
        )
        if self.penalty_operator is not None:
            self.penalty_operator(
                fields,
                out_loss=self._plugin_loss,
                gradient=self.gradient if gradient else None,
                status=self._status,
                context=self.context,
            )
            self.wp.launch(
                self.kernels.scalar_sum,
                dim=1,
                inputs=[self._prior_loss, self._plugin_loss, self._prior_loss],
                stream=self.stream,
                record_tape=False,
            )
            if gradient:
                self.wp.launch_tiled(
                    self.kernels.gradient_magnitudes_tiled,
                    dim=self._voxel_tiles,
                    inputs=[
                        self.gradient,
                        self.materials,
                        self.grid.voxels,
                        self._mappings,
                        self._status,
                    ],
                    block_dim=256,
                    stream=self.stream,
                    record_tape=False,
                )
        self.wp.launch(
            self.kernels.scalar_sum,
            dim=1,
            inputs=[self._data_loss, self._prior_loss, self._total_loss],
            stream=self.stream,
            record_tape=False,
        )

    def projected_trial(self, step: float, *, use_metric: bool = True) -> tuple[float, float]:
        """u=argmin_feasible g^T*(u-f)+(u-f)^T*H*(u-f)/(2*step).

        Accepted fractions f, gradient g and resident trial u have FP32 flat
        material-major (M*voxels,) layout, equivalent to (M,nz,ny,nx), with
        dimensionless fractions. Step is a positive binary64 algorithmic scale.
        The declared material/voxel metric H applies when use_metric=True and the
        identity gives ordinary Euclidean projection otherwise. Feasibility is the
        nonnegative fraction simplex per voxel or the declared nonnegative orthant.
        The returned binary64 pair contains the gradient-mapping norm and actual
        linear slope g^T*(u-f). Accepted fractions/gradient remain unchanged and a
        following evaluate(trial=True) can obtain the candidate's loss. Only field
        coordinates are updated and fixed acquisition/material factors stay unchanged.

        """
        return self._projected_trial(step, use_metric=use_metric)

    def _projected_trial(
        self, step: float, *, use_metric: bool = True, momentum: float = 0.0
    ) -> tuple[float, float]:
        step = finite_scalar(step, "step", minimum=0)
        if step <= 0:
            raise ContractError("step must be positive")
        if type(use_metric) is not bool:
            raise ContractError("use_metric must be a boolean")
        with self.context.scope():
            self._status.zero_()
            selected_kernel = self._project_kernel
            metric_arguments: tuple[Any, ...] = (
                use_metric and self.material_diagonal_metric is not None,
                self._diagonal_device,
            )
            if use_metric and self.material_metric is not None:
                selected_kernel = (
                    self.kernels.projected_metric_step_tiled
                    if self.field_domain == "fractions"
                    else self.kernels.projected_metric_nonnegative_step_tiled
                )
                metric_arguments = self._metric_arguments
            self.wp.launch_tiled(
                selected_kernel,
                dim=self._voxel_tiles,
                inputs=[
                    self.fractions,
                    self.gradient,
                    self.grid.voxels,
                    step,
                    self._inertial_previous if momentum else self.fractions,
                    momentum,
                    use_metric and self.voxel_metric_scale is not None,
                    self.voxel_metric_scale
                    if self.voxel_metric_scale is not None
                    else self.fractions,
                    *metric_arguments,
                    self.trial,
                    self._components,
                    self._mappings,
                    self._status,
                ],
                stream=self.stream,
                block_dim=256,
                record_tape=False,
            )
            self._reduction.finish(
                self._voxel_tiles, self._slope, self._status, source=self._components
            )
            previous, count = self._mappings, self._voxel_tiles
            for following in self._max_buffers:
                self.wp.launch_tiled(
                    self.kernels.max_partials,
                    dim=(count + 255) // 256,
                    inputs=[previous, count, following],
                    block_dim=256,
                    stream=self.stream,
                    record_tape=False,
                )
                previous, count = following, (count + 255) // 256
            self.wp.launch(
                self.kernels.trial_controls,
                dim=1,
                inputs=[previous, self._slope, self._status, self._trial_controls],
                stream=self.stream,
                record_tape=False,
            )
            controls = self._trial_readback.read()
            self._check_code(int(controls[2]))
            mapping, slope = float(controls[0]), float(controls[1])

        if not math.isfinite(mapping) or not math.isfinite(slope):
            raise NumericalError("projected update or slope is nonfinite")
        return mapping, slope

    def _initial_gradient_scale(self) -> float:
        """Read the initial gradient infinity norm through the prepared max tree."""
        with self.context.scope():
            self._status.zero_()
            self.wp.launch_tiled(
                self.kernels.gradient_magnitudes_tiled,
                dim=self._voxel_tiles,
                inputs=[
                    self.gradient,
                    self.materials,
                    self.grid.voxels,
                    self._mappings,
                    self._status,
                ],
                stream=self.stream,
                block_dim=256,
                record_tape=False,
            )
            previous, count = self._mappings, self._voxel_tiles
            for following in self._max_buffers:
                self.wp.launch_tiled(
                    self.kernels.max_partials,
                    dim=(count + 255) // 256,
                    inputs=[previous, count, following],
                    block_dim=256,
                    stream=self.stream,
                    record_tape=False,
                )
                previous, count = following, (count + 255) // 256
            self._check()
            scale = float(self._mapping_readback.read()[0])
        if not math.isfinite(scale):
            raise NumericalError("initial gradient scale is nonfinite")
        return scale if scale > 0 else 1.0

    def solve(
        self,
        callback: Callable[[int, dict[str, Any]], None] | None = None,
        *,
        checkpoint: MaterialReconstructionCheckpoint | None = None,
        objective_id: str | None = None,
        save_checkpoint: Callable[[MaterialReconstructionCheckpoint], None] | None = None,
        checkpoint_relative_tolerance: float = 0.0,
        observe_lifecycle: LifecycleObserver | None = None,
    ) -> dict[str, Any]:
        """f_next=project_feasible(f-step*metric_inverse*grad_F) with accepted Armijo/GLL
        decrease.

        The resident FP32 accepted fractions and gradient use flat material-major
        (M*voxels,) storage, equivalent to (M,nz,ny,nx). Fractions are dimensionless
        and fixed poses, coefficients, spectra and observations define the objective.
        The configured material/voxel metric and feasible simplex/nonnegative orthant
        govern updates. Path/signal storage follows spectral precision, while scalar
        loss/search controls use FP64. The returned mapping records termination,
        accepted loss/history and search diagnostics. Callback receives an iteration
        integer and scalar diagnostic mapping after accepted updates. Checkpoints
        explicitly export the accepted field and verify objective identity on restart.
        Rejected trial losses preserve the incumbent field/gradient, numerical trial
        failures are recorded and accepted-state gradient failures propagate. Lifecycle
        observer exceptions propagate without changing the accepted numerical state.

        """
        with LifecycleRecorder("material", observe_lifecycle) as lifecycle:
            result = self._solve(
                callback,
                checkpoint=checkpoint,
                objective_id=objective_id,
                save_checkpoint=save_checkpoint,
                checkpoint_relative_tolerance=checkpoint_relative_tolerance,
                lifecycle=lifecycle,
            )
            lifecycle.finish(result)
            return result

    def _solve(
        self,
        callback: Callable[[int, dict[str, Any]], None] | None = None,
        *,
        checkpoint: MaterialReconstructionCheckpoint | None = None,
        objective_id: str | None = None,
        save_checkpoint: Callable[[MaterialReconstructionCheckpoint], None] | None = None,
        checkpoint_relative_tolerance: float = 0.0,
        lifecycle: LifecycleRecorder,
    ) -> dict[str, Any]:
        """Projected Armijo/GLL solve; callbacks run only after accepted updates.

        Value evaluations of rejected trials cannot overwrite the accepted
        gradient or fractions. Numerical failure rejects that trial; failures
        in an accepted-state gradient evaluation propagate to the caller.
        Every search failure and rejected numerical evaluation is recorded.
        """
        state = self._prepare_solve_state(
            checkpoint,
            objective_id,
            save_checkpoint,
            checkpoint_relative_tolerance,
        )
        if self.settings.execution == "graph":
            from .material_iteration_graph import iterate_material_graph

            iterate_material_graph(
                self, state, checkpoint, objective_id, callback, save_checkpoint, lifecycle
            )
        else:
            self._iterate_solve(
                state, checkpoint, objective_id, callback, save_checkpoint, lifecycle
            )
        return self._finish_solve(state, checkpoint, lifecycle)

    def _prepare_solve_state(
        self,
        checkpoint: MaterialReconstructionCheckpoint | None,
        objective_id: str | None,
        save_checkpoint: Callable[[MaterialReconstructionCheckpoint], None] | None,
        checkpoint_relative_tolerance: float,
    ) -> _MaterialSolveState:
        state = _MaterialSolveState()
        require_no_tape()
        state.policy = self.settings
        state.checkpoint_relative_tolerance = finite_scalar(
            checkpoint_relative_tolerance, "checkpoint_relative_tolerance", minimum=0
        )
        if objective_id is not None and (
            not isinstance(objective_id, str) or not objective_id.strip()
        ):
            raise ContractError("objective_id must name the immutable fitting model")
        if save_checkpoint is not None and objective_id is None:
            raise ContractError("save_checkpoint requires objective_id")
        if checkpoint is not None:
            self._restore_solve_checkpoint(checkpoint, objective_id, state.policy)
        # A valid new run owns fresh checkpoint metadata. Interrupted id-free
        # runs cannot combine an earlier run's metadata with newly accepted fields.
        self._checkpoint_last_state = None
        self._checkpoint_objective_id = objective_id
        state.resumed = {} if checkpoint is None else copy.deepcopy(checkpoint.state)
        state.start_iteration = int(state.resumed.get("completed_iterations", 0))
        if state.start_iteration < 0 or state.start_iteration > state.policy.iterations:
            raise ContractError("iteration budget precedes the checkpoint")
        state.completed_iterations = state.start_iteration
        state.started = time.perf_counter() - float(state.resumed.get("wall_seconds", 0.0))
        state.step_policy = StepPolicy(state.policy)
        state.history = state.resumed.get("history", [])
        state.rejected = state.resumed.get("rejected", [])
        state.reason = "iteration_budget"
        state.step = min(
            state.policy.maximum_step, state.policy.initial_step * (self._voxel_initial_step or 1.0)
        )
        state.initial_mapping = float(state.resumed.get("initial_mapping", 0.0))
        state.mapping_normaliser = float(state.resumed.get("mapping_normaliser", 1.0))
        state.mapping_step = float(state.resumed.get("mapping_step", state.policy.mapping_step))
        state.relative_mapping = state.policy.gradient_mapping_policy == "relative"
        state.tolerance = float(
            state.resumed.get("tolerance", state.policy.gradient_mapping_tolerance)
        )
        state.accepted = int(state.resumed.get("accepted", 0))
        state.reference_losses = state.resumed.get("reference_losses", [])
        if checkpoint is not None:
            state.step = finite_scalar(state.resumed["next_step"], "checkpoint step", minimum=0)
            self.evaluations = int(state.resumed["evaluations"])
        state.use_secant = state.policy.step_selection == "bb"
        if state.use_secant:
            self._prepare_secant()
        state.use_inertia = state.policy.acceleration == "inertial"
        state.inertial_t = float(state.resumed.get("inertial_t", 1.0))
        if state.use_inertia:
            self._prepare_inertia()
        if checkpoint is not None:
            with self.context.scope():
                for destination, values in (
                    (self._secant_fields, checkpoint.secant_fields),
                    (self._secant_gradient, checkpoint.secant_gradient),
                    (self._inertial_previous, checkpoint.inertial_previous),
                ):
                    if values is not None:
                        if destination is None or values.shape != (self.size,):
                            raise ContractError("checkpoint algorithm arrays have the wrong shape")
                        destination.assign(values)
        state.gradient_current = False
        state.loss = state.mapping = math.inf
        state.base_losses = {}

        return state

    def _remember_solve_state(self, state: _MaterialSolveState) -> None:
        self._checkpoint_last_state = copy.deepcopy(
            {
                "completed_iterations": state.completed_iterations,
                "next_step": state.step,
                "initial_mapping": state.initial_mapping,
                "mapping_normaliser": state.mapping_normaliser,
                "mapping_step": state.mapping_step,
                "tolerance": state.tolerance,
                "accepted": state.accepted,
                "reference_losses": state.reference_losses,
                "inertial_t": state.inertial_t,
                "history": state.history,
                "rejected": state.rejected,
                "evaluations": self.evaluations,
                "accepted_loss": state.loss,
                "voxel_initial_step": self._voxel_initial_step,
                "wall_seconds": time.perf_counter() - state.started,
            }
        )

    def _iterate_solve(
        self,
        state: _MaterialSolveState,
        checkpoint: MaterialReconstructionCheckpoint | None,
        objective_id: str | None,
        callback: Callable[[int, dict[str, Any]], None] | None,
        save_checkpoint: Callable[[MaterialReconstructionCheckpoint], None] | None,
        lifecycle: LifecycleRecorder,
    ) -> None:
        state.lifecycle_started = False
        state.verified_checkpoint = checkpoint is None
        for iteration in range(state.start_iteration, state.policy.iterations):
            state.completed_iterations = iteration + 1
            state.loss = self.evaluate(gradient=True)
            if not state.verified_checkpoint:
                if not math.isclose(
                    state.loss,
                    state.resumed["accepted_loss"],
                    rel_tol=state.checkpoint_relative_tolerance,
                    abs_tol=0,
                ):
                    raise ContractError(
                        "checkpoint accepted objective differs from the current fitting model"
                    )
                state.verified_checkpoint = True
            if not state.lifecycle_started:
                lifecycle.start(
                    objective=state.loss,
                    iteration=state.start_iteration,
                    accepted_steps=state.accepted,
                    resumed=checkpoint is not None,
                )
                state.lifecycle_started = True
            state.gradient_current = True
            state.base_losses = dict(self._last_loss)
            state.reference_losses.append(state.loss)
            del state.reference_losses[: -state.policy.nonmonotone_window]
            reference_loss = max(state.reference_losses)
            if iteration == 0 and state.relative_mapping:
                state.mapping_normaliser = self._initial_gradient_scale()
                state.mapping_step = state.policy.mapping_step / state.mapping_normaliser
                if not math.isfinite(state.mapping_step) or state.mapping_step <= 0:
                    raise NumericalError("relative diagnostic step is not representable")
            state.mapping, _ = self.projected_trial(
                state.mapping_step, use_metric=state.relative_mapping
            )
            if iteration == 0:
                state.initial_mapping = state.mapping
                state.tolerance *= state.mapping_normaliser
                state.tolerance += (
                    state.policy.relative_gradient_mapping_tolerance * state.initial_mapping
                )
            if state.mapping <= state.tolerance:
                state.reason = "projected_gradient_tolerance"
                break
            secant_step = None
            if state.use_secant:
                if iteration > 0:
                    secant_step = self._secant_step(state.policy)
                    if secant_step is not None:
                        state.step = secant_step
                self._remember_secant()
            proposed_step = state.step
            next_inertial_t = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * state.inertial_t**2))
            momentum = (state.inertial_t - 1.0) / next_inertial_t if state.use_inertia else 0.0
            inertial_restart = None
            found = False
            backtracks = 0
            while backtracks < state.policy.maximum_backtracks:
                if state.step == 0:
                    break
                trial_loss = math.inf
                trial_failure = "armijo"
                try:
                    _, slope = self._projected_trial(state.step, momentum=momentum)
                    if slope >= 0:
                        trial_failure = "non_descent"
                    else:
                        trial_loss = self.evaluate(gradient=False, trial=True)
                except NumericalError as error:
                    slope = 0.0
                    trial_failure = "nonfinite"
                    state.rejected.append(
                        {
                            "iteration": iteration + 1,
                            "step": state.step,
                            "backtracks": backtracks,
                            "error": str(error),
                            "momentum": momentum,
                        }
                    )
                if state.step_policy.accepts(
                    state.loss, trial_loss, slope, reference=reference_loss
                ):
                    with self.context.scope():
                        if state.use_inertia:
                            self.wp.copy(
                                self._inertial_previous, self.fractions, stream=self.stream
                            )
                        self.wp.copy(self.fractions, self.trial, stream=self.stream)
                    self._accepted_forward_available = state.policy.cache_accepted_forward
                    if state.use_inertia:
                        state.inertial_t = 1.0 if inertial_restart is not None else next_inertial_t
                    self._prediction_state = "accepted"
                    state.gradient_current = False
                    record: dict[str, Any] = {
                        "iteration": iteration + 1,
                        "accepted": True,
                        "objective_before": state.loss,
                        "line_search_reference": reference_loss,
                        "objective": trial_loss,
                        **self._last_loss,
                        "gradient_mapping_before": state.mapping,
                        "step": state.step,
                        "backtracks": backtracks,
                        "proposed_step": proposed_step,
                        "secant_step": secant_step,
                        "momentum": momentum,
                        "inertial_restart": inertial_restart,
                        "wall_seconds": time.perf_counter() - state.started,
                    }
                    state.history.append(record)
                    state.accepted += 1
                    # Callback failures belong to the recorder/caller; this
                    # update has already been accepted and cannot be retried.
                    state.loss = trial_loss
                    state.step = state.step_policy.grow(state.step)
                    if objective_id is not None:
                        self._remember_solve_state(state)
                    lifecycle.accepted(iteration + 1, state.loss)
                    if callback is not None:
                        callback(iteration + 1, dict(record))
                    if save_checkpoint is not None:
                        save_checkpoint(self.checkpoint())
                    found = True
                    break
                lifecycle.rejected(iteration + 1, trial_failure)
                if momentum:
                    # Discard only the proposal. The old accepted field and its
                    # gradient remain intact; retry the same alpha without inertia.
                    inertial_restart = trial_failure
                    momentum = 0.0
                    continue
                state.step = state.step_policy.backtrack(state.step)
                backtracks += 1
            if not found:
                state.reason = "line_search_failed"
                state.history.append(
                    {
                        "iteration": iteration + 1,
                        "accepted": False,
                        "objective": state.loss,
                        **state.base_losses,
                        "gradient_mapping_before": state.mapping,
                        "step": state.step,
                        "wall_seconds": time.perf_counter() - state.started,
                        "failure": state.reason,
                    }
                )
                break

    def _finish_solve(
        self,
        state: _MaterialSolveState,
        checkpoint: MaterialReconstructionCheckpoint | None,
        lifecycle: LifecycleRecorder,
    ) -> dict[str, Any]:
        if not state.gradient_current:
            # A budget exit follows an accepted update: refresh its gradient.
            final_loss = self.evaluate(gradient=True)
            final_mapping, _ = self.projected_trial(
                state.mapping_step, use_metric=state.relative_mapping
            )
        else:
            # Tolerance/search-failure exits leave the accepted gradient current.
            # A rejected trial may have overwritten shared prediction buffers;
            # restore their accepted values without repeating the volume VJP.
            if self._prediction_state != "accepted":
                final_loss = self.evaluate(gradient=False)
            else:
                final_loss = state.loss
                self._last_loss = state.base_losses
            final_mapping = state.mapping
        if not state.verified_checkpoint and not math.isclose(
            final_loss,
            state.resumed["accepted_loss"],
            rel_tol=state.checkpoint_relative_tolerance,
            abs_tol=0,
        ):
            raise ContractError(
                "checkpoint accepted objective differs from the current fitting model"
            )
        state.loss = final_loss
        if not state.lifecycle_started:
            lifecycle.start(
                objective=state.loss,
                iteration=state.start_iteration,
                accepted_steps=state.accepted,
                resumed=checkpoint is not None,
            )
        lifecycle.iteration = state.completed_iterations
        lifecycle.objective = final_loss
        self._remember_solve_state(state)
        if final_mapping <= state.tolerance:
            state.reason = "projected_gradient_tolerance"
        return self._solve_report(state, final_loss, final_mapping)

    def _solve_report(
        self,
        state: _MaterialSolveState,
        final_loss: float,
        final_mapping: float,
    ) -> dict[str, Any]:
        return {
            "termination": state.reason,
            "field_domain": self.field_domain,
            "observation_model": self.observation_model,
            "accepted_steps": state.accepted,
            "evaluations": self.evaluations,
            "final_objective": final_loss,
            **self._last_loss,
            "initial_gradient_mapping": state.initial_mapping,
            "final_gradient_mapping": final_mapping,
            "mapping_step": state.mapping_step,
            "configured_mapping_step": state.policy.mapping_step,
            "gradient_mapping_policy": state.policy.gradient_mapping_policy,
            "gradient_mapping_normaliser": state.mapping_normaliser,
            "initial_relative_gradient_mapping": state.initial_mapping / state.mapping_normaliser,
            "final_relative_gradient_mapping": final_mapping / state.mapping_normaliser,
            "gradient_mapping_metric": "update" if state.relative_mapping else "euclidean",
            "material_metric": self.material_metric,
            "material_diagonal_metric": self.material_diagonal_metric,
            "material_metric_condition": self._metric_condition,
            "voxel_metric_scale": self._voxel_metric_summary,
            "voxel_curvature_initial_step": self._voxel_initial_step,
            "gradient_mapping_threshold": state.tolerance,
            "step_selection": state.policy.step_selection,
            "nonmonotone_window": state.policy.nonmonotone_window,
            "execution": state.policy.execution,
            "captured_evaluations": len(self._evaluation_graphs),
            "cache_accepted_forward": state.policy.cache_accepted_forward,
            "cached_path_bytes": sum(view.pixels for view in self.views)
            * self.materials
            * (8 if self.precision == "float64" else 4)
            if state.policy.cache_accepted_forward
            else 0,
            "penalty": state.policy.penalty,
            "penalty_operator": None
            if self.penalty_operator is None
            else type(self.penalty_operator).__qualname__,
            "regularisation_per_material": tuple(
                betas
                for betas in (
                    self.settings.regularisation_per_material
                    or (self.settings.regularisation_mm_inverse,) * self.materials
                )
            ),
            "acceleration": state.policy.acceleration,
            "wall_seconds": time.perf_counter() - state.started,
            "history": state.history,
            "rejected_numerical_trials": state.rejected,
        }

    def _checkpoint_signature(self) -> dict[str, Any]:
        return {
            "grid": asdict(self.grid),
            "spectral": asdict(self.spec),
            "field_domain": self.field_domain,
            "observation_model": self.observation_model,
            "material_metric": self.material_metric,
            "material_diagonal_metric": self.material_diagonal_metric,
            "penalty_operator": None
            if self.penalty_operator is None
            else type(self.penalty_operator).__qualname__,
        }

    def checkpoint(self, objective_id: str | None = None) -> MaterialReconstructionCheckpoint:
        """Explicit synchronised host export; preserves BB, inertia and GLL state."""
        require_no_tape()
        objective_id = self._checkpoint_objective_id if objective_id is None else objective_id
        if not isinstance(objective_id, str) or not objective_id.strip():
            raise ContractError("checkpoint requires the immutable fitting-model objective_id")
        if (
            self._checkpoint_objective_id is not None
            and objective_id != self._checkpoint_objective_id
        ):
            raise ContractError("checkpoint fitting-model identity differs")
        if self._checkpoint_last_state is None:
            raise ContractError("solve an accepted state before requesting a checkpoint")
        self.wp.synchronize_stream(self.stream)
        arrays: list[Any] = []
        has_secant = (
            self._checkpoint_last_state["accepted"] > 0 and self.settings.step_selection == "bb"
        )
        for values in (
            self.fractions,
            self._secant_fields if has_secant else None,
            self._secant_gradient if has_secant else None,
            self._inertial_previous if self.settings.acceleration == "inertial" else None,
            self.voxel_metric_scale,
        ):
            if values is None:
                arrays.append(None)
            else:
                host: Any = values.numpy().copy()
                host.setflags(write=False)
                arrays.append(host)
        arrays[0] = arrays[0].reshape(self._shape)
        return MaterialReconstructionCheckpoint(
            objective_id,
            self.settings,
            self._checkpoint_signature(),
            arrays[0],
            arrays[1],
            arrays[2],
            arrays[3],
            arrays[4],
            copy.deepcopy(self._checkpoint_last_state),
        )

    def _prepare_inertia(self) -> None:
        """Allocate and initialise one preceding accepted field outside the loop."""
        with self.context.scope():
            if self._inertial_previous is None:
                self._inertial_previous = self.wp.empty_like(self.fractions)
            self.wp.copy(self._inertial_previous, self.fractions, stream=self.stream)

    def _prepare_secant(self) -> None:
        """Keep two preceding FP32 vectors and secant reduction scratch resident."""
        if self._secant_fields is not None:
            return
        with self.context.scope():
            wp = self.wp
            self._secant_fields = wp.empty_like(self.fractions)
            self._secant_gradient = wp.empty_like(self.gradient)
            self._secant_products = wp.empty(
                self._scalar_tiles, dtype=wp.float64, device=self.device
            )
            self._secant_totals = wp.empty(2, dtype=wp.float64, device=self.device)
            self._secant_numerator = self._secant_totals[0:1]
            self._secant_denominator = self._secant_totals[1:2]
            self._secant_metric = wp.vec3d(*(self.material_metric or (1.0, 0.0, 1.0)))
            self._secant_host = wp.empty(
                2, dtype=wp.float64, device="cpu", pinned=self.context.device.is_cuda
            )
            self._secant_host_status = wp.empty(
                1, dtype=wp.int32, device="cpu", pinned=self.context.device.is_cuda
            )
            self._secant_host_view = self._secant_host.numpy()
            self._secant_host_status_view = self._secant_host_status.numpy()

    def _remember_secant(self) -> None:
        with self.context.scope():
            self.wp.copy(self._secant_fields, self.fractions, stream=self.stream)
            self.wp.copy(self._secant_gradient, self.gradient, stream=self.stream)

    def _secant_step(self, policy: MaterialReconstructionSettings) -> float | None:
        """A secant affects the proposal only, never acceptance or stationarity."""
        with self.context.scope():
            self._status.zero_()
            self.wp.launch_tiled(
                self.kernels.secant_products_tiled,
                dim=self._scalar_tiles,
                inputs=[
                    self.size,
                    self.fractions,
                    self.gradient,
                    self._secant_fields,
                    self._secant_gradient,
                    self.grid.voxels,
                    self.material_metric is not None,
                    self._secant_metric,
                    self.voxel_metric_scale is not None,
                    self.voxel_metric_scale
                    if self.voxel_metric_scale is not None
                    else self.fractions,
                    self.material_diagonal_metric is not None,
                    self._diagonal_device,
                    self._components,
                    self._secant_products,
                ],
                stream=self.stream,
                block_dim=256,
                record_tape=False,
            )
            self._reduction.finish(
                self._scalar_tiles, self._secant_numerator, self._status, source=self._components
            )
            self._reduction.finish(
                self._scalar_tiles,
                self._secant_denominator,
                self._status,
                source=self._secant_products,
            )
            self.wp.copy(self._secant_host, self._secant_totals, stream=self.stream)
            self.wp.copy(self._secant_host_status, self._status, stream=self.stream)
            self.wp.synchronize_stream(self.stream)
            numerator = float(self._secant_host_view[0])
            denominator = float(self._secant_host_view[1])
            # Invalid secant arithmetic is an unusable proposal, not a change
            # to the already checked objective/gradient or a convergence claim.
            invalid = int(self._secant_host_status_view[0]) != 0
        if invalid or not all(math.isfinite(v) and v > 0 for v in (numerator, denominator)):
            return None
        step = numerator / denominator
        if not math.isfinite(step) or step <= 0:
            return None
        return min(policy.maximum_step, max(policy.secant_minimum_step, step))

    def fractions_numpy(self) -> NDArray[np.float32]:
        """Explicit synchronised export of accepted material-major fractions."""
        self.wp.synchronize_stream(self.stream)
        return self.fractions.numpy().reshape(self._shape)

    def predictions_numpy(self) -> tuple[NDArray[np.float32] | NDArray[np.float64], ...]:
        """Export fitting predictions only after evaluating/accepting that state."""
        if self._prediction_state != "accepted":
            raise ContractError("evaluate the accepted state before exporting predictions")
        self.wp.synchronize_stream(self.stream)
        return tuple(
            self.np.stack([channel.mean.numpy() for channel in view.channels]).reshape(view.shape)
            for view in self.views
        )


def _prepare_metric(
    metric: tuple[float, float, float] | None, materials: int
) -> tuple[tuple[float, float, float] | None, tuple[float, float, float] | None, float | None]:
    """Validate the small fixed metric without importing optional dependencies."""
    if metric is None:
        return None, None, None
    if materials != 2:
        raise ContractError("a coupled material_metric requires exactly two materials")
    if not isinstance(metric, tuple) or len(metric) != 3:
        raise ContractError("material_metric must be the symmetric tuple (h00,h01,h11)")
    a, b, d = (finite_scalar(value, "material_metric entry") for value in metric)
    scale = max(abs(a), abs(b), abs(d))
    if scale == 0 or a <= 0 or d <= 0:
        raise ContractError("material_metric must be positive definite")
    a_scaled, b_scaled, d_scaled = a / scale, b / scale, d / scale
    determinant = a_scaled * d_scaled - b_scaled * b_scaled
    maximum = 0.5 * (a_scaled + d_scaled + math.hypot(a_scaled - d_scaled, 2 * b_scaled))
    if determinant <= 0 or maximum <= 0:
        raise ContractError("material_metric must be positive definite")
    minimum = determinant / maximum
    condition = maximum / minimum
    if not math.isfinite(condition) or condition > 1.0e6:
        raise ContractError("material_metric condition must not exceed 1e6")
    inverse = (
        d_scaled / determinant / scale,
        -b_scaled / determinant / scale,
        a_scaled / determinant / scale,
    )
    if not all(math.isfinite(value) for value in inverse):
        raise ContractError("material_metric inverse must be representable in FP64")
    return (a, b, d), inverse, condition

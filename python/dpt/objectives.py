"""Image objectives with an explicit measurement domain and image-space seed.

Poisson half-deviance accepts nonnegative integer observations by default;
``integer_counts=False`` explicitly allows finite nonnegative real observations
with the same half-deviance arithmetic. Count means are nonnegative, and a zero
mean with a positive observation is rejected. Squared error accepts
finite values in the declared domain. Weights are fixed nonnegative per-pixel
coefficients. A mean reduction divides by pixel count, not by the sum of weights.
An explicit uint8 validity mask skips invalid observation lanes before reading
their target or forming residuals, with exact zero loss/seed. It is fixed data,
not a residual-dependent weight. Unmasked predictions/observations remain checked.
"""

# Workspace internals are owned by functions in this module.
# pyright: reportPrivateUsage=false
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, cast

from dpt._reductions import ReductionTree, prepare_reduction
from dpt._runtime import (
    DeviceContext,
    StatusReadback,
    load_kernels,
    prepare_context,
    require_no_tape,
)
from dpt.contracts import ContractError, finite_scalar, integer, raise_device_status

if TYPE_CHECKING:
    import warp as wp


@dataclass(frozen=True, slots=True)
class ObjectiveSpec:
    kind: Literal["squared_error", "poisson"] = "squared_error"
    domain: Literal["counts", "signal", "log_transmission"] = "signal"
    reduction: Literal["sum", "mean"] = "sum"
    weighted: bool = False
    masked: bool = False
    precision: Literal["float32", "float64"] = "float32"
    integer_counts: bool = True

    def __post_init__(self) -> None:
        if self.kind not in ("squared_error", "poisson"):
            raise ContractError("unsupported objective")
        if self.precision not in ("float32", "float64"):
            raise ContractError("objective precision must be float32 or float64")
        if self.domain not in ("counts", "signal", "log_transmission"):
            raise ContractError("declare the objective's measurement domain")
        if self.kind == "poisson" and self.domain != "counts":
            raise ContractError("a Poisson objective requires count-domain observations")
        if type(self.integer_counts) is not bool:
            raise ContractError("integer_counts must be a boolean")
        if (
            self.reduction not in ("sum", "mean")
            or type(self.weighted) is not bool
            or type(self.masked) is not bool
        ):
            raise ContractError("invalid objective reduction or weight policy")


@dataclass(slots=True)
class ObjectiveWorkspace:
    spec: ObjectiveSpec
    max_pixels: int
    context: DeviceContext
    _kernels: Any = field(repr=False)
    _empty: Any = field(repr=False)
    _empty_seed: Any = field(repr=False)
    _empty_valid: Any = field(repr=False)
    _status: Any = field(repr=False)
    _reduction: ReductionTree = field(repr=False)
    _status_readback: StatusReadback = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._status_readback = StatusReadback(self.context, self._status)

    @property
    def scratch_bytes(self) -> int:
        return 4 + self._reduction.scratch_bytes

    def clear_status(self) -> None:
        with self.context.scope():
            self._status.zero_()

    def check_status(self) -> None:
        """Synchronise at an explicit acceptance boundary; failed outputs are unusable."""
        self._raise_status(self._status_readback.read())

    def consume_status(self) -> None:
        """Deliver and reset one completed checkpoint, including reported errors."""
        self._raise_status(self._status_readback.read(reset=True))

    @property
    def host_staging_bytes(self) -> int:
        return self._status_readback.host_bytes

    def _raise_status(self, code: int) -> None:
        messages = None
        if code & 1:
            rule = "unmasked observations must be finite"
            if self.spec.kind == "poisson":
                rule += " and nonnegative"
                if self.spec.integer_counts:
                    rule += " integer counts"
            if self.spec.weighted:
                rule += "; unmasked weights must be finite and nonnegative"
            if self.spec.masked:
                rule += "; validity masks must contain only 0 or 1"
            messages = {1: rule}
        raise_device_status(code, "objectives", messages=messages)

    def validate_observation(
        self,
        observation: Any,
        *,
        weights: Any = None,
        valid: Any = None,
        stream: wp.Stream | None = None,
    ) -> None:
        """Check fixed observations and optional weights without a prediction."""
        require_no_tape()
        ctx = self.context
        ctx.assert_stream(stream)
        size = ctx.array(observation, "observation", dtype=ctx.wp.float32)
        if size > self.max_pixels or (not size and self.spec.reduction == "mean"):
            raise ContractError("observation size is outside the objective capacity/domain")
        if self.spec.weighted:
            ctx.array(weights, "weights", dtype=ctx.wp.float32, shape=(size,))
        elif weights is not None:
            raise ContractError("weights require a weighted objective specification")
        mask = _validity(self, valid, size)
        with ctx.scope():
            self.check_status()
            if size:
                ctx.wp.launch(
                    self._kernels.observation_validation_kernel(
                        self.spec.kind == "poisson",
                        self.spec.weighted,
                        self.spec.masked,
                        self.spec.integer_counts,
                    ),
                    dim=size,
                    inputs=[observation, weights if self.spec.weighted else self._empty, mask],
                    outputs=[self._status],
                    stream=ctx.stream,
                    record_tape=False,
                )
            self.check_status()


def prepare_objective(
    spec: ObjectiveSpec,
    *,
    max_pixels: int,
    device: str | wp.Device = "cuda:0",
    stream: wp.Stream | None = None,
) -> ObjectiveWorkspace:
    capacity = integer(max_pixels, "max_pixels")
    context = prepare_context(device=device, stream=stream)
    wp = context.wp
    kernels = load_kernels("dpt.kernels.objectives")
    reduction = prepare_reduction(context, capacity)
    with context.scope():
        empty = wp.empty(0, dtype=wp.float32, device=context.device)
        empty_seed = wp.empty(
            0,
            dtype=wp.float64 if spec.precision == "float64" else wp.float32,
            device=context.device,
        )
        empty_valid = wp.empty(0, dtype=wp.uint8, device=context.device)
        status = wp.zeros(1, dtype=wp.int32, device=context.device)
    return ObjectiveWorkspace(
        spec, capacity, context, kernels, empty, empty_seed, empty_valid, status, reduction
    )


def _validity(workspace: ObjectiveWorkspace, valid: Any, size: int) -> Any:
    if workspace.spec.masked:
        workspace.context.array(valid, "valid", dtype=workspace.context.wp.uint8, shape=(size,))
        return valid
    if valid is not None:
        raise ContractError("valid requires a masked objective specification")
    return workspace._empty_valid


def evaluate_objective(
    prediction: wp.array[wp.float32] | wp.array[wp.float64],
    observation: wp.array[wp.float32],
    *,
    out_loss: wp.array[wp.float64],
    out_cotangent: wp.array[wp.float32] | wp.array[wp.float64] | None = None,
    weights: wp.array[wp.float32] | None = None,
    valid: wp.array[wp.uint8] | None = None,
    workspace: ObjectiveWorkspace,
    stream: wp.Stream | None = None,
    validate: bool = True,
) -> None:
    """Loss = sum_p valid[p]*weights[p]*ell(prediction[p],observation[p]).

    Ell is the selected Poisson negative log-likelihood or half squared error
    objective and spec.reduction determines sum versus mean.
    Prediction and optional out_cotangent are (P,) in spec.precision, while
    observation and optional weights are FP32 (P,) and valid is uint8 (P,).
    Out_loss is FP64 (1,). Masked entries do not contribute. Poisson prediction
    is an expected count and observation is nonnegative, with integer_counts
    constraining values rather than changing observation storage. The output
    covector differentiates prediction only and has loss units per prediction
    unit. Local arithmetic and the reduction use
    FP64. Ambient tapes are rejected and the covector composes with an explicit
    renderer VJP. Unchecked calls require a completed status checkpoint.

    """
    # Runtime contracts validate the dynamically loaded Warp boundary.
    _api_prediction: Any = cast(Any, prediction)
    _api_observation: Any = cast(Any, observation)
    _api_out_loss: Any = cast(Any, out_loss)
    _api_out_cotangent: Any = cast(Any, out_cotangent)
    _api_weights: Any = cast(Any, weights)
    _api_valid: Any = cast(Any, valid)
    ctx, spec, kernels = workspace.context, workspace.spec, workspace._kernels
    ctx.assert_stream(stream)
    require_no_tape()
    wp = ctx.wp
    value_dtype = wp.float64 if spec.precision == "float64" else wp.float32
    size = ctx.array(_api_prediction, "prediction", dtype=value_dtype)
    if size > workspace.max_pixels:
        raise ContractError("prediction exceeds workspace capacity")
    if size == 0 and spec.reduction == "mean":
        raise ContractError("a mean objective requires at least one pixel")
    ctx.array(_api_observation, "observation", dtype=wp.float32, shape=(size,))
    ctx.array(_api_out_loss, "out_loss", dtype=wp.float64, shape=(1,))
    reads = [("prediction", _api_prediction), ("observation", _api_observation)]
    writes = [("out_loss", _api_out_loss)]
    if spec.weighted:
        ctx.array(_api_weights, "weights", dtype=wp.float32, shape=(size,))
        reads.append(("weights", _api_weights))
    elif _api_weights is not None:
        raise ContractError("weights require a weighted objective specification")
    if _api_out_cotangent is not None:
        ctx.array(_api_out_cotangent, "out_cotangent", dtype=value_dtype, shape=(size,))
        writes.append(("out_cotangent", _api_out_cotangent))
    ctx.disjoint(reads, writes)
    weight_array = _api_weights if spec.weighted else workspace._empty
    validity = _validity(workspace, _api_valid, size)
    if spec.masked:
        reads.append(("valid", _api_valid))
        ctx.disjoint(reads, writes)
    seed = _api_out_cotangent if _api_out_cotangent is not None else workspace._empty_seed
    poisson = spec.kind == "poisson"
    with ctx.scope():
        if validate:
            workspace.check_status()
            if size:
                wp.launch(
                    kernels.validation_kernel(
                        poisson, spec.weighted, spec.masked, spec.precision, spec.integer_counts
                    ),
                    dim=size,
                    inputs=[
                        _api_prediction,
                        _api_observation,
                        weight_array,
                        validity,
                        workspace._status,
                    ],
                    device=ctx.device,
                    stream=ctx.stream,
                    record_tape=False,
                )
            workspace.check_status()
        count = (size + 255) // 256 if ctx.device.is_cuda else size
        if size:
            normalisation = 1.0 / size if spec.reduction == "mean" else 1.0
            wp.launch_tiled(
                kernels.evaluation_kernel(
                    poisson,
                    spec.weighted,
                    _api_out_cotangent is not None,
                    spec.masked,
                    spec.precision,
                ),
                dim=count,
                block_dim=256,
                inputs=[
                    _api_prediction,
                    _api_observation,
                    weight_array,
                    validity,
                    size,
                    normalisation,
                    seed,
                    workspace._reduction.partials[0],
                    workspace._status,
                ],
                device=ctx.device,
                stream=ctx.stream,
                record_tape=False,
            )
        workspace._reduction.finish(count, _api_out_loss, workspace._status)
        if validate:
            workspace.check_status()


def evaluate_primary_objective(
    optical_depth: wp.array[wp.float64],
    observation: wp.array[wp.float32],
    *,
    open_beam: float = 1.0,
    out_prediction: wp.array[wp.float64],
    out_loss: wp.array[wp.float64],
    out_depth_cotangent: wp.array[wp.float64] | None,
    weights: wp.array[wp.float32] | None = None,
    valid: wp.array[wp.uint8] | None = None,
    workspace: ObjectiveWorkspace,
    stream: wp.Stream | None = None,
    validate: bool = True,
) -> None:
    """Lambda[p] = open_beam*exp(-L[p]) and loss = sum_p w[p]*ell(Lambda[p],N[p]).

    Optical_depth, out_prediction and optional out_depth_cotangent are FP64
    (P,), observation and optional weights are FP32 (P,), valid is uint8 (P,)
    and out_loss is FP64 (1,). L is dimensionless and the nonnegative scalar
    open_beam is expected incident counts per pixel. For the Poisson term the
    depth covector is w*(N-Lambda), with the selected mask/normalisation.
    Depth is the differentiated parameter and
    open_beam is fixed. FP64 local arithmetic and reduction avoid an intermediate
    FP32 transmission boundary. A positive count with zero represented mean
    raises instead of clipping the model. The call is an explicit first-order
    composition and rejects ambient tapes.

    """
    # Runtime contracts validate the dynamically loaded Warp boundary.
    _api_optical_depth: Any = cast(Any, optical_depth)
    _api_observation: Any = cast(Any, observation)
    _api_out_prediction: Any = cast(Any, out_prediction)
    _api_out_loss: Any = cast(Any, out_loss)
    _api_out_depth_cotangent: Any = cast(Any, out_depth_cotangent)
    _api_weights: Any = cast(Any, weights)
    _api_valid: Any = cast(Any, valid)
    require_no_tape()
    ctx, spec, wp = workspace.context, workspace.spec, workspace.context.wp
    ctx.assert_stream(stream)
    if spec.domain not in ("counts", "log_transmission"):
        raise ContractError("primary composition requires counts or log transmission")
    beam = finite_scalar(open_beam, "open_beam", minimum=0.0)
    size = ctx.array(_api_optical_depth, "optical_depth", dtype=wp.float64)
    if size > workspace.max_pixels or (not size and spec.reduction == "mean"):
        raise ContractError("primary image is outside the objective capacity/domain")
    ctx.array(_api_observation, "observation", dtype=wp.float32, shape=(size,))
    ctx.array(_api_out_prediction, "out_prediction", dtype=wp.float64, shape=(size,))
    ctx.array(_api_out_depth_cotangent, "out_depth_cotangent", dtype=wp.float64, shape=(size,))
    ctx.array(_api_out_loss, "out_loss", dtype=wp.float64, shape=(1,))
    reads = [("optical_depth", _api_optical_depth), ("observation", _api_observation)]
    if spec.weighted:
        ctx.array(_api_weights, "weights", dtype=wp.float32, shape=(size,))
        reads.append(("weights", _api_weights))
    elif _api_weights is not None:
        raise ContractError("weights require a weighted objective specification")
    mask = _validity(workspace, _api_valid, size)
    if spec.masked:
        reads.append(("valid", _api_valid))
    ctx.disjoint(
        reads,
        [
            ("prediction", _api_out_prediction),
            ("depth seed", _api_out_depth_cotangent),
            ("loss", _api_out_loss),
        ],
    )
    with ctx.scope():
        if validate:
            workspace.validate_observation(
                _api_observation, weights=_api_weights, valid=_api_valid, stream=stream
            )
        count = (size + 255) // 256 if ctx.device.is_cuda else size
        if size:
            wp.launch_tiled(
                workspace._kernels.primary_evaluation_kernel(
                    spec.kind == "poisson",
                    spec.domain == "counts",
                    spec.weighted,
                    spec.masked,
                    spec.integer_counts,
                ),
                dim=count,
                block_dim=256,
                inputs=[
                    _api_optical_depth,
                    beam,
                    _api_observation,
                    _api_weights if spec.weighted else workspace._empty,
                    mask,
                    size,
                    1.0 / size if spec.reduction == "mean" else 1.0,
                ],
                outputs=[
                    _api_out_prediction,
                    _api_out_depth_cotangent,
                    workspace._reduction.partials[0],
                    workspace._status,
                ],
                device=ctx.device,
                stream=ctx.stream,
                record_tape=False,
            )
        workspace._reduction.finish(count, _api_out_loss, workspace._status)
        if validate:
            workspace.check_status()


def reduce_objective_components(
    components: wp.array[wp.float64],
    *,
    out_loss: wp.array[wp.float64],
    workspace: ObjectiveWorkspace,
    stream: wp.Stream | None = None,
    validate: bool = True,
) -> None:
    """Out_loss[0] = sum_p components[p] reduces already weighted objective terms.

    Components is a contiguous FP64 (P,) array with P <= workspace.max_pixels,
    and out_loss is a disjoint FP64 (1,) destination in the same objective units.
    The empty sum is zero. The fixed reduction retains FP64 arithmetic and uses
    prepared scratch without allocating component storage. This call forms no
    covector and does not reevaluate a measurement model. Stochastic estimators
    can supply their independently constructed components through this boundary.

    """
    # Runtime contracts validate the dynamically loaded Warp boundary.
    _api_components: Any = cast(Any, components)
    _api_out_loss: Any = cast(Any, out_loss)
    ctx, wp = workspace.context, workspace.context.wp
    ctx.assert_stream(stream)
    require_no_tape()
    if workspace.spec.reduction != "sum" or workspace.spec.weighted or workspace.spec.masked:
        raise ContractError("component reduction requires an unweighted sum workspace")
    size = ctx.array(_api_components, "components", dtype=wp.float64)
    if size > workspace.max_pixels:
        raise ContractError("components exceed reduction capacity")
    ctx.array(_api_out_loss, "out_loss", dtype=wp.float64, shape=(1,))
    ctx.disjoint([("components", _api_components)], [("out_loss", _api_out_loss)])
    with ctx.scope():
        if validate:
            workspace.check_status()
        workspace._reduction.finish(size, _api_out_loss, workspace._status, source=_api_components)
        if validate:
            workspace.check_status()

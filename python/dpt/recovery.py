"""Composition of the canonical projector, transmission, objective and optimiser.

The fixed volume, observation and reusable image buffers remain on CUDA. Each
evaluation explicitly uploads a twelve-value rigid transform and downloads one
loss, six local pose derivatives and numerical status in one completion payload.
A separate entry checkpoint reports pending unchecked numerical failures.
This is a deterministic primary-model
adapter; spectral/nuisance adapters use the same scaled optimiser contract.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Literal

from dpt._runtime import CompletionReadback, StatusCheckpoint, require_no_tape
from dpt.contracts import ContractError, TrialDomainError, finite_scalar
from dpt.geometry import DetectorGeometry, RigidTransform, chart_gradient, compose_pose
from dpt.objectives import (
    ObjectiveSpec,
    evaluate_objective,
    evaluate_primary_objective,
    prepare_objective,
)
from dpt.projection import ProjectionSpec, prepare_projection, project_optical_depth, projection_vjp
from dpt.registration import (
    Evaluation,
    RecoveryCheckpoint,
    RecoveryPolicy,
    RecoveryResult,
    Vector,
    recover_parameters,
)
from dpt.solver_lifecycle import LifecycleObserver
from dpt.transmission import TransmissionSpec, prepare_transmission, transmission_vjp, transmit
from dpt.volumes import GridSpec


@dataclass(frozen=True, slots=True)
class PoseChart:
    """Dimensionless optimiser coordinates mapped into one fixed SE(3) chart."""

    anchor: RigidTransform = field(default_factory=RigidTransform)
    scales: tuple[float, float, float, float, float, float] = (1.0, 1.0, 1.0, 0.01, 0.01, 0.01)
    rotation_radius_radians: float = math.pi

    def __post_init__(self) -> None:
        if not isinstance(self.anchor, RigidTransform):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise ContractError("pose anchor must be a validated RigidTransform")
        if len(self.scales) != 6:
            raise ContractError("pose scales must contain three mm and three radian values")
        for value in self.scales:
            if finite_scalar(value, "pose scale", minimum=0.0) == 0:
                raise ContractError("pose scales must be strictly positive")
        object.__setattr__(self, "scales", tuple(float(value) for value in self.scales))
        radius = finite_scalar(self.rotation_radius_radians, "rotation radius", minimum=0.0)
        if not 0 < radius <= math.pi:
            raise ContractError("fixed pose chart rotation radius must lie in (0, pi]")

    def increment(self, parameters: Vector) -> Vector:
        if len(parameters) != 6:
            raise ContractError("a rigid-pose evaluator needs six chart parameters")
        values = tuple(a * b for a, b in zip(parameters, self.scales, strict=True))
        if (
            not all(map(math.isfinite, values))
            or math.hypot(*values[3:]) >= self.rotation_radius_radians
        ):
            raise TrialDomainError("trial pose lies outside the declared fixed rotation chart")
        return values

    def pose(self, parameters: Vector) -> RigidTransform:
        increment = self.increment(parameters)
        try:
            return compose_pose(self.anchor, increment)
        except (ValueError, OverflowError) as error:
            raise TrialDomainError("trial transform exceeds the finite rigid-pose range") from error

    def gradient(self, parameters: Vector, local_gradient: Vector) -> Vector:
        gradient = chart_gradient(self.increment(parameters), local_gradient)
        return tuple(a * b for a, b in zip(gradient, self.scales, strict=True))


@dataclass(frozen=True, slots=True)
class PrimaryPoseProblem:
    grid: GridSpec
    geometry: DetectorGeometry
    attenuation: Any
    observation: Any
    objective: ObjectiveSpec = field(default_factory=lambda: ObjectiveSpec(domain="counts"))
    open_beam: float = 1.0
    weights: Any = None
    samples_per_ray: int = 256
    precision: Literal["float32", "float64"] = "float64"
    integration: Literal["midpoint", "cell_gauss"] = "midpoint"


@dataclass(frozen=True, slots=True)
class PoseRecoveryResult:
    pose: RigidTransform
    optimisation: RecoveryResult


class PrimaryPoseEvaluator:
    """Prepared single-stream composition; never share concurrently.

    Attenuation, observations, weights and chart are immutable for a solve.
    The first evaluation checks their device values; later calls rely on that
    immutability and check numerical output flags at explicit completion points.
    FP64 depth, signal and depth cotangents are retained by default so a pose
    line search can resolve the loss changes indicated by its gradient. Explicit
    precision="float32" reproduces the earlier composed storage contract.
    Scratch contents after return may correspond to a rejected trial; the result
    returned by recover_pose identifies the accepted pose and objective. Call
    evaluate explicitly at that pose if a final predicted image is needed.
    """

    def __init__(
        self,
        problem: PrimaryPoseProblem,
        chart: PoseChart,
        *,
        device: str = "cuda:0",
        stream: Any = None,
        execution: Literal["eager", "graph"] = "eager",
    ) -> None:
        if execution not in ("eager", "graph"):
            raise ContractError("execution must be eager or graph")
        self.execution = execution
        self._evaluation_graph: Any = None
        self.problem, self.chart = problem, chart
        if problem.objective.domain not in ("counts", "log_transmission"):
            raise ContractError("primary pose recovery requires counts or log-transmission data")
        self.projection = prepare_projection(
            problem.grid,
            problem.geometry,
            ProjectionSpec(
                problem.samples_per_ray,
                precision=problem.precision,
                integration=problem.integration,
            ),
            device=device,
            stream=stream,
        )
        self.context = self.projection.context
        ctx, wp, count = self.context, self.context.wp, problem.geometry.pixels
        if execution == "graph" and not ctx.device.is_cuda:
            raise ContractError("graph execution requires CUDA")
        ctx.array(
            problem.attenuation, "attenuation", dtype=wp.float32, shape=(problem.grid.voxels,)
        )
        ctx.array(problem.observation, "observation", dtype=wp.float32, shape=(count,))
        if problem.objective.weighted:
            ctx.array(problem.weights, "weights", dtype=wp.float32, shape=(count,))
        elif problem.weights is not None:
            raise ContractError("weights require an explicitly weighted objective")
        self.transmission = (
            prepare_transmission(
                TransmissionSpec(beam="scalar" if problem.objective.domain == "counts" else "none"),
                max_pixels=count,
                device=device,
                stream=ctx.stream,
            )
            if problem.precision == "float32"
            else None
        )
        self.objective = prepare_objective(
            problem.objective, max_pixels=count, device=device, stream=ctx.stream
        )
        with ctx.scope():
            self.pose_device = wp.empty(12, dtype=wp.float64, device=ctx.device)
            self.optical_depth = wp.empty(count, dtype=self.projection.dtype, device=ctx.device)
            self.prediction = wp.empty(count, dtype=self.projection.dtype, device=ctx.device)
            self.image_seed = wp.empty(
                count if problem.precision == "float32" else 0, dtype=wp.float32, device=ctx.device
            )
            self.depth_seed = wp.empty(count, dtype=self.projection.dtype, device=ctx.device)
            self._control_device = wp.empty(7, dtype=wp.float64, device=ctx.device)
            self.loss = self._control_device[:1]
            self.pose_seed = self._control_device[1:]
        # CUDA host staging is pinned; CPU execution needs no CUDA host allocator.
        # Staging has a persistent owner and a fixed-size NumPy view.
        # These are explicit control-plane transfers, not per-pixel Python work.
        self._pose_host = wp.empty(12, dtype=wp.float64, device="cpu", pinned=ctx.device.is_cuda)
        self._pose_view = self._pose_host.numpy()
        self._status_checkpoint = StatusCheckpoint(
            ctx,
            tuple(
                workspace
                for workspace in (self.projection, self.transmission, self.objective)
                if workspace is not None
            ),
        )
        self._completion_readback = CompletionReadback(
            self._status_checkpoint, (self._control_device,)
        )
        self._inputs_checked = False
        self._evaluation_lock = Lock()

    def __call__(self, parameters: Vector) -> Evaluation:
        """f(x)=sum_p ell(n0*exp(-L_p(T(x))),N_p), with T(x)=anchor @ exp((scales*x)^).

        Parameters is a length-6 dimensionless binary64 tuple in the fixed PoseChart.
        Its first three scaled coordinates are Lie translations in mm and the last
        three are rotations in rad. The prepared problem owns fixed references to
        FP32 attenuation (voxels,) in mm^-1 and observation/weights (P,), with
        FP64 (12,) pose staging. Depth, prediction and depth covector retain FP64
        by default or the explicitly selected FP32 boundary. Evaluation returns
        binary64 scalar loss and six fixed-chart derivatives, converting the
        local-right pose product through the chart Jacobian and scales. Grid,
        acquisition, beam and observation are fixed. Images remain on the owning
        device and reusable scratch may represent a trial, so reevaluate accepted
        parameters before exporting its image. Concurrent/recursive calls are rejected.

        """
        require_no_tape()
        if not self._evaluation_lock.acquire(blocking=False):
            raise ContractError("a primary evaluator cannot be used concurrently or recursively")
        completed = False
        try:
            result = self._evaluate(parameters)
            completed = True
            return result
        finally:
            # A successful completion already waited for pinned staging. Failed
            # launches must finish its read before a retry can overwrite it.
            try:
                if not completed and self.context.device.is_cuda:
                    self.context.wp.synchronize_stream(self.context.stream)
            finally:
                self._evaluation_lock.release()

    def _evaluate(self, parameters: Vector) -> Evaluation:
        pose = self.chart.pose(parameters)
        ctx, wp = self.context, self.context.wp
        validate = not self._inputs_checked
        self._status_checkpoint.consume()
        with ctx.scope():
            self._pose_view[:] = pose.packed()
            wp.copy(self.pose_device, self._pose_host, stream=ctx.stream)
            if self.execution == "eager" or validate:
                self._launch_evaluation(validate=validate)
            else:
                if self._evaluation_graph is None:
                    self._launch_evaluation(validate=False)
                    self._status_checkpoint.consume()
                    with wp.ScopedCapture(device=ctx.device, stream=ctx.stream) as capture:
                        self._launch_evaluation(validate=False)
                    self._evaluation_graph = capture.graph
                wp.capture_launch(self._evaluation_graph, stream=ctx.stream)
            (controls,) = self._completion_readback.consume()
            self._inputs_checked = True
        gradient = tuple(float(value) for value in controls[1:])
        return Evaluation(float(controls[0]), self.chart.gradient(parameters, gradient))

    def _launch_evaluation(self, *, validate: bool) -> None:
        """Prepared fixed-pointer launches; graph staging/checkpoints stay outside."""
        ctx, problem = self.context, self.problem
        project_optical_depth(
            problem.attenuation,
            self.pose_device,
            out_L=self.optical_depth,
            workspace=self.projection,
            stream=ctx.stream,
            validate=validate,
        )
        if problem.precision == "float64":
            evaluate_primary_objective(
                self.optical_depth,
                problem.observation,
                open_beam=problem.open_beam,
                weights=problem.weights,
                out_prediction=self.prediction,
                out_loss=self.loss,
                out_depth_cotangent=self.depth_seed,
                workspace=self.objective,
                stream=ctx.stream,
                validate=validate,
            )
        else:
            assert self.transmission is not None
            counts = problem.objective.domain == "counts"
            transmit(
                self.optical_depth,
                problem.open_beam if counts else None,
                out_counts=self.prediction if counts else None,
                out_log_T=None if counts else self.prediction,
                workspace=self.transmission,
                stream=ctx.stream,
                validate=validate,
            )
            evaluate_objective(
                self.prediction,
                problem.observation,
                weights=problem.weights,
                out_loss=self.loss,
                out_cotangent=self.image_seed,
                workspace=self.objective,
                stream=ctx.stream,
                validate=validate,
            )
            transmission_vjp(
                self.optical_depth,
                problem.open_beam if counts else None,
                cotangent_counts=self.image_seed if counts else None,
                cotangent_log_T=None if counts else self.image_seed,
                out_grad_L=self.depth_seed,
                workspace=self.transmission,
                stream=ctx.stream,
                validate=validate,
            )
        projection_vjp(
            problem.attenuation,
            self.pose_device,
            cotangent=self.depth_seed,
            out_pose_tangent=self.pose_seed,
            workspace=self.projection,
            stream=ctx.stream,
            validate=validate,
        )


def recover_pose(
    evaluator: PrimaryPoseEvaluator,
    *,
    initial: Vector | None = None,
    policy: RecoveryPolicy | None = None,
    objective_id: str | None = None,
    checkpoint: RecoveryCheckpoint | None = None,
    save_checkpoint: Callable[[RecoveryCheckpoint], None] | None = None,
    checkpoint_relative_tolerance: float = 0.0,
    observe_lifecycle: LifecycleObserver | None = None,
) -> PoseRecoveryResult:
    """T_accepted = anchor @ exp((scales*x_accepted)^) minimises the prepared primary
    objective.

    Initial is a length-6 dimensionless binary64 tuple, defaulting to zero when
    no checkpoint is supplied. The PrimaryPoseEvaluator holds fixed attenuation,
    acquisition, observation and beam and returns the loss/six derivatives in
    the same immutable chart. RecoveryPolicy controls the safeguarded driver
    without rebasing its curvature coordinates. PoseRecoveryResult contains a
    RigidTransform in world mm and the accepted optimisation records. Checkpoint
    restart retains objective identity and verifies the accepted state. Device
    image scratch can hold the last rejected trial, so reevaluate the returned
    optimisation parameters to obtain the accepted prediction.

    """
    start = (0.0,) * 6 if initial is None and checkpoint is None else initial
    result = recover_parameters(
        evaluator,
        start,
        policy=policy,
        objective_id=objective_id,
        checkpoint=checkpoint,
        save_checkpoint=save_checkpoint,
        checkpoint_relative_tolerance=checkpoint_relative_tolerance,
        observe_lifecycle=observe_lifecycle,
    )
    return PoseRecoveryResult(evaluator.chart.pose(result.parameters), result)

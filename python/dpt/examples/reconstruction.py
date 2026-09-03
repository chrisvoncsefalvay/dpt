"""Fixed-geometry primary-count reconstruction; supplied inputs only.

Uses the canonical Warp projector and first-order volume VJP. Torch supplies
CUDA vector arithmetic with shared storage on the same stream. The examples
README links the recorded studies and their separate numerical acceptance.
"""

from __future__ import annotations

import argparse
import importlib
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

from dpt._reductions import prepare_reduction
from dpt._runtime import HostReadback, StatusCheckpoint, StatusReadback
from dpt.contracts import ContractError, NumericalError, finite_scalar, integer, raise_device_status
from dpt.examples._common import (
    example_parser,
    geometry_from_json,
    grid_from_json,
    load_case,
    pose_from_json,
    required,
    run_example,
    write_array,
)
from dpt.examples._common import (
    mapping as _mapping,
)
from dpt.experiments import describe_device, prepare_output, write_json, write_metadata
from dpt.objectives import ObjectiveSpec, evaluate_objective, prepare_objective
from dpt.projection import ProjectionSpec, prepare_projection, project_optical_depth, projection_vjp
from dpt.transmission import TransmissionSpec, prepare_transmission, transmission_vjp, transmit


@dataclass(frozen=True, slots=True)
class SolverSettings:
    iterations: int
    initial_step_mm_inverse_squared: float
    backtracking_factor: float
    armijo: float
    maximum_backtracks: int
    gradient_mapping_tolerance_mm: float
    regularisation_mm: float
    mapping_step_mm_inverse_squared: float | None = None
    warm_start_step: bool = False

    @classmethod
    def read(cls, settings: dict[str, Any]) -> SolverSettings:
        settings = _mapping(settings, "reconstruction.solver")
        for name in (
            "iterations",
            "initial_step_mm_inverse_squared",
            "backtracking_factor",
            "armijo",
            "maximum_backtracks",
            "gradient_mapping_tolerance_mm",
            "regularisation_mm",
        ):
            required(settings, name, "reconstruction.solver")
        result = cls(
            iterations=integer(
                required(settings, "iterations", "reconstruction.solver"), "iterations", minimum=1
            ),
            initial_step_mm_inverse_squared=finite_scalar(
                required(settings, "initial_step_mm_inverse_squared", "reconstruction.solver"),
                "initial step",
                minimum=0,
            ),
            backtracking_factor=finite_scalar(
                required(settings, "backtracking_factor", "reconstruction.solver"), "backtracking"
            ),
            armijo=finite_scalar(
                required(settings, "armijo", "reconstruction.solver"), "Armijo coefficient"
            ),
            maximum_backtracks=integer(
                required(settings, "maximum_backtracks", "reconstruction.solver"),
                "maximum_backtracks",
                minimum=1,
            ),
            gradient_mapping_tolerance_mm=finite_scalar(
                required(settings, "gradient_mapping_tolerance_mm", "reconstruction.solver"),
                "gradient tolerance",
                minimum=0,
            ),
            regularisation_mm=finite_scalar(
                required(settings, "regularisation_mm", "reconstruction.solver"),
                "regularisation coefficient",
                minimum=0,
            ),
            mapping_step_mm_inverse_squared=(
                finite_scalar(
                    settings["mapping_step_mm_inverse_squared"], "mapping step", minimum=0
                )
                if settings.get("mapping_step_mm_inverse_squared") is not None
                else None
            ),
            warm_start_step=settings.get("warm_start_step", False),
        )
        if type(result.warm_start_step) is not bool:
            raise ContractError("warm_start_step must be a boolean")
        if result.initial_step_mm_inverse_squared <= 0:
            raise ContractError("Set initial_step_mm_inverse_squared to a positive value.")
        if (
            result.mapping_step_mm_inverse_squared is not None
            and result.mapping_step_mm_inverse_squared <= 0
        ):
            raise ContractError("Set mapping_step_mm_inverse_squared to a positive value.")
        if not 0 < result.backtracking_factor < 1 or not 0 < result.armijo < 1:
            raise ContractError("Set backtracking_factor and armijo strictly between zero and one.")
        return result


def _settings(case: Any) -> dict[str, Any]:
    settings = _mapping(required(case.config, "reconstruction", "case"), "reconstruction")
    SolverSettings.read(_mapping(required(settings, "solver", "reconstruction"), "solver"))
    grid_from_json(required(settings, "grid", "reconstruction"))
    pose_from_json(required(settings, "fixed_pose", "reconstruction"))
    integer(required(settings, "samples_per_ray", "reconstruction"), "samples_per_ray", minimum=1)
    required(settings, "initial_volume", "reconstruction")
    views = required(settings, "views", "reconstruction")
    if not isinstance(views, list) or not views:
        raise ContractError("reconstruction.views needs at least one calibrated count view")
    for value in cast(list[Any], views):
        view = _mapping(value, "reconstruction.views entry")
        geometry_from_json(required(view, "geometry", "reconstruction view"))
        required(view, "counts", "reconstruction view")
        required(view, "open_beam", "reconstruction view")
    return settings


class Reconstruction:
    """Serial full-data evaluations with persistent CUDA buffers and checked VJPs."""

    def __init__(self, case: Any, *, device: str, torch: Any, wp: Any) -> None:
        self.torch, self.wp, self.device = torch, wp, device
        settings = _settings(case)
        self.policy = SolverSettings.read(
            _mapping(required(settings, "solver", "reconstruction"), "reconstruction.solver")
        )
        self.grid = grid_from_json(settings["grid"])
        pose = pose_from_json(settings["fixed_pose"])
        samples = integer(
            required(settings, "samples_per_ray", "reconstruction"), "samples_per_ray", minimum=1
        )
        initial = case.array(
            required(settings, "initial_volume", "reconstruction"),
            shape=self.grid.shape,
            units="mm^-1",
        )
        if (initial < 0).any():
            raise ContractError(
                "initial_volume contains negative attenuation; supply values in mm^-1."
            )
        if not required(settings, "views", "reconstruction"):
            raise ContractError("Supply at least one calibrated view in reconstruction.views.")
        self.torch_stream = torch.cuda.current_stream(device=device)
        wp.init()
        self.stream = wp.stream_from_torch(self.torch_stream)
        self.mu = torch.tensor(initial, device=device).flatten()
        self.trial = torch.empty_like(self.mu)
        self.gradient = torch.empty_like(self.mu)
        self.delta = torch.empty_like(self.mu)
        self.edge32 = torch.empty_like(self.mu)
        self.volume64 = torch.empty(self.grid.voxels, dtype=torch.float64, device=device)
        self.edge64 = torch.empty_like(self.volume64)
        self.scalar = torch.empty((), dtype=torch.float64, device=device)
        self.pose = wp.array(pose.packed(), dtype=wp.float64, device=device)
        self.mu_wp = wp.from_torch(self.mu)
        self.trial_wp = wp.from_torch(self.trial)
        self.gradient_wp = wp.from_torch(self.gradient)
        self.views: list[dict[str, Any]] = []
        for entry in required(settings, "views", "reconstruction"):
            geometry = geometry_from_json(entry["geometry"])
            observed = case.array(
                required(entry, "counts", "reconstruction.views entry"),
                shape=geometry.shape,
                units="counts",
            )
            beam = case.array(
                required(entry, "open_beam", "reconstruction.views entry"),
                shape=geometry.shape,
                units="counts",
            )
            if (observed < 0).any() or (observed % 1 != 0).any() or (observed > 2**24).any():
                raise ContractError(
                    "counts must be nonnegative integers no greater than 2^24; "
                    "supply unprocessed counts exactly representable in float32."
                )
            if (beam <= 0).any():
                raise ContractError("open_beam must contain positive calibrated mean counts.")
            projection = prepare_projection(
                self.grid,
                geometry,
                ProjectionSpec(samples, active_pose=False, active_volume=True),
                device=device,
                stream=self.stream,
            )
            view: dict[str, Any] = {
                "projection": projection,
                "transmission": prepare_transmission(
                    TransmissionSpec(beam="per-pixel"),
                    max_pixels=geometry.pixels,
                    device=device,
                    stream=self.stream,
                ),
                "objective": prepare_objective(
                    ObjectiveSpec(kind="poisson", domain="counts", reduction="sum"),
                    max_pixels=geometry.pixels,
                    device=device,
                    stream=self.stream,
                ),
                "observed": wp.array(observed.reshape(-1), dtype=wp.float32, device=device),
                "beam": wp.array(beam.reshape(-1), dtype=wp.float32, device=device),
                "shape": geometry.shape,
            }
            for name in ("depth", "prediction", "image_seed", "depth_seed"):
                view[name] = wp.empty(geometry.pixels, dtype=wp.float32, device=device)
            view["loss"] = wp.empty(1, dtype=wp.float64, device=device)
            self.views.append(view)
        # All uploads above are complete before either framework consumes them.
        # Subsequent Torch/Warp work shares exactly one stream.
        wp.synchronize_device(device)
        self.context = self.views[0]["projection"].context
        with self.context.scope():
            self._status = wp.zeros(1, dtype=wp.int32, device=self.context.device)
            self._view_losses = wp.empty(
                len(self.views), dtype=wp.float64, device=self.context.device
            )
            self._view_total = wp.empty(1, dtype=wp.float64, device=self.context.device)
        self._status_readback = StatusReadback(self.context, self._status)
        self._loss_readback = HostReadback(self.context, self._view_total)
        self._loss_reduction = prepare_reduction(self.context, len(self.views))
        for index, view in enumerate(self.views):
            view["loss"] = self._view_losses[index : index + 1]
            view["objective"].validate_observation(view["observed"], stream=self.stream)
        self._checkpoint = StatusCheckpoint(
            self.context,
            (
                self,
                *(
                    view[name]
                    for view in self.views
                    for name in ("projection", "transmission", "objective")
                ),
            ),
        )
        self.edges: list[tuple[Any, Any, Any, Any, Any, float]] = []
        shaped64 = self.volume64.view(self.grid.shape)
        shaped_gradient = self.gradient.view(self.grid.shape)
        cell_volume = math.prod(self.grid.spacing_mm)
        for axis, spacing in enumerate(reversed(self.grid.spacing_mm)):
            if self.grid.shape[axis] == 1:
                continue
            low = [slice(None)] * 3
            high = [slice(None)] * 3
            low[axis], high[axis] = slice(None, -1), slice(1, None)
            lower, upper = tuple(low), tuple(high)
            shape = shaped64[lower].shape
            count = shaped64[lower].numel()
            coefficient = self.policy.regularisation_mm * cell_volume / spacing**2
            if not math.isfinite(coefficient):
                raise ContractError(
                    "Grid spacing and regularisation overflow; rescale their units."
                )
            self.edges.append(
                (
                    shaped64[lower],
                    shaped64[upper],
                    shaped_gradient[lower],
                    shaped_gradient[upper],
                    (self.edge64[:count].view(shape), self.edge32[:count].view(shape)),
                    coefficient,
                )
            )
        self._edge_losses = torch.zeros(len(self.edges), dtype=torch.float64, device=device)
        self._penalty_total = torch.empty((), dtype=torch.float64, device=device)

    def clear_status(self) -> None:
        with self.context.scope():
            self._status.zero_()

    def check_status(self) -> None:
        raise_device_status(self._status_readback.read(), "reconstruction loss reduction")

    def evaluate(self, volume: Any, volume_wp: Any, *, gradient: bool) -> float:
        """Evaluate every supplied view before an update; add the prior once."""
        if gradient:
            self.gradient.zero_()
        self._checkpoint.check()
        self.views[0]["projection"].validate_inputs(volume_wp, self.pose, stream=self.stream)
        for view in self.views:
            project_optical_depth(
                volume_wp,
                self.pose,
                workspace=view["projection"],
                out_L=view["depth"],
                stream=self.stream,
                validate=False,
            )
            transmit(
                view["depth"],
                view["beam"],
                workspace=view["transmission"],
                out_counts=view["prediction"],
                stream=self.stream,
                validate=False,
            )
            evaluate_objective(
                view["prediction"],
                view["observed"],
                workspace=view["objective"],
                out_loss=view["loss"],
                out_cotangent=view["image_seed"] if gradient else None,
                stream=self.stream,
                validate=False,
            )
            if gradient:
                transmission_vjp(
                    view["depth"],
                    view["beam"],
                    cotangent_counts=view["image_seed"],
                    workspace=view["transmission"],
                    out_grad_L=view["depth_seed"],
                    stream=self.stream,
                    validate=False,
                )
                projection_vjp(
                    volume_wp,
                    self.pose,
                    cotangent=view["depth_seed"],
                    workspace=view["projection"],
                    out_mu=self.gradient_wp,
                    accumulate=True,
                    stream=self.stream,
                    validate=False,
                    check_volume_gradient=False,
                )
        if gradient:
            self.views[0]["projection"].enqueue_volume_gradient_check(
                self.gradient_wp, stream=self.stream
            )
        with self.context.scope():
            self._loss_reduction.finish(
                len(self.views), self._view_total, self._status, source=self._view_losses
            )
        penalty = self.regularise(volume, gradient=gradient)
        self._checkpoint.consume()
        loss = float(self._loss_readback.read()[0]) + penalty
        if not math.isfinite(loss):
            raise NumericalError("The objective is non-finite; inspect counts and model range.")
        if gradient and not bool(self.torch.isfinite(self.gradient).all().item()):
            raise NumericalError("The volume gradient overflowed; discard this evaluation.")
        return loss

    def regularise(self, volume: Any, *, gradient: bool) -> float:
        """Quadratic physical-space differences; omit edges beyond the grid."""
        self.volume64.copy_(volume)
        self._edge_losses.zero_()
        for index, (low, high, grad_low, grad_high, scratch, coefficient) in enumerate(self.edges):
            difference, derivative = scratch
            self.torch.sub(high, low, out=difference)
            if gradient:
                # Scale in FP64 before the FP32 accumulation boundary.
                difference.mul_(coefficient)
                derivative.copy_(difference)
                grad_low.sub_(derivative)
                grad_high.add_(derivative)
                # Restore unscaled differences for the objective.
                self.torch.sub(high, low, out=difference)
            difference.square_()
            self.torch.sum(difference, dim=(0, 1, 2), out=self._edge_losses[index])
            self._edge_losses[index].mul_(0.5 * coefficient)
        self.torch.sum(self._edge_losses, dim=0, out=self._penalty_total)
        return float(self._penalty_total.item())

    def inner_product(self, left: Any, right: Any) -> float:
        self.volume64.copy_(left)
        self.edge64.copy_(right)
        self.volume64.mul_(self.edge64)
        self.torch.sum(self.volume64, dim=0, out=self.scalar)
        return float(self.scalar.item())

    def displacement(self, step: float) -> tuple[float, float]:
        # min(g, mu / step) evaluates the projected-gradient mapping
        # without cancellation between almost identical accepted/trial voxels.
        self.volume64.copy_(self.gradient)
        self.edge64.copy_(self.mu).div_(step)
        self.torch.minimum(self.volume64, self.edge64, out=self.volume64)
        self.torch.abs(self.volume64, out=self.edge64)
        mapping = float(self.edge64.max().item())
        self.volume64.mul_(-step).add_(self.mu).clamp_(min=0)
        self.trial.copy_(self.volume64)
        self.torch.sub(self.trial, self.mu, out=self.delta)
        slope = self.inner_product(self.gradient, self.delta)
        return mapping, slope

    def solve(
        self, callback: Callable[[int, dict[str, Any]], None] | None = None
    ) -> dict[str, Any]:
        """Solve with optional observation of accepted updates only.

        The callback receives the stable one-based accepted iteration and a
        detached history dictionary after ``mu`` has been updated. It may read
        or export the accepted fields but must not mutate solver state. Callback
        exceptions propagate with the last accepted volume intact.

        An optional fixed mapping step separates the stationarity diagnostic
        from the initial Armijo trial. Omitting it preserves the original
        diagnostic at ``initial_step_mm_inverse_squared``.
        """
        policy = self.policy
        mapping_step = (
            policy.initial_step_mm_inverse_squared
            if policy.mapping_step_mm_inverse_squared is None
            else policy.mapping_step_mm_inverse_squared
        )
        history: list[dict[str, Any]] = []
        reason = "iteration_budget"
        accepted_step = policy.initial_step_mm_inverse_squared
        gradient_current = False
        for iteration in range(policy.iterations):
            loss = self.evaluate(self.mu, self.mu_wp, gradient=True)
            gradient_current = True
            step = (
                accepted_step if policy.warm_start_step else policy.initial_step_mm_inverse_squared
            )
            mapping, _ = self.displacement(mapping_step)
            if not math.isfinite(mapping):
                raise NumericalError("The trial update overflowed; reduce the initial step.")
            if mapping <= policy.gradient_mapping_tolerance_mm:
                reason = "projected_gradient_tolerance"
                break
            accepted = False
            for attempt in range(policy.maximum_backtracks):
                if step == 0:
                    break
                _, slope = self.displacement(step)
                if not math.isfinite(slope) or slope >= 0:
                    step *= policy.backtracking_factor
                    continue
                # Trial value evaluation leaves the accepted gradient unchanged.
                # Domain/range errors abort with the last accepted volume intact.
                trial_loss = self.evaluate(self.trial, self.trial_wp, gradient=False)
                if trial_loss < loss and trial_loss <= loss + policy.armijo * slope:
                    self.mu.copy_(self.trial)
                    history.append(
                        {
                            "iteration": iteration + 1,
                            "loss_before": loss,
                            "loss_after": trial_loss,
                            "step_mm_inverse_squared": step,
                            "backtracks": attempt,
                            "mapping_mm_before": mapping,
                        }
                    )
                    accepted = True
                    accepted_step = step
                    gradient_current = False
                    if callback is not None:
                        callback(iteration + 1, dict(history[-1]))
                    break
                step *= policy.backtracking_factor
            if not accepted:
                reason = "line_search_failed"
                break
        # Trial buffers may describe a rejected volume; refresh accepted outputs.
        final_loss = self.evaluate(self.mu, self.mu_wp, gradient=not gradient_current)
        final_mapping, _ = self.displacement(mapping_step)
        if final_mapping <= policy.gradient_mapping_tolerance_mm:
            reason = "projected_gradient_tolerance"
        return {
            "termination": reason,
            "accepted_steps": len(history),
            "final_objective": final_loss,
            "final_gradient_mapping_mm": final_mapping,
            "history": history,
            "warm_start_step": policy.warm_start_step,
        }


def main() -> None:
    run_example(
        example_parser("Reconstruct attenuation from supplied calibrated primary counts."), _run
    )


def _run(args: argparse.Namespace) -> None:
    case = load_case(args.case)
    _settings(case)
    if not str(args.device).startswith("cuda:"):
        raise ContractError("Select a CUDA device, for example --device cuda:0.")
    torch: Any = importlib.import_module("torch")
    wp: Any = importlib.import_module("warp")
    output = prepare_output(
        args.output,
        configuration=case.config,
        metadata={
            "application": "reconstruction",
            "execution_acceptance": "requires separate numerical and physical assessment",
        },
    )
    # Never build a Torch autograd graph around the manually supplied VJP.
    with torch.cuda.device(args.device), torch.no_grad():
        try:
            solver = Reconstruction(case, device=args.device, torch=torch, wp=wp)
            write_metadata(
                output,
                **describe_device(solver.views[0]["projection"].context),
                gpu_name=torch.cuda.get_device_name(args.device),
                torch_version=torch.__version__,
                volume_units="mm^-1",
                objective="summed Poisson half-deviance plus quadratic regularisation",
            )
            report = solver.solve()
            wp.synchronize_stream(solver.stream)
            write_array(
                output, "attenuation.npy", solver.mu.cpu().numpy().reshape(solver.grid.shape)
            )
            for index, view in enumerate(solver.views):
                write_array(
                    output,
                    f"predictions/view-{index:04d}.npy",
                    view["prediction"].numpy().reshape(view["shape"]),
                )
            write_json(output / "solver.json", report)
        finally:
            # Drain pending work before exception paths release shared owners.
            torch.cuda.synchronize(args.device)


if __name__ == "__main__":
    main()

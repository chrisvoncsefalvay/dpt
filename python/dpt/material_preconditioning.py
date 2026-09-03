"""Fixed voxel scaling from fitting-model Fisher information, prepared on CUDA.

This module estimates a metric; it does not change a loss, gradient, constraint
or stationarity test. It reads the accepted field and fixed acquisition inputs,
never observed counts or reference composition. All image/voxel arrays remain
resident. Preparation is deliberately outside the repeated solver loop.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from dpt._reductions import prepare_reduction
from dpt._runtime import load_kernels, require_no_tape
from dpt.contracts import ContractError, NumericalError, finite_scalar
from dpt.material_projection import project_material_paths
from dpt.projection import project_optical_depth, projection_vjp
from dpt.spectral import spectral_signal, spectral_vjp

if TYPE_CHECKING:
    from dpt.material_reconstruction import MaterialReconstruction


@dataclass(frozen=True, slots=True)
class FisherScaling:
    """Resident positive FP32 voxel metric and scalar preparation diagnostics."""

    scale: Any
    diagnostics: dict[str, Any]

    def install(self, solver: MaterialReconstruction, *, initialise_step: bool = True) -> None:
        """Install fixed scaling and optionally retain its absolute curvature.

        With mean-normalised D, the natural trial scale is 1/normaliser. This
        only initialises a proposal: observed nonlinear Poisson curvature can
        exceed local Fisher curvature, so the usual search still safeguards it.
        """
        if type(initialise_step) is not bool:
            raise ContractError("initialise_step must be a boolean")
        solver.set_voxel_metric_scale(
            self.scale,
            curvature_normaliser=self.diagnostics["normaliser"] if initialise_step else None,
        )


def prepare_fisher_scaling(
    solver: MaterialReconstruction, *, minimum_relative_curvature: float = 1.0e-6
) -> FisherScaling:
    """D_raw = A^T*(kappa*(A*1)) + penalty_curvature, with kappa=lambda_max(H^-1/2*B*H^-1/2).

    The prepared two-material Poisson solver supplies B=sum_channels j*j^T/lambda
    from canonical spectral derivatives and the declared positive 2-by-2 metric
    H. A is its nonnegative sampled-field line integral. The returned FisherScaling
    owns FP32 scale (voxels,), formed by flooring D_raw relative to its largest
    curvature and dividing by the floored arithmetic mean. Diagnostics contain
    binary64 normaliser, floor, extrema and counts. This gives a dimensionless
    fixed voxel metric, while Armijo still decides actual updates. Custom penalties
    must provide add_fisher_curvature with a nonnegative smooth majorant.
    Fields/volume adjoints retain FP32 and Fisher algebra/reductions retain FP64.
    No field/gradient is changed, but reused path/image scratch invalidates prediction
    exports until reevaluation. All-zero total curvature is rejected.

    """
    relative_floor = finite_scalar(
        minimum_relative_curvature, "minimum_relative_curvature", minimum=0
    )
    if not 0 < relative_floor <= 1:
        raise ContractError("minimum_relative_curvature must lie in (0,1]")
    custom_curvature = (
        None
        if solver.penalty_operator is None
        else getattr(solver.penalty_operator, "add_fisher_curvature", None)
    )
    if solver.penalty_operator is not None and not callable(custom_curvature):
        raise ContractError("Fisher scaling needs an explicit curvature bound for a custom penalty")
    if solver.materials != 2 or solver.material_metric is None:
        raise ContractError("Fisher scaling requires two materials and a fixed positive metric")
    if solver.observation_model != "poisson":
        raise ContractError("Fisher scaling currently requires the independent Poisson model")
    require_no_tape()
    ctx, wp, np = solver.context, solver.wp, solver.np
    ctx.assert_stream(solver.stream)
    kernels = load_kernels("dpt.kernels.material_preconditioning")
    a, b, c = solver.material_metric
    matrix = np.asarray(((a, b), (b, c)), dtype=np.float64)
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    minimum_eigenvalue = float(eigenvalues[0])
    inverse_root = (eigenvectors / np.sqrt(eigenvalues)) @ eigenvectors.T
    if minimum_eigenvalue <= 0 or not np.isfinite(inverse_root).all():
        raise ContractError("material metric inverse square root is not representable")
    whiten = wp.vec3d(inverse_root[0, 0], inverse_root[0, 1], inverse_root[1, 1])
    precision = solver.spectral_workspace.spec.precision
    double = precision == "float64"
    value_type = wp.float64 if double else wp.float32
    voxels = solver.grid.voxels
    maximum_pixels = max(view.pixels for view in solver.views)
    reduction = prepare_reduction(ctx, voxels)

    def check_status() -> None:
        wp.synchronize_stream(ctx.stream)
        if int(status.numpy()[0]):
            raise NumericalError("Fisher voxel scaling is not finite and representable")

    with ctx.scope():
        ones = wp.full(voxels, 1.0, dtype=wp.float32, device=ctx.device)
        projected = wp.zeros(voxels, dtype=wp.float32, device=ctx.device)
        curvature = wp.empty(voxels, dtype=wp.float64, device=ctx.device)
        penalty_curvature = (
            curvature
            if custom_curvature is None
            else wp.zeros(voxels, dtype=wp.float64, device=ctx.device)
        )
        scale = wp.empty(voxels, dtype=wp.float32, device=ctx.device)
        ray_ones = wp.full(maximum_pixels, 1.0, dtype=value_type, device=ctx.device)
        mean = wp.empty(maximum_pixels, dtype=value_type, device=ctx.device)
        derivative = wp.empty(2 * maximum_pixels, dtype=value_type, device=ctx.device)
        length = wp.empty(maximum_pixels, dtype=value_type, device=ctx.device)
        seed = wp.empty(maximum_pixels, dtype=value_type, device=ctx.device)
        fisher = wp.empty(maximum_pixels, dtype=wp.vec3d, device=ctx.device)
        extrema = wp.array((math.inf, 0.0), dtype=wp.float64, device=ctx.device)
        counts = wp.zeros(2, dtype=wp.int32, device=ctx.device)
        status = wp.zeros(1, dtype=wp.int32, device=ctx.device)
        total = wp.empty(1, dtype=wp.float64, device=ctx.device)
        solver.invalidate_predictions()
        solver.spectral_workspace.check_status()
        solver.views[0].accepted.validate_inputs(stream=ctx.stream)
        for view in solver.views:
            pixels = view.pixels
            projection = view.accepted.projection
            current_mean, current_derivative = mean[:pixels], derivative[: 2 * pixels]
            current_length, current_seed = length[:pixels], seed[:pixels]
            current_fisher, current_ones = fisher[:pixels], ray_ones[:pixels]
            current_fisher.zero_()
            project_material_paths(view.pose, workspace=view.accepted, stream=ctx.stream)
            for channel in view.channels:
                spectral_signal(
                    view.paths,
                    solver.coefficients,
                    channel.weights,
                    channel.response,
                    out_mean=current_mean,
                    workspace=solver.spectral_workspace,
                    stream=ctx.stream,
                )
                spectral_vjp(
                    view.paths,
                    solver.coefficients,
                    channel.weights,
                    channel.response,
                    cotangent=current_ones,
                    out_grad_paths=current_derivative,
                    workspace=solver.spectral_workspace,
                    stream=ctx.stream,
                )
                wp.launch(
                    kernels.get_add_fisher(double),
                    dim=pixels,
                    inputs=[current_mean, current_derivative, pixels, current_fisher, status],
                    stream=ctx.stream,
                    record_tape=False,
                )
            project_optical_depth(
                ones,
                view.pose,
                out_L=current_length,
                workspace=projection,
                stream=ctx.stream,
            )
            wp.launch(
                kernels.get_fisher_row_seed(double),
                dim=pixels,
                inputs=[current_fisher, current_length, whiten, current_seed, status],
                stream=ctx.stream,
                record_tape=False,
            )
            check_status()
            projection_vjp(
                ones,
                view.pose,
                cotangent=current_seed,
                out_mu=projected,
                workspace=projection,
                stream=ctx.stream,
                accumulate=True,
            )
        if custom_curvature is not None:
            custom_curvature(
                solver.fractions,
                out_curvature=penalty_curvature,
                minimum_metric_eigenvalue=minimum_eigenvalue,
                status=status,
                context=ctx,
            )
            check_status()
        wp.launch(
            kernels.add_regulariser,
            dim=voxels,
            inputs=[
                projected,
                *solver.grid.shape[::-1],
                wp.vec3d(*solver.penalty_coefficients),
                minimum_eigenvalue,
                custom_curvature is not None,
                penalty_curvature,
                curvature,
                extrema,
                counts,
                status,
            ],
            stream=ctx.stream,
            record_tape=False,
        )
        check_status()
        low, high = map(float, extrema.numpy())
        if high <= 0:
            raise ContractError("no positive Fisher or regularisation curvature; scaling undefined")
        floor = high * relative_floor
        if not math.isfinite(floor) or floor <= 0:
            raise NumericalError("the declared relative curvature floor is not representable")
        wp.launch(
            kernels.floor_curvature,
            dim=voxels,
            inputs=[curvature, floor, counts],
            stream=ctx.stream,
            record_tape=False,
        )
        reduction.finish(voxels, total, status, source=curvature)
        check_status()
        normaliser = float(total.numpy()[0]) / voxels
        wp.launch(
            kernels.normalise_curvature,
            dim=voxels,
            inputs=[curvature, normaliser, scale, status],
            stream=ctx.stream,
            record_tape=False,
        )
        check_status()
        zero_data, floored = map(int, counts.numpy())
    return FisherScaling(
        scale,
        {
            "method": "local Poisson Fisher row majorant plus regulariser row majorant",
            "field_state": "accepted field at preparation; fixed for the following solve",
            "material_metric": solver.material_metric,
            "minimum_material_metric_eigenvalue": minimum_eigenvalue,
            "normalisation": "arithmetic mean of floored curvature",
            "normaliser": normaliser,
            "natural_initial_step": 1.0 / normaliser,
            "minimum_raw_curvature": low,
            "maximum_raw_curvature": high,
            "minimum_relative_curvature": relative_floor,
            "curvature_floor": floor,
            "minimum_scale": max(low, floor) / normaliser,
            "maximum_scale": high / normaliser,
            "zero_data_curvature_voxels": zero_data,
            "floored_voxels": floored,
            "voxels": voxels,
            "views": len(solver.views),
            "scale_bytes": voxels * 4,
            "path_signal_precision": precision,
            "custom_penalty_curvature": None
            if solver.penalty_operator is None
            else type(solver.penalty_operator).__qualname__,
            "accumulation": "canonical FP32 volume VJP; FP64 Fisher algebra and sum reduction",
            "scope": "positive algorithmic metric; not observed-Hessian or identifiability proof",
        },
    )

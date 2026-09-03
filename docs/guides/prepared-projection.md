# Prepare a projection for repeated use

A projection workspace fixes the grid, acquisition, quadrature and execution stream. A `PreparedProjectionInputs` record additionally owns a field and rigid-pose snapshot. Prepare this record when repeated evaluations use the same inputs, then pass it to the forward operator or an explicit VJP.

## Choose the numerical settings

For tensor calculations, prepare [`dpt.torch.Projector`](../pytorch.md) with the grid, acquisition, `precision="fp32"` or `"fp64"`, and sample count. Reuse the module while field values or poses change.

The explicit Warp interface below uses one `ProjectionSpec`. It fixes integration, active derivatives, buffer types and validation alongside the prepared workspace. `ProjectionGradients` groups explicit volume, local-right pose and packed-storage pose destinations. Low-level mixed sampling modes remain available for numerical comparisons. These modes retain FP64 accumulation.

## Own the inputs

This example projects a constant 1 mm cube. The field coefficient is 0.25 mm⁻¹, so its optical depth is 0.25 and the derivative with respect to that coefficient is 1 mm.

```python
import math

import warp as wp

from dpt.geometry import DetectorGeometry, RigidTransform
from dpt.projection import (
    ProjectionSpec, prepare_projection, project_optical_depth, projection_vjp,
)
from dpt.volumes import GridSpec

device = "cuda:0"
grid = GridSpec((1, 1, 1), (1.0, 1.0, 1.0))
detector = DetectorGeometry(
    (-2.0, 0.0, 0.0), (2.0, 0.0, 0.0),
    (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 1.0), (1, 1),
)
workspace = prepare_projection(
    grid, detector,
    ProjectionSpec(32, active_pose=False, active_volume=True, precision="float64"),
    device=device,
)
mu = wp.array([0.25], dtype=wp.float32, device=device)
pose = wp.array(RigidTransform().packed(), dtype=wp.float64, device=device)
inputs = workspace.prepare_inputs(mu, pose)
workspace.warm(volume_gradient_precision="float64")

depth = wp.empty(1, dtype=wp.float64, device=device)
cotangent = wp.array([1.0], dtype=wp.float64, device=device)
gradient = wp.empty(1, dtype=wp.float64, device=device)
project_optical_depth(inputs, workspace=workspace, out_L=depth)
projection_vjp(
    inputs, cotangent=cotangent, workspace=workspace, out_mu=gradient,
    volume_gradient_precision="float64",
)
assert math.isclose(float(depth.numpy()[0]), 0.25, abs_tol=1e-12)
assert math.isclose(float(gradient.numpy()[0]), 1.0, abs_tol=1e-12)
```

`prepare_inputs` copies the field and pose on the workspace stream and validates those copies. Editing `mu` or `pose` afterwards leaves the snapshot intact. Prepare another record to project changed values. Each record belongs to its original workspace configuration. Its `device_bytes` reports the retained field and pose storage.

`warm()` compiles the selected forward and active VJP variants without evaluating rays or allocating image/gradient buffers. Use `volume_gradient_precision="float64"` when the planned VJP writes an FP64 volume destination. Preparation, compilation and repeated evaluation are separate operations.

## Choose validation checkpoints

| Policy | Field and pose contents | Completion |
|---|---|---|
| `once` | Validate owned snapshots during preparation | Check diagnostics on each evaluation. Validate each new VJP cotangent |
| `always` | Rescan current inputs on every call | Check diagnostics before and after execution |
| `never` | Caller validates the current input epoch | Caller checks diagnostics before accepting outputs |

The default specification is `validate="once"`. Raw `mu, pose` calls under that default rescan inputs each time, because mutable external arrays have no retained validation epoch. An explicit `validate="once"` call requires a prepared record. Boolean `True` and `False` select `always` and `never` respectively.

For an unchecked sequence, call `workspace.validate_inputs(mu, pose, cotangent=cotangent)` after the last input change, keep those arrays intact through the sequence, and call `workspace.check_status()` before using the outputs. Array dtype, shape, device, stream and overlap checks still apply. The [execution rules](../concepts/execution.md) explain persistent diagnostics and run boundaries.

Outputs are caller-owned. Prepared records use explicit `projection_vjp`. A Warp tape records raw FP32 field arrays and requires their lifetime through backward. See [differentiation](differentiation.md) for the six-coordinate local-right pose cotangent and twelve-coordinate storage cotangent.

## Select storage and arithmetic independently

| Option | Field storage or sampling arithmetic |
|---|---|
| `storage_precision="float32"` | FP32 field values |
| `storage_precision="float16"` | Native FP16 values, decoded while sampling |
| `storage_precision="bfloat16"` | Native BF16 values, decoded while sampling |
| `sampling_precision="float64"` | FP64 coordinates, interpolation and accumulation |
| `sampling_precision="mixed_float32"` | FP32 local marching and weights, FP64 field products and sums |
| `sampling_precision="fast_float32"` | FP32 local marching, weights and cell partial sums, FP64 ray sums |

`precision` selects optical-depth and cotangent storage separately. FP16/BF16 volume VJPs write FP32 or FP64 destinations and differentiate the decoded field values. Evaluate these VJPs through explicit calls.

Both FP32 sampling policies require midpoint integration, `active_pose=False`, and grid axis counts at most `2**24`. Ray setup and clipping remain FP64. `fast_float32` adds cell rounding to the forward map, so its rounded-weight VJP has an adjoint error. Compare its forward error and [adjoint diagnostics](operator-diagnostics.md) on the fields and acquisitions you intend to use.

## Gather a fixed-pose adjoint

For a fixed pose, `prepare_projection_adjoint` captures the ray setup for the discrete transpose `A(pose)ᵀ`. The gather gives each voxel its own FP64 sum in ray and quadrature order. It uses the forward operator's midpoint positions, clipping and trilinear corner weights.

Continuing the cube example above:

```python
from dpt.projection import prepare_projection_adjoint, projection_adjoint_gather

adjoint = prepare_projection_adjoint(inputs, workspace=workspace)
projection_adjoint_gather(
    adjoint, cotangent=cotangent, workspace=workspace, out_mu=gradient,
    volume_gradient_precision="float64",
)
assert math.isclose(float(gradient.numpy()[0]), 1.0, abs_tol=1e-12)
```

The plan owns its pose and ray snapshots. Flat cone panels use conservative projected voxel-support bounds to select detector candidates. Arbitrary ray bundles retain every ray and restrict the midpoint indices. Source-plane and ill-conditioned bounds use the complete panel. Preparation validates every canonical ray and midpoint before accepting the plan.

Prepared gathering uses `sampling_precision="float64"` and midpoint integration. Its `candidate_rays`, `quadrature_nodes`, `work_bound` and `device_bytes` report the retained calculation. Set `max_work`, `max_prepare_voxels`, `max_rays_per_voxel` and `max_bytes` explicitly when preparing a larger calculation. Execution checks its work budget before writing the destination.

Reuse the plan with different image cotangents. Rebuild it when the pose or geometry changes, and use `accumulate=True` to add the transpose to an existing destination. The raw-pose gather also accepts strict cell-Gauss integration and checks its all-ray work bound before launch.

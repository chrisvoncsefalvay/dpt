# Primary projection

The projector integrates a non-negative sampled attenuation field over each finite source-to-detector segment. It returns optical depth, before spectral summation, detector response or observation noise. Use the [geometry conventions](../concepts/geometry.md) to place the sampled field and detector.

## Project a constant cube on CUDA

This complete example uses the same analytic 1 mm cube as [the first calculation](../getting-started/first-calculation.md). Use `device="cpu"` with the `warp` extra for CPU execution, or run it on CUDA after a successful CUDA preflight. It allocates a one-pixel output and requests retained FP64 optical depth.

```python
import math

import warp as wp

from dpt.geometry import DetectorGeometry, RigidTransform
from dpt.projection import ProjectionSpec, prepare_projection, project_optical_depth
from dpt.volumes import GridSpec

device = "cuda:0"
grid = GridSpec((1, 1, 1), (1.0, 1.0, 1.0))
detector = DetectorGeometry(
    (-2.0, 0.0, 0.0), (2.0, 0.0, 0.0),
    (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 1.0), (1, 1),
)
workspace = prepare_projection(
    grid, detector, ProjectionSpec(32, precision="float64"), device=device
)
mu = wp.array([0.3], dtype=wp.float32, device=device)
pose = wp.array(RigidTransform().packed(), dtype=wp.float64, device=device)
depth = wp.empty(detector.pixels, dtype=wp.float64, device=device)
project_optical_depth(mu, pose, workspace=workspace, out_L=depth)

measured = float(depth.numpy()[0])
assert math.isclose(measured, 0.3, rel_tol=0.0, abs_tol=2e-8)
print(measured)
```

The assertion accounts for the FP32 representation of the coefficient 0.3. FP64 output retains the integral of the supplied FP32 field.

`depth.numpy()` copies the result to the host for this demonstration. Keep production images on the device and pass them directly to compatible downstream operators.

## Select quadrature

`ProjectionSpec(..., integration="midpoint")` uses a fixed number of samples per intersecting ray. Increasing that count can reduce quadrature error. Voxel discretisation error depends separately on the grid and supplied field.

`integration="cell_gauss"` splits the ray at interpolation planes and uses two Gauss nodes on each cubic segment. It ignores `samples_per_ray`, although the specification still requires a positive count. This integrates the declared trilinear field up to arithmetic error. Derivatives follow the active intersection branches described in [differentiation](differentiation.md).

## Request a pose VJP

The default specification permits active pose. To differentiate a scalar function whose optical-depth cotangent is one for this pixel, allocate that cotangent and a six-coordinate destination:

```python
from dpt.projection import projection_vjp

cotangent = wp.array([1.0], dtype=wp.float64, device=device)
local_pose_gradient = wp.empty(6, dtype=wp.float64, device=device)
projection_vjp(
    mu, pose, workspace=workspace, cotangent=cotangent,
    out_pose_tangent=local_pose_gradient,
)
```

This returns a six-coordinate local-right pose cotangent. The constant central-ray fixture is useful for forward units, but is a poor pose-identification problem: symmetry can make several pose directions locally uninformative. Use oriented, spatially varying reference fields and step-size sweeps to test pose derivatives, as the regression suite does.

For a volume cotangent, prepare with `active_volume=True` and use the declared `out_mu` destination. Consult the [exact API](../reference/api-operators.md) and [differentiation guide](differentiation.md) for active flags, accumulation and tape support.

## Form a measurement

Optical depth is dimensionless. Transmission is `exp(-depth)`, and expected primary counts additionally require calibrated open-beam counts. DPT's FP32 transmission interface and retained-FP64 primary objective have different storage contracts. Choose operators with matching storage precision. The [operator reference](../reference/operators.md) identifies the compatible boundaries.

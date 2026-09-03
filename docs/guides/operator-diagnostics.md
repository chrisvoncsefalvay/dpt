# Check a discrete linear operator

An adjoint belongs to a particular discretisation. Check the supplied forward/adjoint pair before using it for reconstruction or choosing a step from its operator norm. `dpt.operator_checks` accepts host NumPy callables and includes an adapter for the fixed-pose projection and volume VJP.

## Compare inner products

For `A: R^n -> R^m`, the Euclidean adjoint satisfies

```text
<Ax, y> = <x, A*y>
relative_error = abs(<Ax,y> - <x,A*y>)
                 / max(||Ax|| ||y||, ||x|| ||A*y||)
```

Norm-product scaling keeps the diagnostic useful when the inner products nearly cancel. A zero map has zero relative error. `seed` fixes the normal random vectors, so a reported result includes a reproducible draw.

```python
import math

import numpy as np

from dpt.operator_checks import LinearOperator, check_adjoint, estimate_operator_norm

matrix = np.array([[1.0, 2.0, -3.0], [-4.0, 5.0, 6.0]])
operator = LinearOperator(
    domain_shape=(3,), range_shape=(2,),
    forward=lambda x: matrix @ x,
    adjoint=lambda y: matrix.T @ y,
)
check = check_adjoint(operator, seed=17, tolerance=1e-12)
assert check.passed

norm = estimate_operator_norm(operator, seed=17, iterations=100, tolerance=1e-10)
assert norm.converged
assert math.isclose(norm.value, float(np.linalg.norm(matrix, ord=2)), rel_tol=1e-10)
```

Callables consume and return finite real arrays with the declared shapes. `AdjointCheck` reports both inner products, absolute and relative errors, tolerance and pass status. Repeat with several seeds and representative operator configurations to explore different directions.

## Estimate the spectral norm

`estimate_operator_norm` uses power iteration on `A*A`. `OperatorNorm.value` estimates `||A||₂`. `residual` measures the relative eigenvector residual and `converged` records whether that residual reached the requested tolerance within `iterations`.

A linear forward and matched adjoint give this estimate its spectral-norm meaning. For half squared least squares, a step proportional to `1 / ||A||₂**2` uses the objective's gradient Lipschitz scale. Select any additional safety factor and regularisation contribution for the optimisation problem you are solving. Inspect convergence before using the estimate to set a step.

## Diagnose the production projection pair

For this diagnostic, a fixed-pose projection is a linear map from signed field values to line integrals. Prepare `field_domain="signed_linear"`, `active_volume=True` and `active_pose=False`. The adapter requires FP32 field storage and owns FP64 volume-gradient scratch.

```python
from dpt.geometry import DetectorGeometry, RigidTransform
from dpt.operator_checks import projection_linear_operator
from dpt.projection import ProjectionSpec, prepare_projection
from dpt.volumes import GridSpec

grid = GridSpec((1, 1, 1), (1.0, 1.0, 1.0))
detector = DetectorGeometry(
    (-2.0, 0.0, 0.0), (2.0, 0.0, 0.0),
    (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 1.0), (1, 1),
)
workspace = prepare_projection(
    grid, detector,
    ProjectionSpec(
        32, active_pose=False, active_volume=True,
        precision="float64", field_domain="signed_linear",
    ),
    device="cuda:0",
)
projection = projection_linear_operator(workspace, RigidTransform())
assert check_adjoint(projection, tolerance=1e-6).passed
projection_norm = estimate_operator_norm(projection, iterations=20)
assert projection_norm.converged
assert math.isclose(projection_norm.value, 1.0, rel_tol=1e-6)
```

Its domain is `(grid.voxels,)` in x-fast order and its range is `(geometry.pixels,)` in row-major order. Diagnostic calls explicitly upload host inputs and read results back after synchronising the owning stream. Keep this interface for operator inspection. Production iterations can retain their arrays on the device.

The diagnostic tests the selected midpoint or cell-Gauss discretisation and its implemented volume VJP. Storage conversion and FP32 sampling options affect the measured error. Compare configurations with the [projection precision guide](prepared-projection.md).

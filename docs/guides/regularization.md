# Regularise a voxel field

Use total variation (TV) to favour piecewise constant fields, L1 to favour
sparse coefficients, and a support mask to fix known exterior voxels at zero.
Prepare one workspace for the grid and penalty, then reuse it for value,
subgradient and proximal calculations.

## Choose the discrete penalty

`RegularizationSpec` defines

```text
R(x) = tv_weight * sum(phi(Dx)) + l1_weight * sum(abs(x))
```

`D` takes forward differences along the grid's local x, y and z axes, divided
by the corresponding spacing in millimetres. Isotropic TV uses the Euclidean
norm of each three-component difference. Anisotropic TV sums its absolute
components. The outward difference at an outer grid boundary is zero.

The sum has no implicit voxel-volume factor. Choose weights for the field's
units and the declared spacing. Multiply them by the voxel volume explicitly
when your discretisation calls for it. At a zero value or zero difference,
the reported subgradient is zero.

This two-voxel example has a single x difference of `-4 / mm`. Its TV/L1
penalty is `0.2 * 4 + 0.1 * 2 = 1`.

```python
import math
import warp as wp

from dpt.volumes import GridSpec
from dpt.regularization import (
    RegularizationSpec, prepare_regularization,
    regularization_value_and_gradient,
)

grid = GridSpec((1, 1, 2), (0.5, 1.0, 1.0))
spec = RegularizationSpec(tv_weight=0.2, l1_weight=0.1)
workspace = prepare_regularization(grid, spec, device="cuda:0")
field = wp.array([1.0, -1.0], dtype=wp.float64, device="cuda:0")
gradient = wp.empty(2, dtype=wp.float64, device="cuda:0")
value = regularization_value_and_gradient(
    field, out_gradient=gradient, workspace=workspace,
)
assert math.isclose(value, 1.0, rel_tol=1e-14)
assert all(math.isclose(float(actual), expected, rel_tol=1e-14)
           for actual, expected in zip(gradient.numpy(), (0.5, -0.5), strict=True))
```

Flatten fields with x varying fastest, from `(nz, ny, nx)`. Input, gradient
and output use the precision declared in the spec. Internal reductions and
proximal iterations use FP64.

## Take a proximal step

`regularization_prox` minimises

```text
0.5 * sum((x - field)**2) + step * R(x)
```

Continue the example with a step of one. TV reduces the difference between
the two coefficients and L1 draws both towards zero, giving `(0.5, -0.5)`.

```python
from dpt.regularization import RegularizationProxPolicy, regularization_prox

output = wp.empty_like(field)
result = regularization_prox(
    field, step=1.0, out=output, workspace=workspace,
    policy=RegularizationProxPolicy(absolute_gap=1e-13, relative_gap=1e-12),
)
assert result.converged
assert result.gap < 2e-12
assert all(math.isclose(float(actual), expected, abs_tol=2e-6)
           for actual, expected in zip(output.numpy(), (0.5, -0.5), strict=True))
```

L1 and support projection have a closed-form solution. TV uses
Chambolle–Pock iterations and stops when the primal-dual gap meets the
policy's absolute and relative tolerances. Inspect `result.reason`:

| Reason | Result |
| --- | --- |
| `closed_form` | Exact L1 soft thresholding and support projection |
| `primal_dual_gap` | The requested gap tolerance was met |
| `iteration_budget` | A feasible iterate and its gap. Increase the budget if needed |

`primal`, `dual` and `gap` describe the FP64 iterate. A declared FP32 output
adds one final rounding when that iterate is written.

## Fix a support region

Pass a flat device `uint8` mask containing zeros and ones. Preparation copies
the mask. Excluded voxels stay at zero, and TV includes differences against
those zeros at internal support boundaries.

```python
support = wp.array([1, 0], dtype=wp.uint8, device="cuda:0")
masked = prepare_regularization(
    grid, RegularizationSpec(l1_weight=0.1), support=support, device="cuda:0",
)
certificate = regularization_prox(field, step=1.0, out=output, workspace=masked)
assert certificate.reason == "closed_form"
assert math.isclose(float(output.numpy()[0]), 0.9, rel_tol=1e-14)
assert float(output.numpy()[1]) == 0.0
```

Penalty evaluation requires zeros outside the support. A proximal input may
contain arbitrary finite values there: the step projects them to zero. The
subgradient is zero at inactive coordinates.

Workspaces own their scratch arrays and stream. Give each concurrent call its
own workspace and keep output arrays disjoint from inputs. Only scalar loss,
status and certificate data cross to the host during a call. The examples
download arrays to check their small analytical results.

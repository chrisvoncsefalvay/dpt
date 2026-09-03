# Geometry and units

DPT uses millimetres for spatial coordinates and keV for photon energy. Shape tuples describe storage, whilst spacing tuples describe geometric axes. Projection uses sample and pixel centres. Transport uses lower cell and pixel faces.

## Deterministic coordinates

`RigidTransform` is the object-to-world map `T_WO`. Its row-major rotation acts on column vectors, and its translation is in mm. Rotations must be right-handed and orthonormal to `1e-10`. `packed()` returns twelve FP64 values: nine rotation entries followed by three translations.

`GridSpec.shape` is `(nz, ny, nx)`, with x varying fastest in the flat field. `spacing_mm` is `(dx, dy, dz)`, regardless of shape order. `origin_mm` is the first sample centre in object coordinates. `orientation` maps grid axes into object axes. World-space ray points are mapped back through the inverse pose and grid orientation before interpolation.

The sampled field's support extends half a spacing beyond its outer sample centres. Interpolation clamps to the outer samples inside this support and returns zero outside. Non-zero boundary samples therefore create a discontinuity at the support boundary.

`DetectorGeometry` declares a point source and a detector plane. `origin_mm` is the centre of pixel `(0, 0)`. Unit vector `u` advances columns, and unit vector `v` advances rows. Pixel spacing is `(du, dv)`, shape is `(height, width)`, and flat index is `row * width + column`.

## Transport coordinates

`MaterialGrid` is axis-aligned. Its origin is the lower cell face, spacing is `(dx, dy, dz)`, and storage shape remains `(nz, ny, nx)`. Material IDs select supplied piecewise-constant coefficients. The upper extent is `origin + shape_xyz * spacing`.

`PlanarDetector` lies on a constant-z plane above the entire grid. Its `lower_xy_mm` specifies lower pixel faces, its spacing is `(dx, dy)`, and its shape is **`(width, height)`**. Pixel support is half-open. Use this interface for axis-aligned transport experiments.

!!! warning "Origins and shape order differ"

    Convert grid faces, pixel faces and shape order explicitly when moving a fixture between APIs. An unchanged tuple can describe a different physical support.

| Convention | Deterministic API | Transport API |
|---|---|---|
| Grid origin | First sample centre | Lower cell faces |
| Grid field | Trilinear sampled values | Piecewise-constant material labels |
| Detector origin | First pixel centre | Lower pixel faces |
| Detector shape | `(height, width)` | `(width, height)` |
| Detector orientation | Explicit orthonormal in-plane basis | Constant-z plane above the grid |

For axis-aligned grids with the same intended support, a transport lower face is a deterministic sample-centre origin minus half a sample spacing. This aligns the outer support. Transport still uses a piecewise-constant field, whilst the deterministic projector interpolates trilinearly.

## Pose increments and gradients

`compose_pose(anchor, increment)` applies a local right SE(3) increment. The six entries are translation followed by rotation, with translation in mm and rotation in radians. A `PoseChart` supplies physical scales and a fixed anchor so that optimisation can use dimensionless coordinates.

Projection can return a six-coordinate local-right cotangent or twelve unconstrained pose-storage partials. Transform the local cotangent into the optimiser's chart. See [differentiation](../guides/differentiation.md) for the conversion and tape conventions.

At interpolation knots, grazing rays and tied intersection bounds, projection returns the derivative of the selected branch. The forward map can be non-differentiable at these boundaries.

## Keep units at ingestion

Attenuation coefficients are mm⁻¹, material paths are mm, fractions and optical depth are dimensionless, and source weights must have an explicitly interpreted normalisation. Material mass-attenuation data require a declared density conversion before use. Convert centimetres, Hounsfield units, mass coefficients and scanner coordinates to the required quantities before calling the operators.

Record the conversion, acquisition frame and coefficient source alongside the arrays.

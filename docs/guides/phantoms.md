# Synthetic phantoms and analytic chords

Use a small analytic phantom to distinguish geometry, voxel discretisation and
quadrature errors. `Ellipsoid.optical_depth` evaluates the continuous finite-ray
integral, whilst `sample_ellipsoids` creates the voxel-centre values used by the
sampled-field projector. The two quantities converge as the grid resolves the
boundary.

## Check a central chord

An ellipsoid is defined by three positive semi-axes in mm, a centre, a rigid
orientation and a non-negative constant attenuation coefficient in mm⁻¹. A
central ray along its first axis has length twice that semi-axis.

```python
import math

from dpt.phantoms import Ellipsoid

model = Ellipsoid((2.0, 1.5, 1.0), 0.02)
central_depth = model.optical_depth((-3.0, 0.0, 0.0), (3.0, 0.0, 0.0))
assert math.isclose(central_depth, 0.08, rel_tol=1e-14)
assert model.optical_depth((-3.0, 3.0, 0.0), (3.0, 3.0, 0.0)) == 0.0
```

The integral clips to the finite source/target segment. A ray that ends inside
the ellipsoid includes only its traversed chord, and a tangent contributes zero
length. The orientation maps the ellipsoid's local axes into object coordinates.

## Sample the same physical object

This complete example samples the ellipsoid on a 16³ grid, then compares the
central sampled-field integral with its analytic chord. Install the `analysis`
extra for NumPy.

```python
import math

from dpt.geometry import DetectorGeometry, RigidTransform
from dpt.phantoms import Ellipsoid, sample_ellipsoids
from dpt.validation.projection import integrate_sampled_field
from dpt.volumes import GridSpec

spacing = 0.5
origin = -4.0 + spacing / 2
grid = GridSpec((16, 16, 16), (spacing, spacing, spacing), (origin, origin, origin))
model = Ellipsoid((2.0, 1.5, 1.0), 0.02)
values = sample_ellipsoids(grid, (model,))
panel = DetectorGeometry(
    (-6.0, 0.0, 0.0), (6.0, 0.0, 0.0),
    (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 1.0), (1, 1),
)
depth = integrate_sampled_field(
    values.reshape(-1).tolist(), grid, panel, RigidTransform(), 0, 0,
)
analytic = model.optical_depth(panel.source_mm, panel.origin_mm)
assert values.shape == grid.shape and values.dtype.name == "float32"
assert math.isfinite(depth)
assert abs(depth - analytic) < 2 * spacing * model.attenuation_mm_inverse
```

The sampled array is C-contiguous and uses z/y/x storage with x fastest. Its
default dtype is FP32, and `dtype="float64"` selects FP64. Grid orientation and
sample-centre origin determine every physical sample location. Overlapping
ellipsoids add their attenuation. For material fractions, assign a dimensionless
inside value and check the resulting fraction domain before material projection.

The reference integrates the trilinearly interpolated sampled field with cell
Gauss quadrature. It cannot remove the sphere's voxel-boundary error. Refine
spacing at fixed physical support to study that error, then vary midpoint
samples on the same grid to study quadrature separately.

## Construct circular cone views

`circular_cone_beam` returns flat panels facing a common centre. At zero angle,
the source lies on positive object x and the detector on negative x, with
columns along positive y and rows along positive z.

```python
import math

from dpt.phantoms import circular_cone_beam

panels = circular_cone_beam(
    (0.0, math.pi / 2), source_radius_mm=30.0, detector_radius_mm=30.0,
    shape_rows_columns=(3, 5), spacing_column_row_mm=(0.5, 0.7),
)
assert len(panels) == 2 and panels[0].shape == (3, 5)
assert panels[0].source_mm == (30.0, 0.0, 0.0)
assert math.dist(panels[0].pixel_centre(1, 2), (-30.0, 0.0, 0.0)) < 1e-14
```

Both radii are distances from the centre, so the source-to-panel-centre distance
is their sum. Shapes use `(rows, columns)` and spacing uses `(column, row)`.
This helper's zero-angle convention differs from `ConeBeamIntrinsics`, whose
source starts on negative world y. Retain the chosen geometry with your arrays.

Use the [sphere reconstruction](../getting-started/quickstart.md) for generated
Poisson views and a complete constrained solve, or [imaging volumes](imaging-volumes.md)
for a sphere carrying ingestion metadata.

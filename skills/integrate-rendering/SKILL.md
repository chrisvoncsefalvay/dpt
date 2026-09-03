---
name: integrate-rendering
description: Use installed DPT operations to project real CT volumes and compose differentiable spectral, detector and transport calculations with explicit physical inputs.
---

# Integrate DPT imaging

Use the installed `dpt` CLI and tensor API. Begin with the actual image file and
its physical transform. Keep projection and differentiation on the selected
device. Save arrays and viewable images after the calculation.

```sh
dpt project chest.nii.gz --units HU --views ap lateral --energy-kev 60 --output projections
```

For DICOM, supply the image-series directory. A directory containing several
image series reports its actual UIDs and requires `--series-uid` selection.

```python
import dpt

ct = dpt.read_volume("chest.nii.gz", units="HU")
mu = dpt.ct_to_attenuation(ct, energy_kev=60.0)
views = dpt.standard_views(ct.grid, ("ap", "lateral"), shape=(512, 512))
optical_depth = dpt.project(mu, views, device="cuda", precision="fp32")
transmission = (-optical_depth).exp()
```

`ct_to_attenuation` uses the declared 60 keV water-scaled model, with
`mu_water = 0.02059 mm^-1` and relative density `max(1 + HU/1000, 0)`.
It preserves the original HU volume. Supplied calibrated attenuation keeps its
`mm^-1` units. Scanner kVp and image filenames do not provide material calibration.
See [loading and physical geometry](../../docs/geometry.md).

For gradients, construct `dpt.torch.Projector` once and pass the attenuation
tensor to its forward call. Its output is `(views, rows, columns)` dimensionless
optical depth. The optional FP64 `pose=(tx,ty,tz,rx,ry,rz)` is a right increment
in mm/radians around the prepared anchor. `packed_pose=` instead supplies twelve
FP64 entries for row-major rotation followed by translation. Keep image display conversion outside
the loss. See [PyTorch differentiation](../../docs/pytorch.md).

All distances are mm and energies are keV. Volume storage is `(nz, ny, nx)`.
Sampled-grid origins denote the first voxel centre. Deterministic fields are
trilinear with half-voxel outer support. The readers preserve NIfTI scaling and
DICOM value transforms once and express physical geometry in RAS coordinates.
Explicit acquisitions must use the same frame as the volume.

Spectral rendering needs declared material fields, attenuation tables, source
spectrum and detector response. Incident photons per detector pixel are distinct
from total sampled source histories. Apply energy quadrature and quantum
efficiency exactly once. An attenuation field alone cannot determine scatter.
See [spectrum, detector and transport operations](../../docs/components.md).

Transport material grids contain cell labels and use lower cell faces, unlike
trilinear sampled fields. Use the documented grid conversion and compose the
object pose once. Preserve source-law normalisation, complete-history checks and
the distinction between Monte Carlo estimation variance and physical detector
noise. Primary plus scatter already forms the total transport image.

Save numerical optical depth and transmission before PNG windowing. The CLI
writes arrays, PNGs and resolved physical settings. The installed
`dpt.rendering.save_projection_outputs` helper performs the explicit final device
readback. Examples use this same path rather than a separate renderer.

# Detector acceptance and response

Build the detector model in three steps: find how many photons reach each
pixel, choose what an interaction contributes to the signal, then describe how
that signal spreads and acquires noise.

```mermaid
flowchart LR
    A[Source and panel geometry] --> B[Pixel acceptance]
    B --> C[Incident photons by energy]
    C --> D[Interaction probability and signal]
    D --> E[Supplied point spread function]
    E --> F[Image and noise power]
```

## Find the pixel acceptance

`pixel_solid_angle` writes each pixel's solid angle in steradians and its
centre incidence cosine. It accepts planar panels and source-centred
cylindrical panels. For an isotropic point source, multiply the emitted
photon population by `solid_angle / (4 * pi)` to obtain the incident population.
A beam model supplies its own angular distribution.

This complete example uses a single planar pixel. Coordinates and pixel
spacings are in millimetres. Output arrays have shape `(views, rows, columns)`.

```python
import warp as wp

from dpt.detector_physics import prepare_pixel_solid_angle, pixel_solid_angle
from dpt.geometry import DetectorGeometry

panel = DetectorGeometry(
    (0.0, 0.0, -10.0), (0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
    (0.5, 0.5), (1, 1),
)
acceptance = prepare_pixel_solid_angle(panel, device="cuda:0")
solid_angle = wp.empty((1, 1, 1), dtype=wp.float64, device="cuda:0")
cosine = wp.empty_like(solid_angle)
pixel_solid_angle(
    out_solid_angle=solid_angle, out_incidence_cosine=cosine,
    workspace=acceptance,
)
```

## Choose the interaction signal

For total attenuation `mu` in inverse millimetres, normal thickness `t` in
millimetres and incidence cosine `c`, the first-interaction probability is

```text
eta = 1 - exp(-mu * t / c)
```

Choose a `FirstInteractionModel` to assign a signal to that interaction:

| Model | Signal per first interaction |
| --- | --- |
| `interaction-counting` | One count |
| `ideal_full_energy()` | Incident energy in keV |
| `supplied-deposition` | Supplied conditional mean deposited energy in keV |
| `photoelectric_only(...)` | Incident energy for the photoelectric branch, zero for other branches |

Supplied deposition carries its own provenance. The photoelectric model takes
matching total and photoelectric coefficient tables and retains both sources.
Use [material coefficients](physical-data.md) to obtain or read those tables.

Continue the single-pixel example with an analytic attenuation fixture of
`0.5 / mm` at 20 keV and a counting response. The output has energy-major
shape `(energies, pixels)`.

```python
import math

from dpt.detector_physics import (
    DetectorResponse, FirstInteractionModel, prepare_scintillator_response,
    scintillator_response, scintillator_response_vjp,
)
from dpt.materials import MaterialTable

material = MaterialTable("analytic detector", (20.0,), (0.5,), "mm^-1")
model = FirstInteractionModel("interaction-counting")
recipe = DetectorResponse(material, (20.0,), 0.7, model, "above")
slab = prepare_scintillator_response(
    material, (20.0,), model=model, edge_side="above",
    max_pixels=1, precision="float64", device="cuda:0",
)
thickness = wp.array([0.7], dtype=wp.float64, device="cuda:0")
response = wp.empty((1, 1), dtype=wp.float64, device="cuda:0")
scintillator_response(
    thickness, cosine.flatten(), out_response=response, workspace=slab,
)
assert math.isclose(float(response.numpy()[0, 0]), -math.expm1(-0.35), rel_tol=1e-14)

cotangent = wp.array([[1.0]], dtype=wp.float64, device="cuda:0")
grad_thickness = wp.empty(1, dtype=wp.float64, device="cuda:0")
grad_cosine = wp.empty_like(grad_thickness)
scintillator_response_vjp(
    thickness, cosine.flatten(), cotangent=cotangent,
    out_thickness=grad_thickness, out_incidence_cosine=grad_cosine,
    workspace=slab,
)
assert math.isclose(float(grad_thickness.numpy()[0]), 0.5 * math.exp(-0.35), rel_tol=1e-14)
```

Thickness and incidence cosine are active parameters. The prepared coefficients,
energy nodes, interpolation choice and interaction scores stay fixed. The VJP
writes FP64 derivatives for each pixel. The response may use FP32 or FP64 storage.

## Calculate MTF, noise power and DQE

`DetectorTransferSpec` describes incident photon populations, first and second
per-photon signal moments, a passive point spread function and white read noise.
Missed photons contribute zero to both moments. Generic supplied deposition needs
its conditional second moment as well as its mean.

Continue with 100 incident photons per pixel, a three-pixel spread and read-noise
variance of one count squared. Frequencies are `(column, row)` in cycles/mm,
within the sampled Nyquist band.

```python
from dpt.detector_transfer import (
    DetectorTransferSpec, prepare_detector_transfer, detector_transfer,
)

spec = DetectorTransferSpec.from_response(
    recipe, (100.0,), psf_shape=(1, 3), psf_weights=(0.25, 0.5, 0.25),
    spacing_mm=(0.5, 0.5), white_read_noise_variance=1.0,
)
transfer = prepare_detector_transfer(spec, max_frequencies=2, device="cuda:0")
frequencies = wp.array([[0.0, 0.0], [0.5, 0.0]], dtype=wp.float64, device="cuda:0")
mtf, nps, dqe = (wp.empty(2, dtype=wp.float64, device="cuda:0") for _ in range(3))
detector_transfer(
    frequencies, out_mtf=mtf, out_nps=nps, out_dqe=dqe, workspace=transfer,
)
assert all(math.isclose(float(value), expected, rel_tol=1e-14)
           for value, expected in zip(mtf.numpy(), (1.0, 0.5), strict=True))
assert (dqe.numpy() >= 0).all() and (dqe.numpy() <= 1).all()
```

MTF and DQE are dimensionless. NPS has units of signal squared times mm²:
counts² mm² here, or keV² mm² for an energy signal. The calculation describes
the sampled interior lattice under a common fractional modulation of all energy
bins. A finite image uses its declared blur boundary rule.

All three operators reuse prepared workspaces on their owning stream and write
caller-owned arrays. Finish status checks before reading results or using them
in another calculation. The example reads arrays only for its assertions.

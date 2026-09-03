# Beam hardening and water-equivalent calibration

A polychromatic beam changes its spectrum as it passes through a material because its attenuation depends on energy. A single effective attenuation coefficient therefore gives different path lengths at different thicknesses. `WaterEquivalentCalibration` evaluates the spectral attenuation curve and inverts it to recover a path length in the calibration material.

For incident detected weights $w_e$ and linear attenuation coefficients $\mu_e$, the optical depth is

$$
p(l) = -\log\left(\frac{\sum_e w_e\exp(-\mu_e l)}{\sum_e w_e}\right).
$$

Use a water coefficient table at the required composition and density to obtain water-equivalent lengths. The [physical data guide](physical-data.md) explains coefficient sources, units and interpolation at absorption edges. Source weights contain energy-integrated photon populations, and detector response multiplies them before calibration. A common exposure scale cancels from the ratio.

## Prepare the calibration

This calculation uses an explicitly synthetic four-energy table so that it runs without downloading physical data. Supply the material, spectrum and detector response for the acquisition when preparing a measured calibration.

```python
import hashlib
import numpy as np
import warp as wp

from dpt.beam_hardening import BeamHardeningSpec, WaterEquivalentCalibration
from dpt.materials import MaterialTable, Provenance

def fixture_provenance(description, payload):
    return Provenance(
        source="Analytic fixture defined in this calculation",
        sha256=hashlib.sha256(payload).hexdigest(),
        rights="Apache-2.0",
        description=description,
    )

energies = (20.0, 40.0, 80.0, 120.0)
material = MaterialTable(
    name="Synthetic four-energy calibration material",
    energies_kev=energies,
    coefficients=(0.4, 0.12, 0.035, 0.018),
    units="mm^-1",
    provenance=fixture_provenance(
        "Synthetic coefficients in inverse mm", b"mu=(0.4,0.12,0.035,0.018)"
    ),
)
spec = BeamHardeningSpec(
    material=material,
    energies_kev=energies,
    source_weights=(0.1, 0.4, 0.35, 0.15),
    detector_response=(0.7, 0.9, 1.0, 0.8),
    maximum_length_mm=1000.0,
    edge_side="above",
    spectrum_provenance=fixture_provenance(
        "Synthetic integrated photon populations", b"w=(0.1,0.4,0.35,0.15)"
    ),
    response_provenance=fixture_provenance(
        "Synthetic detected-photon response", b"R=(0.7,0.9,1.0,0.8)"
    ),
)
calibration = WaterEquivalentCalibration(spec, shape=(3,), device="cpu")
lengths = wp.array([0.0, 20.0, 100.0], dtype=wp.float64, device="cpu")
depth = calibration.forward(lengths)
recovered = calibration.inverse(depth)
np.testing.assert_allclose(recovered.numpy(), lengths.numpy(), rtol=2e-14, atol=0)
```

Lengths use mm and optical depth is dimensionless. Inputs and outputs are contiguous binary64 Warp arrays with the prepared shape. Select `device="cuda:0"` to prepare the same operator on CUDA. Preparation owns the fixed spectral inputs and reusable image buffers.

## Differentiate the curve

The forward derivative is the attenuation coefficient averaged over the transmitted spectrum:

$$
\frac{dp}{dl} = \frac{\sum_e w_e\mu_e\exp(-\mu_e l)}{\sum_e w_e\exp(-\mu_e l)},
\qquad
\frac{dl}{dp} = \left(\frac{dp}{dl}\right)^{-1}.
$$

Pass an image-shaped cotangent to the corresponding VJP. The two derivative states are retained independently, which lets a calculation inspect both derivatives after forward and inverse evaluation.

```python
seed = wp.array([0.2, -0.3, 0.5], dtype=wp.float64, device="cpu")
grad_length = calibration.forward_vjp(seed).numpy().copy()
grad_depth = calibration.inverse_vjp(seed).numpy().copy()
```

Returned device arrays belong to the calibration and are reused by later calls. Copy an array when retaining several images or gradients. Each checked call completes work on the owning stream and reports its status. With `check=False`, call `check_status()` before consuming the result. Lengths and optical depths must lie within the prepared calibration range, and inversion reports an error if its execution bound cannot resolve the binary64 bracket.

- Source: `python/dpt/beam_hardening.py`
- [Spectral signal and detector operators](spectral-detector.md)

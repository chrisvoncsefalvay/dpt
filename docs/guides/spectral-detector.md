# Spectral signal and detectors

A spectral image sums contributions from energy bins, each with its own attenuation and detector response. DPT evaluates this discrete sum and its derivatives using the supplied coefficients, spectrum and response.

## Declare the spectrum and basis

Project dimensionless material fields into material-major path lengths `A[m, p]` in mm. Supply total primary attenuation coefficients `mu[m, k]` in mm⁻¹. Declare the basis mixing/density assumption, coefficient sources, spectrum provenance and detector-response provenance in the spectral specification.

For each pixel, the defining mean is:

```text
mean[p] = sum_k weights[k,p] * response[k,p]
                * exp(-sum_m coefficients[m,k] * paths[m,p])
```

Weights and responses can instead be shared across pixels when the specification declares that layout. Flat arrays retain material-major or energy-major ordering. Declare shared layouts in the specification.

A spectral density in photons/keV needs explicit quadrature weights in keV. `Spectrum(..., representation="density")` requires them. `representation="bin-fluence"` accepts populations already integrated over each bin and rejects a second energy weight. Source acceptance and inverse-square effects must already be represented in the supplied detector-plane populations.

## Check a two-bin calculation

The independent Decimal reference makes the discrete arithmetic inspectable without CUDA. The following complete base-package example uses one material, one pixel and two energy bins. All numbers are analytic fixtures.

```python
import math

from dpt.validation.spectral import spectral_reference

result = spectral_reference(
    paths=(2.0,), coefficients=(0.25, 0.5),
    weights=(100.0, 50.0), response=(1.0, 1.0),
    materials=1, energies=2,
)
expected = 100.0 * math.exp(-0.5) + 50.0 * math.exp(-1.0)
expected_derivative = -25.0 * math.exp(-0.5) - 25.0 * math.exp(-1.0)
assert math.isclose(float(result.mean[0]), expected, rel_tol=1e-14)
assert math.isclose(float(result.grad_paths[0]), expected_derivative, rel_tol=1e-14)
print(f"mean={float(result.mean[0]):.8f}")
```

The unit response here makes the mean an expected photon count. A response representing energy deposition would give a different unit and observation law. The reference evaluates the defining formula and analytic first derivatives on the CPU. Native spectral evaluation uses Warp on the prepared CPU or CUDA device.

## Prepare the native composition

Prepare material projection and spectral workspaces with compatible path precision and pixel capacity. The spectral specification determines shared/per-pixel storage and which parameters are active. Forward evaluation writes the expected mean into a caller-owned buffer. `spectral_vjp` accepts an image cotangent and writes requested path, coefficient, population or response cotangents.

FP64 paths, means and path cotangents retain weighted deep-attenuation tails. Coefficient, population and response inputs remain FP32. Shared parameter-gradient destinations may use FP32 or FP64. Pixel-specific gradients use FP32. Check diagnostics after reverse evaluation.

Spectral and material-projection interfaces use explicit VJPs and reject ambient tape recording. Compose their VJPs as shown in [the reverse calculation](differentiation.md).

## Model the detector observation

Calibration applies gain, exposure and offset to an expected signal. Both gain and exposure cannot be fitted simultaneously through their product. Blur uses a supplied non-negative stencil with zero extension at the boundary, without edge renormalisation.

For photon counting, `spectral_bin_counts` requires response probabilities in `[0,1]`. Poisson and compound Poisson samplers accept FP32 or FP64 rate arrays, so you can pass a retained-FP64 spectral output directly. Sampling uses an `ObservationIdentity` to define the random experiment.

Use Poisson count objectives for integer count observations. Energy-weighted signals with read noise belong to the declared signal domain. The [operator reference](../reference/operators.md) supplies the exact domain and dtype boundaries, and [parameter recovery](recovery.md) describes fitting the model.

Use [detector acceptance and response](detector-physics.md) to calculate pixel
acceptance, derive a response from slab thickness and incidence, and evaluate
MTF, noise power and DQE for a supplied point spread function.

## Retain the electronic-noise experiment

Electronic read noise evaluates `Y[p] = signal[p] + sigma * Z[p]`, with
independent standard normal `Z[p]` and `sigma` in the signal units. Inputs and
outputs are disjoint caller-owned FP32 `(pixels,)` arrays, with FP64 local
arithmetic before the output rounding. The calculation uses the supplied values directly.

`GaussianReadNoiseExecution()` uses the default `"scalar-box-muller"` policy.
Selecting `"box-muller-pairs"` uses both members of a normal pair at each
absolute even-pixel anchor. Odd offsets, final pixels and chunk boundaries
retain the same absolute image. The policies have the same observation law
and different sample identities. The workspace policy must match
`ObservationIdentity.read_noise_algorithm`.

This complete CPU call declares the paired policy and its matching identity:

```python
import numpy as np
import warp as wp

from dpt.detector import (
    GaussianReadNoiseExecution, ObservationIdentity,
    add_gaussian_read_noise, prepare_detector,
)

policy = GaussianReadNoiseExecution("box-muller-pairs")
workspace = prepare_detector(
    max_pixels=5, device="cpu", read_noise_execution=policy,
)
signal = wp.ones(5, dtype=wp.float32, device="cpu")
output = wp.empty_like(signal)
add_gaussian_read_noise(
    signal, standard_deviation=2.7, out_signal=output,
    identity=ObservationIdentity(17, 3, pixel_offset=1,
                                 read_noise_algorithm=policy.algorithm),
    workspace=workspace,
)
assert np.isfinite(output.numpy()).all()
```

Keep the observation identity when replaying common random numbers. Use a
fresh observation ID for an independent noise realisation.

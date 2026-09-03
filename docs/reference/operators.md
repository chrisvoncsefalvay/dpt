# Deterministic operators

DPT separates ray integration, attenuation, detector response and the objective. Each boundary declares a physical quantity, its storage precision and the inputs that may be differentiated. A detector image and a photon-counting mean can have the same array shape without representing the same measurement.

The low-level production operators use caller-owned, contiguous Warp arrays on a prepared CPU or CUDA device. CUDA execution owns one stream. Preparation allocates persistent scratch. Evaluation writes into supplied destinations and rejects overlapping input/output storage. Prepared projection inputs own field/pose snapshots. The spectral pipeline owns reusable images and gradient destinations. The [execution contract](../concepts/execution.md) explains preparation, stream ownership and persistent diagnostics. Complete signatures are in the [API reference](api.md).

## Storage and layout

Here, `V` is the voxel count, `P` the detector pixel count, `M` the material count and `K` the energy-bin count. FP32 means binary32 storage and FP64 means binary64 storage. Destination storage uses the declared dtype.

| Boundary | Input storage | Output and cotangent storage |
|---|---|---|
| Optical-depth projection | Field `(V,)` FP32, FP16 or BF16 as declared, pose `(12,)` FP64 | Depth/cotangent `(P,)`, FP32 or FP64 as declared |
| Material projection | Fields `(M*V,)` FP32, pose `(12,)` FP64 | Paths/cotangent `(M*P,)` in projection precision |
| Transmission | Depth, device beam and cotangents FP32 | Selected images and gradients FP32 |
| Spectral signal | Paths in declared precision, coefficients/weights/response FP32 | Mean/image cotangent/path gradient in declared precision, shared factors FP32 or FP64, pixel-specific factors FP32 |
| Calibration and blur | Images and calibration arrays FP32 | Images FP32, shared calibration gradients FP32 or FP64 |
| Objective | Prediction in declared precision, observation/weights FP32, validity mask `uint8` | Scalar loss FP64, prediction cotangent in declared precision |

Detector images are row-major. Material paths and fields are material-major, with the pixel/voxel index varying fastest. Spectral coefficients have layout `(M,K)`, and per-pixel spectral weights and responses have layout `(K,P)`. Low-level operator buffers use flat contiguous layouts. Typed pipeline boundaries also accept the documented material/image shapes without changing that element order.

## Optical depth and material paths

`project_optical_depth(mu, pose, *, workspace, out_L, ...)` integrates a non-negative attenuation field, in mm⁻¹, over finite source-to-detector segments. The result is dimensionless optical depth. The default sampling policy uses FP64 coordinates, interpolation and integration accumulators. Native FP16/BF16 field storage and fixed-pose FP32 sampling are explicit choices in `ProjectionSpec`. See [prepared projection](../guides/prepared-projection.md) for validation epochs, warm-up and precision selection.

`ProjectionSpec` selects midpoint integration or `cell_gauss`. Midpoint uses `samples_per_ray`. Cell Gauss splits at interpolation planes and integrates each cubic segment with two nodes, ignoring that sample count. This quadrature choice determines the discrete forward map and its derivative. See the [primary-projection guide](../guides/primary-projection.md) and [geometry conventions](../concepts/geometry.md).

`project_material_paths` applies the same projector to dimensionless basis fields and writes path lengths in mm. The default `fractions` domain requires non-negative fractions summing to at most one, with an explicit `2^-24` allowance for FP32 rounding. The `nonnegative` domain permits equivalent-basis fields without a sum cap. Supply any CT-to-material conversion and density normalisation when preparing these fields.

## Transmission

`transmit` exposes any requested subset of transmission `exp(-L)`, expected counts `n0*exp(-L)`, log transmission `-L` and removed primary fraction `1-exp(-L)`. Counts require a declared beam representation: fixed Python scalar, device scalar or per-pixel array. An active beam must be a device array.

The removed fraction and `optical_depth_from_removed` use cancellation-safe formulae, including portable FP64 series near zero. Counts are formed from the retained attenuation factor before output rounding. This matters when a small transmission multiplied by a bright beam remains representable. The inverse requires finite `0 <= delta < 1`.

## Spectral signal

`spectral_signal` evaluates

```text
mean[p] = sum_k weights[k,p] * response[k,p]
                * exp(-sum_m coefficients[m,k] * paths[m,p])
```

Paths are in mm and coefficients in mm⁻¹. Weights are integrated expected photon populations per energy bin. A `Spectrum` supplied as photons/keV requires explicit energy quadrature weights. Apply quadrature once when converting spectral density to bin fluence. Supply the required inverse-square scaling and spectrum normalisation with the inputs.

Response describes the declared first moment and output unit. `spectral_bin_counts` instead requires detection probabilities in `[0,1]` and materialises energy-major count means for observation generation. It has no tape rule. Its output follows spectral precision. Poisson and compound Poisson samplers accept FP32 or FP64 rates and preserve the supplied rate precision. The FP64 spectral path combines attenuation and factor exponents before final rounding, so underflow of an isolated exponential need not erase a representable weighted signal or VJP.

The [spectral pipeline](../guides/spectral-pipeline.md) owns checked fixed factors and composes material projection with this signal. Its `forward(fields, pose)` saves owned inputs for the next explicit `vjp(grad_image)`. Images and typed gradient arrays are reused. Clone a result you need to retain. `Spectrum.filtered` applies a supplied attenuation table and retains the filter recipe and thickness derivative.

## Detector and objective

`calibrate` computes `gain*exposure*mean + offset`, with positive gain and non-negative exposure. Choose either gain or exposure as the active scale; the model depends on their product. Fitted gain and offset must be shared scalars. `blur` uses a supplied non-negative, odd-sized stencil with sum at most one and zero extension at detector edges. The stencil keeps its supplied weights at the edges. `blur_transpose` applies the corresponding discrete transpose.

Observation generators are separate from these deterministic operators. Poisson count sampling writes `uint64`. Compound Poisson sampling and Gaussian read noise write FP32 signals. `ObservationIdentity` fixes the random experiment. Reusing an identity supplies common random numbers. Independent replicates use independent identities. Sampling functions have no differentiation rule.

`evaluate_objective` writes half squared error or Poisson half-deviance into `out_loss` and, when requested, the prediction derivative into `out_cotangent`. Poisson requires count-domain, non-negative integer observations at included pixels. A zero predicted mean with a positive included observation is outside its trial domain. With `masked=True`, `uint8` validity values of zero bypass value checks and contribute zero. `mean` reduction still divides by the declared image length. This validity mask differs from the registration driver's multiplicative weights, whose excluded observations must already be zero. `evaluate_primary_objective` retains FP64 depth, prediction and depth cotangent through primary transmission, avoiding an FP32 intermediate. It supports count and log-transmission domains.

The [differentiation guide](../guides/differentiation.md) covers cotangent/VJP composition, buffer ownership and pose coordinates.

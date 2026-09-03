# Deterministic operator API

## Flat/dark logarithmic calibration

Correct each frame with its flat and dark acquisitions, then compute optical depth
as `-log((signal-dark)/(flat-dark))`. The prepared operator retains the frame used
by its reverse product. [Calibration source](#dpt.calibration.LogCalibration)
is included below.

::: dpt.calibration.LogCalibration
    options:
      show_source: true

::: dpt.calibration.LogCalibrationResult

::: dpt.calibration.LogCalibrationGradients

Preparation fixes device metadata and scratch. The [operator contracts](operators.md) explain units, layouts and retained precision. [differentiation](../guides/differentiation.md) explains reverse ownership.

## `dpt.projection`


::: dpt.projection.ProjectionGradients

::: dpt.projection.ProjectionSpec

::: dpt.projection.ProjectionWorkspace

::: dpt.projection.PreparedProjectionInputs
    options:
      show_source: true

::: dpt.projection.PreparedProjectionAdjoint

::: dpt.projection.prepare_projection

::: dpt.projection.prepare_projection_adjoint
    options:
      show_source: true

::: dpt.projection.project_optical_depth

::: dpt.projection.projection_vjp

::: dpt.projection.projection_adjoint_gather

::: dpt.projection.projection_pose_sensitivities

## `dpt.batched_projection`

Prepare one shared attenuation grid and compact per-view acquisition metadata. `prepare_batched_projection` accepts flat, parallel or curved panels. `prepare_cone_projection` additionally supports cone-calibration derivatives for SOD, SDD, column offset and row offset.

Forward inputs are a flat attenuation field in mm⁻¹ and packed FP64 object-to-world poses `(V, 12)`. Images and cotangents have shape `(V, rows, columns)` and the retained projection precision. `batched_projection_vjp` writes FP64 `(V, 6)` local-right pose cotangents, FP64 `(V, 4)` intrinsic cotangents and the requested shared-volume cotangent. Active flags select the destinations. Sum intrinsic rows for a calibration shared across views.

The workspace owns its stream, generated rays and derivative reductions. Each call refreshes rays from its supplied poses. The [multiview example](../guides/batched-geometry.md#project-and-differentiate-per-view-motion) supplies all layouts and derivative destinations.


::: dpt.batched_projection.BatchedProjectionWorkspace

::: dpt.batched_projection.prepare_batched_projection

::: dpt.batched_projection.prepare_cone_projection

::: dpt.batched_projection.project_batched_optical_depth

::: dpt.batched_projection.batched_projection_vjp
    options:
      show_source: true

::: dpt.batched_projection.projection_ray_vjp

## Finite detector apertures

`ApertureGeometry` groups explicit rays into pixels and supplies their
normalised incident/response weights. The expected count averages transmitted
contributions within each group:

```text
counts[p] = beam[p] * sum_j weights[j] * exp(-depth[j])
```

The sum runs over pixel `p`'s subrays. Depth is dimensionless and beam values
are expected incident counts per pixel. Preparation binds caller-owned FP64
`depths` and `depth_cotangents`, one value per subray. The VJP receives an FP64
count `cotangent` per pixel and writes the requested beam, field or pose
derivative. Choose `out_pose_tangent` for six local-right cotangents or
`out_pose_matrix` for twelve packed-storage partials.


::: dpt.aperture.ApertureGeometry

::: dpt.aperture.ApertureWorkspace

::: dpt.aperture.prepare_aperture_projection

::: dpt.aperture.project_aperture_counts

::: dpt.aperture.aperture_projection_vjp

## `dpt.material_projection`


::: dpt.material_projection.MaterialProjectionWorkspace

::: dpt.material_projection.prepare_material_projection

::: dpt.material_projection.project_material_paths

`prepare_material_projection(cotangent_paths=...)` binds the material-major path cotangents consumed by `material_projection_vjp`. Select `out_pose_tangent` for six FP64 local-right SE(3) cotangents or `out_pose_matrix` for twelve FP64 storage partials, ordered as the row-major rotation followed by translation. Supply one pose destination per call.

The packed derivative is `sum(m,p) cotangent_paths[m,p] * d paths[m,p] / d pose[12]`. It differentiates the projector's packed storage directly, allowing an external rigid chart to apply its own chain rule. [Torch's spectral adapter](api-torch.md#material-and-spectral-derivatives) uses these twelve partials before differentiating `compose_pose`.

::: dpt.material_projection.material_projection_vjp
    options:
      show_source: true

## `dpt.transmission`


::: dpt.transmission.TransmissionOutputs

::: dpt.transmission.TransmissionSpec

::: dpt.transmission.TransmissionWorkspace

::: dpt.transmission.TransmissionError

::: dpt.transmission.GradientRangeError

::: dpt.transmission.prepare_transmission

::: dpt.transmission.transmit

::: dpt.transmission.transmission_vjp

::: dpt.transmission.optical_depth_from_removed

::: dpt.transmission.optical_depth_from_removed_vjp

## `dpt.spectral`


::: dpt.spectral.Spectrum

::: dpt.spectral.SpectrumFilter

::: dpt.spectral.SpectralInputs
    options:
      show_source: true

::: dpt.spectral.SpectralSpec

::: dpt.spectral.SpectralWorkspace

::: dpt.spectral.prepare_spectral

::: dpt.spectral.spectral_signal

::: dpt.spectral.spectral_vjp

::: dpt.spectral.spectral_bin_counts

## `dpt.pipeline`

Use the [spectral pipeline guide](../guides/spectral-pipeline.md) for a complete material-field forward/VJP.

::: dpt.pipeline.SpectralGradients

::: dpt.pipeline.SpectralPipeline
    options:
      show_source: true

## `dpt.response_matrix`

`ResponseMatrixInputs` owns FP32 coefficients `(M,K)`, photon populations `(K,)` or `(K,P)` and a pixel-shared response bank `(B,K)`. `ResponseMatrixPipeline.forward(fields, pose)` computes one material-path array and returns band images `(B,P)`. `vjp` applies the selected native band derivatives, combines common cotangents and returns a typed `ResponseMatrixGradients` record.

Projection and spectral precision agree. `active_paths` supports active material fields or pose. Spectral flags select coefficient, population and response gradients. Each band retains its native derivative representability boundary. Finite common cotangents are summed exactly and rounded once to their public storage precision before one material reverse. Log-output VJPs require positive physical means for every band/pixel, including zero-cotangent lanes.

Factors are copied at ingestion, fields and pose are copied at forward, and cotangents are copied at reverse. Images and gradients use reusable owned buffers. Read the [response-bank guide](../guides/spectral-pipeline.md#share-material-paths-across-detector-bands) for units, active destinations and the complete two-band example.


::: dpt.response_matrix.ResponseMatrixInputs

::: dpt.response_matrix.ResponseMatrixPipeline
    options:
      show_source: true

::: dpt.response_matrix.ResponseMatrixGradients

## `dpt.spectral_inputs`


::: dpt.spectral_inputs.read_spectral_inputs

## `dpt.detector`


::: dpt.detector.GaussianReadNoiseExecution

::: dpt.detector.DetectorWorkspace

::: dpt.detector.prepare_detector

::: dpt.detector.CalibrationSpec

::: dpt.detector.calibrate

::: dpt.detector.calibration_vjp

::: dpt.detector.BlurSpec

::: dpt.detector.BlurWorkspace

::: dpt.detector.prepare_blur

::: dpt.detector.blur

::: dpt.detector.blur_transpose

::: dpt.detector.ObservationIdentity

::: dpt.detector.sample_poisson_counts

::: dpt.detector.sample_compound_poisson

::: dpt.detector.add_gaussian_read_noise

## `dpt.detector_physics`

Read [detector acceptance and response](../guides/detector-physics.md) for the
complete calculation, units and interaction models.


::: dpt.detector_physics.FirstInteractionModel

::: dpt.detector_physics.DetectorResponse

::: dpt.detector_physics.SolidAngleWorkspace

::: dpt.detector_physics.prepare_pixel_solid_angle

::: dpt.detector_physics.pixel_solid_angle

::: dpt.detector_physics.ScintillatorWorkspace

::: dpt.detector_physics.prepare_scintillator_response

::: dpt.detector_physics.scintillator_response

::: dpt.detector_physics.scintillator_response_vjp

## `dpt.detector_transfer`

The sampled-lattice model retains the supplied signal moments, point spread
function and read-noise variance. Frequencies are in cycles/mm. NPS is in
signal² mm², while MTF and DQE are dimensionless.


::: dpt.detector_transfer.DetectorTransferSpec

::: dpt.detector_transfer.DetectorTransferWorkspace

::: dpt.detector_transfer.prepare_detector_transfer

::: dpt.detector_transfer.detector_transfer

## `dpt.objectives`


::: dpt.objectives.ObjectiveSpec

::: dpt.objectives.ObjectiveWorkspace

::: dpt.objectives.prepare_objective

::: dpt.objectives.evaluate_objective

::: dpt.objectives.evaluate_primary_objective

::: dpt.objectives.reduce_objective_components

## `dpt.autodiff`


::: dpt.autodiff.CheckedWorkspace

::: dpt.autodiff.FirstOrderPass

## Beam-hardening calibration

The [complete calibration](../guides/beam-hardening.md) maps a supplied material path in mm to polychromatic optical depth and its inverse, retaining independent forward/inverse derivative states.


::: dpt.beam_hardening.BeamHardeningSpec

::: dpt.beam_hardening.BeamHardeningExecution

::: dpt.beam_hardening.WaterEquivalentCalibration

## Photon-counting readout

The [photon-counting guide](../guides/photon-counting.md) distinguishes calibrated charge-cloud means, rectangular stationary pileup and finite event observations, with separate pulse/count and energy-accounting outputs.


::: dpt.photon_counting.ChargeCloudPattern

::: dpt.photon_counting.ChargeSharingSpec

::: dpt.photon_counting.ChargeSharingIdentity

::: dpt.photon_counting.ChargeSharingWorkspace

::: dpt.photon_counting.prepare_charge_sharing

::: dpt.photon_counting.charge_sharing_mean

::: dpt.photon_counting.charge_sharing_vjp

::: dpt.photon_counting.sample_charge_sharing

::: dpt.photon_counting.RectangularPileupSpec

::: dpt.photon_counting.RectangularPileupWorkspace

::: dpt.photon_counting.prepare_rectangular_pileup

::: dpt.photon_counting.rectangular_pileup_mean

::: dpt.photon_counting.rectangular_pileup_vjp

::: dpt.photon_counting.rectangular_pileup_readout

## Precision contracts

`precision_contract(spec)` reports each named buffer dtype separately from local arithmetic, coordinates, accumulation and reduction, without importing an optional numerical runtime.


::: dpt.precision.ArrayPrecision

::: dpt.precision.PrecisionContract

::: dpt.precision.precision_contract

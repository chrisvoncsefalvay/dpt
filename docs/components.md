# Spectrum, detector and transport components

Projection returns optical depth. Supply the material fields, spectrum and detector calibration required by the chosen image model.

## Primary transmission

For optical depth L, the surviving primary fraction is `exp(-L)`. Multiplying by the incident photons **per detector pixel** gives expected primary counts. The supported transmission, calibration, blur and objective operators have composable Torch forms in `dpt.torch`. Their units and derivative arguments are documented in the [Torch API](reference/api-torch.md).

## Materials and spectrum

Material projection integrates supplied material fields on the same physical grid. Spectral integration combines those paths with energy-dependent coefficients, quadrature weights and a declared detector response. Supply a conversion model when deriving material fields from HU.

Use the [spectral/detector guide](guides/spectral-detector.md) and [complete spectral pipeline](guides/spectral-pipeline.md) for these operators. Further supported components include [photon counting](guides/photon-counting.md), [detector acceptance/transfer](guides/detector-physics.md) and [beam hardening](guides/beam-hardening.md).

## Stochastic transport

Transport samples source histories and complete event paths. The source normalisation, history count, detector quantity and estimator define the resulting signal. Incomplete histories raise an error.

The [transport guide](guides/transport.md) covers preparation and estimators. [Physical interactions](guides/physical-transport.md), [bound Compton scattering](guides/bound-compton.md), [sampling designs](guides/transport-designs.md) and [partition/replay execution](guides/transport-execution.md) provide the necessary scientific detail.

## Optimisation outputs

The supplied optimisation drivers save settings, arrays, solver termination reasons and histories directly. Checkpoints retain numerical optimiser state where supported. See [solver details](guides/recovery.md) and [regularisation](guides/regularization.md).

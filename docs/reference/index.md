# API and mathematical reference

The normal path is `read_volume` → `ct_to_attenuation` → `standard_views` → `project`, or a prepared `dpt.torch.Projector` for repeated differentiable calls. Specialist operators live in their corresponding submodules.

## Main interfaces

::: dpt.imaging.read_volume

::: dpt.imaging.ct_to_attenuation

::: dpt.imaging.standard_views

## Detailed API

- [Geometry, physical grids and image readers](api-geometry.md)
- [Deterministic operators](api-operators.md)
- [PyTorch operators](api-torch.md)
- [Transport interfaces](api-transport.md)
- [Recovery and line search](api-recovery.md)
- [Numerical diagnostics and physical data](api-diagnostics.md)

## Mathematical and execution details

- [Execution and ownership](../concepts/execution.md)
- [Geometry and units](../concepts/geometry.md)
- [The physical models](../concepts/physical-model.md)
- [Build a multiview acquisition](../guides/batched-geometry.md)
- [Beam hardening and water-equivalent calibration](../guides/beam-hardening.md)
- [Bound Compton scattering](../guides/bound-compton.md)
- [Detector acceptance and response](../guides/detector-physics.md)
- [Differentiating a calculation](../guides/differentiation.md)
- [Imaging volumes](../guides/imaging-volumes.md)
- [Check a discrete linear operator](../guides/operator-diagnostics.md)
- [Synthetic phantoms and analytic chords](../guides/phantoms.md)
- [Photon counting](../guides/photon-counting.md)
- [Material coefficients and reference data](../guides/physical-data.md)
- [Physical interactions and energy accounting](../guides/physical-transport.md)
- [Prepare a projection for repeated use](../guides/prepared-projection.md)
- [Primary projection](../guides/primary-projection.md)
- [Fitting parameters](../guides/recovery.md)
- [Regularise a voxel field](../guides/regularization.md)
- [Simulate a detector expectation](../guides/simulation.md)
- [Spectral signal and detectors](../guides/spectral-detector.md)
- [Compose material projection and a spectrum](../guides/spectral-pipeline.md)
- [Imaging and transport with Torch](../guides/torch.md)
- [Randomise complete transport designs](../guides/transport-designs.md)
- [Partition and replay photon histories](../guides/transport-execution.md)
- [Differentiate transport geometry](../guides/transport-geometry.md)
- [Select detector scores and transport derivatives](../guides/transport-readout.md)
- [Photon histories](../guides/transport.md)
- [Preparing a case](../applications/cases.md)
- [Application drivers](../applications/drivers.md)

# Install and make a projection

Use Python 3.12 or later. Install CUDA-enabled PyTorch for your GPU, then run from the source checkout:

```bash
python -m pip install '.[gpu,imaging]'
```

The import is `dpt`. The distribution is `differentiable-photon-transport`. To run the figure scripts and download the samples:

```bash
python -m pip install matplotlib SimpleITK
python tools/download_samples.py all --data-dir "$HOME/.cache/dpt/samples"
```

The downloader verifies immutable Slicer source checksums, converts NRRD with SimpleITK and checks every voxel and all physical corners. CTChest is 512 × 512 × 139. CTBrain is 512 × 507 × 76 with an oblique transform. Downloaded data retain their source terms and stay outside Git and the wheel.

## Project the chest

```python
import dpt

ct = dpt.read_volume("chest.nii.gz", units="HU")
mu = dpt.ct_to_attenuation(ct, energy_kev=60.0)
views = dpt.standard_views(ct.grid, ("ap", "lateral"), shape=(512, 512))
images = dpt.project(mu, views, device="cuda", precision="fp32")
transmission = images.neg().exp()
```

`images` has shape `(views, rows, columns)` and contains dimensionless optical depth. `transmission` is the surviving primary fraction. Both tensors remain on CUDA.

```bash
dpt project chest.nii.gz --units HU --views ap lateral --energy-kev 60 --output projections
python examples/project_ct.py chest.nii.gz --units HU --output chest-example
```

The CLI saves `optical_depth.npy`, `transmission.npy`, per-view PNGs and `settings.json`. PNG windowing affects display only. The example calls the same renderer and additionally saves a CT/projection comparison in PNG, SVG and PDF. Its first figure render fetches the checksum-pinned official Geist font into the user cache.

## State the attenuation model

For the supplied HU example, DPT uses `max(1 + HU/1000, 0) * 0.02059` in inverse millimetres. This model supports 60 keV water at density 1 g/cm³, using [NIST's water mass attenuation table](https://physics.nist.gov/PhysRefData/XrayMassCoef/ComTab/water.html). The HU input remains separate and unchanged.

Supply calibrated attenuation with `units="mm^-1"`, or use the [material/spectral operators](../components.md) with material and acquisition data. Unknown value units are rejected by the rendering path.

For repeated differentiation, construct a [Projector](../pytorch.md) once.

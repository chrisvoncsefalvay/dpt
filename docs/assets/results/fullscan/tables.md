## DPT: completed resident PyTorch calls

| Arithmetic | Operation | Median / p95 (ms) | Sampled process VRAM (GiB) |
|---|---|---|---|
| FP64 | Forward | 4914.62 / 4920.19 | 2.83 |
| FP64 | Volume VJP | 4294.51 / 4312.27 | 2.83 |
| FP64 | Pose VJP | 9481.86 / 9522.05 | 2.91 |
| FP64 | Forward + volume/pose VJP | 16848.30 / 16869.51 | 3.68 |
| FP32 | Forward | 512.60 / 514.35 | 1.78 |
| FP32 | Volume VJP | 545.79 / 546.97 | 1.78 |
| FP32 | Pose VJP | 548.91 / 550.70 | 1.86 |
| FP32 | Forward + volume/pose VJP | 1475.54 / 1478.70 | 2.27 |

## ASTRA and TIGRE: their native algorithms

| Implementation | Boundary | Operation | Rounds × repeats | Median / p95 (ms) | Sampled process VRAM (GiB) | Image relative L2 vs DPT FP64 | Measured adjoint defect |
|---|---|---|---|---|---|---|---|
| astra-ss1 | resident | forward | 3 × 100 | 38.10 / 38.48 | 1.87 | 0.000827 | 0.0273 |
| astra-ss1 | resident | backprojection | 3 × 100 | 9.07 / 9.53 | 2.16 | 0.000827 | 0.0273 |
| astra-ss1 | torch | forward | 3 × 100 | 38.31 / 38.88 | 1.87 | 0.000827 | 0.0273 |
| astra-ss1 | torch | backprojection | 3 × 100 | 9.10 / 9.24 | 2.16 | 0.000827 | 0.0273 |
| astra-ss1 | torch | iteration | 3 × 100 | 49.81 / 50.65 | 2.86 | 0.000827 | 0.0273 |
| astra-ss1 | host | forward | 3 × 100 | 94.99 / 98.36 | 1.10 | 0.000827 | 0.0273 |
| astra-ss1 | host | backprojection | 3 × 100 | 98.75 / 111.83 | 1.37 | 0.000827 | 0.0273 |
| tigre-siddon | torch | forward | 3 × 10 | 227.63 / 232.50 | 1.66 | 0.00315 | 8.7e-05 |
| tigre-siddon | torch | backprojection | 3 × 10 | 239.30 / 254.08 | 1.66 | 0.00315 | 8.7e-05 |
| tigre-siddon | torch | iteration | 3 × 10 | 481.82 / 507.62 | 1.81 | 0.00315 | 8.7e-05 |
| tigre-siddon | host | forward | 3 × 100 | 177.07 / 182.56 | 0.63 | 0.00315 | 8.7e-05 |
| tigre-siddon | host | backprojection | 3 × 100 | 67.83 / 71.69 | 0.63 | 0.00315 | 8.7e-05 |
| tigre-interpolated-1 | host | forward | 3 × 100 | 149.74 / 163.98 | 0.54 | 0.000614 | — |
| tigre-interpolated-0.5 | host | forward | 3 × 100 | 152.87 / 160.88 | 0.54 | 0.000494 | — |
| tigre-interpolated-0.25 | host | forward | 3 × 10 | 285.95 / 288.46 | 0.54 | 0.000482 | — |

Image and backprojection errors use the saved FP64 arrays identified in [the numerical reference manifest](competitor_reference.json). DPT's complete iteration requests both volume and pose gradients. Competitor iterations request volume gradients only. The volume-only table below gives the corresponding DPT measurement.


ASTRA uses cuda3d native slice traversal and voxel-driven backprojection, with detector and voxel supersampling both set to one. Its resident rows link CUDA arrays through DLPack. TIGRE Siddon uses piecewise-constant voxel intersections and explicit `Atb(backprojection_type='matched')`. Its interpolation rows vary the native `accuracy` setting. TIGRE Torch rows include the installed wrapper's host staging and required native transfers. All rows report completed wall time and sampled NVML process memory.


## Numerical agreement

| Arithmetic | Image identical to same-mode reference | Volume VJP relative L2 vs reference | Volume adjoint discrepancy |
|---|---|---|---|
| FP64 | True | 6.83789e-08 | 5.16702e-11 |
| FP32 | True | 4.62664e-08 | 3.49037e-10 |

FP32 against FP64 on the same stored field and physical rays: maximum optical-depth error 2.26569e-05, RMSE 9.92433e-07, relative L2 5.2597e-07. These measurements describe this resampled chest and acquisition.
This untimed comparison uses the selected runs' saved images and their recorded NPY and raw-array checksums. Its plot point uses the measured FP32 forward median.


Nonidentity-pose guards use views 0, 37 and 113, the fixed asymmetric detector objective and the saved finite-difference steps. Volume differences account for the realised FP32 input perturbation. Independent CPU rays use an explicit slab intersection and eight interpolation weights.

| Arithmetic | Volume FD absolute error | Volume error bound | Pose FD absolute error | Pose error bound | CPU ray max absolute error |
|---|---|---|---|---|---|
| FP32 | 2.23742e-08 | 5.20989e-05 | 5.54787e-07 | 4.48884e-05 | 3.6849e-06 |
| FP64 | 2.58342e-13 | 2.14396e-07 | 1.97916e-10 | 6.32217e-06 | 7.10543e-15 |

## Fresh-process startup

Each precision has three independent empty-cache processes, each followed by a new process reusing that pair of compilation caches. The first-image clock starts at benchmark child entry, after interpreter startup and initial standard-library imports. It includes scientific-package imports, CUDA initialisation, loading the prepared attenuation field, upload, module construction and the first completed public projection. Lazy workspace preparation and JIT are inside the first public call. Original CT resampling/calibration, output download and subsequent correctness checks are outside the interval. The OS filesystem cache remains in its observed state.

| Arithmetic | New-process cache state | Processes | First device image median / p95 (s) | First public forward median / p95 (s) | Warm completed wall median / p95 (ms) |
|---|---|---|---|---|---|
| FP64 | Empty compilation caches | 3 | 7.115 / 7.140 | 6.075 / 6.098 | 4881.53 / 4914.38 |
| FP64 | Reused compilation caches | 3 | 6.463 / 6.560 | 5.445 / 5.523 | 4913.73 / 4920.32 |
| FP32 | Empty compilation caches | 3 | 2.888 / 2.944 | 1.813 / 1.845 | 509.60 / 512.15 |
| FP32 | Reused compilation caches | 3 | 2.094 / 2.109 | 1.047 / 1.057 | 511.09 / 564.48 |

Warm rows pool three calls per process. The reported p95 values are sample quantiles. First-image p95 uses three processes. Raw phase timings, module events and actual process dates remain in the compact results. Measurement intervals: 2026-10-20T03:56:17.356468+00:00 to 2026-10-20T03:59:17.016647+00:00.

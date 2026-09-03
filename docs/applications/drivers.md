# Application drivers

Use the application drivers to fit a rigid pose, reconstruct an attenuation volume or rank candidate views. Each driver needs a [case manifest](cases.md), calibrated geometry and a CUDA environment. The output includes the configuration, input files and solver status.

For a complete two-image pose-fitting workflow with JSON tools for a coding agent, see [the worked agentic imaging example](agentic-imaging.md).

| Module | Calculation | Fixed information |
|---|---|---|
| `dpt.examples.registration` | Fit one shared rigid pose to calibrated radiographs | Known attenuation volume and per-view geometry |
| `dpt.examples.reconstruction` | Estimate one non-negative attenuation field | Calibrated geometry, fixed pose and primary-count model |
| `dpt.examples.acquisition_design` | Rank supplied next-view candidates | Known volume, pose, target points and current precision |

For joint spectral material reconstruction, use `dpt.material_reconstruction`. The acquisition-design driver returns a ranked selection for the supplied candidates.

## Run a supplied case

Use the Python environment containing DPT and its GPU dependencies. The common command interface is:

```bash
python -m dpt.examples.registration \
  --case /data/registration/case.json \
  --output /data/runs/registration-001 \
  --device cuda:0

python -m dpt.examples.reconstruction \
  --case /data/reconstruction/case.json \
  --output /data/runs/reconstruction-001 \
  --device cuda:0

python -m dpt.examples.acquisition_design \
  --case /data/design/case.json \
  --output /data/runs/design-001 \
  --device cuda:0
```

These commands require supplied case files. `--case` and `--output` are required, and `--device` defaults to `cuda:0`. `--help` displays the interface without launching a calculation. For a source checkout without an installed package, set `PYTHONPATH=python` from the checkout root.

The output directory stores arrays, settings and solver results. The [primary-projection guide](../guides/primary-projection.md) describes the numerical image-formation boundary used by these applications.

## Check solver completion

The JSON result reports the termination reason, stationarity flag and final diagnostics. Arrays are saved as NumPy files.

## Registration results

Registration sums per-view masked Poisson half-deviances in one fixed pose chart. It starts at zero chart coordinates, meaning the supplied anchor pose. Precision defaults to FP64, retaining primary intermediates and count predictions. The line search seeks strong Wolfe conditions and can retain the best checked Armijo trial when Wolfe conditions are unavailable. Curvature-history updates remain separately guarded.

`optimisation.json` contains the accepted object-to-world transform, chart, policy, precision, integration mode, solver result and stationarity flag. `fit-report.json` contains final per-view half-deviances, included pixel counts, evaluation accounting and optional independent geometric errors. `prediction-{view-id}.npy` contains each final prediction.

Predictions are recomputed at the accepted parameters before export: the last line-search trial may have left different values in scratch. Optional reference pose and landmarks are read only after fitting. `fit-report.json` stores their source and uncertainty alongside the geometric errors. The [geometry conventions](../concepts/geometry.md) define the chart and transform direction.

## Reconstruction results

Reconstruction uses non-negative projected updates with Armijo backtracking. The objective is summed Poisson half-deviance plus a quadratic spatial penalty. The penalty uses physical voxel spacing and volume weights, with no neighbour beyond the array boundary. Torch provides constrained updates and the regulariser, whilst Warp provides image formation and explicit VJPs on a shared CUDA stream.

`attenuation.npy` is the accepted field in grid shape and `mm^-1`. `predictions/view-NNNN.npy` contains accepted-state predictions in view order. `solver.json` reports termination, accepted steps, final objective, final projected-gradient mapping and update history. The mapping's fixed step comes from `mapping_step_mm_inverse_squared` when supplied, otherwise the initial step. The mapping measures stationarity for the declared discretised objective.

## Acquisition-design results

Design ranks candidates within `maximum_cost` by mean trace of local target-position covariance, in mm². It combines each candidate's ideal Poisson Fisher information with the supplied current precision. Exact score ties prefer lower stated cost, then candidate name. Target points describe the known task, and existing observations belong in current precision once.

`design.json` records the selected candidate, budget exclusions, diagnostics and matrix row order. `candidate_information.npy` and `candidate_local_covariance.npy` contain `(6,6)` matrices in that order and the scaled pose coordinates. `current_local_covariance.npy` records the `(6,6)` baseline. Precision defaults to FP32 projection storage. Conditioning and rank limits default to `1e12` and `1e-10` respectively.

The score is predicted local uncertainty under the fixed ideal Poisson model. Candidate feasibility and acquisition costs are supplied in the case manifest.

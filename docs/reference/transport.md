# Transport contracts

The transport interfaces estimate detector scores from original photon histories on a fixed material grid. Their contract specifies the supplied physical model, random identity and failure ownership.

For a worked analytic slab, see [photon transport](../guides/transport.md). The [simulation guide](../guides/simulation.md) provides a prepared source-to-estimate interface with resident mean and variance images. The [API index](../reference/api.md) lists the module entry points.

## Geometry and physical inputs

`MaterialGrid(origin_mm, spacing_mm, shape, materials)` describes axis-aligned, piecewise-constant cells. Storage shape is `(nz, ny, nx)` with x fastest, spacing follows x/y/z, and origin denotes the lower cell face. `shape_xyz` reverses the storage extents for geometric calculations.

`PlanarDetector(lower_xy_mm, spacing_xy_mm, shape, z_mm)` uses lower pixel faces and shape **`(width, height)`**. Flat pixel index is `row * width + column`. Support is half-open, and the detector plane must lie strictly above the entire grid. Deterministic sampled volumes and detectors use centres and different detector shape ordering. See [geometry conventions](../concepts/geometry.md).

`TransportSpec` requires a grid, detector, positive source `energy_kev` and nonempty `coefficient_provenance`. Absorption and scattering coefficients are supplied in `mm^-1`, stored material-major with energy fastest. Their sum is total extinction. An omitted `coefficient_energies_kev` declares a single elastic source-energy node. Otherwise nodes increase strictly and contain the source energy. Interpolation is linear and extrapolation is forbidden.

The default `isotropic-elastic` law preserves energy. `free-electron-compton` samples unpolarised Klein–Nishina scattering from stationary free electrons. [Bound-electron Compton](../guides/bound-compton.md) adds shell momentum profiles and Doppler broadening. Supply a `PhysicalTransport` record for bound Compton, Rayleigh form factors, fluorescence, local deposition or Woodcock tracking. The [physical transport guide](../guides/physical-transport.md) describes its tables. Polarisation and pair production are outside these models. `coefficient_provenance` records the supplied coefficient source, and the coefficient-energy table must cover the transported energies above the local-deposition cutoff.

Default scoring is `photon-count`. `energy-kev` scores incident energy times photon weight. Energy scores use keV; photon-count scores use counts. Escaped scores already include primary and scattered contributions.

## Preparation and array ownership

```text
prepare_transport(
    spec, *, material_ids, absorption, scattering, max_histories,
    device="cuda:0", stream=None, validate=True,
)
```

Preparation binds caller-owned material IDs and coefficient tables, uploads the declared energy nodes and allocates a status word. Material buffers remain immutable for the workspace lifetime. Preparation accepts `device="cpu"` or an available CUDA device. CPU execution requires `stream=None`, whilst CUDA work uses the owning stream.

| Array | Warp dtype | Flat length and meaning |
| --- | --- | --- |
| `material_ids` | `int32` | Product of grid shape, zero-based material IDs |
| `absorption`, `scattering` | `float64` | Materials × energy nodes, nonnegative `mm^-1` |
| `positions`, `directions` | `vec3d` | Batch count, positions in mm, unit directions |
| `weights` | `float64` | Batch count, nonnegative source importance weights |
| `density` | `float64` | Materials, strictly positive dimensionless scales |

Arrays must be contiguous, on the owning device and stream, and disjoint where required. Dynamic inputs remain unchanged until execution completes. Explicit upload and cross-stream ordering belong to the caller. See [execution contracts](../concepts/execution.md).

## History execution and completion

```text
trace_histories(
    positions, directions, weights, density, *, batch, workspace,
    out_pixel, out_score, out_energy, out_events, out_status,
    source_amplitude=1.0, stream=None, validate=True,
)
```

Every output has batch length. Pixel, event count and status use `int32`, whilst score and final energy use `float64`. `out_pixel == -1` denotes a miss or absorption. Every launched original history remains in the denominator, including zero contributions. Source amplitude is a separate non-negative multiplier. No area, exposure or inverse-square normalisation is inferred from source weights.

`estimator="analogue"` samples total-extinction flights and absorption/scattering events. `continuous-absorption` samples scattering flights while integrating absorption along the complete realised path, including escape. Both estimators retain low-weight histories.

Checked calls synchronise and verify status. `validate=False` requires valid current inputs and permits asynchronous execution, but `workspace.check_status()` remains mandatory before accepting an estimate. Low-level status persists across launches: a later checked success cannot erase an earlier unchecked failure. `clear_status()` is an explicit independent-run boundary after failed outputs have been discarded.

`TransportError`, a `ContractError` subtype, covers invalid physical/storage inputs and energy-support failures. `IncompleteHistoryError` means an event, crossing or angular-rejection guard exhausted. Discard the entire estimate. Arithmetic failures raise `NumericalError`. High-level evaluators manage their own run boundaries and internal workspace lifetimes.

## Identity, derivatives and uncertainty

`HistoryBatch(seed, first_history, count, source_namespace="deterministic")` declares original identities. Seed and start are unsigned 64-bit values. Count is a positive signed-32-bit value and the range must not wrap. Independent histories require independent source draws and flight counters. Declare distinct source namespaces, or use `deterministic` for a deterministic source.

Replay retains the integer random schedule with unchanged inputs. Floating-point atomic sums need not be bitwise identical across scheduling, builds or devices. Open-interval uniforms have 32 random bits, which limits extreme-tail experiments.

`derivative_histories` replaces forward outputs with `out_pixel`, signed FP64 `out_derivative` and `out_status`, and adds a `TransportParameter`. Supported kinds are `log-material-density`, `source-amplitude` and `log-source-amplitude`. Log-density derivatives combine collision/scattering likelihood terms and integrated extinction. Continuous absorption also differentiates its explicit weight. Direct amplitude derivatives retain the base score at zero amplitude. This parameter family keeps geometry, conditional angular laws and energy support fixed. Use the [geometry pullback](../guides/transport-geometry.md) for object, source and detector coordinates, and the [coefficient-node and source-logit interfaces](../guides/transport-readout.md#differentiate-coefficient-nodes-and-source-probabilities) for logarithmic coefficients or normalised source probabilities.

`prepare_estimators(workspace)` prepares reusable moment scratch. `history_mean` includes all original histories. `history_moments` additionally writes marginal variance of the mean and requires at least two IID original histories. Stratified designs need independent complete-batch replicates. Pixel correlations prevent summing marginal variances into a general loss uncertainty.

Independent squared-loss/gradient products require independent factors. Common random numbers may connect candidate and incumbent within each factor. See [recovery](../guides/recovery.md) for absolute log charts, replicate acceptance and held-out final checks.

## Prepared simulation results

`prepare_simulation(transport, ...)` owns reusable source/history buffers and moment scratch. `simulate(density, source=..., batch=..., workspace=...)` samples a library `ParallelBeam` or `PointConeSource`, checks completion and returns a typed `SimulationResult`. Its FP64 `mean` and `estimated_variance_of_mean` arrays have shape `(pixels,)`. `image_shape` records the detector layout. For `N >= 2` IID original histories, marginal variance of the mean is `sum_i (X_i - mean)**2 / (N * (N - 1))`, including zeros for misses and absorption.

The returned arrays alias the simulation workspace and are overwritten by its next call. Batch identity and seven terminal-status counts are immutable snapshots. Simulation transfers diagnostic counts/status words and retains the images on the device. Read the [simulation guide](../guides/simulation.md) for byte budgets, scoring units and uncertainty for correlated pixel statistics.

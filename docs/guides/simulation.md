# Simulate a detector expectation

`simulate` samples a library source, traces a batch of original photon histories and returns a typed `SimulationResult`. Its mean and estimated variance images remain on the prepared device. The result also records the original-history identity and a terminal-status histogram.

## Prepare a reusable simulation

This example uses continuous absorption through a 1 mm slab with coefficient 0.3 mm⁻¹. Every axial history has the same score, so the mean is `exp(-0.3)` and its estimated sampling variance is zero.

```python
import math

import warp as wp

from dpt.transport import HistoryBatch, HistoryStatus, ParallelBeam, prepare_simulation, simulate
from dpt.transport.forward import prepare_transport
from dpt.transport.model import MaterialGrid, PlanarDetector, TransportSpec

device, count = "cuda:0", 128
spec = TransportSpec(
    grid=MaterialGrid((-1.0, -1.0, 0.0), (2.0, 2.0, 1.0), (1, 1, 1), 1),
    detector=PlanarDetector((-1.0, -1.0), (2.0, 2.0), (1, 1), 2.0),
    energy_kev=80.0,
    coefficient_provenance="analytic slab, absorption=0.3/mm, scattering=0/mm",
    estimator="continuous-absorption",
)
transport = prepare_transport(
    spec,
    material_ids=wp.array([0], dtype=wp.int32, device=device),
    absorption=wp.array([0.3], dtype=wp.float64, device=device),
    scattering=wp.array([0.0], dtype=wp.float64, device=device),
    max_histories=count, device=device,
)
workspace = prepare_simulation(transport, max_histories=count)
result = simulate(
    wp.array([1.0], dtype=wp.float64, device=device),
    source=ParallelBeam((0.0, 0.0, -1.0), (0.0, 0.0)),
    batch=HistoryBatch(seed=92761, first_history=0, count=count),
    workspace=workspace,
)
assert math.isclose(float(result.mean.numpy()[0]), math.exp(-0.3), rel_tol=1e-12)
assert float(result.estimated_variance_of_mean.numpy()[0]) < 1e-28
assert result.status_histogram.count(HistoryStatus.ESCAPED) == count
assert result.status_histogram.total == count
```

The density array contains positive, dimensionless material coefficient scales in FP64, with shape `(materials,)`. `ParallelBeam` samples a rectangle. Zero widths give the deterministic axial source above. `PointConeSource` samples a supplied cone acquisition. Both library sources assign independent original-history draws and retain their sampler identity in `result.batch`.

## Read the estimate

For pixel `p`, let `X_i[p]` be the score from original history `i`. Misses and absorbed photons have score zero. With `N >= 2` IID original histories,

```text
mean[p] = sum_i X_i[p] / N
estimated_variance_of_mean[p]
        = sum_i (X_i[p] - mean[p])**2 / (N * (N - 1))
```

Both arrays are flat FP64 images of shape `(pixels,)`. `result.image_shape` supplies `(rows, columns)`. The variance image estimates marginal variance of the mean. For a pixel sum, region statistic or nonlinear loss, retain correlations by estimating uncertainty from independent complete-batch replicates. Stratified sampling likewise uses complete-batch replicates rather than this IID within-batch formula.

`result.estimator` identifies analogue or continuous absorption. `result.scoring` identifies photon-count or energy-keV scoring, so mean units are score units per launched history and variance units are their square. Source amplitude multiplies the score explicitly. Detection, exposure and source normalisation follow the declared physical model.

The seven histogram entries follow `HistoryStatus`: escaped, absorbed, event budget, crossing budget, numerical failure, energy support and angle budget. An escaped history can miss the detector. Inspect the mean for detector signal and the histogram for termination. Every launched original history is counted.

## Budget and retain storage

`prepare_simulation` allocates source/history buffers, moment scratch, two estimate images and diagnostic staging. Its additional byte budget is

```text
84 * max_histories + 36 * pixels + 56
```

Material arrays belong to the transport workspace and sit outside that additional budget. `max_bytes` defaults to 256 MiB and is checked before allocation. `tally_strategy="plain"` is the default. Select `"grouped"` explicitly for a workload whose tally performance benefits from grouping.

The workspace runs sequentially on its transport device and stream. Its next simulation overwrites both result images. Clone them with `wp.clone` to keep device results. Batch metadata and histogram counts are immutable snapshots.

Each call checks input contents and completion. Incomplete resource-limited histories, energy-support violations and arithmetic failures raise before returning a result. Seven terminal counts and completion status words cross to the host. The images stay resident. The example's `.numpy()` calls are explicit demonstration readbacks.

For supplied source arrays, single-history calculations or derivative estimators, use the lower-level [history interfaces](transport.md) and [transport contracts](../reference/transport.md).

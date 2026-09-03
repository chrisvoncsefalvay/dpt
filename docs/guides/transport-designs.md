# Randomise complete transport designs

IID Philox draws remain the default transport sampling law. `SamplingDesign` selects randomised stratification or scrambled Sobol points across the complete bounded photon path, including source, collision, angular rejection, fluorescence, next-event and roulette draws.

## Declare the coordinate budget

The default `layout="legacy-v1"` reserves

```text
dimensions = 3 + max_branches * max_events * (12 + 6 * max_angle_trials)
```

The three source dimensions hold position and energy selection. Event roles keep fixed coordinates across model configurations. The design must cover the prepared transport's event/rejection budgets and any readout splitting branches, with at most 21,201 coordinates. Exhausting a declared path budget invalidates the estimate.

Both laws use a power-of-two complete history count. Stratification applies independently keyed affine permutations of dyadic strata with per-stratum jitter. Scrambled Sobol uses Joe–Kuo D(6) directions, a binary left linear matrix scramble and an independent digital shift. The packaged direction asset retains its source licence and attribution.

This host calculation inspects a complete design without importing Warp:

```python
import numpy as np

from dpt.transport.sampling import sampling_uniforms
from dpt.transport.sampling_identity import SamplingDesign, SamplingIdentity

design = SamplingDesign("stratified", max_events=4, max_angle_trials=8)
identity = SamplingIdentity(design, seed=41, first_history=0, count=16)
draws = sampling_uniforms(identity, event=0, domain=0)
assert draws.shape == (16, 4)
assert ((draws > 0.0) & (draws < 1.0)).all()
assert sorted(np.floor(16 * draws[:, 0]).astype(int)) == list(range(16))
```

`prepare_transport_sampling(transport, batch=batch, design=design)` uploads the coordinate records once and returns a structural `HistoryBatch` token. Pass that token to source/history/readout calls. Preparing a new seed replaces the resident randomisation, so finish its current calls before preparing another design.

## Select the Compton proposal layout

`TransportExecution(compton_sampler="legacy-rejection")` and
`SamplingDesign(layout="legacy-v1")` retain the default sample coordinates.
An explicit `compton_sampler="packed-uniform-rejection"` tests two
Klein–Nishina proposal/acceptance pairs per random block and draws the
independent azimuth from the collision block. It preserves the conditional
scattering law and angular-trial budget while changing the realised draws.

A complete stratified or Sobol design for this policy uses
`layout="packed-compton-v2"`. Its coordinate budget is

```text
event_stride = 12 + 4 * ceil(max_angle_trials / 2) + 3 * max_angle_trials
dimensions = 3 + max_branches * max_events * event_stride
```

The final term reserves three coordinates per Rayleigh trial. Preparation
rejects a sampler/layout mismatch. Record the execution policy and design
layout with the seed when reproducing an experiment.

This complete host calculation declares matching records:

```python
from dpt.transport import TransportExecution
from dpt.transport.sampling_identity import SamplingDesign

execution = TransportExecution(
    max_events=4, max_angle_trials=8,
    compton_sampler="packed-uniform-rejection",
)
design = SamplingDesign(
    "stratified", max_events=execution.max_events,
    max_angle_trials=execution.max_angle_trials, layout="packed-compton-v2",
)
assert design.event_stride == 52
assert design.dimensions == 211
```

## Preserve identity through partitioning

Every point's index is `history - sampling.first_history`. `batch.shard(parts, rank)` preserves the complete design identity and index while dividing its original-history range. Reordering, chunking or moving a shard therefore retains the same source/path draws. Disjoint subsets of one scramble remain dependent, and disjoint complete ranges with the same randomisation seed also remain dependent designs.

Use `history_mean` for a complete design's mean. `history_moments` and other IID within-history variance interfaces reject these non-IID batches. Pool a partition with `pool_design_shard_means` before treating it as one complete replicate.

## Estimate uncertainty across complete designs

Independent randomisation seeds define the sampling units. For `R >= 2` complete means,

```text
mean = sum_r mean_r / R
variance_of_mean = sum_r (mean_r - mean)**2 / (R * (R - 1))
```

`design_replicate_moments` accepts finite FP64 means shaped `(replicates, ...)` and matching complete structural batches. It verifies that laws, budgets and counts agree and seeds differ. Its marginal variance has squared measurement units, while a scalar objective or combined detector statistic retains correlations by operating on complete replicate values.

Replay can use the same complete randomisation for common-random-number comparisons between two parameter points. Independent factors in nonlinear squared-loss estimators still require different randomisations. Independent factors use independent scrambles.

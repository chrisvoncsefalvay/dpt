# Select detector scores and transport derivatives

`TransportReadout` declares how each complete original photon history contributes to detector pixels and energy channels. It can select primary, scattered or total signal, apply next-event connections, split the source and use weight roulette. The returned mean includes every original history, including misses and absorption.

## Prepare a primary readout

The primary channel contains paths with zero real interactions. Woodcock null candidates preserve that classification. `mode="scatter"` selects paths with real interactions and `mode="total"` includes both, so adding a separate primary image to total signal would count the primary contribution twice.

This complete CPU example uses next-event estimation of the primary beam through a 1 mm absorption slab. Its mean is `exp(-0.3)` even though the analogue flight law contains random absorption.

```python
import math

import warp as wp

from dpt.transport import (
    HistoryBatch, MaterialGrid, PlanarDetector, TransportReadout,
    TransportSpec, prepare_transport_readout, trace_readout,
)
from dpt.transport.forward import prepare_transport

device, count = "cpu", 32
transport = prepare_transport(
    TransportSpec(
        MaterialGrid((-1.0, -1.0, 0.0), (2.0, 2.0, 1.0), (1, 1, 1), 1),
        PlanarDetector((-1.0, -1.0), (2.0, 2.0), (1, 1), 2.0),
        80.0, "Synthetic absorption 0.3/mm", estimator="analogue",
    ),
    material_ids=wp.array([0], dtype=wp.int32, device=device),
    absorption=wp.array([0.3], dtype=wp.float64, device=device),
    scattering=wp.array([0.0], dtype=wp.float64, device=device),
    max_histories=count, device=device,
)
readout = prepare_transport_readout(
    transport, spec=TransportReadout(mode="primary", next_event=True,
                                     tally_strategy="deterministic"),
)
mean = wp.empty(1, dtype=wp.float64, device=device)
variance = wp.empty_like(mean)
trace_readout(
    wp.array([(0.0, 0.0, -1.0)] * count, dtype=wp.vec3d, device=device),
    wp.array([(0.0, 0.0, 1.0)] * count, dtype=wp.vec3d, device=device),
    wp.ones(count, dtype=wp.float64, device=device),
    wp.ones(1, dtype=wp.float64, device=device),
    batch=HistoryBatch(41, 0, count), workspace=readout,
    out_mean=mean, out_variance_of_mean=variance,
)
assert math.isclose(float(mean.numpy()[0]), math.exp(-0.3), rel_tol=1e-12)
assert 0.0 <= float(variance.numpy()[0]) < 1e-28
```

`energy_edges_kev` selects half-open bins of **incident** photon energy. An empty tuple uses one channel, while a nonempty edge sequence excludes initial energies outside its support. Channels are energy-major and pixels within each channel are row-major. Mean, optional variance and optional derivative mean are caller-owned FP64 `(energy_bins * pixels,)` arrays. Photon-count scoring has count-score units per original history and energy scoring has keV-score units. The `.numpy()` calls above explicitly read the analytic example back to the host.

## Retain complete-history contributions

Next-event estimation scores uncollided transmission directly and samples a uniform detector-area connection at each real scattering or fluorescent emission. Each connection includes its conditional phase density, projected area divided by squared distance and intervening attenuation. These connections replace direct escape contributions.

`split_branches=K` gives each source descendant importance weight `weight/K` with distinct collision counters. Descendants share their original source sample and their contributions are combined before forming the mean or uncertainty. Weight roulette acts on the live continuous-absorption weight at interactions: survivors receive reciprocal survival-probability weight.

For `N` IID original histories with complete score `Y[i,c]`,

```text
mean[c] = sum_i Y[i,c] / N
variance_of_mean[c] = sum_i (Y[i,c] - mean[c])**2 / (N * (N - 1))
```

The variance requires at least two IID originals. Stratified/Sobol sampling uses [independent complete designs](transport-designs.md), and a combined pixel/energy objective uses complete replicate statistics to retain correlations. Descendants and detector connections are contributions within one original history.

`tally_strategy="deterministic"` uses a compensated reduction in increasing original-history order and repeats bits for fixed device, buffers and identity. `"plain"` uses FP64 atomics. The readout workspace owns dense original-history score scratch, checked against `max_bytes` before allocation, while outputs remain caller-owned. Calls reuse the prepared device/stream sequentially and report incomplete or invalid histories before accepting an estimate.

## Differentiate coefficient nodes and source probabilities

`CoefficientParameter("absorption", material, energy_node)` or `"scattering"` selects a dimensionless logarithmic coefficient-node coordinate. At intermediate energies its derivative includes that node's linear interpolation basis. Other coefficients, geometry and conditional laws stay fixed. A zero node stays fixed under this multiplicative coordinate.

`coefficient_derivative_histories` exports separately weighted likelihood and explicit-score contributions:

```text
d E[X] / d eta = E[X * d log(p) / d eta + d X / d eta]
```

Analogue transport differentiates the selected collision partial and complete survival depth. Continuous absorption differentiates scattering likelihood and the explicit absorption weight separately. The two FP64 per-history outputs sum to the expected-score derivative.

`SourceSpectrum` stores positive normalised probabilities at fixed keV support energies. For logits `a`, the probability score is `1[selected=k] - p[k]`. `SpectrumParameter(k)` and `spectrum_derivative_histories` replay complete histories for that logit, while `spectrum_logit_derivative` applies the score to retained energy-index records. Source amplitude carries total exposure independently of these normalised probabilities.

A coefficient or spectral-logit parameter can instead be bound at `prepare_transport_readout(..., parameter=...)`, then `trace_readout` writes its matching `out_derivative_mean`. For a squared-error objective, estimate its residual and derivative using independent complete batches before multiplying them.

## Choose rate arithmetic

`precision="reference"` retains FP64 physical rates. `"mixed-rates"` interpolates tables and forms rate sums/density products in FP32 while retaining FP64 geometry, optical depths, attenuation and detector scores. These define distinct numerical flight laws, so compare them on the scene before selecting an approximation. Coefficient-node derivatives require reference rates. Buffer storage is FP64 under both policies.

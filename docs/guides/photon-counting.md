# Photon counting

Photon-counting readout operates on pulses, so a spatially shared photon can
produce several coincident counts and photons within a temporal integration
gate can produce one pulse with their combined energy. The package provides
supplied charge-cloud outcomes and an explicit rectangular integration gate,
with separate deterministic means, parameter derivatives and event readout.

## Charge sharing

`ChargeSharingSpec` declares the finite detector shape, incident energies in
keV, energy-bin boundaries and calibrated cloud patterns. Each
`ChargeCloudPattern` supplies distinct pixel offsets and nonnegative energy
fractions whose binary64 sum is one. The energy-major probability table selects
one entire pattern for each event and each energy row also sums to one. A
`Provenance` record identifies the supplied calibration or analytic law.

For incident energy \(E_e\), pattern \(m\) produces fragment energies
\(E_e f_{mf}\) at offsets \(o_{mf}\), with
\(\sum_f f_{mf}=1\). Every positive fragment inside the detector produces a
pulse at the incident event's timestamp. Fragments outside the detector escape
and retain their energy in the escape ledger. Bins are left-closed/right-open,
and pulses below the first edge or at/above the final edge have separate energy
ledgers. Zero-energy components never count.

`sample_charge_sharing` accepts int32 incident pixel and energy-bin indices
`[N]` and float64 times `[N]`. It writes pulse destinations, times, energies and
bin indices `[N,F]`, selected patterns `[N]` and an energy ledger `[N,4]` ordered
as accepted, below threshold, overflow and edge escape. An escaped destination
is `-1`. Bin codes are `-1` below threshold, `-2` overflow and `-3` for an escaped
or zero-energy component. One selected cloud preserves correlations between
neighbouring pulse counts.

For expected incident photons \(\Phi_{ep}\), `charge_sharing_mean` evaluates

\[
\mathbb E[y_{bq}]=\sum_{e,p,m,f}\Phi_{ep}\pi_{em}
\mathbf1(q=p+o_{mf})\mathbf1(T_b\le E_e f_{mf}<T_{b+1}).
\]

Its float64 input has shape `[E,P]`, count output `[B,P]` and energy ledger
`[4,P]`, with the ledger indexed by the incident pixel. The four energy rows
sum to \(\sum_e\Phi_{ep}E_e\). `charge_sharing_vjp` applies the exact transpose
to signed count and energy-ledger seeds, with the cloud calibration fixed.

`ChargeSharingIdentity` reserves the complete Philox counter address
`((observation_id << 32) | event_index, 0xffffffff, 1)` and a 64-bit seed.
Existing Poisson proposal events are below \(2^{31}\), electronic-noise events
use zero and transport domains are zero or at least `0x80000000`, which makes
these random addresses disjoint. Event offsets preserve draws when a list is
split, and repeating an identity repeats the same experiment.

## Rectangular pileup and dead time

`RectangularPileupSpec` declares positive integer energy marks, their
probabilities, an energy quantum in keV and integer readout edges. The
stationary incident process is Poisson with rate \(r\) per second, with
independent marks. A live arrival opens an integration gate of duration
\(\Delta\) and every photon in `[start,start + Δ)` contributes its full energy
to the same pulse. A subsequent blind interval of duration \(\tau\) discards
arrivals, without extending the gate or blind interval. The gate opens on any
positive input pulse, independently of the final energy comparator.

The model is nonparalysable and its stationary cluster rate is

\[
h=\frac{r}{1+r(\Delta+\tau)},\qquad
S=E_0+\sum_{i=1}^{N}E_i,\quad N\sim\operatorname{Poisson}(r\Delta).
\]

With a zero integration gate this reduces to the conventional nonparalysable
rate \(r/(1+r\tau)\). The rate approaches \(r\) as \(r\to0\) and
\(1/(\Delta+\tau)\) as \(r\to\infty\) when the combined duration is positive.
Paralysable electronics extend their inactive interval after every arrival and
have a different count-rate law, \(r\exp(-r\tau)\). The rectangular model does
not extend its inactive interval.

For lattice mark probabilities \(p_j\), the additional-energy law satisfies

\[
c_0=e^{-\lambda},\qquad
c_n=\frac{\lambda}{n}\sum_{j\le n}j p_jc_{n-j},\qquad
g_n=\sum_{j\le n}p_jc_{n-j},\quad\lambda=r\Delta.
\]

The Warp implementation stores log probabilities, so an underflowed initial
Poisson term cannot erase measurable probability near a later energy mode.
Only the finite readout energies are resolved, and the entire remaining mass
and energy are represented as overflow. Roundoff-sized negative complements
within \(2\times10^{-12}\) are rounded to zero, while a larger negative
complement is a numerical error.

`rectangular_pileup_mean` accepts float64 per-pixel rate, integration-gate and
blind-time arrays `[P]`. It writes bin rates `[B,P]` and accounting `[6,P]`:

| Row | Quantity | Unit |
| --- | --- | --- |
| 0 | Below-threshold pulse rate | s⁻¹ |
| 1 | Overflow pulse rate | s⁻¹ |
| 2 | Below-threshold energy rate | keV/s |
| 3 | Overflow energy rate | keV/s |
| 4 | Accepted energy rate | keV/s |
| 5 | Blind-interval energy loss | keV/s |

The bin rates plus accounting rows 0 and 1 sum to \(h\), and energy rows 2–5
sum to \(r\mathbb E[E]\). Total pulse energy is
\(h(1+r\Delta)\mathbb E[E]\) and blind loss is
\(hr\tau\mathbb E[E]\), so energy beyond the top comparator is retained.
All parameters and outputs must fit binary64, including
\(r(\Delta+\tau)\) and \(r\Delta\).

`rectangular_pileup_vjp` differentiates the complete mean and accounting with
respect to rate, gate duration and blind duration. Its signed cotangents have
shapes `[B,P]` and `[6,P]`, and its parameter outputs have shape `[P]`. It uses
\(c'_n=\sum_jp_jc_{n-j}-c_n\), which remains defined at zero rate, and
recomputes the law from the supplied parameter point. Mark probabilities and
bin boundaries remain fixed, and realised event draws have no pathwise VJP.

## Event readout

`rectangular_pileup_readout` accepts float64 times and energies `[P,N]`, int32
valid-prefix counts `[P]` and float64 gate/blind durations `[P]`. Times are
finite, nonnegative and nondecreasing, and energies are positive. It writes
pulse arrays `[P,N]`, pulse counts `[P]`, uint64 bin counts `[B,P]` and an
energy ledger `[4,P]` ordered as accepted, below threshold, overflow and blind
loss. Unused pulse slots are zero and pulses rejected by comparators remain in
the pulse arrays. At zero gate duration each starting event forms its own
pulse, including coincident events. An arrival exactly at the end of the blind
interval starts a new pulse.

An event observation starts live and integrates the available prefix of its
final gate, whereas the mean operator describes a stationary process.
Continuous energies are accepted by event readout, while the mean uses the
declared energy lattice. Charge-sharing output can therefore be passed to
temporal readout after grouping by pixel and timestamp, without rounding the
fragment energies to the mean model's lattice.

The executable example `dpt.examples.photon_counting.read_shared_events` performs
this composition on supplied NumPy event arrays and explicit CPU/CUDA device
selection. It retains weak fragments until temporal integration, since several
subthreshold fragments can form an accepted pulse. Its final readout energy
plus charge-cloud edge escape equals the incident energy.

## Preparation and validation

Preparation uploads immutable calibration tables and allocates bounded
recursion scratch. Event and output arrays remain caller-owned. Workspaces
retain the selected CPU/CUDA device and CUDA stream, and operators reject
overlapping input/output buffers. Checked calls report fixed-input and
representation errors, while unchecked calls retain a sticky diagnostic for
`workspace.check_status()`. Clearing a diagnostic is explicit.

## Read a small event observation

This complete CPU example uses a synthetic one-pixel charge cloud with all
energy in the incident pixel. Two 20 keV events arrive within one integration
gate, producing a 40 keV pulse. The final ledger preserves the incident energy.

```python
import hashlib

import numpy as np

from dpt.examples.photon_counting import read_shared_events
from dpt.materials import Provenance
from dpt.photon_counting import (
    ChargeCloudPattern, ChargeSharingIdentity, ChargeSharingSpec,
    RectangularPileupSpec,
)

provenance = Provenance(
    source="Synthetic one-pixel event law defined in this calculation",
    sha256=hashlib.sha256(b"one-pixel-cloud-20keV").hexdigest(),
    rights="Apache-2.0", description="Synthetic charge and pulse law",
)
charge = ChargeSharingSpec(
    (1, 1), (20.0,), (ChargeCloudPattern(((0, 0),), (1.0,)),),
    (1.0,), (10.0, 30.0, 50.0), provenance,
)
readout = RectangularPileupSpec(
    10.0, (2,), (1.0,), (1, 3, 5), provenance,
)
result = read_shared_events(
    charge, readout,
    np.array([0, 0], dtype=np.int32), np.array([0, 0], dtype=np.int32),
    np.array([0.0, 0.0001], dtype=np.float64),
    np.array([0.001], dtype=np.float64), np.array([0.0], dtype=np.float64),
    identity=ChargeSharingIdentity(41, 1), device="cpu",
)
assert result.pulse_counts[0] == 1
assert result.pulse_energies_kev[0, 0] == 40.0
assert result.bin_counts[1, 0] == 1
assert result.readout_accounting_kev.sum() == 40.0
```

The helper accepts NumPy event arrays and returns host snapshots, so its
charge-fragment transfer and timestamp grouping are explicit observation
boundaries. Lower-level operators accept caller-owned device arrays for
resident mean/VJP and event calculations.

## Scientific references

[Taguchi et al., Medical Physics 37 (2010)](https://doi.org/10.1118/1.3429056)

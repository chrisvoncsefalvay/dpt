# Transport API

These interfaces operate on original-history identities and fixed piecewise-constant material grids. Read the [transport contracts](transport.md) before using the estimators.

## `dpt.transport.model`


::: dpt.transport.model.TransportError

::: dpt.transport.model.IncompleteHistoryError

::: dpt.transport.model.HistoryStatus

::: dpt.transport.model.MaterialGrid

::: dpt.transport.model.PlanarDetector

::: dpt.transport.model.OrientedDetector

::: dpt.transport.model.TransportPhysics

::: dpt.transport.model.TransportExecution

::: dpt.transport.model.TransportSpec

## `dpt.transport.bound_compton`

Use [bound Compton scattering](../guides/bound-compton.md) to bind shell occupancies, momentum profiles and ionisation energies to the transport model. The joint density and its interior scores describe the same conditional law used by trajectory and next-event sampling.


::: dpt.transport.bound_compton.ComptonShell

::: dpt.transport.bound_compton.BoundComptonTable

::: dpt.transport.bound_compton.xraylib_bound_compton

::: dpt.transport.bound_compton.bound_compton_joint_pdf

::: dpt.transport.bound_compton.bound_compton_joint_scores

## `dpt.transport.rng`


::: dpt.transport.rng.HistoryBatch

::: dpt.transport.rng.HistoryShard

::: dpt.transport.rng.require_independent

## `dpt.transport.source`


::: dpt.transport.source.ParallelBeam

::: dpt.transport.source.PointConeSource

::: dpt.transport.source.sample_parallel_beam

::: dpt.transport.source.sample_point_cone

## `dpt.transport.forward`


::: dpt.transport.forward.TransportWorkspace

::: dpt.transport.forward.prepare_transport

::: dpt.transport.forward.trace_histories

## `dpt.transport.derivatives`

These interfaces differentiate material density and source amplitude through
`TransportParameter`. Their `CAPABILITIES` record describes that parameter
family. Geometry coordinates use the separate preparation and pullback below.


::: dpt.transport.derivatives.DerivativeCapability

::: dpt.transport.derivatives.TransportParameter

::: dpt.transport.derivatives.TransportVJPWorkspace

::: dpt.transport.derivatives.prepare_transport_vjp

::: dpt.transport.derivatives.transport_vjp

::: dpt.transport.derivatives.derivative_columns

::: dpt.transport.derivatives.derivative_histories

## Physical transport geometry

The [geometry guide](../guides/transport-geometry.md) prepares object, source,
launch-target and detector coordinates. It combines interior, material-interface,
source-support and detector-boundary contributions per original photon before
contracting a signed detector cotangent. Translations and spacings are in mm.
Rotations are in radians about the selected world pivot.

Outputs retain the selected coordinate order. The raw second moment is computed
from complete original-photon derivative contributions. Use independent complete
replicates to estimate uncertainty for stratified or scrambled Sobol designs.


::: dpt.transport.geometry_parameters.GeometryParameter

::: dpt.transport.geometry_histories.GeometryPhysicalMedium

::: dpt.transport.geometry_sampling.GeometrySamplingDesign

::: dpt.transport.geometry_sampling.GeometrySamplingIdentity

::: dpt.transport.geometry_derivatives.TransportGeometryIdentity

::: dpt.transport.geometry_derivatives.TransportGeometryWorkspace

::: dpt.transport.geometry_derivatives.prepare_transport_geometry

::: dpt.transport.geometry_derivatives.transport_geometry_vjp

## `dpt.transport.voxel_derivatives`

The [spatial density example](../guides/transport.md#reverse-into-a-spatial-density-field) contracts a detector cotangent into every selected voxel's log-density coordinate. Active voxel multipliers are strictly positive. An inactive zero multiplier represents fixed vacuum. The material model, voxel field and active mask are immutable for the prepared lifetime.

Gradient and optional marginal variance arrays are caller-owned FP64 buffers of shape `(nz * ny * nx,)`, flattened x fastest. The gradient is the mean of complete original-history contributions, including zero scores for misses, absorption and unvisited cells. IID variance combines repeated visits within each original history before reduction. With an independent stochastic cotangent, the reported variance is conditional on that cotangent. Retain both batches' identities.

`slots_per_history` bounds distinct active visited cells per history. Preparation checks its power-of-two capacity, memory budget and signed 32-bit indexing. Calls reuse the sparse scratch on the owning transport stream. A failed resource or numerical check invalidates the complete estimate. Explicitly unchecked calls require `workspace.check_status()` before output use.


::: dpt.transport.voxel_derivatives.VoxelDensityVJPResult

::: dpt.transport.voxel_derivatives.VoxelDensityVJPWorkspace

::: dpt.transport.voxel_derivatives.prepare_voxel_density_vjp
    options:
      show_source: true

::: dpt.transport.voxel_derivatives.voxel_density_vjp
    options:
      show_source: true

## `dpt.transport.estimators`


::: dpt.transport.estimators.EstimatorWorkspace

::: dpt.transport.estimators.prepare_estimators

::: dpt.transport.estimators.FusedMeanWorkspace

::: dpt.transport.estimators.prepare_fused_mean

::: dpt.transport.estimators.trace_mean

::: dpt.transport.estimators.history_mean

::: dpt.transport.estimators.history_moments

::: dpt.transport.estimators.independent_squared_gradient

::: dpt.transport.estimators.independent_squared_loss

::: dpt.transport.estimators.ReplicateEstimate

::: dpt.transport.estimators.MeanUncertainty

::: dpt.transport.estimators.summarise_replicates

## `dpt.transport.simulation`

See [detector simulation](../guides/simulation.md) for resident mean/variance images, original-history denominators and result lifetimes.

::: dpt.transport.simulation.HistoryStatusHistogram

::: dpt.transport.simulation.SimulationResult

::: dpt.transport.simulation.SimulationWorkspace

::: dpt.transport.simulation.prepare_simulation
    options:
      show_source: true

::: dpt.transport.simulation.simulate
    options:
      show_source: true

## `dpt.transport.sharding`

Use [partitioning and replay](../guides/transport-execution.md) to retain original-history identities and combine unequal shards.

::: dpt.transport.sharding.ShardMoments

::: dpt.transport.sharding.PooledHistoryMoments

::: dpt.transport.sharding.pool_shard_moments
    options:
      show_source: true

## `dpt.transport.graph`

A fixed batch keeps the same counter addresses while material-density values change.

::: dpt.transport.graph.TransportGraph

::: dpt.transport.graph.prepare_transport_graph
    options:
      show_source: true

## `dpt.transport.inverse`


::: dpt.transport.inverse.TransportSquaredOracle

::: dpt.transport.inverse.prepare_transport_inverse

## Physical interactions

Read [physical interactions and energy accounting](../guides/physical-transport.md) for supplied atomic tables, coherent partials, single-vacancy line probabilities, local deposition and fixed-majorant traversal.


::: dpt.transport.physical.AtomicScatteringTable

::: dpt.transport.physical.FluorescenceLine

::: dpt.transport.physical.PhysicalTransport

::: dpt.transport.physical.xraylib_atomic_scattering

::: dpt.transport.physical.xraylib_fluorescence_lines

## Energy ledger

The analogue ledger retains unweighted initial/final/deposited keV per history and a source-weighted local deposition tally, with every original history in the denominator.


::: dpt.transport.energy_balance.trace_energy_balance

## Coefficient-node derivatives

A selected logarithmic absorption/scattering node differentiates its interpolation basis and returns separately weighted likelihood and explicit-score contributions.


::: dpt.transport.parameters.CoefficientParameter

::: dpt.transport.parameters.coefficient_derivative_histories

## Detector scores

The [readout guide](../guides/transport-readout.md) selects total/primary/scatter scores, incident-energy bins, next-event connections, splitting, roulette and fixed-order reduction before complete-history moments.


::: dpt.transport.readout.TransportReadout

::: dpt.transport.readout.TransportReadoutWorkspace

::: dpt.transport.readout.prepare_transport_readout

::: dpt.transport.readout.trace_readout

## Complete design identities

The [sampling-design guide](../guides/transport-designs.md) fixes all bounded source/path coordinates and retains the complete randomisation identity through partitioning.


::: dpt.transport.sampling_identity.SamplingDesign

::: dpt.transport.sampling_identity.SamplingIdentity

## Randomised design execution

Sampling uncertainty is estimated across independent complete designs or Sobol scrambles, with each partition pooled before it supplies a replicate.


::: dpt.transport.sampling.PreparedSampling

::: dpt.transport.sampling.prepare_transport_sampling

::: dpt.transport.sampling.sampling_uniforms

::: dpt.transport.sampling.pool_design_shard_means

::: dpt.transport.sampling.design_replicate_moments

## Batched inverse replicates

The bounded replica axis preserves complete independent pair order while keeping detector products resident and returning small gradient/change/proposal-metric controls.


::: dpt.transport.batched.BatchedTransportSquaredOracle

::: dpt.transport.batched.prepare_transport_batched

"""Photon transport with explicit statistical and derivative contracts.

Importing this package is CPU-safe. Warp is loaded only by workspace preparation;
no material coefficients, sampled sources, or result images are bundled.
"""

from .batched import BatchedTransportSquaredOracle, prepare_transport_batched
from .bound_compton import (
    BoundComptonTable,
    ComptonShell,
    bound_compton_joint_pdf,
    bound_compton_joint_scores,
    xraylib_bound_compton,
)
from .derivatives import (
    TransportParameter,
    TransportVJPWorkspace,
    derivative_columns,
    derivative_histories,
    prepare_transport_vjp,
    transport_vjp,
)
from .energy_balance import trace_energy_balance
from .estimators import FusedMeanWorkspace, prepare_estimators, prepare_fused_mean, trace_mean
from .geometry_derivatives import (
    TransportGeometryIdentity,
    TransportGeometryWorkspace,
    prepare_transport_geometry,
    transport_geometry_vjp,
)
from .geometry_histories import GeometryPhysicalMedium
from .geometry_parameters import GeometryParameter
from .geometry_sampling import GeometrySamplingDesign, GeometrySamplingIdentity
from .graph import TransportGraph, prepare_transport_graph
from .model import (
    HistoryStatus,
    IncompleteHistoryError,
    MaterialGrid,
    OrientedDetector,
    PlanarDetector,
    TransportError,
    TransportExecution,
    TransportPhysics,
    TransportSpec,
)
from .parameters import CoefficientParameter, coefficient_derivative_histories
from .physical import (
    AtomicScatteringTable,
    FluorescenceLine,
    PhysicalTransport,
    xraylib_atomic_scattering,
    xraylib_fluorescence_lines,
)
from .readout import (
    TransportReadout,
    TransportReadoutWorkspace,
    prepare_transport_readout,
    trace_readout,
)
from .rng import HistoryBatch, HistoryShard, SourceIdentity
from .sampling import (
    PreparedSampling,
    design_replicate_moments,
    pool_design_shard_means,
    prepare_transport_sampling,
    sampling_uniforms,
)
from .sampling_identity import SamplingDesign, SamplingIdentity
from .sharding import PooledHistoryMoments, ShardMoments, pool_shard_moments
from .simulation import (
    HistoryStatusHistogram,
    SimulationResult,
    SimulationWorkspace,
    prepare_simulation,
    simulate,
)
from .source import ParallelBeam, PointConeSource, sample_parallel_beam, sample_point_cone
from .source_spectrum import (
    SourceSpectrum,
    SpectrumParameter,
    spectrum_derivative_histories,
    spectrum_logit_derivative,
)
from .voxel_derivatives import (
    VoxelDensityVJPResult,
    VoxelDensityVJPWorkspace,
    prepare_voxel_density_vjp,
    voxel_density_vjp,
)

__all__ = [
    "AtomicScatteringTable",
    "BatchedTransportSquaredOracle",
    "BoundComptonTable",
    "CoefficientParameter",
    "ComptonShell",
    "FluorescenceLine",
    "FusedMeanWorkspace",
    "GeometryParameter",
    "GeometryPhysicalMedium",
    "GeometrySamplingDesign",
    "GeometrySamplingIdentity",
    "HistoryBatch",
    "HistoryShard",
    "HistoryStatus",
    "HistoryStatusHistogram",
    "IncompleteHistoryError",
    "MaterialGrid",
    "OrientedDetector",
    "ParallelBeam",
    "PhysicalTransport",
    "PlanarDetector",
    "PointConeSource",
    "PooledHistoryMoments",
    "PreparedSampling",
    "SamplingDesign",
    "SamplingIdentity",
    "ShardMoments",
    "SimulationResult",
    "SimulationWorkspace",
    "SourceIdentity",
    "SourceSpectrum",
    "SpectrumParameter",
    "TransportError",
    "TransportExecution",
    "TransportGeometryIdentity",
    "TransportGeometryWorkspace",
    "TransportGraph",
    "TransportParameter",
    "TransportPhysics",
    "TransportReadout",
    "TransportReadoutWorkspace",
    "TransportSpec",
    "TransportVJPWorkspace",
    "VoxelDensityVJPResult",
    "VoxelDensityVJPWorkspace",
    "bound_compton_joint_pdf",
    "bound_compton_joint_scores",
    "coefficient_derivative_histories",
    "derivative_columns",
    "derivative_histories",
    "design_replicate_moments",
    "pool_design_shard_means",
    "pool_shard_moments",
    "prepare_estimators",
    "prepare_fused_mean",
    "prepare_simulation",
    "prepare_transport_batched",
    "prepare_transport_geometry",
    "prepare_transport_graph",
    "prepare_transport_readout",
    "prepare_transport_sampling",
    "prepare_transport_vjp",
    "prepare_voxel_density_vjp",
    "sample_parallel_beam",
    "sample_point_cone",
    "sampling_uniforms",
    "simulate",
    "spectrum_derivative_histories",
    "spectrum_logit_derivative",
    "trace_energy_balance",
    "trace_mean",
    "trace_readout",
    "transport_geometry_vjp",
    "transport_vjp",
    "voxel_density_vjp",
    "xraylib_atomic_scattering",
    "xraylib_bound_compton",
    "xraylib_fluorescence_lines",
]

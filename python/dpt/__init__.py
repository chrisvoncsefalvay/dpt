"""Real-volume X-ray projection and differentiation.

Use specialist submodules for calibrated geometry, spectral response, detector
models and stochastic transport.
"""

from dpt.api import project
from dpt.contracts import ContractError, TrialDomainError
from dpt.geometry import BatchedGeometry, DetectorGeometry, RigidTransform
from dpt.imaging import ct_to_attenuation, read_volume, standard_views
from dpt.imaging_io import ImagingVolume
from dpt.volumes import GridSpec

__version__ = "1.0.0"

__all__ = [
    "BatchedGeometry",
    "ContractError",
    "DetectorGeometry",
    "GridSpec",
    "ImagingVolume",
    "RigidTransform",
    "TrialDomainError",
    "ct_to_attenuation",
    "project",
    "read_volume",
    "standard_views",
]

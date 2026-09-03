"""PyTorch modules and composable tensor imaging operations.

This namespace requires the Torch extra. CUDA/Warp work begins when an
operator is prepared or evaluated.
"""

from dpt.interop.torch import SpectralProjection, project_optical_depth, project_spectral_signal

from .geometry import compose_pose
from .modules import Projector
from .operators import TransmissionOutput, affine_calibrate, blur, objective, transmit

__all__ = [
    "Projector",
    "SpectralProjection",
    "TransmissionOutput",
    "affine_calibrate",
    "blur",
    "compose_pose",
    "objective",
    "project_optical_depth",
    "project_spectral_signal",
    "transmit",
]

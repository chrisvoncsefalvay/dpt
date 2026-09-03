"""Optional framework adapters; importing this namespace loads no framework."""

from dpt.interop import torch, torch_transport
from dpt.interop.arrays import CUDAArrayProvider, DLPackProvider, as_warp_array

__all__ = ["CUDAArrayProvider", "DLPackProvider", "as_warp_array", "torch", "torch_transport"]

"""Tiny Warp kernel used only to prove that CUDA execution is real.

This is infrastructure validation, not photon-transport code or benchmark evidence.
"""

import warp as wp  # pyright: ignore[reportMissingImports]


@wp.kernel
def probe_add_one(values: wp.array(dtype=wp.int32)) -> None:
    """Mutate a small device array so the preflight exercises a CUDA kernel."""
    index = wp.tid()
    values[index] = values[index] + 1

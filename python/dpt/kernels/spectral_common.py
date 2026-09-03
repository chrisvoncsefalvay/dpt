"""Deterministic flattened spectral indexing and binary64 compensated updates.

Callers choose shared energy factors explicitly; energy-major indexing is only
for per-pixel factors. Dimensions and index ranges are validated by the host.
"""

# Warp annotations are executable DSL expressions.
# pyright: reportInvalidTypeForm=false, reportUnknownParameterType=false
# pyright: reportUnknownMemberType=false, reportUntypedFunctionDecorator=false
# pyright: reportMissingImports=false
import warp as wp

STRICT = {"fast_math": False, "fuse_fp": True, "enable_backward": False}
wp.set_module_options(STRICT)


@wp.func
def material_major_index(material: int, sample: int, samples: int) -> int:
    """Index paths (M,P) or coefficients (M,K), with the second axis contiguous."""
    return material * samples + sample


@wp.func
def energy_pixel_index(energy: int, pixel: int, pixels: int) -> int:
    """Index a per-pixel spectral factor or bin output in (K,P) order."""
    return energy * pixels + pixel


@wp.func
def compensated_sum_update(
    total: wp.float64, compensation: wp.float64, contribution: wp.float64
) -> tuple[wp.float64, wp.float64]:
    """Retain the original four-operation update and return both scalar states."""
    corrected = contribution - compensation
    updated = total + corrected
    next_compensation = (updated - total) - corrected
    return updated, next_compensation


@wp.func
def compensated_product_sum_update(
    total: wp.float64, compensation: wp.float64, left: wp.float64, right: wp.float64
) -> tuple[wp.float64, wp.float64]:
    """Keep multiplication in the corrected term for the original fusion policy."""
    corrected = left * right - compensation
    updated = total + corrected
    next_compensation = (updated - total) - corrected
    return updated, next_compensation

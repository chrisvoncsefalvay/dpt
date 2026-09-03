"""Typed native column masks for certified physically redundant interfaces."""

# Warp annotations are checked by its runtime DSL compiler.
# pyright: reportPrivateUsage=false, reportInvalidTypeForm=false
import importlib
from typing import Any

import pytest

wp: Any = importlib.import_module("warp")
native: Any = importlib.import_module("dpt.transport.geometry_driver_kernels")
interfaces: Any = importlib.import_module("dpt.transport.geometry_interface_kernels")

wp.set_module_options({"enable_backward": False, "fast_math": False, "fuse_fp": False})


@wp.kernel
def inspect_mask(
    column: native.NativeGeometryColumn, axis: int, face: int, out: wp.array(dtype=wp.int32)
):
    out[0] = native._redundant_plane(column, axis, face)


@pytest.mark.parametrize(
    "mask,axis,face,expected",
    [
        (None, 0, 1, 0),
        ((0, 1, 0, 0, 0, 0, 0), 0, 1, 1),
        ((0, 1, 0, 0, 0, 0, 0), 2, 1, 0),
        ((0, 2, 0, 0, 0, 0, 0), 0, 1, -1),
        ((0,) * 6, 0, 1, -1),
        ((0,) * 7, 0, 3, -1),
    ],
)
def test_native_column_mask_exact_dense_plane_ownership(device, mask, axis, face, expected):
    grid = interfaces.InterfaceGrid()
    grid.origin = wp.vec3d(0.0)
    grid.spacing = wp.vec3d(1.0)
    grid.shape = wp.vec3i(2, 1, 1)
    grid.rotation = wp.mat33d(1, 0, 0, 0, 1, 0, 0, 0, 1)
    column = native.NativeGeometryColumn()
    column.grid = grid
    if mask is not None:
        column.redundant_planes = wp.array(mask, dtype=wp.int32, device=device)
    out = wp.empty(1, dtype=wp.int32, device=device)
    wp.launch(inspect_mask, dim=1, inputs=[column, axis, face, out], device=device)
    assert out.numpy()[0] == expected

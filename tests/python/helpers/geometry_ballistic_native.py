"""Shared exact scientific helpers for the native controls."""

# pyright: reportInvalidTypeForm=false, reportUnknownMemberType=false
# pyright: reportUntypedFunctionDecorator=false
import importlib
from typing import Any

wp: Any = importlib.import_module("warp")

native: Any = importlib.import_module("dpt.transport.geometry_ballistic_kernels")

interfaces: Any = importlib.import_module("dpt.transport.geometry_interface_kernels")

coarea: Any = importlib.import_module("dpt.transport.compton_geometry_boundary_kernels")


@wp.kernel
def _crossings(
    ray: coarea.NativeBoundaryRay,
    grid: interfaces.InterfaceGrid,
    indices: wp.array(dtype=wp.int32),
    output: wp.array2d(dtype=wp.float64),
    status: wp.array(dtype=wp.int32),
):
    index = wp.tid()
    crossing = native.selected_plane_crossing(ray, indices[index], grid)
    status[index] = crossing.status
    if crossing.status == 0:
        output[index, 0] = crossing.distance_mm
        output[index, 1] = crossing.inverse_normal_projection
        output[index, 2] = wp.float64(crossing.patch.axis)
        output[index, 3] = wp.float64(crossing.patch.face)
        output[index, 4] = wp.float64(crossing.patch.negative_cell)
        output[index, 5] = wp.float64(crossing.patch.positive_cell)
        for axis in range(3):
            output[index, 6 + axis] = crossing.point.point_mm[axis]
            output[index, 9 + axis] = crossing.point.point_dot[axis]
            output[index, 12 + axis] = crossing.incoming.origin_dot[axis]
            output[index, 15 + axis] = crossing.incoming.direction_dot[axis]


@wp.kernel
def _patches(
    grid: interfaces.InterfaceGrid,
    indices: wp.array(dtype=wp.int32),
    output: wp.array2d(dtype=wp.int32),
):
    index = wp.tid()
    patch = native.select_material_patch(indices[index], grid)
    plane = native.select_material_plane(indices[index], grid)
    output[index, 0] = patch.status
    if patch.status == 0:
        output[index, 1] = patch.patch.axis
        output[index, 2] = patch.patch.face
        output[index, 3] = patch.patch.tangential_cells[0]
        output[index, 4] = patch.patch.tangential_cells[1]
        output[index, 5] = patch.patch.negative_cell
        output[index, 6] = patch.patch.positive_cell
    output[index, 7] = plane.status
    output[index, 8] = native.material_plane_count(grid)
    output[index, 9] = native.material_patch_count(grid)


def _grid(shape=(2, 3, 4), spacing=(0.7, 0.5, 0.25), *, scale=1.0, rotated=False):
    grid = interfaces.InterfaceGrid()
    grid.origin = wp.vec3d(0.0)
    grid.spacing = wp.vec3d(*(scale * value for value in spacing))
    grid.shape = wp.vec3i(*shape)
    grid.rotation = wp.mat33d(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    grid.translation = wp.vec3d(0.0)
    grid.motion_translation = wp.vec3d(0.1, -0.2, 0.3)
    if rotated:
        grid.rotation = wp.mat33d(0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
        grid.translation = wp.vec3d(0.5 * scale, -0.25 * scale, 0.125 * scale)
    return grid


def _ray(origin=(0.35, 0.6, -1.0), direction=(0.0, 0.0, 1.0)):
    ray = coarea.NativeBoundaryRay()
    ray.origin_mm = wp.vec3d(*origin)
    ray.direction = wp.vec3d(*direction)
    return ray


def _run(ray, grid, indices, device):
    output = wp.zeros((len(indices), 18), dtype=wp.float64, device=device)
    status = wp.zeros(len(indices), dtype=wp.int32, device=device)
    wp.launch(
        _crossings,
        dim=len(indices),
        inputs=[ray, grid, wp.array(indices, dtype=wp.int32, device=device), output, status],
        device=device,
    )
    return output.numpy(), status.numpy()


def _decode(grid, indices, device):
    output = wp.zeros((len(indices), 10), dtype=wp.int32, device=device)
    wp.launch(
        _patches,
        dim=len(indices),
        inputs=[grid, wp.array(indices, dtype=wp.int32, device=device), output],
        device=device,
    )
    return output.numpy()

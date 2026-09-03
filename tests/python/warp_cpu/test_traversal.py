"""Non-dyadic lattice faces use the same floating construction as traversal."""

import importlib
from typing import Any

import numpy as np
import pytest

from dpt.transport.kernels import (
    Parameters,
    distances_to_faces,
    enter_grid,
    scatter_direction,
)

# Warp's array annotations are runtime DSL expressions.
# pyright: reportInvalidTypeForm=false
wp: Any = importlib.import_module("warp")


@wp.kernel
def inspect_grid(
    position: wp.array(dtype=wp.vec3d),
    direction: wp.array(dtype=wp.vec3d),
    params: Parameters,  # pyright: ignore[reportGeneralTypeIssues]
    cells: wp.array(dtype=wp.vec3i),
    distances: wp.array(dtype=wp.vec3d),
    status: wp.array(dtype=wp.int32),
):
    i = wp.tid()
    entry = enter_grid(position[i], direction[i], params)
    cells[i] = entry.cell
    distances[i] = distances_to_faces(entry.position, direction[i], entry.cell, params)
    status[i] = entry.status


@wp.kernel
def inspect_scatter(
    directions: wp.array(dtype=wp.vec3d),
    cosine: wp.float64,
    azimuth: wp.float64,
    out: wp.array(dtype=wp.vec3d),
):
    i = wp.tid()
    out[i] = scatter_direction(directions[i], cosine, azimuth)


@pytest.mark.parametrize("spacing", [0.1, 0.3, 0.11])
def test_lattice_entry_and_edge_aimed_rays_have_nonnegative_face_distances(spacing, warp_device):
    p = Parameters()
    p.origin = wp.vec3d(-0.7, 0.13, -0.31)
    p.shape = wp.vec3i(17, 13, 11)
    p.spacing = wp.vec3d(spacing, spacing, spacing)
    positions = []
    directions = []
    for i in range(17):
        for j in range(13):
            positions.append((-0.7 + i * spacing, 0.13 + j * spacing, -0.31 - spacing))
            directions.append((0.0, 0.0, 1.0))
    for i in range(1, 17):
        positions.append((-0.7 + i * spacing, 0.13 + 3 * spacing, -0.31 + 4 * spacing))
        directions.append((-1.0, 0.0, 0.0))
    # Oblique pencils aimed at reconstructed grid edges, including shared faces.
    for i in range(1, 17):
        for j in range(1, 13):
            target = np.array((-0.7 + i * spacing, 0.13 + j * spacing, -0.31))
            vector = np.array((0.17, -0.23, 1.0))
            vector /= np.linalg.norm(vector)
            positions.append(tuple(target - vector * spacing))
            directions.append(tuple(vector))
    x = wp.array(positions, dtype=wp.vec3d, device=warp_device)
    d = wp.array(directions, dtype=wp.vec3d, device=warp_device)
    cells = wp.empty(len(positions), dtype=wp.vec3i, device=warp_device)
    faces = wp.empty(len(positions), dtype=wp.vec3d, device=warp_device)
    status = wp.zeros(len(positions), dtype=wp.int32, device=warp_device)
    wp.launch(
        inspect_grid, dim=len(positions), inputs=[x, d, p, cells, faces, status], device=warp_device
    )
    assert not np.any(status.numpy())
    assert np.all(np.isfinite(faces.numpy())) and np.all(faces.numpy() >= 0)
    np.testing.assert_array_equal(
        cells.numpy()[: 17 * 13, :2], np.array([(i, j) for i in range(17) for j in range(13)])
    )
    np.testing.assert_array_equal(cells.numpy()[17 * 13 : 17 * 13 + 16, 0], np.arange(16))


def test_scatter_preserves_polar_angle_and_normalisation(warp_device):
    directions = np.array(
        [[0, 0, 1], [0, 0, -1], [1, 0, 0], [0.3, 0.4, np.sqrt(0.75)]], dtype=np.float64
    )
    x = wp.array(directions, dtype=wp.vec3d, device=warp_device)
    out = wp.empty_like(x)
    for cosine in (-1.0, -0.3, 0.0, 0.7, 1.0):
        wp.launch(
            inspect_scatter, dim=len(directions), inputs=[x, cosine, 0.37, out], device=warp_device
        )
        np.testing.assert_allclose(np.linalg.norm(out.numpy(), axis=1), 1.0, atol=1e-14)
        np.testing.assert_allclose(np.sum(out.numpy() * directions, axis=1), cosine, atol=1e-14)

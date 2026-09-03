"""Shared fixtures and independent oracles: material fisher inputs."""

import importlib
import math
from typing import Any, Literal

from dpt.geometry import DetectorGeometry, RigidTransform
from dpt.material_reconstruction import (
    MaterialReconstructionSettings,
    MaterialReconstructionView,
)
from dpt.materials import Provenance
from dpt.spectral import SpectralSpec
from dpt.validation.projection import integrate_sampled_field
from dpt.volumes import GridSpec

np: Any = importlib.import_module("numpy")


def _inputs(
    *,
    beta: float = 0.0,
    missed: bool = False,
    coefficients: bool = True,
    precision: Literal["float32", "float64"] = "float32",
) -> dict[str, Any]:
    grid = GridSpec((2, 2, 3), (1.0, 1.2, 0.8), (-1.0, -0.6, -0.4))
    provenance = Provenance(
        source="Independent small-volume Beer-Lambert/Fisher test",
        sha256="0" * 64,
        rights="Original test fixture, Apache-2.0",
        description="Numerical coefficient fixture; no anatomical or physical-material claim",
    )
    spectral = SpectralSpec(
        materials=2,
        energies=2,
        coefficients_provenance=(provenance, provenance),
        spectrum_provenance=provenance,
        response_provenance=provenance,
        output_unit="counts",
        input_description="Independent numerical count channels",
        precision=precision,
    )
    geometry = DetectorGeometry(
        (-20.0, 0.0, 0.0),
        (20.0, -0.8 if not missed else 20.0, -0.4),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.8, 0.8),
        (2, 3),
    )
    mu = np.array([[0.11, 0.03], [0.02, 0.13]], dtype=np.float32)
    if not coefficients:
        mu.fill(0)
    weights = np.array([[7000, 3000], [1000, 9000]], dtype=np.float32)
    response = np.ones((2, 2), dtype=np.float32)
    counts = np.full((2, 2, 3), 8100, dtype=np.float32)
    view = MaterialReconstructionView(geometry, RigidTransform(), counts, weights, response)
    fields = np.random.default_rng(512).uniform(0.08, 0.32, (2, *grid.shape)).astype(np.float32)
    return {
        "grid": grid,
        "views": (view,),
        "coefficients": mu,
        "spectral_spec": spectral,
        "initial_fractions": fields,
        "settings": MaterialReconstructionSettings(regularisation_mm_inverse=beta),
        "samples_per_ray": 256,
        "material_metric": (2.0, 0.3, 1.0),
    }


def _reference(inputs: dict[str, Any]) -> tuple[Any, Any, Any]:
    grid, view = inputs["grid"], inputs["views"][0]
    matrix = np.array([[2.0, 0.3], [0.3, 1.0]])
    eigenvalues, vectors = np.linalg.eigh(matrix)
    inverse_root = (vectors / np.sqrt(eigenvalues)) @ vectors.T
    mu = inputs["coefficients"].astype(np.float64)
    fields = inputs["initial_fractions"].reshape(2, -1).astype(np.float64)
    weights = view.weights.astype(np.float64) * view.response.astype(np.float64)
    projection_rows: list[list[float]] = []
    for row in range(view.geometry.shape[0]):
        for col in range(view.geometry.shape[1]):
            integrals: list[float] = []
            for voxel in range(grid.voxels):
                basis = np.zeros(grid.voxels, dtype=np.float32)
                basis[voxel] = 1
                integrals.append(
                    integrate_sampled_field(basis, grid, view.geometry, view.pose, row, col)
                )
            projection_rows.append(integrals)
    projection = np.array(projection_rows)
    blocks: list[float] = []
    fisher = np.zeros((2 * grid.voxels, 2 * grid.voxels))
    for ray in projection:
        paths = fields @ ray
        contributions = weights * np.exp(-(paths @ mu))[None, :]
        means = contributions.sum(axis=1)
        derivatives = -(contributions @ mu.T)
        block = derivatives.T @ (derivatives / means[:, None])
        blocks.append(float(np.linalg.eigvalsh(inverse_root @ block @ inverse_root)[-1]))
        for derivative, mean in zip(derivatives, means, strict=True):
            jacobian = np.kron(derivative, ray)
            fisher += np.outer(jacobian, jacobian) / mean
    raw = projection.T @ (np.asarray(blocks) * projection.sum(axis=1))
    laplacian = np.zeros((grid.voxels, grid.voxels))
    beta = inputs["settings"].regularisation_mm_inverse
    for z in range(grid.shape[0]):
        for y in range(grid.shape[1]):
            for x in range(grid.shape[2]):
                voxel = x + grid.shape[2] * (y + grid.shape[1] * z)
                for coordinate, limit, stride, spacing in zip(
                    (x, y, z),
                    grid.shape[::-1],
                    (1, grid.shape[2], grid.shape[2] * grid.shape[1]),
                    grid.spacing_mm,
                    strict=True,
                ):
                    if coordinate + 1 < limit:
                        coefficient = beta * math.prod(grid.spacing_mm) / spacing**2
                        neighbour = voxel + stride
                        laplacian[voxel, voxel] += coefficient
                        laplacian[neighbour, neighbour] += coefficient
                        laplacian[voxel, neighbour] -= coefficient
                        laplacian[neighbour, voxel] -= coefficient
    raw += 2 * np.diag(laplacian) / eigenvalues[0]
    fisher += np.kron(np.eye(2), laplacian)
    return raw, fisher, matrix

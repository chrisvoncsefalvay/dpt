"""Exact-cell references include tied planes and the constant half-cell extension."""

# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false

import importlib
from typing import Any

import pytest

from dpt.geometry import DetectorGeometry, RigidTransform
from dpt.projection import ProjectionSpec, prepare_projection, project_optical_depth
from dpt.validation.projection import integrate_sampled_field
from dpt.volumes import GridSpec

wp: Any = importlib.import_module("warp")
np: Any = importlib.import_module("numpy")
pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("shape", [(3, 3, 3), (1, 1, 1)])
@pytest.mark.parametrize(
    "start,end",
    [
        ((-4.0, 0.13, 0.27), (5.0, 1.7, 1.9)),
        ((-4.0, 1.0, 1.0), (5.0, 1.0, 1.0)),
        ((5.0, 1.0, 1.0), (-4.0, 1.0, 1.0)),
        ((-4.0, -4.0, -4.0), (5.0, 5.0, 5.0)),
        ((-0.4, -0.3, -0.2), (-0.1, -0.2, -0.1)),
        ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        ((-4.0, -3.0, 0.0), (5.0, -3.0, 0.0)),
        ((-0.5, -0.5, -4.0), (-0.5, -0.5, 5.0)),
    ],
)
def test_exact_cell_forward_matches_sorted_plane_cpu_oracle(
    shape: tuple[int, int, int], start: Any, end: Any
) -> None:
    grid = GridSpec(shape, (1.0, 1.0, 1.0), (0.0, 0.0, 0.0))
    z, y, x = np.indices(shape, dtype=np.float64)
    # A nonzero xyz coefficient exercises the cubic term along diagonal rays.
    values = (
        (0.02 + 0.013 * x + 0.007 * y + 0.003 * z + 0.009 * x * y * z).astype(np.float32).ravel()
    )
    normal_axis = int(np.argmax(abs(np.asarray(end) - np.asarray(start))))
    tangent_axes = [axis for axis in range(3) if axis != normal_axis]
    basis = np.eye(3)
    geometry = DetectorGeometry(
        start,
        end,
        tuple(basis[tangent_axes[0]]),
        tuple(basis[tangent_axes[1]]),
        (0.1, 0.1),
        (1, 1),
    )
    expected = integrate_sampled_field(values.tolist(), grid, geometry, RigidTransform(), 0, 0)
    work = prepare_projection(
        grid, geometry, ProjectionSpec(1, precision="float64", integration="cell_gauss")
    )
    field = wp.array(values, dtype=wp.float32, device="cuda:0")
    pose = wp.array(RigidTransform().packed(), dtype=wp.float64, device="cuda:0")
    depth = wp.empty(1, dtype=wp.float64, device="cuda:0")
    project_optical_depth(field, pose, workspace=work, out_L=depth)
    assert float(depth.numpy()[0]) == pytest.approx(expected, rel=2e-14, abs=1e-15)

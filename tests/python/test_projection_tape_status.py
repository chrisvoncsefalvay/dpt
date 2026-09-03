"""Portable production-kernel regressions for projection and tape ownership."""

import importlib
from typing import Any

import pytest

from dpt._runtime import cpu_test_context
from dpt.autodiff import FirstOrderPass
from dpt.contracts import NumericalError
from dpt.geometry import DetectorGeometry, RigidTransform
from dpt.projection import ProjectionSpec, prepare_projection, project_optical_depth
from dpt.volumes import GridSpec


def test_tape_entry_preserves_pending_unchecked_projection_failure() -> None:
    wp: Any = importlib.import_module("warp")
    with cpu_test_context():
        grid = GridSpec((1, 1, 1), (1.0, 1.0, 1.0))
        detector = DetectorGeometry(
            (-2.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 1.0),
            (1, 1),
        )
        workspace = prepare_projection(grid, detector, ProjectionSpec(8), device="cpu")
        field = wp.array([float("nan")], dtype=wp.float32, device="cpu")
        pose = wp.array(RigidTransform().packed(), dtype=wp.float64, device="cpu")
        output = wp.empty(1, dtype=wp.float32, device="cpu")
        project_optical_depth(field, pose, workspace=workspace, out_L=output, validate=False)
        with pytest.raises(NumericalError):
            with FirstOrderPass([workspace]):
                pytest.fail("the tape must not open after an unchecked numerical failure")
        with pytest.raises(NumericalError):
            workspace.check_status()
        workspace.clear_status()
        owner = FirstOrderPass([workspace])
        with owner:
            pass
        owner.close()

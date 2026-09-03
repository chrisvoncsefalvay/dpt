"""Geometry collision marks match the complete production impulse sampler."""

# Warp annotations are runtime DSL types.
# pyright: reportInvalidTypeForm=false, reportUnknownMemberType=false
# pyright: reportUntypedFunctionDecorator=false, reportPrivateUsage=false
import importlib
from typing import Any

import numpy as np
import pytest

from dpt.transport.forward import prepare_transport
from helpers.geometry_conditioned_cases import CASES, physical_case

wp: Any = importlib.import_module("warp")
native: Any = importlib.import_module("dpt.transport.physical_geometry_kernels")
canonical: Any = importlib.import_module("dpt.transport.bound_compton_kernels")
wp.set_module_options({"enable_backward": False, "fast_math": False, "fuse_fp": False})


@wp.kernel
def _compare_addressed_marks(
    energy: wp.float64,
    parameters: native.Parameters,
    output: wp.array2d(dtype=wp.float64),
    status: wp.array2d(dtype=wp.int32),
):
    index = wp.tid()
    physical = parameters.physics
    physical.random_branch = 123
    actual = native._bound_angle(
        energy, 0, wp.uint64(19), wp.uint64(index), wp.uint32(2), 64, physical
    )
    expected = canonical.bound_scatter_event(
        energy, 0, wp.uint64(19), wp.uint64(index), wp.uint32(2), 64, physical
    )
    expected_mark = wp.vec4d()
    if expected[3] == wp.float64(0.0):
        expected_mark = canonical.bound_energy_event(
            energy, expected[0], 0, wp.uint64(19), wp.uint64(index), wp.uint32(2), 0, physical
        )
    output[index, 0] = actual.cosine
    output[index, 1] = actual.azimuth
    output[index, 2] = actual.energy
    output[index, 3] = actual.binding
    output[index, 4] = actual.momentum
    output[index, 5] = expected[0]
    output[index, 6] = expected[1]
    output[index, 7] = expected[2]
    output[index, 8] = expected_mark[1]
    output[index, 9] = expected_mark[2]
    status[index, 0] = actual.status
    status[index, 1] = int(expected[3])
    status[index, 2] = int(expected_mark[3])
    status[index, 3] = actual.used_mark


@pytest.mark.parametrize("name,energy,entries", CASES)
def test_geometry_native_conditioned_bound_marks_match_production(device, name, energy, entries):
    spec, physical, _ = physical_case(name, energy, entries, trials=64)
    workspace = prepare_transport(
        spec,
        material_ids=wp.array([0], dtype=wp.int32, device=device),
        absorption=wp.array([0.0, 0.0], dtype=wp.float64, device=device),
        scattering=wp.array([1.0, 1.0], dtype=wp.float64, device=device),
        max_histories=3,
        device=device,
        physical=physical,
    )
    output = wp.empty((3, 10), dtype=wp.float64, device=device)
    status = wp.empty((3, 4), dtype=wp.int32, device=device)
    wp.launch(
        _compare_addressed_marks,
        3,
        inputs=[energy, workspace._parameters],
        outputs=[output, status],
        device=device,
    )
    values, completion = output.numpy(), status.numpy()
    assert np.all(completion[:, :3] == 0)
    assert np.all(completion[:, 3] == 1)
    assert np.isfinite(values).all()
    np.testing.assert_array_equal(values[:, :5], values[:, 5:])
    assert np.all((values[:, 0] >= -1.0) & (values[:, 0] <= 1.0))
    assert np.all((values[:, 2] > 0.0) & (values[:, 2] <= energy - values[:, 3]))

"""Scale-safe native direction-jet admission against independent host norms."""

# Warp kernel annotations are runtime DSL types.
# pyright: reportInvalidTypeForm=false, reportUnknownMemberType=false
# pyright: reportUntypedFunctionDecorator=false, reportPrivateUsage=false
import importlib
import math
from typing import Any

import numpy as np
import pytest

from helpers.physical_geometry_native import (
    SEED,
    _selected_histories,
    _workspace,
)
from helpers.physical_geometry_native import (
    actual_tables as actual_tables,
)

wp: Any = importlib.import_module("warp")
native: Any = importlib.import_module("dpt.transport.physical_geometry_kernels")


@wp.kernel(
    module="unique",
    enable_backward=False,
    module_options={"enable_backward": False, "max_unroll": 1},
)
def _admission(
    direction: wp.array(dtype=wp.vec3d),
    direction_dot: wp.array(dtype=wp.vec3d),
    output: wp.array(dtype=wp.int32),
):
    index = wp.tid()
    output[index] = native._ray_jet_valid(
        wp.vec3d(wp.float64(0.0)),
        direction[index],
        wp.vec3d(wp.float64(0.0)),
        direction_dot[index],
    )


@wp.kernel(
    module="unique",
    enable_backward=False,
    module_options={"enable_backward": False, "max_unroll": 1},
)
def _physical_admission(
    direction_dot: wp.array(dtype=wp.vec3d),
    energies: wp.array(dtype=wp.float64),
    absorption: wp.array(dtype=wp.float64),
    scattering: wp.array(dtype=wp.float64),
    parameters: native.Parameters,
    history: wp.uint64,
    output: wp.array2d(dtype=wp.int32),
    tangent: wp.array2d(dtype=wp.float64),
):
    index = wp.tid()
    origin = wp.vec3d(wp.float64(0.0))
    direction = wp.vec3d(wp.float64(1.0), wp.float64(0.0), wp.float64(0.0))
    collision = native.conditional_physical_event_jet(
        origin,
        direction,
        origin,
        direction_dot[index],
        wp.float64(80.0),
        0,
        wp.float64(1.0),
        energies,
        absorption,
        scattering,
        parameters,
        wp.uint64(0x123456789ABCDEF0),
        history,
        wp.uint32(1),
        2,
    )
    terminal = native.terminal_physical_kernel_jet(
        origin,
        direction,
        origin,
        direction_dot[index],
        wp.vec3d(wp.float64(0.3), wp.float64(0.2), wp.float64(3.0)),
        origin,
        wp.float64(80.0),
        0,
        0,
        -1,
        wp.float64(75.0),
        energies,
        absorption,
        scattering,
        parameters,
    )
    output[index, 0] = collision.status
    output[index, 1] = terminal.status
    for component in range(3):
        tangent[index, component] = collision.direction_dot[component]
        tangent[index, component + 3] = terminal.direction_dot[component]
    tangent[index, 6] = terminal.energy_dot


def _host_admission(direction, direction_dot):
    # math.hypot scales before squaring, and these independent controls stay
    # within its finite result range. fsum avoids a cancellation-order oracle.
    magnitude = math.hypot(*direction_dot)
    orthogonality = abs(math.fsum(a * b for a, b in zip(direction, direction_dot, strict=True)))
    assert math.isfinite(magnitude) and math.isfinite(orthogonality)
    return int(orthogonality <= 1e-12 * max(1.0, magnitude))


@pytest.mark.parametrize(
    "direction,direction_dot",
    [
        ((1.0, 0.0, 0.0), (0.0, 0.4, -0.2)),
        ((1.0, 0.0, 0.0), (1e-4, 0.4, -0.2)),
        ((1.0, 0.0, 0.0), (1e200, 0.0, 0.0)),
        ((1.0, 0.0, 0.0), (0.0, 1e200, 0.0)),
        ((1.0, 0.0, 0.0), (0.5e188, 1e200, 0.0)),
        ((1.0, 0.0, 0.0), (2e188, 1e200, 0.0)),
        ((1.0, 0.0, 0.0), (0.0, 1e308, 1e308)),
        ((math.sqrt(0.5), math.sqrt(0.5), 0.0), (1e200, -1e200, 0.0)),
    ],
)
def test_direction_jet_admission_matches_independent_scale_safe_norm(
    warp_device, direction, direction_dot
):
    output = wp.empty(1, dtype=wp.int32, device=warp_device)
    wp.launch(
        _admission,
        1,
        inputs=[
            wp.array([direction], dtype=wp.vec3d, device=warp_device),
            wp.array([direction_dot], dtype=wp.vec3d, device=warp_device),
        ],
        outputs=[output],
        device=warp_device,
    )
    np.testing.assert_equal(output.numpy(), [_host_admission(direction, direction_dot)])


@pytest.mark.parametrize(
    "direction_dot,expected_status",
    [
        ((1e200, 0.0, 0.0), 4),
        ((0.0, 1e200, 0.0), 0),
    ],
)
def test_large_jet_admission_reaches_both_production_physical_helpers(
    warp_device, actual_tables, direction_dot, expected_status
):
    assert SEED == 0x123456789ABCDEF0
    workspace = _workspace(warp_device, actual_tables)
    history = int(_selected_histories(workspace)[-1])
    output = wp.empty((1, 2), dtype=wp.int32, device=warp_device)
    tangent = wp.empty((1, 7), dtype=wp.float64, device=warp_device)
    wp.launch(
        _physical_admission,
        1,
        inputs=[
            wp.array([direction_dot], dtype=wp.vec3d, device=warp_device),
            workspace._energies,
            workspace.absorption,
            workspace.scattering,
            workspace._parameters,
            wp.uint64(history),
        ],
        outputs=[output, tangent],
        device=warp_device,
    )
    np.testing.assert_equal(output.numpy(), [[expected_status, expected_status]])
    if expected_status == 0:
        assert np.all(np.isfinite(tangent.numpy()))
        assert np.any(np.abs(tangent.numpy()) > 1e190)

"""Surviving conditioned bound terminal jets retain complete joint scores."""

# Warp annotations are runtime DSL types.
# pyright: reportInvalidTypeForm=false, reportUnknownMemberType=false
# pyright: reportUntypedFunctionDecorator=false, reportPrivateUsage=false
import importlib
import math
from decimal import Decimal, localcontext
from typing import Any

import numpy as np
import pytest

from dpt.transport.bound_compton import (
    bound_compton_angle_proposal,
    bound_compton_conditional_energy,
    bound_compton_joint_scores,
)
from dpt.transport.forward import prepare_transport
from helpers.geometry_conditioned_cases import TERMINAL_CASES, physical_case

wp: Any = importlib.import_module("warp")
native: Any = importlib.import_module("dpt.transport.physical_geometry_kernels")
canonical: Any = importlib.import_module("dpt.transport.bound_compton_kernels")
wp.set_module_options({"enable_backward": False, "fast_math": False, "fuse_fp": False})


def _decimal_cosine_score(energy, cosine, outgoing, table):
    """Differentiate the original joint law in Decimal, holding E and E' fixed.

    Its angular normaliser is constant in this derivative. The step follows the
    shell profile width in exact Decimal arithmetic, including widths below an
    ordinary binary64 cosine increment. This avoids a finite difference across
    the entire narrow peak or subtraction of underflowed raw profile factors.
    """
    with localcontext() as context:
        context.prec = 90
        e, c, out = map(Decimal.from_float, (energy, cosine, outgoing))
        mass = Decimal.from_float(510.99895069)
        ratio = out / e

        def log_joint(angle):
            factor = 1 + e / mass * (1 - angle)
            q = ((1 - ratio) ** 2 + 2 * ratio * (1 - angle)).sqrt()
            momentum = (factor * ratio - 1) / q
            logs = []
            for shell in table.shells:
                if out < e - Decimal.from_float(shell.binding_kev):
                    height = Decimal.from_float(shell.momentum_height)
                    amount = height * abs(momentum)
                    logs.append(
                        Decimal.from_float(shell.occupancy).ln()
                        + height.ln()
                        + (1 + 2 * amount).ln()
                        - 2 * amount * (1 + amount)
                    )
            largest = max(logs)
            profile = largest + sum(((value - largest).exp() for value in logs), Decimal(0)).ln()
            jacobian_numerator = factor - angle + ratio * (1 - factor * angle)
            klein_nishina = -2 * factor.ln() + (1 / factor + factor - 1 + angle**2).ln()
            return klein_nishina + profile + jacobian_numerator.ln() - e.ln() - 3 * q.ln()

        height = max(Decimal.from_float(shell.momentum_height) for shell in table.shells)
        step = min(Decimal("1e-6"), (1 - abs(c)) / 1000, 1 / (height * 10**12))
        assert step > 0
        result = (
            log_joint(c - 2 * step)
            - 8 * log_joint(c - step)
            + 8 * log_joint(c + step)
            - log_joint(c + 2 * step)
        ) / (12 * step)
        return float(result)


@wp.kernel
def _terminal_jets(
    endpoint: wp.vec3d,
    energy: wp.float64,
    outgoing: wp.float64,
    energies: wp.array(dtype=wp.float64),
    absorption: wp.array(dtype=wp.float64),
    scattering: wp.array(dtype=wp.float64),
    parameters: native.Parameters,
    output: wp.array2d(dtype=wp.float64),
    metadata: wp.array2d(dtype=wp.int32),
):
    role = wp.tid()
    position_dot = wp.vec3d()
    direction_dot = wp.vec3d()
    endpoint_dot = wp.vec3d()
    if role == 0:  # Object-frame rotation of the incoming prefix.
        direction_dot = wp.vec3d(wp.float64(0.2), wp.float64(-0.3), wp.float64(0.0))
    elif role == 1:  # Source rigid ray jet.
        position_dot = wp.vec3d(wp.float64(-0.2), wp.float64(0.1), wp.float64(0.0))
        direction_dot = wp.vec3d(wp.float64(0.1), wp.float64(0.2), wp.float64(0.0))
    else:  # Attached detector endpoint translation.
        endpoint_dot = wp.vec3d(wp.float64(0.2), wp.float64(-0.1), wp.float64(0.3))
    result = native.terminal_physical_kernel_jet(
        wp.vec3d(),
        wp.vec3d(wp.float64(0.0), wp.float64(0.0), wp.float64(1.0)),
        position_dot,
        direction_dot,
        endpoint,
        endpoint_dot,
        energy,
        0,
        0,
        -1,
        outgoing,
        energies,
        absorption,
        scattering,
        parameters,
    )
    scores = canonical.bound_joint_scores(energy, result.cosine, outgoing, 0, parameters.physics)
    output[role, 0] = result.cosine
    output[role, 1] = result.cosine_dot
    output[role, 2] = result.density
    output[role, 3] = result.density_dot
    output[role, 4] = scores[0]
    output[role, 5] = canonical.bound_joint_pdf(
        energy, result.cosine, outgoing, 0, parameters.physics
    )
    metadata[role, 0] = result.status
    metadata[role, 1] = result.present


@pytest.mark.parametrize("name,energy,entries", TERMINAL_CASES)
def test_bound_terminal_conditioning_preserves_all_geometry_cosine_jets(
    device, name, energy, entries
):
    spec, physical, table = physical_case(name, energy, entries, trials=64)
    cosine, _ = bound_compton_angle_proposal(energy, table, 0.5)
    outgoing, _, _ = bound_compton_conditional_energy(energy, cosine, table, 0.5, 0.5)
    assert outgoing > physical.cutoff_kev
    endpoint = (math.sqrt((1 - cosine) * (1 + cosine)), 0.0, cosine)
    workspace = prepare_transport(
        spec,
        material_ids=wp.array([0], dtype=wp.int32, device=device),
        absorption=wp.array([0.0, 0.0], dtype=wp.float64, device=device),
        scattering=wp.array([1.0, 1.0], dtype=wp.float64, device=device),
        max_histories=3,
        device=device,
        physical=physical,
    )
    output = wp.empty((3, 6), dtype=wp.float64, device=device)
    status = wp.empty((3, 2), dtype=wp.int32, device=device)
    wp.launch(
        _terminal_jets,
        3,
        inputs=[
            wp.vec3d(*endpoint),
            energy,
            outgoing,
            workspace._energies,
            workspace.absorption,
            workspace.scattering,
            workspace._parameters,
        ],
        outputs=[output, status],
        device=device,
    )
    values, completion = output.numpy(), status.numpy()
    assert np.all(completion[:, 0] == 0) and np.all(completion[:, 1] == 1)
    assert np.isfinite(values).all() and np.all(values[:, 2] > 0)
    np.testing.assert_array_equal(values[:, 2], values[:, 5])
    np.testing.assert_allclose(
        values[:, 3], values[:, 2] * values[:, 4] * values[:, 1], rtol=2e-14, atol=1e-300
    )
    for actual_cosine, cosine_dot, density, density_dot, score, _ in values:
        expected = bound_compton_joint_scores(energy, actual_cosine, outgoing, table)[0]
        assert score == pytest.approx(expected, rel=2e-11, abs=2e-12)
        difference = _decimal_cosine_score(energy, actual_cosine, outgoing, table)
        assert score == pytest.approx(difference, rel=2e-11, abs=2e-12)
        assert abs(cosine_dot) > 0
        assert density_dot / density == pytest.approx(expected * cosine_dot, rel=2e-11, abs=2e-12)

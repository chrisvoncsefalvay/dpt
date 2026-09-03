"""Canonical-address collision parity and physical terminal geometry controls."""

# Warp annotations are runtime DSL types.
# pyright: reportInvalidTypeForm=false, reportUnknownMemberType=false
# pyright: reportUntypedFunctionDecorator=false, reportPrivateUsage=false
import importlib
import math
from dataclasses import replace
from decimal import Decimal, localcontext
from typing import Any, cast

import numpy as np
import pytest

from dpt.geometry import Vector3
from dpt.transport.geometry_tangents import RayTangent
from dpt.transport.physical_geometry_events import (
    AddressedUniform,
    PhysicalEventDraws,
    conditional_physical_event,
    terminal_physical_kernels,
)
from dpt.transport.rng import HistoryBatch
from dpt.transport.sampling import _philox_words, prepare_transport_sampling, sampling_uniforms
from dpt.transport.sampling_identity import SamplingDesign
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

INCIDENT = RayTangent((0.2, -0.1, 0.3), (0.0, 0.0, 1.0), (0.1, -0.2, 0.3), (0.4, -0.7, 0.0))
CHANNELS = {
    "isotropic": 0,
    "free-compton": 1,
    "bound-compton": 2,
    "rayleigh": 3,
    "fluorescence": 4,
    "absorption": 5,
    "cutoff": 6,
}
STATUSES = {"continuing": 0, "absorbed": 1, "angle-budget": 6}


@wp.kernel(
    module="unique",
    enable_backward=False,
    module_options={"enable_backward": False, "max_unroll": 1},
)
def _collisions(
    histories: wp.array(dtype=wp.uint64),
    energy: wp.float64,
    density_factor: wp.float64,
    energies: wp.array(dtype=wp.float64),
    absorption: wp.array(dtype=wp.float64),
    scattering: wp.array(dtype=wp.float64),
    parameters: native.Parameters,
    seed: wp.uint64,
    event: wp.uint32,
    branch: int,
    output: wp.array2d(dtype=wp.float64),
    metadata: wp.array2d(dtype=wp.int64),
):
    index = wp.tid()
    result = native.conditional_physical_event_jet(
        wp.vec3d(wp.float64(0.2), wp.float64(-0.1), wp.float64(0.3)),
        wp.vec3d(wp.float64(0.0), wp.float64(0.0), wp.float64(1.0)),
        wp.vec3d(wp.float64(0.1), wp.float64(-0.2), wp.float64(0.3)),
        wp.vec3d(wp.float64(0.4), wp.float64(-0.7), wp.float64(0.0)),
        energy,
        0,
        density_factor,
        energies,
        absorption,
        scattering,
        parameters,
        seed,
        histories[index],
        event,
        branch,
    )
    for component in range(3):
        output[index, component] = result.direction[component]
        output[index, component + 3] = result.direction_dot[component]
        output[index, component + 6] = result.origin[component]
        output[index, component + 9] = result.origin_dot[component]
    output[index, 12] = result.energy
    output[index, 13] = result.deposited
    output[index, 14] = result.total_rate
    output[index, 15] = result.binding
    output[index, 16] = result.momentum
    metadata[index, 0] = wp.int64(result.status)
    metadata[index, 1] = wp.int64(result.channel)
    metadata[index, 2] = wp.int64(result.line_index)
    metadata[index, 3] = wp.int64(result.layout)
    metadata[index, 4] = wp.int64(result.used_roles)
    metadata[index, 5] = wp.int64(result.angular_domain)
    metadata[index, 6] = wp.int64(result.mark_domain)


@wp.kernel(
    module="unique",
    enable_backward=False,
    module_options={"enable_backward": False, "max_unroll": 1},
)
def _terminals(
    endpoint: wp.vec3d,
    endpoint_dot: wp.vec3d,
    energy: wp.float64,
    queries: wp.array(dtype=wp.vec2i),
    mark: wp.float64,
    energies: wp.array(dtype=wp.float64),
    absorption: wp.array(dtype=wp.float64),
    scattering: wp.array(dtype=wp.float64),
    parameters: native.Parameters,
    output: wp.array2d(dtype=wp.float64),
    metadata: wp.array2d(dtype=wp.int32),
):
    index = wp.tid()
    result = native.terminal_physical_kernel_jet(
        wp.vec3d(wp.float64(0.2), wp.float64(-0.1), wp.float64(0.3)),
        wp.vec3d(wp.float64(0.0), wp.float64(0.0), wp.float64(1.0)),
        wp.vec3d(wp.float64(0.1), wp.float64(-0.2), wp.float64(0.3)),
        wp.vec3d(wp.float64(0.4), wp.float64(-0.7), wp.float64(0.0)),
        endpoint,
        endpoint_dot,
        energy,
        0,
        queries[index][0],
        queries[index][1],
        mark,
        energies,
        absorption,
        scattering,
        parameters,
    )
    for component in range(3):
        output[index, component] = result.direction[component]
        output[index, component + 3] = result.direction_dot[component]
    output[index, 6] = result.cosine
    output[index, 7] = result.cosine_dot
    output[index, 8] = result.energy
    output[index, 9] = result.energy_dot
    output[index, 10] = result.density
    output[index, 11] = result.density_dot
    metadata[index, 0] = result.status
    metadata[index, 1] = result.present
    metadata[index, 2] = result.channel
    metadata[index, 3] = result.line_index
    metadata[index, 4] = result.measure


@wp.kernel(
    module="unique",
    enable_backward=False,
    module_options={"enable_backward": False, "max_unroll": 1},
)
def _normalisations(energies: wp.array(dtype=wp.float64), output: wp.array(dtype=wp.float64)):
    index = wp.tid()
    output[index] = native._kn_normalisation(energies[index])


def _launch_collisions(workspace, histories, *, energy=10.0, factor=2.5, event=1, branch=2):
    device = workspace.context.device
    output = wp.empty((len(histories), 17), dtype=wp.float64, device=device)
    metadata = wp.empty((len(histories), 7), dtype=wp.int64, device=device)
    wp.launch(
        _collisions,
        len(histories),
        inputs=[
            wp.array(histories, dtype=wp.uint64, device=device),
            energy,
            factor,
            workspace._energies,
            workspace.absorption,
            workspace.scattering,
            workspace._parameters,
            wp.uint64(SEED),
            wp.uint32(event),
            branch,
        ],
        outputs=[output, metadata],
        device=device,
    )
    return output.numpy(), metadata.numpy()


def _addressed(workspace, history, *, event=1, branch=2, identity=None):
    spec = workspace.spec
    packed = spec.compton_sampler == "packed-uniform-rejection"
    layout = (
        "bound-compton-v3"
        if spec.scattering_law == "bound-electron-compton"
        else ("packed-compton-v2" if packed else "legacy-v1")
    )
    domains = [0, 2, 3]
    if layout == "bound-compton-v3":
        domains.append(0x10000000)
    compton_blocks = (spec.max_angle_trials + 1) // 2 if packed else spec.max_angle_trials
    domains += [0x80000000 + trial for trial in range(compton_blocks)]
    domains += [0xC0000000 + trial for trial in range(spec.max_angle_trials)]
    domains = np.asarray(domains, dtype=np.uint64) + np.uint64(branch * 65536)
    values: Any
    if identity is None:
        words = _philox_words(
            SEED, np.full(len(domains), history, dtype=np.uint64), event, cast(Any, domains)
        )
        values = (words.astype(np.float64) + 0.5) / 2**32
    else:
        values = np.asarray(
            [
                sampling_uniforms(identity, histories=[history], event=event, domain=int(domain))[0]
                for domain in domains
            ]
        )
    return PhysicalEventDraws(
        event,
        branch,
        layout,
        tuple(
            AddressedUniform(event, int(domain), lane, float(value))
            for domain, lanes in zip(domains, values, strict=True)
            for lane, value in enumerate(lanes)
            if identity is None or math.isfinite(float(value))
        ),
        "Canonical counter addresses, pinned Philox or declared resident coordinate design",
    )


@pytest.mark.parametrize(
    "law,packed",
    [
        ("isotropic-elastic", False),
        ("free-electron-compton", False),
        ("free-electron-compton", True),
        ("bound-electron-compton", False),
    ],
)
def test_actual_collision_address_energy_and_rotation_parity(
    warp_device, actual_tables, law, packed
):
    workspace = _workspace(warp_device, actual_tables, law, packed=packed)
    histories = _selected_histories(workspace)
    output, metadata = _launch_collisions(workspace, histories)
    witnessed = set()
    for index, history in enumerate(histories):
        expected = conditional_physical_event(
            INCIDENT,
            10.0,
            0,
            workspace.spec,
            actual_tables[1],
            actual_tables[2],
            workspace.physical,
            _addressed(workspace, int(history)),
            density_factor=2.5,
        )
        assert metadata[index, 0] == STATUSES[expected.status]
        assert metadata[index, 1] == CHANNELS[expected.channel]
        assert metadata[index, 3] == (2 if law == "bound-electron-compton" else int(packed))
        assert metadata[index, 4] & 1
        roles = 0
        for used in expected.used_uniforms:
            base = used.domain - 2 * 65536
            if base == 0:
                roles |= 1
            elif base == 3:
                roles |= 2
            elif base == 2:
                roles |= 4
            elif base == 0x10000000:
                roles |= 32
            elif base >= 0xC0000000:
                roles |= 16
            elif base >= 0x80000000:
                roles |= 8
        assert metadata[index, 4] == roles
        assert metadata[index, 2] == (-1 if expected.line_index is None else expected.line_index)
        np.testing.assert_allclose(
            output[index, 12:15],
            [
                expected.energy_kev,
                expected.deposited_kev,
                expected.total_mm_inverse,
            ],
            rtol=3e-12,
            atol=3e-12,
        )
        if expected.outgoing is not None:
            witnessed.add(expected.channel)
            np.testing.assert_allclose(
                output[index, :12],
                [
                    *expected.outgoing.direction,
                    *expected.outgoing.direction_dot,
                    *expected.outgoing.origin_mm,
                    *expected.outgoing.origin_dot,
                ],
                rtol=3e-12,
                atol=3e-12,
            )
            np.testing.assert_allclose(output[index, 12] + output[index, 13], 10.0, rtol=1e-14)
            assert abs(np.dot(output[index, :3], output[index, 3:6])) < 2e-13
        if expected.channel == "bound-compton" and expected.outgoing is not None:
            assert expected.shell_index is not None
            assert expected.momentum is not None
            binding = actual_tables[4].shells[expected.shell_index].binding_kev
            np.testing.assert_allclose(
                output[index, 15:17], [binding, expected.momentum], rtol=3e-12
            )
            assert metadata[index, 6] == 0x10000000 + 2 * 65536
    assert {"fluorescence", "rayleigh"} <= witnessed
    assert CHANNELS[next(iter(witnessed - {"fluorescence", "rayleigh"}))] in (0, 1, 2)


def _launch_terminals(
    workspace, *, endpoint=(1.4, 0.7, 3.0), velocity=(0.3, 0.1, -0.2), energy=80.0, mark=75.0
):
    device = workspace.context.device
    queries = [(0, -1), (1, -1), (2, 0), (2, 1)]
    output = wp.empty((4, 12), dtype=wp.float64, device=device)
    metadata = wp.empty((4, 5), dtype=wp.int32, device=device)
    wp.launch(
        _terminals,
        4,
        inputs=[
            wp.vec3d(*endpoint),
            wp.vec3d(*velocity),
            energy,
            wp.array(queries, dtype=wp.vec2i, device=device),
            mark,
            workspace._energies,
            workspace.absorption,
            workspace.scattering,
            workspace._parameters,
        ],
        outputs=[output, metadata],
        device=device,
    )
    return output.numpy(), metadata.numpy()


@pytest.mark.parametrize(
    "law", ["isotropic-elastic", "free-electron-compton", "bound-electron-compton"]
)
def test_actual_terminal_mixed_measure_and_jet_parity(warp_device, actual_tables, law):
    workspace = _workspace(warp_device, actual_tables, law)
    output, metadata = _launch_terminals(workspace)
    expected = terminal_physical_kernels(
        INCIDENT,
        (1.4, 0.7, 3),
        80.0,
        0,
        workspace.spec,
        actual_tables[1],
        actual_tables[2],
        workspace.physical,
        endpoint_dot=(0.3, 0.1, -0.2),
        bound_outgoing_kev=75.0 if law == "bound-electron-compton" else None,
    )
    assert np.all(metadata[:, :2] == [0, 1])
    for index in range(4):
        channel = metadata[index, 2]
        line = metadata[index, 3]
        kernel = next(
            item
            for item in expected
            if CHANNELS[item.channel] == channel and (channel != 4 or item.line_index == line)
        )
        np.testing.assert_allclose(
            output[index],
            [
                *kernel.outgoing.direction,
                *kernel.outgoing.direction_dot,
                kernel.cosine,
                kernel.cosine_dot,
                kernel.energy_kev,
                kernel.energy_dot,
                kernel.density,
                kernel.density_dot,
            ],
            rtol=2e-11,
            atol=2e-13,
        )
        assert metadata[index, 4] == int(kernel.measure == "solid-angle-energy")
    probabilities = output[2:, 10] * 4 * math.pi
    a, s = actual_tables[1][2], actual_tables[2][2]
    assert workspace.physical is not None
    np.testing.assert_allclose(
        probabilities,
        [a / (a + s) * line.probabilities[2] for line in workspace.physical.fluorescence[0]],
        rtol=3e-14,
    )


def test_actual_bound_terminal_cosine_derivative_finite_difference(warp_device, actual_tables):
    workspace = _workspace(warp_device, actual_tables, "bound-electron-compton")
    output, _ = _launch_terminals(workspace)
    # Hold the incident direction fixed and move only the endpoint to check a
    # separate derivative direction without differentiating the host oracle.
    epsilon = 1e-5
    baseline = replace(INCIDENT, origin_dot=(0, 0, 0), direction_dot=(0, 0, 0))
    velocity = (0.3, 0.1, -0.2)
    reference = terminal_physical_kernels(
        baseline,
        (1.4, 0.7, 3),
        80,
        0,
        workspace.spec,
        actual_tables[1],
        actual_tables[2],
        workspace.physical,
        endpoint_dot=velocity,
        bound_outgoing_kev=75,
    )
    bound = next(item for item in reference if item.channel == "bound-compton")
    values = []
    for step in (-epsilon, epsilon):
        endpoint = cast(
            Vector3, tuple(a + step * b for a, b in zip((1.4, 0.7, 3), velocity, strict=True))
        )
        values.append(
            next(
                item.density
                for item in terminal_physical_kernels(
                    baseline,
                    endpoint,
                    80,
                    0,
                    workspace.spec,
                    actual_tables[1],
                    actual_tables[2],
                    workspace.physical,
                    bound_outgoing_kev=75,
                )
                if item.channel == "bound-compton"
            )
        )
    np.testing.assert_allclose(
        bound.density_dot, (values[1] - values[0]) / (2 * epsilon), rtol=2e-7
    )
    assert np.isfinite(output[0, 11])


def test_native_cutoff_energy_and_address_ledger(warp_device, actual_tables):
    workspace = _workspace(warp_device, actual_tables)
    output, metadata = _launch_collisions(workspace, [2**32 + 1], energy=0.05)
    assert tuple(metadata[0, :2]) == (1, 6)
    assert metadata[0, 4] == 0
    np.testing.assert_equal(output[0, 12:14], [0, 0.05])
    workspace._parameters.max_angle_trials = 65537
    _, metadata = _launch_collisions(workspace, [2**32 + 1])
    assert metadata[0, 0] == 4


@pytest.mark.parametrize("factor", [1e308, 1e-320])
def test_native_scaled_rate_guard(warp_device, actual_tables, factor):
    workspace = _workspace(warp_device, actual_tables)
    # These are representability guards on the exact admitted physical rates.
    # The subnormal factor alone is valid if its product remains positive.
    _, metadata = _launch_collisions(workspace, [2**32 + 1], factor=factor)
    total = (actual_tables[1][1] + actual_tables[2][1]) * factor
    if not math.isfinite(total) or total <= 0:
        assert metadata[0, 0] == 4
    else:
        assert metadata[0, 0] in (0, 1, 6)


@pytest.mark.parametrize("law", ["free-electron-compton", "bound-electron-compton"])
def test_native_real_rejection_exhaustion_is_fatal(warp_device, actual_tables, law):
    workspace = _workspace(warp_device, actual_tables, law, trials=1)
    histories = _selected_histories(workspace)
    output, metadata = _launch_collisions(workspace, histories)
    failed = metadata[:, 0] == 6
    assert np.any(failed)
    assert np.all(metadata[failed, 1] != 5)
    assert np.all(output[failed, 13] == 0)
    if law == "bound-electron-compton":
        bound_failed = failed & (metadata[:, 1] == 2)
        assert np.any(bound_failed)
        assert np.all(metadata[bound_failed, 4] == 11)
        assert np.all(metadata[bound_failed, 6] == 0)


def test_native_kn_total_matches_independent_decimal_reference(warp_device):
    energies = (1e-9, 0.001, 0.01, 0.1, 1.0, 25.0, 80.0, 5000.0, 5e6, 5e8, 1e155)
    output = wp.empty(len(energies), dtype=wp.float64, device=warp_device)
    wp.launch(
        _normalisations,
        len(energies),
        inputs=[
            wp.array(energies, dtype=wp.float64, device=warp_device),
        ],
        outputs=[output],
        device=warp_device,
    )
    expected = []
    with localcontext() as context:
        context.prec = 100
        one, two = Decimal(1), Decimal(2)
        for energy in energies:
            alpha = Decimal(str(energy)) / Decimal("510.99895069")
            logarithm = (one + two * alpha).ln()
            total = two * (
                (one + alpha)
                / alpha**2
                * (two * (one + alpha) / (one + two * alpha) - logarithm / alpha)
                + logarithm / (two * alpha)
                - (one + 3 * alpha) / (one + two * alpha) ** 2
            )
            expected.append(float(two * Decimal(str(math.pi)) * total))
    np.testing.assert_allclose(output.numpy(), expected, rtol=3e-13, atol=0)


def test_native_scaled_positive_rate_underflow_is_fatal(warp_device, actual_tables):
    workspace = _workspace(warp_device, actual_tables)
    _, metadata = _launch_collisions(
        workspace,
        [2**32 + 1],
        energy=200.0,
        factor=5e-324,
    )
    assert (actual_tables[1][-1] + actual_tables[2][-1]) * 5e-324 == 0
    assert metadata[0, 0] == 4


@pytest.mark.parametrize(
    "law,packed,design_law,layout",
    [
        ("free-electron-compton", False, "stratified", "legacy-v1"),
        ("free-electron-compton", True, "scrambled-sobol", "packed-compton-v2"),
        ("bound-electron-compton", False, "stratified", "bound-compton-v3"),
    ],
)
def test_resident_addresses_match_host_declared_design(
    warp_device, actual_tables, law, packed, design_law, layout
):
    workspace = _workspace(warp_device, actual_tables, law, packed=packed, trials=8)
    batch = prepare_transport_sampling(
        workspace,
        batch=HistoryBatch(SEED, 0, 16),
        design=SamplingDesign(design_law, 4, 8, max_branches=3, layout=layout),
    )
    output, metadata = _launch_collisions(workspace, np.arange(16), energy=80, event=1, branch=2)
    for index in range(16):
        expected = conditional_physical_event(
            INCIDENT,
            80,
            0,
            workspace.spec,
            actual_tables[1],
            actual_tables[2],
            workspace.physical,
            _addressed(workspace, index, identity=batch.sampling),
            density_factor=2.5,
        )
        assert metadata[index, 0] == STATUSES[expected.status]
        assert metadata[index, 1] == CHANNELS[expected.channel]
        np.testing.assert_allclose(
            output[index, 12:15],
            [
                expected.energy_kev,
                expected.deposited_kev,
                expected.total_mm_inverse,
            ],
            rtol=3e-12,
            atol=3e-12,
        )

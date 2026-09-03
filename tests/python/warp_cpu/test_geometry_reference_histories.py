"""Complete native physical futures against the independent host partition law."""

# Warp decorators/annotations are runtime DSL records.
# pyright: reportInvalidTypeForm=false, reportUnknownMemberType=false
# pyright: reportUntypedFunctionDecorator=false, reportPrivateUsage=false
import importlib
import math
from dataclasses import replace
from typing import Any, Literal, cast

import numpy as np
import pytest

from dpt.transport.forward import prepare_transport
from dpt.transport.geometry_flights import optical_flight_tangent
from dpt.transport.geometry_histories import (
    GeometryPhysicalMedium,
    complete_geometry_history,
    conditional_geometry_collision,
)
from dpt.transport.geometry_tangents import RayTangent
from dpt.transport.model import MaterialGrid, PlanarDetector, TransportError, TransportSpec
from dpt.transport.rng import HistoryBatch
from dpt.transport.sampling import prepare_transport_sampling
from dpt.transport.sampling_identity import SamplingDesign
from helpers.geometry_reference_native import (
    _Addressed,
    _launch,
)
from helpers.physical_geometry_native import SEED, _selected_histories
from helpers.physical_geometry_native import actual_tables as actual_tables

wp: Any = importlib.import_module("warp")
native: Any = importlib.import_module("dpt.transport.geometry_reference_kernels")
canonical: Any = importlib.import_module("dpt.transport.kernels")


@wp.kernel
def _canonical_cutoff(
    origin: wp.vec3d,
    direction: wp.vec3d,
    materials: wp.array(dtype=wp.int32),
    voxel: wp.array(dtype=wp.float64),
    energies: wp.array(dtype=wp.float64),
    absorption: wp.array(dtype=wp.float64),
    scattering: wp.array(dtype=wp.float64),
    density: wp.array(dtype=wp.float64),
    parameters: native.Parameters,
    seed_image: wp.array(dtype=wp.float64),
    output: wp.array(dtype=wp.float64),
):
    result = canonical.walk(
        origin,
        direction,
        wp.float64(1.0),
        wp.float64(1.0),
        wp.uint64(SEED),
        wp.uint64(4294967297),
        -1,
        wp.vec4i(-1),
        0,
        materials,
        voxel,
        energies,
        absorption,
        scattering,
        density,
        parameters,
    )
    output[0] = wp.float64(0.0)
    if result.pixel >= 0:
        output[0] = result.score * seed_image[result.pixel]


def _prepared(
    device,
    tables,
    *,
    law: Literal["isotropic-elastic", "free-electron-compton", "bound-electron-compton"] = (
        "free-electron-compton"
    ),
    packed=False,
    scoring: Literal["photon-count", "energy-kev"] = "photon-count",
    shape=(1, 1, 1),
    spacing=(0.01, 0.01, 0.01),
    voxel=None,
    events=4,
    trials=256,
    crossings=1024,
    cutoff=None,
):
    energies, absorption, scattering, physical, shells = tables
    if law == "bound-electron-compton":
        physical = replace(physical, bound_compton=(shells,))
    if cutoff is not None:
        physical = replace(physical, cutoff_kev=cutoff)
    spec = TransportSpec(
        MaterialGrid((0, 0, 0), spacing, shape, 1),
        PlanarDetector((-10, -10), (5, 5), (4, 4), max(3.0, shape[0] * spacing[2] + 1.0)),
        80.0,
        "Pinned xraylib4.3.0 Al density2.6989g/cm3, cm^-1 to mm^-1",
        energies,
        scattering_law=law,
        scoring=scoring,
        max_events=events,
        max_angle_trials=trials,
        max_crossings=crossings,
        compton_sampler="packed-uniform-rejection" if packed else "legacy-rejection",
    )
    cells = math.prod(shape)
    voxels = (1.0,) * cells if voxel is None else tuple(voxel)
    workspace = prepare_transport(
        spec,
        material_ids=wp.array([0] * cells, dtype=wp.int32, device=device),
        absorption=wp.array(absorption, dtype=wp.float64, device=device),
        scattering=wp.array(scattering, dtype=wp.float64, device=device),
        voxel_density=wp.array(voxels, dtype=wp.float64, device=device),
        max_histories=64,
        device=device,
        physical=physical,
    )
    density = wp.array([1.0], dtype=wp.float64, device=device)
    medium = GeometryPhysicalMedium((0,) * cells, absorption, scattering, (1.0,), voxels, physical)
    return workspace, medium, density


@pytest.mark.parametrize(
    "law,packed,scoring",
    [
        ("isotropic-elastic", False, "photon-count"),
        ("free-electron-compton", False, "energy-kev"),
        ("free-electron-compton", True, "photon-count"),
        ("bound-electron-compton", False, "energy-kev"),
    ],
)
def test_complete_actual_physics_futures_host_parity(
    warp_device, actual_tables, law, packed, scoring
):
    workspace, medium, density = _prepared(
        warp_device, actual_tables, law=law, packed=packed, scoring=scoring
    )
    histories = _selected_histories(workspace)
    output, metadata, ray, seed = _launch(workspace, density, histories, forced=0)
    channels = set()
    terminals = {"escaped": 0, "absorbed": 1, "local-deposition": 2}
    for index, history in enumerate(histories):
        design = _Addressed(workspace.spec, int(history), 2, 1)
        expected = complete_geometry_history(
            ray,
            10.0,
            workspace.spec,
            medium,
            design,
            lineage=(int(history),),
            cotangent=seed,
            forced_collision_cell=0,
        )
        first = conditional_geometry_collision(
            expected.prefixes[0], workspace.spec, medium, design, (int(history),)
        )
        channels.add(first.channel)
        np.testing.assert_allclose(
            output[index], (expected.response, expected.energy_kev), rtol=2e-12, atol=2e-12
        )
        assert tuple(metadata[index]) == (
            0,
            len(expected.prefixes),
            terminals[expected.terminal],
            4,
        )
    noncoherent = {
        "isotropic-elastic": "isotropic",
        "free-electron-compton": "free-compton",
        "bound-electron-compton": "bound-compton",
    }[law]
    assert channels == {"absorption", "fluorescence", "rayleigh", noncoherent}
    assert np.any(output[:, 0] != 0), (
        "actual continuing channels must reach signed detector response"
    )


@pytest.mark.parametrize("energy", [10.0, 80.0, 200.0])
def test_prefix_full_partition_actual_extinction(warp_device, actual_tables, energy):
    workspace, medium, density = _prepared(
        warp_device, actual_tables, shape=(3, 1, 1), spacing=(1, 1, 1), voxel=(0.0, 0.5, 2.0)
    )
    ray = RayTangent((0.5, 0.5, -0.5), (0.0, 0.0, 1.0))
    histories = [2**32 + i for i in range(1, 17)]
    output, metadata, _, _ = _launch(
        workspace, density, histories, ray=ray, energy=energy, prefix=True
    )
    for index, history in enumerate(histories):
        design = _Addressed(workspace.spec, history, 2, 1)
        expected = optical_flight_tangent(
            ray,
            workspace.spec,
            extinction_mm_inverse=medium.extinction(energy, workspace.spec),
            optical_depth=-math.log(design.uniform((history,), 0, 0, 0)),
        )
        np.testing.assert_allclose(
            output[index, :3], expected.ray.origin_mm, rtol=1e-13, atol=1e-13
        )
        np.testing.assert_allclose(
            output[index, 3:],
            (expected.distance_mm or 0.0, expected.total_optical_depth),
            rtol=1e-13,
            atol=1e-13,
        )
        assert tuple(metadata[index]) == (
            0,
            -1 if expected.cell is None else expected.cell,
            int(expected.terminal == "escape"),
            3,
            1,
        )


@pytest.mark.parametrize("face,sign", [(1, -1), (1, 1), (0, -1), (0, 1)])
def test_exact_material_side_signed_response(warp_device, actual_tables, face, sign):
    workspace, medium, density = _prepared(
        warp_device, actual_tables, shape=(1, 1, 2), spacing=(1, 1, 1), voxel=(0.0, 1.0)
    )
    side = (0, face, cast(Literal[-1, 1], sign))
    ray = RayTangent((float(face), 0.5, -0.5), (0.0, 0.0, 1.0))
    histories = [2**32 + i for i in range(1, 9)]
    output, metadata, _, seed = _launch(workspace, density, histories, side=side, ray=ray)
    for index, history in enumerate(histories):
        expected = complete_geometry_history(
            ray,
            10.0,
            workspace.spec,
            medium,
            _Addressed(workspace.spec, history, 2, 1),
            lineage=(history,),
            cotangent=seed,
            initial_material_side=side,
        )
        np.testing.assert_allclose(
            output[index], (expected.response, expected.energy_kev), rtol=2e-12, atol=2e-12
        )
        assert metadata[index, 0] == 0
        assert metadata[index, 1] == len(expected.prefixes)
    if face == 1 and sign < 0:
        assert np.all(output[:, 0] != 0) and np.all(metadata[:, 1] == 0)


@pytest.mark.parametrize(
    "law,packed,design_law",
    [
        ("free-electron-compton", True, "scrambled-sobol"),
        ("bound-electron-compton", False, "stratified"),
    ],
)
def test_full_futures_resident_coordinates_preserve_lineage(
    warp_device, actual_tables, law, packed, design_law
):
    workspace, medium, density = _prepared(
        warp_device, actual_tables, law=law, packed=packed, trials=8
    )
    layout = "bound-compton-v3" if law == "bound-electron-compton" else "packed-compton-v2"
    batch = prepare_transport_sampling(
        workspace,
        batch=HistoryBatch(SEED, 2**32 + 1, 16),
        design=SamplingDesign(design_law, 16, 8, 2, layout=layout),
    )
    histories = [batch.first_history + i for i in range(batch.count)]
    output, metadata, ray, seed = _launch(
        workspace, density, histories, energy=80.0, forced=0, branch=1, offset=8
    )
    passed = 0
    for index, history in enumerate(histories):
        design = _Addressed(workspace.spec, history, 1, 8, batch.sampling)
        try:
            expected = complete_geometry_history(
                ray,
                80.0,
                workspace.spec,
                medium,
                design,
                lineage=(history,),
                cotangent=seed,
                forced_collision_cell=0,
            )
        except TransportError:
            assert metadata[index, 0] == 4
        else:
            passed += 1
            np.testing.assert_allclose(
                output[index], (expected.response, expected.energy_kev), rtol=3e-12, atol=3e-12
            )
            assert metadata[index, 0] == 0 and metadata[index, 1] == len(expected.prefixes)
        assert metadata[index, 3] == 4, (
            "coordinate widening must leave physical Parameters unchanged"
        )
    assert passed >= 12
    _, invalid, _, _ = _launch(
        workspace, density, [batch.first_history - 1], forced=0, branch=1, offset=8
    )
    assert invalid[0, 0] == 4, "a foreign original history cannot use an IID fallback"


def test_whole_partition_guard_rejects_after_early_collision(warp_device, actual_tables):
    workspace, medium, density = _prepared(
        warp_device, actual_tables, shape=(2, 1, 1), spacing=(1, 1, 1), crossings=1
    )
    ray = RayTangent((0.5, 0.5, 0.01), (0.0, 0.0, 1.0))
    history = 2**32 + 1
    design = _Addressed(workspace.spec, history, 2, 1)
    with pytest.raises(TransportError, match="crossing guard"):
        optical_flight_tangent(
            ray,
            workspace.spec,
            extinction_mm_inverse=medium.extinction(10.0, workspace.spec),
            optical_depth=-math.log(design.uniform((history,), 0, 0, 0)),
        )
    _, prefix, _, _ = _launch(workspace, density, [history], ray=ray, prefix=True)
    _, response, _, _ = _launch(workspace, density, [history], ray=ray)
    assert prefix[0, 0] == response[0, 0] == 4


def test_invalid_side_cell_and_address_fail_before_indexing(warp_device, actual_tables):
    workspace, _, density = _prepared(warp_device, actual_tables)
    queries = [
        [-2, -1, 0, 0, 2, 1, 16],
        [1, -1, 0, 0, 2, 1, 16],
        [-1, 3, 0, 1, 2, 1, 16],
        [-1, 0, 2, 1, 2, 1, 16],
        [-1, 0, 0, 0, 2, 1, 16],
        [-1, 2, 0, 1, 2, 1, 16],
        [0, -1, 0, 0, 2, 16, 16],
        [0, -1, 0, 0, 4096, 1, 16],
        [0, -1, 0, 0, 2, 1, 3],
    ]
    _, metadata, _, _ = _launch(workspace, density, [2**32 + 1] * len(queries), queries=queries)
    assert np.all(metadata[:, 0] == 4)


def test_zero_density_forced_collision_is_fatal_but_vacuum_flight_completes(
    warp_device, actual_tables
):
    workspace, _, density = _prepared(warp_device, actual_tables, voxel=(0.0,))
    histories = [2**32 + 1]
    flight, metadata, _, _ = _launch(workspace, density, histories)
    _, forced, _, _ = _launch(workspace, density, histories, forced=0)
    assert metadata[0, 0] == 0 and metadata[0, 1] == 0 and flight[0, 0] != 0
    assert forced[0, 0] == 4


def test_physical_event_budget_stays_distinct_from_counter_limit(warp_device, actual_tables):
    workspace, medium, density = _prepared(
        warp_device, actual_tables, events=1, spacing=(100, 100, 100)
    )
    histories = _selected_histories(workspace)[-4:]
    ray = RayTangent((50.0, 50.0, 50.0), (0.0, 0.0, 1.0))
    _, metadata, _, seed = _launch(workspace, density, histories, forced=0, ray=ray)
    for history in histories:
        with pytest.raises(TransportError, match="event guard"):
            complete_geometry_history(
                ray,
                10.0,
                workspace.spec,
                medium,
                _Addressed(workspace.spec, int(history), 2, 1),
                lineage=(int(history),),
                cotangent=seed,
                forced_collision_cell=0,
            )
    assert np.all(metadata[:, 0] == 4) and np.all(metadata[:, 1] == 1)
    assert np.all(metadata[:, 3] == 1)


@pytest.mark.parametrize(
    "energy,status,terminal", [(0.05, 0, 2), (0.0, 4, 0), (201.0, 4, 0), (math.inf, 4, 0)]
)
def test_cutoff_and_energy_admission_are_distinct(
    warp_device, actual_tables, energy, status, terminal
):
    workspace, _, density = _prepared(warp_device, actual_tables)
    output, metadata, _, _ = _launch(workspace, density, [2**32 + 1], energy=energy)
    assert metadata[0, 0] == status
    if status == 0:
        assert metadata[0, 2] == terminal and metadata[0, 1] == 0
        assert tuple(output[0]) == (0.0, energy)


def test_angular_exhaustion_invalidates_complete_futures(warp_device, actual_tables):
    workspace, medium, density = _prepared(warp_device, actual_tables, trials=1)
    histories = _selected_histories(workspace)
    output, metadata, ray, seed = _launch(workspace, density, histories, forced=0)
    angular_failures = 0
    for index, history in enumerate(histories):
        try:
            expected = complete_geometry_history(
                ray,
                10.0,
                workspace.spec,
                medium,
                _Addressed(workspace.spec, int(history), 2, 1),
                lineage=(int(history),),
                cotangent=seed,
                forced_collision_cell=0,
            )
        except TransportError as error:
            assert metadata[index, 0] == 4
            if "angular guard" in str(error):
                angular_failures += 1
        else:
            assert metadata[index, 0] == 0
            np.testing.assert_allclose(
                output[index], (expected.response, expected.energy_kev), rtol=2e-12, atol=2e-12
            )
    assert angular_failures > 0


@pytest.mark.parametrize(
    "energy,where",
    [
        (0.05, "outside"),
        (0.1, "outside"),
        (0.05, "inside"),
        (0.1, "inside"),
        (0.05, "upper"),
        (0.1, "upper"),
        (201.0, "outside"),
        (201.0, "upper"),
    ],
)
def test_initial_cutoff_box_support_matches_canonical_walk(
    warp_device, actual_tables, energy, where
):
    workspace, medium, density = _prepared(warp_device, actual_tables)
    x = {"inside": 0.005, "outside": 2.0, "upper": 0.01}[where]
    ray = RayTangent((x, 0.005, 0.005), (0.0, 0.0, 1.0))
    output, metadata, _, seed = _launch(
        workspace, density, [2**32 + 1], energy=energy, ray=ray, offset=0, branch=0
    )
    expected = complete_geometry_history(
        ray,
        energy,
        workspace.spec,
        medium,
        _Addressed(workspace.spec, 2**32 + 1, 0, 0),
        lineage=(2**32 + 1,),
        cotangent=seed,
    )
    assert metadata[0, 0] == 0 and metadata[0, 1] == 0
    assert metadata[0, 2] == (2 if where == "inside" else 0)
    np.testing.assert_allclose(output[0], (expected.response, expected.energy_kev), rtol=0, atol=0)
    workspace._parameters.source_energy = energy
    result = wp.empty(1, dtype=wp.float64, device=warp_device)
    wp.launch(
        _canonical_cutoff,
        1,
        inputs=[
            wp.vec3d(*ray.origin_mm),
            wp.vec3d(*ray.direction),
            workspace.material_ids,
            workspace.voxel_density,
            workspace._energies,
            workspace.absorption,
            workspace.scattering,
            density,
            workspace._parameters,
            wp.array(seed, dtype=wp.float64, device=warp_device),
        ],
        outputs=[result],
        device=warp_device,
    )
    assert result.numpy()[0] == output[0, 0]


def test_post_fluorescence_cutoff_equality_deposits_before_escape(warp_device, actual_tables):
    cutoff = max(line.energy_kev for line in actual_tables[3].fluorescence[0])
    workspace, medium, density = _prepared(warp_device, actual_tables, cutoff=cutoff)
    histories = _selected_histories(workspace)[4:8]
    output, metadata, ray, seed = _launch(workspace, density, histories, forced=0)
    equalities = 0
    for index, history in enumerate(histories):
        expected = complete_geometry_history(
            ray,
            10.0,
            workspace.spec,
            medium,
            _Addressed(workspace.spec, int(history), 2, 1),
            lineage=(int(history),),
            cotangent=seed,
            forced_collision_cell=0,
        )
        assert expected.terminal == "local-deposition" and len(expected.prefixes) == 1
        equalities += int(expected.energy_kev == cutoff)
        assert tuple(metadata[index]) == (0, 1, 2, 4)
        assert tuple(output[index]) == (0.0, expected.energy_kev)
    assert equalities > 0


def test_continuing_real_channels_clear_parallel_side(warp_device, actual_tables):
    workspace, medium, density = _prepared(
        warp_device, actual_tables, shape=(1, 1, 2), spacing=(1, 1, 1), voxel=(0.0, 1.0)
    )
    histories = _selected_histories(workspace)[4:]
    ray = RayTangent((1.0, 0.5, 0.5), (0.0, 0.0, 1.0))
    side = (0, 1, 1)
    output, metadata, _, seed = _launch(workspace, density, histories, ray=ray, forced=1, side=side)
    channels = set()
    for index, history in enumerate(histories):
        design = _Addressed(workspace.spec, int(history), 2, 1)
        expected = complete_geometry_history(
            ray,
            10.0,
            workspace.spec,
            medium,
            design,
            lineage=(int(history),),
            cotangent=seed,
            initial_material_side=side,
            forced_collision_cell=1,
        )
        first = conditional_geometry_collision(
            expected.prefixes[0], workspace.spec, medium, design, (int(history),)
        )
        assert first.outgoing is not None and first.outgoing.direction[0] != 0.0
        channels.add(first.channel)
        assert metadata[index, 0] == 0 and metadata[index, 1] == len(expected.prefixes)
        np.testing.assert_allclose(
            output[index], (expected.response, expected.energy_kev), rtol=3e-12, atol=3e-12
        )
    assert channels == {"fluorescence", "rayleigh", "free-compton"}


def test_finite_seed_energy_product_overflow_invalidates_response(warp_device, actual_tables):
    workspace, medium, density = _prepared(
        warp_device, actual_tables, voxel=(0.0,), scoring="energy-kev"
    )
    output, metadata, ray, seed = _launch(
        workspace, density, [2**32 + 1], energy=80.0, seed_values=(1e308,) * 16
    )
    assert metadata[0, 0] == 4 and output[0, 0] == 0.0
    with pytest.raises(TransportError, match=r"representable|finite|overflow"):
        complete_geometry_history(
            ray,
            80.0,
            workspace.spec,
            medium,
            _Addressed(workspace.spec, 2**32 + 1, 2, 1),
            lineage=(2**32 + 1,),
            cotangent=seed,
        )

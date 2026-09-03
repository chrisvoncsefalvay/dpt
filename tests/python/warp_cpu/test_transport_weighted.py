"""Continuous-absorption CUDA checks with independent geometric/Decimal values.

These are analytic transport fixtures, not anatomical data or material evidence.
Exact no-scattering references and path identities complement (and do not replace)
independent scattering quadrature and statistical derivative acceptance.
"""

# Optional Warp and pytest stubs do not describe their complete runtime API.
# pyright: reportUnknownMemberType=false
import importlib
import math
from dataclasses import replace
from decimal import Decimal, localcontext
from typing import Any

import pytest

from dpt.transport.derivatives import TransportParameter, derivative_histories
from dpt.transport.forward import TransportWorkspace, prepare_transport, trace_histories
from dpt.transport.model import (
    IncompleteHistoryError,
    MaterialGrid,
    PlanarDetector,
    TransportError,
    TransportSpec,
)
from dpt.transport.rng import HistoryBatch
from dpt.validation.transport_weighted import primary_single_scatter

wp: Any = importlib.import_module("warp")
DEVICE = "cpu"


@pytest.fixture(autouse=True)
def _device_scope(warp_device: str):
    global DEVICE
    DEVICE = warp_device
    with wp.ScopedDevice(DEVICE):
        yield


def _array(values: Any, dtype: Any = None) -> Any:
    return wp.array(values, dtype=wp.float64 if dtype is None else dtype, device=DEVICE)


def _outputs(count: int) -> dict[str, Any]:
    return {
        f"out_{name}": wp.empty(count, dtype=dtype, device=DEVICE)
        for name, dtype in (
            ("pixel", wp.int32),
            ("score", wp.float64),
            ("energy", wp.float64),
            ("events", wp.int32),
            ("status", wp.int32),
        )
    }


def _slab(
    count: int,
    *,
    absorption: float = 0.7,
    scattering: float = 0.0,
    density: float = 1.0,
    weight: float = 1.0,
    **changes: Any,
) -> tuple[TransportWorkspace, list[Any], dict[str, Any]]:
    spec = TransportSpec(
        MaterialGrid((-10.0, -10.0, 0.0), (20.0, 20.0, 1.0), (1, 1, 1), 1),
        PlanarDetector((-20.0, -20.0), (40.0, 40.0), (1, 1), 2.0),
        80.0,
        "analytic continuous-absorption test coefficients",
        estimator="continuous-absorption",
    )
    spec = replace(spec, **changes)
    workspace = prepare_transport(
        spec,
        material_ids=_array([0], wp.int32),
        absorption=_array([absorption] * len(spec.energy_nodes)),
        scattering=_array([scattering] * len(spec.energy_nodes)),
        device=DEVICE,
        max_histories=count,
    )
    return (
        workspace,
        [
            _array([[0.0, 0.0, -1.0]] * count, wp.vec3d),
            _array([[0.0, 0.0, 1.0]] * count, wp.vec3d),
            _array([weight] * count),
            _array([density]),
        ],
        _outputs(count),
    )


def _derivative(
    workspace: TransportWorkspace,
    inputs: list[Any],
    outputs: dict[str, Any],
    batch: HistoryBatch,
    parameter: TransportParameter,
    amplitude: float = 1.0,
) -> Any:
    values = wp.empty(batch.count, dtype=wp.float64, device=DEVICE)
    derivative_histories(
        *inputs,
        parameter=parameter,
        batch=batch,
        workspace=workspace,
        out_pixel=outputs["out_pixel"],
        out_derivative=values,
        out_status=outputs["out_status"],
        source_amplitude=amplitude,
    )
    return values.numpy()


def _decimal_score(tau: float, *factors: float) -> float:
    with localcontext() as context:
        context.prec = 100
        value = (-Decimal.from_float(tau)).exp()
        for factor in factors:
            value *= Decimal.from_float(factor)
        return float(value)


def _reference_ray(
    grid: MaterialGrid,
    material_ids: list[int],
    absorption: list[float],
    densities: list[float],
    position: tuple[float, float, float],
    direction: tuple[float, float, float],
) -> tuple[Decimal, list[Decimal]]:
    """Intersect the ray independently with every box, with no DDA or face walk."""
    with localcontext() as context:
        context.prec = 80
        depths = [Decimal(0) for _ in densities]
        origin = [Decimal.from_float(value) for value in grid.origin_mm]
        spacing = [Decimal.from_float(value) for value in grid.spacing_mm]
        start = [Decimal.from_float(value) for value in position]
        slope = [Decimal.from_float(value) for value in direction]
        nx, ny, nz = grid.shape_xyz
        for z in range(nz):
            for y in range(ny):
                for x in range(nx):
                    entry, exit_ = Decimal(0), Decimal("Infinity")
                    for axis, cell in enumerate((x, y, z)):
                        lower = origin[axis] + cell * spacing[axis]
                        upper = lower + spacing[axis]
                        if slope[axis] == 0:
                            if not lower <= start[axis] < upper:
                                exit_ = Decimal(-1)
                                break
                        else:
                            crossings = (
                                (lower - start[axis]) / slope[axis],
                                (upper - start[axis]) / slope[axis],
                            )
                            entry = max(entry, min(crossings))
                            exit_ = min(exit_, max(crossings))
                    if exit_ > entry:
                        material = material_ids[(z * ny + y) * nx + x]
                        depths[material] += (
                            (exit_ - entry)
                            * Decimal.from_float(absorption[material])
                            * Decimal.from_float(densities[material])
                        )
        return (-sum(depths, Decimal(0))).exp(), depths


@pytest.mark.parametrize("depth", [0.0, 0.1, 1.0, 3.0, 6.0])
@pytest.mark.parametrize("density", [0.5, 1.25, 2.0])
def test_absorption_values_and_all_supported_derivatives(depth: float, density: float) -> None:
    workspace, inputs, outputs = _slab(33, absorption=depth, density=density)
    batch = HistoryBatch(50183, 2**45, 33)
    amplitude = 1.7
    trace_histories(
        *inputs, batch=batch, workspace=workspace, source_amplitude=amplitude, **outputs
    )
    with localcontext() as context:
        context.prec = 80
        tau = Decimal.from_float(depth) * Decimal.from_float(density)
        base = (-tau).exp()
        value = Decimal.from_float(amplitude) * base
        expected = float(value)
        density_derivative = float(-tau * value)
    assert outputs["out_score"].numpy().tolist() == pytest.approx([expected] * 33, rel=3e-14)
    assert (outputs["out_pixel"].numpy() == 0).all()
    assert (outputs["out_events"].numpy() == 0).all()
    assert (outputs["out_status"].numpy() == 0).all()
    for parameter, reference in (
        (TransportParameter("log-material-density", 0), density_derivative),
        (TransportParameter("source-amplitude"), float(base)),
        (TransportParameter("log-source-amplitude"), expected),
    ):
        actual = _derivative(workspace, inputs, outputs, batch, parameter, amplitude)
        assert actual.tolist() == pytest.approx([reference] * 33, rel=4e-14, abs=0.0)


def test_asymmetric_voxels_oblique_reverse_tied_faces_vacuum_and_misses() -> None:
    grid = MaterialGrid((-1.5, -1.25, 0.0), (0.75, 1.25, 0.5), (3, 2, 4), 3)
    materials = [(x + 2 * y + z) % 3 for z in range(3) for y in range(2) for x in range(4)]
    absorption = [0.0, 0.375, 1.125]
    densities = [0.7, 1.25, 0.8]
    diagonal = 1 / math.sqrt(3)
    rays = [
        ((0.0, 0.0, -1.0), (0.0, 0.0, 1.0)),
        ((-2.0, -0.2, -1.0), (0.6, 0.0, 0.8)),
        ((2.0, 0.2, -1.0), (-0.6, 0.0, 0.8)),
        ((-1.5, -1.25, 0.0), (diagonal, diagonal, diagonal)),
        ((0.0, 0.0, 0.75), (0.0, 0.0, 1.0)),
        ((5.0, 5.0, -1.0), (0.0, 0.0, 1.0)),
        ((11.0, 0.0, -1.0), (0.0, 0.0, 1.0)),
        ((0.0, 0.0, -1.0), (0.0, 0.0, -1.0)),
        ((1.5, 0.0, 0.0), (0.0, 0.0, 1.0)),
    ]
    count = len(rays)
    spec = TransportSpec(
        grid,
        PlanarDetector((-10.0, -10.0), (1.0, 1.0), (20, 20), 3.0),
        80.0,
        "analytic asymmetric box fixture",
        estimator="continuous-absorption",
    )
    workspace = prepare_transport(
        spec,
        material_ids=_array(materials, wp.int32),
        absorption=_array(absorption),
        scattering=_array([0.0] * 3),
        device=DEVICE,
        max_histories=count,
    )
    inputs = [
        _array([point for point, _ in rays], wp.vec3d),
        _array([direction for _, direction in rays], wp.vec3d),
        _array([1.0] * count),
        _array(densities),
    ]
    outputs = _outputs(count)
    batch = HistoryBatch(97, 2**33, count)
    trace_histories(*inputs, batch=batch, workspace=workspace, **outputs)
    references = [_reference_ray(grid, materials, absorption, densities, *ray) for ray in rays]
    expected_scores = [
        float(value) if index not in (6, 7) else 0.0 for index, (value, _) in enumerate(references)
    ]
    assert outputs["out_score"].numpy().tolist() == pytest.approx(expected_scores, rel=4e-14)
    assert (outputs["out_events"].numpy() == 0).all()
    for material in range(3):
        values = _derivative(
            workspace, inputs, outputs, batch, TransportParameter("log-material-density", material)
        )
        expected = [
            float(-value * depths[material]) if index not in (6, 7) else 0.0
            for index, (value, depths) in enumerate(references)
        ]
        assert values.tolist() == pytest.approx(expected, rel=5e-14, abs=1e-16)


@pytest.mark.parametrize(
    "tau,weight,amplitude,energy",
    [
        (800.0, 1.0, 1e300, 1.0),
        (800.0, 1e300, 1e300, 80.0),
        (800.0, 1e-300, 1e300, 80.0),
        (1.0, 1e308, 1e-308, 80.0),
        (746.0, 1.0, 1.0, 1.0),
        (800.0, 1.0, 1e-300, 1.0),
        (10000.0, 1e308, 1e308, 80.0),
    ],
)
def test_output_range_is_formed_from_log_attenuation_and_original_factors(
    tau: float, weight: float, amplitude: float, energy: float
) -> None:
    workspace, inputs, outputs = _slab(
        3,
        absorption=tau,
        weight=weight,
        energy_kev=energy,
        scoring="energy-kev",
    )
    batch = HistoryBatch(1691, 0, 3)
    trace_histories(
        *inputs, batch=batch, workspace=workspace, source_amplitude=amplitude, **outputs
    )
    expected = _decimal_score(tau, weight, amplitude, energy)
    actual = float(outputs["out_score"].numpy()[0])
    assert actual == pytest.approx(expected, rel=4e-14, abs=math.ulp(expected))
    derivative = _derivative(
        workspace,
        inputs,
        outputs,
        batch,
        TransportParameter("log-material-density", 0),
        amplitude,
    )
    reference = _decimal_score(tau, weight, amplitude, energy, -tau)
    assert float(derivative[0]) == pytest.approx(reference, rel=4e-14, abs=math.ulp(reference))
    if tau == 746:
        assert actual == 0.0
        assert float(derivative[0]) < 0.0


def test_source_derivative_retains_attenuation_at_zero_amplitude() -> None:
    workspace, inputs, outputs = _slab(3, absorption=800.0, weight=1e300)
    batch = HistoryBatch(883, 0, 3)
    trace_histories(*inputs, batch=batch, workspace=workspace, source_amplitude=0.0, **outputs)
    assert (outputs["out_score"].numpy() == 0.0).all()
    values = _derivative(
        workspace, inputs, outputs, batch, TransportParameter("source-amplitude"), 0.0
    )
    assert values.tolist() == pytest.approx([_decimal_score(800.0, 1e300)] * 3, rel=4e-14, abs=0.0)
    density = _derivative(
        workspace, inputs, outputs, batch, TransportParameter("log-material-density", 0), 0.0
    )
    assert (density == 0.0).all()
    with pytest.raises(TransportError, match="strictly positive"):
        _derivative(
            workspace, inputs, outputs, batch, TransportParameter("log-source-amplitude"), 0.0
        )


def test_density_derivative_can_be_finite_when_forward_score_would_overflow() -> None:
    workspace, inputs, outputs = _slab(3, absorption=1e-4, weight=1e308, scoring="energy-kev")
    batch = HistoryBatch(4243, 0, 3)
    values = _derivative(
        workspace, inputs, outputs, batch, TransportParameter("log-material-density", 0)
    )
    expected = _decimal_score(1e-4, 1e308, 80.0, -1e-4)
    assert values.tolist() == pytest.approx([expected] * 3, rel=4e-14)
    with pytest.raises(FloatingPointError):
        trace_histories(*inputs, batch=batch, workspace=workspace, **outputs)


def test_zero_absorption_preserves_analogue_paths_draws_and_derivatives() -> None:
    count = 257
    weighted, inputs, outputs = _slab(count, absorption=0.0, scattering=0.8)
    analogue = prepare_transport(
        replace(weighted.spec, estimator="analogue"),
        material_ids=weighted.material_ids,
        absorption=weighted.absorption,
        scattering=weighted.scattering,
        device=DEVICE,
        max_histories=count,
    )
    batch = HistoryBatch(752719, 2**49, count)
    trace_histories(*inputs, batch=batch, workspace=analogue, **outputs)
    expected = {name: array.numpy().copy() for name, array in outputs.items()}
    trace_histories(*inputs, batch=batch, workspace=weighted, **outputs)
    for name, array in outputs.items():
        assert (array.numpy() == expected[name]).all()
    parameter = TransportParameter("log-material-density", 0)
    analogue_derivative = _derivative(analogue, inputs, outputs, batch, parameter)
    weighted_derivative = _derivative(weighted, inputs, outputs, batch, parameter)
    assert (weighted_derivative == analogue_derivative).all()


@pytest.mark.parametrize("compton", [False, True])
def test_scattering_density_identity_and_chunk_replay(compton: bool) -> None:
    count = 1025
    extra: dict[str, Any] = {}
    if compton:
        extra = {
            "scattering_law": "free-electron-compton",
            "coefficient_energies_kev": (0.0, 40.0, 80.0),
            "scoring": "energy-kev",
        }
    workspace, inputs, outputs = _slab(count, absorption=0.4, scattering=0.8, **extra)
    if compton:
        # Energy-dependent tables with mu_s / mu_a = 2 at every interpolated energy.
        workspace = prepare_transport(
            workspace.spec,
            material_ids=workspace.material_ids,
            absorption=_array([0.2, 0.4, 0.8]),
            scattering=_array([0.4, 0.8, 1.6]),
            device=DEVICE,
            max_histories=count,
        )
    batch = HistoryBatch(38271, 2**44, count)
    trace_histories(*inputs, batch=batch, workspace=workspace, **outputs)
    expected = {name: array.numpy().copy() for name, array in outputs.items()}
    derivative = _derivative(
        workspace, inputs, outputs, batch, TransportParameter("log-material-density", 0)
    )
    assert (derivative > 0.0).any()
    assert (derivative < 0.0).any()
    for index in range(count):
        score = float(expected["out_score"][index])
        if score > 0:
            energy = float(expected["out_energy"][index]) if compton else 1.0
            depth = -math.log(score / energy)
            reference = score * (int(expected["out_events"][index]) - 3 * depth)
            assert float(derivative[index]) == pytest.approx(reference, rel=2e-12, abs=2e-13)
        else:
            assert float(derivative[index]) == 0.0
    for first, size in ((0, 127), (127, 898)):
        trace_histories(
            inputs[0][first : first + size],
            inputs[1][first : first + size],
            inputs[2][first : first + size],
            inputs[3],
            batch=HistoryBatch(batch.seed, batch.first_history + first, size),
            workspace=workspace,
            **{name: array[first : first + size] for name, array in outputs.items()},
        )
    for name, array in outputs.items():
        assert (array.numpy() == expected[name]).all()


@pytest.mark.parametrize("source_weight", [0.0, 1.0])
def test_zero_weight_or_underflow_never_bypasses_event_guard(source_weight: float) -> None:
    workspace, inputs, outputs = _slab(
        257, absorption=1e12, scattering=1e6, weight=source_weight, max_events=1
    )
    with pytest.raises(IncompleteHistoryError):
        trace_histories(*inputs, batch=HistoryBatch(827, 0, 257), workspace=workspace, **outputs)
    assert (outputs["out_status"].numpy() == 2).all()
    assert not (outputs["out_status"].numpy() == 1).any()


def test_crossing_guard_covers_zero_scattering_segments() -> None:
    workspace, inputs, outputs = _slab(3)
    spec = replace(
        workspace.spec,
        grid=MaterialGrid((-10.0, -10.0, 0.0), (20.0, 20.0, 0.5), (2, 1, 1), 1),
        max_crossings=1,
    )
    workspace = prepare_transport(
        spec,
        material_ids=_array([0, 0], wp.int32),
        absorption=workspace.absorption,
        scattering=workspace.scattering,
        device=DEVICE,
        max_histories=3,
    )
    with pytest.raises(IncompleteHistoryError):
        trace_histories(*inputs, batch=HistoryBatch(819, 0, 3), workspace=workspace, **outputs)
    assert (outputs["out_status"].numpy() == 3).all()


def test_compton_energy_support_is_checked_even_after_weight_underflow() -> None:
    workspace, inputs, outputs = _slab(
        257,
        absorption=1e12,
        scattering=100.0,
        scattering_law="free-electron-compton",
        coefficient_energies_kev=(79.999, 80.0),
    )
    with pytest.raises(TransportError, match="energy support"):
        trace_histories(*inputs, batch=HistoryBatch(92319, 0, 257), workspace=workspace, **outputs)
    assert (outputs["out_status"].numpy() == 5).any()
    assert not (outputs["out_status"].numpy() == 1).any()


@pytest.mark.parametrize("compton", [False, True])
def test_values_and_density_derivatives_against_independent_scatter_integral(compton: bool) -> None:
    """The derivative reference is differentiated quadrature plus its own tail bound."""
    count = 2**20
    density = 1.25
    absorption = (0.8, 0.5, 0.25) if compton else (0.3,)
    scattering = (0.001, 0.00075, 0.0005) if compton else (0.0005,)
    nodes = (0.0, 40.0, 80.0) if compton else (80.0,)
    law = "free-electron-compton" if compton else "isotropic-elastic"
    reference = primary_single_scatter(
        absorption=absorption,
        scattering=scattering,
        coefficient_energies_kev=nodes,
        scattering_law=law,
        density=density,
    )
    assert reference.quadrature_value_error < 1e-12
    assert reference.quadrature_derivative_error < 1e-12
    assert reference.previous_value_error < 1e-10
    assert reference.previous_derivative_error < 1e-10
    assert reference.value_tail_bound < 2e-6
    assert reference.derivative_tail_bound < 1e-5
    # Independently perturb the quadrature itself to check its differentiated
    # integrand; this does not replay paths from either CUDA estimator.
    step = 1e-4
    differences: list[float] = []
    for half_step in (step, step / 2):
        lower = primary_single_scatter(
            absorption=absorption,
            scattering=scattering,
            coefficient_energies_kev=nodes,
            scattering_law=law,
            density=density * math.exp(-half_step),
        )
        upper = primary_single_scatter(
            absorption=absorption,
            scattering=scattering,
            coefficient_energies_kev=nodes,
            scattering_law=law,
            density=density * math.exp(half_step),
        )
        differences.append((upper.mean - lower.mean) / (2 * half_step))
    richardson = (4 * differences[1] - differences[0]) / 3
    assert richardson == pytest.approx(reference.log_density_derivative, rel=2e-10, abs=2e-12)
    spec = TransportSpec(
        MaterialGrid((-0.5, -0.5, 0.0), (1.0, 1.0, 1.0), (1, 1, 1), 1),
        PlanarDetector((-1.0, -1.0), (2.0, 2.0), (1, 1), 2.0),
        80.0,
        "analytic thin cuboid, independent primary/single-scatter quadrature",
        coefficient_energies_kev=nodes,
        scattering_law=law,
        estimator="continuous-absorption",
    )
    workspace = prepare_transport(
        spec,
        material_ids=_array([0], wp.int32),
        absorption=_array(absorption),
        scattering=_array(scattering),
        device=DEVICE,
        max_histories=count,
    )
    inputs = [
        _array([[0.0, 0.0, -1.0]] * count, wp.vec3d),
        _array([[0.0, 0.0, 1.0]] * count, wp.vec3d),
        _array([1.0] * count),
        _array([density]),
    ]
    outputs = _outputs(count)
    batch = HistoryBatch(953671 if compton else 658771, 2**40, count)
    trace_histories(*inputs, batch=batch, workspace=workspace, **outputs)
    values = outputs["out_score"].numpy()
    derivatives = _derivative(
        workspace, inputs, outputs, batch, TransportParameter("log-material-density", 0)
    )
    for samples, expected, tail, quadrature_error in (
        (values, reference.mean, reference.value_tail_bound, reference.quadrature_value_error),
        (
            derivatives,
            reference.log_density_derivative,
            reference.derivative_tail_bound,
            reference.quadrature_derivative_error,
        ),
    ):
        mean = float(samples.mean())
        standard_error = float(samples.std(ddof=1)) / math.sqrt(count)
        # A predeclared seven-SE stochastic check, not a universal coverage claim.
        # The omitted derivative orders have their own proven bound, not the
        # value-tail bound reused as an unsupported gradient error estimate.
        allowance = 7 * standard_error + tail + quadrature_error + 64 * math.ulp(expected)
        assert allowance < 5e-4
        assert abs(mean - expected) < allowance


def test_single_scatter_signal_exceeds_entire_predeclared_allowance() -> None:
    """A drop-all-scatter estimator cannot fit this interval around the oracle."""
    count = 2**22
    reference = primary_single_scatter(
        absorption=(0.3,),
        scattering=(0.004,),
        coefficient_energies_kev=(80.0,),
        scattering_law="isotropic-elastic",
        density=1.0,
        half_width_mm=10.0,
        detector_half_width_mm=10.0,
    )
    spec = TransportSpec(
        MaterialGrid((-10.0, -10.0, 0.0), (20.0, 20.0, 1.0), (1, 1, 1), 1),
        PlanarDetector((-10.0, -10.0), (20.0, 20.0), (1, 1), 2.0),
        80.0,
        "analytic wide cuboid: detectable single scatter",
        coefficient_energies_kev=(80.0,),
        estimator="continuous-absorption",
    )
    workspace = prepare_transport(
        spec,
        material_ids=_array([0], wp.int32),
        absorption=_array([0.3]),
        scattering=_array([0.004]),
        device=DEVICE,
        max_histories=count,
    )
    inputs = [
        wp.full(count, wp.vec3d(0.0, 0.0, -1.0), dtype=wp.vec3d, device=DEVICE),
        wp.full(count, wp.vec3d(0.0, 0.0, 1.0), dtype=wp.vec3d, device=DEVICE),
        wp.ones(count, dtype=wp.float64, device=DEVICE),
        _array([1.0]),
    ]
    outputs = _outputs(count)
    trace_histories(
        *inputs, batch=HistoryBatch(197031, 2**40, count), workspace=workspace, **outputs
    )
    values = outputs["out_score"].numpy()
    allowance = (
        7 * float(values.std(ddof=1)) / math.sqrt(count)
        + reference.value_tail_bound
        + reference.quadrature_value_error
    )
    assert allowance < reference.single_scatter
    assert abs(float(values.mean()) - reference.mean) < allowance
    # The counterfactual retained-primary/drop-all-scatter mutant is excluded.
    assert abs(reference.primary - reference.mean) > allowance

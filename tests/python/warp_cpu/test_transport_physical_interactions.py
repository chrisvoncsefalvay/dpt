"""Independent partial-physics, null-process and energy-accounting checks.

The source data are queried from the installed pinned provider. Beer-law and
marked-Poisson identities provide independent transport/score expectations,
and the provider's differential API supplies the atomic angular reference.
"""

# Warp decorators and type annotations are runtime DSL expressions.
# pyright: reportInvalidTypeForm=false, reportUnknownParameterType=false
# pyright: reportUnknownMemberType=false, reportUntypedFunctionDecorator=false
import importlib
import json
import math
from collections.abc import Callable
from dataclasses import replace
from decimal import Decimal, localcontext
from itertools import pairwise
from typing import Any, Literal

import numpy as np
import pytest
from scipy import stats

from dpt.materials import Provenance
from dpt.transport.derivatives import (
    TransportParameter,
    derivative_histories,
)
from dpt.transport.energy_balance import trace_energy_balance
from dpt.transport.forward import prepare_transport, trace_histories
from dpt.transport.model import (
    MaterialGrid,
    PlanarDetector,
    TransportError,
    TransportSpec,
)
from dpt.transport.physical import (
    HC_KEV_ANGSTROM,
    AtomicScatteringTable,
    PhysicalTransport,
    xraylib_atomic_scattering,
    xraylib_fluorescence_lines,
)
from dpt.transport.rng import HistoryBatch
from dpt.transport.source_spectrum import SourceSpectrum

wp: Any = importlib.import_module("warp")
physical_kernels: Any = importlib.import_module("dpt.transport.physical_kernels")
transport_kernels: Any = importlib.import_module("dpt.transport.kernels")


@wp.kernel
def _sample_rayleigh(
    energy: wp.float64,
    physical: physical_kernels.PhysicalParameters,
    out_values: wp.array(dtype=wp.vec4d),
):
    index = wp.tid()
    out_values[index] = physical_kernels.rayleigh_scatter_event(
        energy, 0, wp.uint64(51681), wp.uint64(index), wp.uint32(0), 4096, physical
    )


@wp.kernel
def _sample_fluorescence(
    energy: wp.float64,
    coordinate: wp.vec2d,
    physical: physical_kernels.PhysicalParameters,
    out_values: wp.array(dtype=wp.vec4d),
):
    index = wp.tid()
    out_values[index] = physical_kernels.fluorescence_event(
        energy,
        0,
        coordinate,
        2,
        wp.uint64(918181),
        wp.uint64(index),
        wp.uint32(0),
        physical,
    )


@wp.kernel
def _rayleigh_pdf(
    energy: wp.float64,
    physical: physical_kernels.PhysicalParameters,
    cosines: wp.array(dtype=wp.float64),
    out_values: wp.array(dtype=wp.float64),
):
    index = wp.tid()
    out_values[index] = physical_kernels.rayleigh_phase_pdf(energy, 0, cosines[index], physical)


@wp.kernel
def _woodcock_recorded_scores(
    positions: wp.array(dtype=wp.vec3d),
    directions: wp.array(dtype=wp.vec3d),
    weights: wp.array(dtype=wp.float64),
    density: wp.array(dtype=wp.float64),
    material_ids: wp.array(dtype=wp.int32),
    voxel_density: wp.array(dtype=wp.float64),
    energies: wp.array(dtype=wp.float64),
    absorption: wp.array(dtype=wp.float64),
    scattering: wp.array(dtype=wp.float64),
    parameters: transport_kernels.Parameters,
    records: transport_kernels.VoxelRecords,
    out: wp.array(dtype=wp.vec4d),
    out_counts: wp.array(dtype=wp.vec2i),
    out_status: wp.array(dtype=wp.int32),
):
    index = wp.tid()
    local_records = records
    local_records.history = index
    result = transport_kernels.walk_recorded(
        positions[index],
        directions[index],
        weights[index],
        wp.float64(1.0),
        wp.uint64(738191),
        wp.uint64(index),
        0,
        wp.vec4i(0, 1, -1, -1),
        2,
        material_ids,
        voxel_density,
        energies,
        absorption,
        scattering,
        density,
        parameters,
        local_records,
    )
    out[index] = wp.vec4d(
        result.density_score,
        result.density_scores[0],
        result.density_scores[1],
        result.energy,
    )
    out_counts[index] = wp.vec2i(result.events, result.real_events)
    out_status[index] = result.status


def _workspace(
    device: str,
    *,
    physical: PhysicalTransport,
    energies: tuple[float, ...],
    absorption: tuple[float, ...],
    scattering: tuple[float, ...],
    count: int = 65536,
    scattering_law: Literal["isotropic-elastic", "free-electron-compton"] = "isotropic-elastic",
) -> Any:
    spec = TransportSpec(
        MaterialGrid((-1.0, -1.0, 0.0), (2.0, 2.0, 1.0), (1, 1, 1), 1),
        PlanarDetector((-100.0, -100.0), (200.0, 200.0), (1, 1), 2.0),
        energies[-1],
        "declared partial physical model with pinned atomic tables",
        coefficient_energies_kev=energies,
        scattering_law=scattering_law,
        scoring="energy-kev",
        max_events=1024,
    )
    return prepare_transport(
        spec,
        material_ids=wp.array([0], dtype=wp.int32, device=device),
        absorption=wp.array(absorption, dtype=wp.float64, device=device),
        scattering=wp.array(scattering, dtype=wp.float64, device=device),
        max_histories=count,
        device=device,
        physical=physical,
    )


def _source(device: str, count: int) -> tuple[Any, Any, Any, Any]:
    return (
        wp.array(np.tile((0.0, 0.0, -1.0), (count, 1)), dtype=wp.vec3d, device=device),
        wp.array(np.tile((0.0, 0.0, 1.0), (count, 1)), dtype=wp.vec3d, device=device),
        wp.ones(count, dtype=wp.float64, device=device),
        wp.ones(1, dtype=wp.float64, device=device),
    )


def _outputs(device: str, count: int) -> dict[str, Any]:
    return {
        "out_" + name: wp.empty(count, dtype=dtype, device=device)
        for name, dtype in (
            ("pixel", wp.int32),
            ("score", wp.float64),
            ("energy", wp.float64),
            ("events", wp.int32),
            ("status", wp.int32),
        )
    }


def test_rayleigh_angular_law_energy_and_phase_pdf(
    warp_device: str, record_testsuite_property: Callable[[str, object], None]
) -> None:
    provider = pytest.importorskip("xraylib")
    energy = 20.0
    q = (0.0, *np.geomspace(0.005, energy / HC_KEV_ANGSTROM, 1024))
    table = xraylib_atomic_scattering(13, q)
    assert table.provider is not None
    physical = PhysicalTransport((table,), (1.0,))
    workspace = _workspace(
        warp_device,
        physical=physical,
        energies=(energy,),
        absorption=(0.0,),
        scattering=(1.0,),
    )
    count = 65536
    values = wp.empty(count, dtype=wp.vec4d, device=warp_device)
    wp.launch(
        _sample_rayleigh,
        dim=count,
        inputs=[energy, workspace._parameters.physics, values],
        device=warp_device,
    )
    observed = values.numpy()
    assert np.all(observed[:, 3] == 0) and np.all(observed[:, 2] == energy)
    edges = np.linspace(-1.0, 1.0, 17)
    nodes, weights = np.polynomial.legendre.leggauss(32)
    masses = []
    for cosine_left, cosine_right in pairwise(edges):
        q_left = energy / HC_KEV_ANGSTROM * math.sqrt((1 - cosine_right) / 2)
        q_right = energy / HC_KEV_ANGSTROM * math.sqrt((1 - cosine_left) / 2)
        integral = 0.0
        for left_node, right_node in pairwise(q):
            left, right = max(left_node, q_left), min(right_node, q_right)
            if right <= left:
                continue
            sample_q = (left + right) / 2 + (right - left) / 2 * nodes
            sample_c = 1 - 2 * (sample_q * HC_KEV_ANGSTROM / energy) ** 2
            form = np.interp(sample_q, q, table.form_factor)
            integral += (
                (right - left) / 2 * np.sum(weights * sample_q * form**2 * (1 + sample_c**2))
            )
        masses.append(integral)
    masses = np.array(masses)
    probability = masses / masses.sum()
    counts = np.histogram(observed[:, 0], edges)[0]
    alpha = 1e-6 / len(probability)
    lower, upper = (
        stats.binom.ppf(alpha / 2, count, probability),
        stats.binom.ppf(1 - alpha / 2, count, probability),
    )
    assert np.all((counts >= lower) & (counts <= upper))
    cosines = np.array([-0.9, 0.0, 0.5, 0.99])
    out_pdf = wp.empty(len(cosines), dtype=wp.float64, device=warp_device)
    wp.launch(
        _rayleigh_pdf,
        dim=len(cosines),
        inputs=[
            energy,
            workspace._parameters.physics,
            wp.array(cosines, dtype=wp.float64, device=warp_device),
            out_pdf,
        ],
        device=warp_device,
    )
    represented_integral = 0.0
    for left, right in pairwise(q):
        sample_q = (left + right) / 2 + (right - left) / 2 * nodes
        sample_c = 1 - 2 * (sample_q * HC_KEV_ANGSTROM / energy) ** 2
        form = np.interp(sample_q, q, table.form_factor)
        represented_integral += (
            (right - left) / 2 * np.sum(weights * sample_q * form**2 * (1 + sample_c**2))
        )
    represented_integral *= 4 * (HC_KEV_ANGSTROM / energy) ** 2
    form = np.interp(energy / HC_KEV_ANGSTROM * np.sqrt((1 - cosines) / 2), q, table.form_factor)
    expected = (1 + cosines**2) * form**2 / (2 * math.pi * represented_integral)
    np.testing.assert_allclose(out_pdf.numpy(), expected, rtol=3e-13, atol=0)
    # The upstream differential cross section supplies an independent physical
    # shape check at supported transfer values, with constants cancelled by ratio.
    provider_shape = np.array([provider.DCS_Rayl(13, energy, math.acos(c)) for c in cosines])
    np.testing.assert_allclose(
        provider_shape / provider_shape[1], expected / expected[1], rtol=1e-4
    )
    record_testsuite_property(
        "Rayleigh-physical-reference",
        json.dumps(
            {
                "histories": count,
                "energy_kev": energy,
                "atomic_number": 13,
                "counts": counts.tolist(),
                "probabilities": probability.tolist(),
                "binomial_family_alpha": 1e-6,
                "provider": table.provider.version,
                "q_table_sha256": table.provenance.sha256,
                "represented_phase_pdf_max_relative_error": float(
                    np.max(abs(out_pdf.numpy() / expected - 1))
                ),
            },
            sort_keys=True,
        ),
    )


def test_polyenergetic_beer_law_and_fixed_majorant_null_score(
    warp_device: str, record_testsuite_property: Callable[[str, object], None]
) -> None:
    pytest.importorskip("xraylib")
    table = xraylib_atomic_scattering(13, (0.0, 0.1, 1.0, 10.0))
    provenance = Provenance(
        "analytic Beer-law marked Poisson reference",
        "0" * 64,
        "mathematical identity",
        "Two prescribed source lines.",
    )
    physical = PhysicalTransport(
        (table,),
        (0.0, 0.0),
        majorant_mm_inverse=(2.0, 2.0),
        spectrum=SourceSpectrum((40.0, 80.0), (0.25, 0.75), provenance.source),
    )
    workspace = _workspace(
        warp_device,
        physical=physical,
        energies=(40.0, 80.0),
        absorption=(0.5, 1.0),
        scattering=(0.0, 0.0),
    )
    count = 65536
    source = _source(warp_device, count)
    outputs = _outputs(warp_device, count)
    batch = HistoryBatch(19482, 0, count)
    trace_histories(*source, batch=batch, workspace=workspace, **outputs)
    score = outputs["out_score"].numpy()
    exact = 0.25 * 40 * math.exp(-0.5) + 0.75 * 80 * math.exp(-1)
    assert abs(score.mean() - exact) < 7 * score.std(ddof=1) / math.sqrt(count)
    derivative = wp.empty(count, dtype=wp.float64, device=warp_device)
    derivative_histories(
        *source,
        batch=batch,
        workspace=workspace,
        parameter=TransportParameter("log-material-density", 0),
        out_pixel=outputs["out_pixel"],
        out_derivative=derivative,
        out_status=outputs["out_status"],
    )
    observed = derivative.numpy()
    exact_derivative = -0.25 * 40 * 0.5 * math.exp(-0.5) - 0.75 * 80 * math.exp(-1)
    assert abs(observed.mean() - exact_derivative) < 7 * observed.std(ddof=1) / math.sqrt(count)
    # Null counts conditional on transmission have Poisson intensity M-mu.
    # A null contribution -mu/(M-mu) gives E[score|escape] = -mu*L.
    assert np.any(outputs["out_events"].numpy() > 1)
    record_testsuite_property(
        "Woodcock-polyenergetic-Beer-reference",
        json.dumps(
            {
                "histories": count,
                "mean_energy_kev": float(score.mean()),
                "exact_mean": exact,
                "mean_log_density_derivative": float(observed.mean()),
                "exact_derivative": exact_derivative,
                "majorant_mm_inverse": [2, 2],
                "physical_rates": [0.5, 1],
                "null_score": "-mu/(M-mu)",
                "real_score": 1,
            },
            sort_keys=True,
        ),
    )
    bad_density = wp.array([2.0], dtype=wp.float64, device=warp_device)
    with pytest.raises(TransportError, match="invalid device"):
        trace_histories(*source[:3], bad_density, batch=batch, workspace=workspace, **outputs)


def test_fluorescence_local_deposition_and_cutoff_convergence(
    warp_device: str, record_testsuite_property: Callable[[str, object], None]
) -> None:
    provider = pytest.importorskip("xraylib")
    energies = (0.5, 1.0, 1.4864, 1.4869, 2.0, 5.0, 10.0, 20.0, 40.0, 80.0)
    table = xraylib_atomic_scattering(13, (0.0, *np.geomspace(0.005, 10.0, 512)))
    lines = xraylib_fluorescence_lines(13, energies, lines=("KL2_LINE", "KL3_LINE"))
    rho = 2.6989
    absorption = tuple(provider.CS_Photo(13, e) * rho / 10 for e in energies)
    rayleigh = tuple(provider.CS_Rayl(13, e) * rho / 10 for e in energies)
    scattering = tuple(
        (provider.CS_Rayl(13, e) + provider.CS_Compt(13, e)) * rho / 10 for e in energies
    )
    physical = PhysicalTransport((table,), rayleigh, fluorescence=(lines,), cutoff_kev=0.5)
    count = 8192
    source = _source(warp_device, count)
    outputs = _outputs(warp_device, count)
    extras = {
        "out_" + name: wp.empty(count, dtype=dtype, device=warp_device)
        for name, dtype in (
            ("initial_energy", wp.float64),
            ("deposited_energy", wp.float64),
            ("spectrum_index", wp.int32),
        )
    }
    local = wp.empty(1, dtype=wp.float64, device=warp_device)
    escaped_means = []
    for cutoff in (5.0, 2.0, 1.0, 0.5):
        workspace = _workspace(
            warp_device,
            physical=replace(physical, cutoff_kev=cutoff),
            energies=energies,
            absorption=absorption,
            scattering=scattering,
            count=count,
            scattering_law="free-electron-compton",
        )
        trace_energy_balance(
            *source,
            batch=HistoryBatch(952811, 0, count),
            workspace=workspace,
            **outputs,
            **extras,
            out_local_deposition=local,
        )
        initial = extras["out_initial_energy"].numpy()
        deposited = extras["out_deposited_energy"].numpy()
        final = outputs["out_energy"].numpy()
        np.testing.assert_allclose(initial, final + deposited, rtol=0, atol=1e-12)
        assert local.numpy()[0] == pytest.approx(deposited.sum(), rel=2e-12)
        escaped_means.append(final.mean())
    # Coupled cutoff histories agree exactly when lowering below every emitted
    # line and all actually sampled recoil energies, or reveal the omitted energy.
    assert np.all(np.diff(escaped_means) >= -1e-12)
    assert escaped_means[-1] - escaped_means[-2] <= escaped_means[-2] - escaped_means[0] + 1e-12
    record_testsuite_property(
        "Aluminium-energy-balance-cutoff-reference",
        json.dumps(
            {
                "histories": count,
                "source_energy_kev": 80,
                "cutoffs_kev": [5, 2, 1, 0.5],
                "escaped_energy_means_kev": escaped_means,
                "energy_residual_max_abs_kev": float(np.max(abs(initial - final - deposited))),
                "local_deposition_sum_kev": float(local.numpy()[0]),
                "line_energies_kev": [line.energy_kev for line in lines],
                "line_table_sha256": [line.provenance.sha256 for line in lines],
            },
            sort_keys=True,
        ),
    )


def test_heterogeneous_woodcock_scores_match_independent_poisson_moments(
    warp_device: str, record_testsuite_property: Callable[[str, object], None]
) -> None:
    pytest.importorskip("xraylib")
    table = xraylib_atomic_scattering(13, (0.0, 0.1, 1.0, 10.0))
    physical = PhysicalTransport((table, table), (0.0, 0.0), majorant_mm_inverse=(2.0,))
    count = 65536
    spec = TransportSpec(
        MaterialGrid((-1.0, -1.0, 0.0), (2.0, 2.0, 0.5), (2, 1, 1), 2),
        PlanarDetector((-1.0, -1.0), (2.0, 2.0), (1, 1), 2.0),
        80.0,
        "analytic two-layer marked Poisson extinction model",
    )
    workspace = prepare_transport(
        spec,
        material_ids=wp.array([0, 1], dtype=wp.int32, device=warp_device),
        absorption=wp.array([0.3, 0.7], dtype=wp.float64, device=warp_device),
        scattering=wp.zeros(2, dtype=wp.float64, device=warp_device),
        voxel_density=wp.array([2.0, 0.5], dtype=wp.float64, device=warp_device),
        max_histories=count,
        device=warp_device,
        physical=physical,
    )
    source = _source(warp_device, count)
    density = wp.array([1.2, 0.8], dtype=wp.float64, device=warp_device)
    outputs = _outputs(warp_device, count)
    batch = HistoryBatch(817731, 0, count)
    trace_histories(*source[:3], density, batch=batch, workspace=workspace, **outputs)
    rates = np.array([0.3 * 1.2 * 2, 0.7 * 0.8 * 0.5])
    exact = math.exp(-float(rates.sum()) * 0.5)
    scores = outputs["out_score"].numpy()
    assert abs(scores.mean() - exact) < 7 * math.sqrt(exact * (1 - exact) / count)
    escaped = scores != 0
    null_mean = float((2 - rates).sum()) * 0.5
    events = outputs["out_events"].numpy()[escaped]
    assert abs(events.mean() - null_mean) < 7 * math.sqrt(null_mean / len(events))
    derivative = wp.empty(count, dtype=wp.float64, device=warp_device)
    means = []
    for material, rate in enumerate(rates):
        derivative_histories(
            *source[:3],
            density,
            parameter=TransportParameter("log-material-density", material),
            batch=batch,
            workspace=workspace,
            out_pixel=outputs["out_pixel"],
            out_derivative=derivative,
            out_status=outputs["out_status"],
        )
        observed = derivative.numpy()
        expected = -rate * 0.5 * exact
        # Conditional on escape, N_i~Poisson((M-mu_i)L_i), so the
        # null-score second moment is mu_i^2[L_i/(M-mu_i)+L_i^2].
        second_moment = exact * rate**2 * (0.5 / (2 - rate) + 0.25)
        variance = second_moment - expected**2
        assert abs(observed.mean() - expected) < 7 * math.sqrt(variance / count)
        means.append(float(observed.mean()))
    record_testsuite_property(
        "Heterogeneous-Woodcock-Poisson-reference",
        json.dumps(
            {
                "histories": count,
                "cell_rates": rates.tolist(),
                "cell_lengths_mm": [0.5, 0.5],
                "survival": float(scores.mean()),
                "exact_survival": exact,
                "escaped_null_count_mean": float(events.mean()),
                "exact_conditional_null_mean": null_mean,
                "log_density_derivative_means": means,
                "exact_log_density_derivatives": (-rates * 0.5 * exact).tolist(),
            },
            sort_keys=True,
        ),
    )


def test_copper_cutoff_refinement_transports_actual_fluorescent_energy(
    warp_device: str, record_testsuite_property: Callable[[str, object], None]
) -> None:
    provider = pytest.importorskip("xraylib")
    z = 29
    lines_requested = ("KL2_LINE", "KL3_LINE")
    line_energies = tuple(
        provider.LineEnergy(z, getattr(provider, name)) for name in lines_requested
    )
    energies = (1.0, 2.0, 5.0, *line_energies, 10.0, 20.0)
    table = xraylib_atomic_scattering(z, (0.0, *np.geomspace(0.005, 10.0, 512)))
    lines = xraylib_fluorescence_lines(z, energies, lines=lines_requested)
    rho = 8.96
    absorption = tuple(provider.CS_Photo(z, e) * rho / 10 for e in energies)
    rayleigh = tuple(provider.CS_Rayl(z, e) * rho / 10 for e in energies)
    scattering = tuple(
        (provider.CS_Rayl(z, e) + provider.CS_Compt(z, e)) * rho / 10 for e in energies
    )
    physical = PhysicalTransport((table,), rayleigh, fluorescence=(lines,), cutoff_kev=1)
    count = 8192
    source = _source(warp_device, count)
    outputs = _outputs(warp_device, count)
    extras = {
        "out_" + name: wp.empty(count, dtype=dtype, device=warp_device)
        for name, dtype in (
            ("initial_energy", wp.float64),
            ("deposited_energy", wp.float64),
            ("spectrum_index", wp.int32),
        )
    }
    local = wp.empty(1, dtype=wp.float64, device=warp_device)
    means, final_arrays = [], []
    for cutoff in (10.0, 5.0, 2.0, 1.0):
        workspace = _workspace(
            warp_device,
            physical=replace(physical, cutoff_kev=cutoff),
            energies=energies,
            absorption=absorption,
            scattering=scattering,
            count=count,
            scattering_law="free-electron-compton",
        )
        trace_energy_balance(
            *source,
            batch=HistoryBatch(711741, 0, count),
            workspace=workspace,
            **outputs,
            **extras,
            out_local_deposition=local,
        )
        final = outputs["out_energy"].numpy()
        np.testing.assert_allclose(
            extras["out_initial_energy"].numpy(),
            final + extras["out_deposited_energy"].numpy(),
            atol=1e-12,
            rtol=0,
        )
        means.append(float(final.mean()))
        final_arrays.append(final)
    assert means[1] > means[0]
    np.testing.assert_array_equal(final_arrays[1], final_arrays[2])
    np.testing.assert_array_equal(final_arrays[2], final_arrays[3])
    assert any(np.any(final_arrays[-1] == line_energy) for line_energy in line_energies)
    record_testsuite_property(
        "Copper-cutoff-convergence-reference",
        json.dumps(
            {
                "histories": count,
                "source_energy_kev": 20,
                "cutoffs_kev": [10, 5, 2, 1],
                "escaped_energy_means_kev": means,
                "fluorescence_line_energies_kev": line_energies,
                "refinements_below5kev_bit_identical": True,
            },
            sort_keys=True,
        ),
    )


def test_real_null_material_and_voxel_scores_obey_path_normalisation(
    warp_device: str, record_testsuite_property: Callable[[str, object], None]
) -> None:
    pytest.importorskip("xraylib")
    table = xraylib_atomic_scattering(13, (0.0, 0.1, 1.0, 10.0))
    physical = PhysicalTransport((table, table), (0.0, 0.0), majorant_mm_inverse=(2.0,))
    count = 32768
    spec = TransportSpec(
        MaterialGrid((-1.0, -1.0, 0.0), (2.0, 2.0, 0.5), (2, 1, 1), 2),
        PlanarDetector((-1.0, -1.0), (2.0, 2.0), (1, 1), 2.0),
        80.0,
        "conservative isotropic two-material path-law normalisation reference",
    )
    workspace = prepare_transport(
        spec,
        material_ids=wp.array([0, 1], dtype=wp.int32, device=warp_device),
        absorption=wp.zeros(2, dtype=wp.float64, device=warp_device),
        scattering=wp.array([0.6, 0.4], dtype=wp.float64, device=warp_device),
        voxel_density=wp.array([5 / 6, 1.25], dtype=wp.float64, device=warp_device),
        max_histories=count,
        device=warp_device,
        physical=physical,
    )
    positions, directions, weights, _ = _source(warp_device, count)
    density = wp.ones(2, dtype=wp.float64, device=warp_device)
    out = wp.empty(count, dtype=wp.vec4d, device=warp_device)
    counts = wp.empty(count, dtype=wp.vec2i, device=warp_device)
    status = wp.empty(count, dtype=wp.int32, device=warp_device)
    records = transport_kernels.VoxelRecords()
    records.enabled, records.capacity = 1, 4
    records.active = wp.ones(2, dtype=wp.int32, device=warp_device)
    records.voxel = wp.full(count * 4, -1, dtype=wp.int32, device=warp_device)
    records.score = wp.zeros(count * 4, dtype=wp.float64, device=warp_device)
    records.status = wp.zeros(1, dtype=wp.int32, device=warp_device)
    wp.launch(
        _woodcock_recorded_scores,
        dim=count,
        device=warp_device,
        inputs=[
            positions,
            directions,
            weights,
            density,
            *workspace._model_inputs(),
            records,
            out,
            counts,
            status,
        ],
    )
    observed, events = out.numpy(), counts.numpy()
    assert np.all(status.numpy() == 0) and records.status.numpy()[0] == 0
    assert np.all(observed[:, 3] == 80) and np.any(events[:, 1] > 0)
    np.testing.assert_array_equal(observed[:, 0], observed[:, 1])
    voxel_scores = records.score.numpy().reshape(count, 4)
    voxel_ids = records.voxel.numpy().reshape(count, 4)
    for voxel in (0, 1):
        expected = np.sum(np.where(voxel_ids == voxel, voxel_scores, 0), axis=1)
        np.testing.assert_array_equal(expected, observed[:, voxel + 1])
    joint = observed[:, 1] + observed[:, 2]
    # Every candidate mark is real with probability1/4 and null with
    # probability3/4, giving independent LR increments1 and-1/3.
    # Counts come from the realised physical event ledger, not the LR arithmetic.
    expected_joint = events[:, 1] - (events[:, 0] - events[:, 1]) / 3
    np.testing.assert_allclose(joint, expected_joint, atol=3e-14, rtol=0)
    for material in (0, 1):
        score = observed[:, material + 1]
        assert abs(score.mean()) < 7 * score.std(ddof=1) / math.sqrt(count)
    martingale_difference = joint**2 - events[:, 0] / 3
    assert abs(martingale_difference.mean()) < 7 * martingale_difference.std(ddof=1) / math.sqrt(
        count
    )
    record_testsuite_property(
        "Woodcock-real-null-voxel-normalisation-reference",
        json.dumps(
            {
                "histories": count,
                "real_events": int(events[:, 1].sum()),
                "null_events": int((events[:, 0] - events[:, 1]).sum()),
                "material_score_means": observed[:, 1:3].mean(axis=0).tolist(),
                "global_score_second_moment": float(np.mean(joint**2)),
                "exact_predictable_quadratic_variation_mean": float(np.mean(events[:, 0] / 3)),
                "voxel_and_material_records_bit_identical": True,
            },
            sort_keys=True,
        ),
    )


def test_native_fluorescence_branch_probabilities_and_vacancy_threshold(
    warp_device: str, record_testsuite_property: Callable[[str, object], None]
) -> None:
    provider = pytest.importorskip("xraylib")
    table = xraylib_atomic_scattering(13, (0.0, 0.1, 1.0, 10.0))
    lines = xraylib_fluorescence_lines(13, (1.0, 10.0), lines=("KL2_LINE", "KL3_LINE"))
    physical = PhysicalTransport((table,), (0.0, 0.0), fluorescence=(lines,), cutoff_kev=1.0)
    workspace = _workspace(
        warp_device,
        physical=physical,
        energies=(1.0, 10.0),
        absorption=(1.0, 1.0),
        scattering=(0.0, 0.0),
    )
    count = 65536
    values = wp.empty(count, dtype=wp.vec4d, device=warp_device)
    wp.launch(
        _sample_fluorescence,
        dim=count,
        device=warp_device,
        inputs=[10.0, wp.vec2d(0.0, 1.0), workspace._parameters.physics, values],
    )
    observed = values.numpy()
    probabilities = np.array(
        [
            (1 - 1 / provider.JumpFactor(13, provider.K_SHELL))
            * provider.FluorYield(13, provider.K_SHELL)
            * provider.RadRate(13, getattr(provider, name))
            for name in ("KL2_LINE", "KL3_LINE")
        ]
    )
    line_counts = np.array([np.count_nonzero(observed[:, 2] == line.energy_kev) for line in lines])
    lower = stats.binom.ppf(1e-6 / 4, count, probabilities)
    upper = stats.binom.ppf(1 - 1e-6 / 4, count, probabilities)
    assert np.all((line_counts >= lower) & (line_counts <= upper))
    assert np.all(
        (observed[:, 2] == 0)
        | (observed[:, 2] == lines[0].energy_kev)
        | (observed[:, 2] == lines[1].energy_kev)
    )
    radiative = observed[:, 3] == 0
    assert np.all(abs(observed[radiative, 0]) <= 1)
    assert np.all((observed[radiative, 1] >= 0) & (observed[radiative, 1] < 2 * math.pi))
    wp.launch(
        _sample_fluorescence,
        dim=count,
        device=warp_device,
        inputs=[1.0, wp.vec2d(0.0, 0.0), workspace._parameters.physics, values],
    )
    assert np.all(values.numpy()[:, 2] == 0)
    record_testsuite_property(
        "Native-fluorescence-branch-reference",
        json.dumps(
            {
                "histories": count,
                "conditional_line_probabilities": probabilities.tolist(),
                "line_counts": line_counts.tolist(),
                "below_vacancy_threshold_emission_count": 0,
                "binomial_family_alpha": 1e-6,
            },
            sort_keys=True,
        ),
    )


def test_checked_woodcock_rescues_finite_density_product(warp_device: str) -> None:
    """Checked majorant validation must agree with the resident walker law."""
    with localcontext() as context:
        context.prec = 80
        exact_rate = Decimal.from_float(1.0e200) ** 2 * Decimal.from_float(1.0e-300)
        assert float(exact_rate) < 1.0e101
    provenance = Provenance(
        "analytic finite-product regression; no calibrated atomic data",
        "0" * 64,
        "Apache-2.0",
        "unused bounded angular fixture with zero scattering",
    )
    table = AtomicScatteringTable(1, (0.0, 1.0), (1.0, 1.0), (0.0, 0.0), provenance)
    physical = PhysicalTransport((table,), (0.0,), majorant_mm_inverse=(1.0e101,))
    spec = TransportSpec(
        MaterialGrid((-1.0, -1.0, 0.0), (2.0, 2.0, 1.0), (1, 1, 1), 1),
        PlanarDetector((-1.0, -1.0), (2.0, 2.0), (1, 1), 2.0),
        1.0,
        "analytic finite product and strict majorant; absorption only",
        max_events=256,
    )
    workspace = prepare_transport(
        spec,
        material_ids=wp.array([0], dtype=wp.int32, device=warp_device),
        voxel_density=wp.array([1.0e200], dtype=wp.float64, device=warp_device),
        absorption=wp.array([1.0e-300], dtype=wp.float64, device=warp_device),
        scattering=wp.array([0.0], dtype=wp.float64, device=warp_device),
        max_histories=8,
        physical=physical,
        device=warp_device,
    )
    positions, directions, weights, _ = _source(warp_device, 8)
    density = wp.array([1.0e200], dtype=wp.float64, device=warp_device)
    results = _outputs(warp_device, 8)
    trace_histories(
        positions,
        directions,
        weights,
        density,
        batch=HistoryBatch(79201, 0, 8),
        workspace=workspace,
        **results,
    )
    np.testing.assert_array_equal(results["out_status"].numpy(), 1)
    np.testing.assert_array_equal(results["out_score"].numpy(), 0.0)
    workspace.check_status()

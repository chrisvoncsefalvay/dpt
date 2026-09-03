"""Independent shell provenance, analytical-profile and kinematic references."""

import math
from dataclasses import replace
from itertools import pairwise

import pytest
from scipy.integrate import quad

from dpt.transport.bound_compton import (
    BoundComptonTable,
    ComptonShell,
    bound_compton_joint_pdf,
    bound_compton_joint_scores,
    impulse_angular_weight,
    impulse_energy,
    profile_cdf,
    profile_inverse,
    shell_accessibility,
    xraylib_bound_compton,
)
from dpt.transport.model import ELECTRON_REST_ENERGY_KEV, TransportError
from dpt.transport.sampling_identity import SamplingDesign


@pytest.mark.parametrize("z,total_shells", [(1, 1), (6, 3), (13, 6)])
def test_actual_biggs_configuration_and_provider_values(z: int, total_shells: int) -> None:
    import xraylib

    table = xraylib_bound_compton(z)
    assert len(table.shells) == total_shells
    assert sum(s.occupancy for s in table.shells) == z
    assert table.provider is not None and table.provider.version == "4.3.0"
    assert len(table.provenance.sha256) == 64
    for index, shell in enumerate(table.shells):
        assert shell.profile_zero == xraylib.ComptonProfile_Partial(z, index, 0.0)
        assert shell.binding_kev == xraylib.EdgeEnergy(z, index)
    if z == 6:
        assert table.shells[2].occupancy == 2
        assert xraylib.ElectronConfig(6, xraylib.L2_SHELL) == pytest.approx(2 / 3, rel=1e-6)


def test_missing_real_binding_data_is_rejected_without_omitting_shell() -> None:
    with pytest.raises(TransportError, match="Z=26, occupied shell N1"):
        xraylib_bound_compton(26)


def test_incident_score_rejects_exact_shell_threshold() -> None:
    table = xraylib_bound_compton(13)
    energy = table.shells[0].binding_kev
    outgoing = 0.97 * energy
    assert len(bound_compton_joint_scores(energy, 0.0, outgoing, table)) == 2
    with pytest.raises(TransportError, match=r"incident Compton score.*shell threshold"):
        bound_compton_joint_scores(energy, 0.0, outgoing, table, normalisation_energy_score=0.0)


@pytest.mark.parametrize("height", [0.8, 5.0, 120.0])
def test_infinite_profile_normalisation_and_inverse_against_density_quadrature(
    height: float,
) -> None:
    # Independent integration of the published analytical profile, including
    # both infinite tails, rather than comparison against the production CDF.
    def density(p: float) -> float:
        return (
            height
            * (1 + 2 * height * abs(p))
            * math.exp(-2 * height * abs(p) - 2 * height**2 * p**2)
        )

    assert quad(density, -math.inf, math.inf, epsabs=1e-12)[0] == pytest.approx(1, abs=2e-12)
    for probability in (1e-12, 0.01, 0.2, 0.5, 0.8, 0.99, 1 - 1e-12):
        momentum = profile_inverse(probability, height)
        if momentum <= 0:
            integral = quad(density, -math.inf, momentum, epsabs=1e-13, epsrel=1e-13)[0]
        else:
            integral = 0.5 + quad(density, 0, momentum, epsabs=1e-13, epsrel=1e-13)[0]
        assert integral == pytest.approx(probability, abs=2e-12)
        assert profile_cdf(momentum, height) == pytest.approx(probability, abs=2e-15)


@pytest.mark.parametrize("energy,cosine", [(5.0, -0.9), (80.0, 0.3), (500.0, 0.9), (5000.0, -0.9)])
def test_energy_momentum_constraint_and_zero_momentum_compton_limit(
    energy: float, cosine: float
) -> None:
    for momentum in (-0.8, -0.01, 0.0):
        outgoing = impulse_energy(energy, cosine, momentum, 0.0)
        transfer = math.sqrt(energy**2 + outgoing**2 - 2 * energy * outgoing * cosine)
        # Independent conservation equation for the longitudinal impulse
        # momentum, using the physical photon momenta directly in keV/c.
        residual = outgoing - energy + energy * outgoing * (1 - cosine) / ELECTRON_REST_ENERGY_KEV
        assert residual == pytest.approx(momentum * transfer, abs=2e-11, rel=2e-12)
        assert 0 < outgoing <= energy
    stationary = energy / (1 + energy / ELECTRON_REST_ENERGY_KEV * (1 - cosine))
    assert impulse_energy(energy, cosine, 0, 0) == pytest.approx(stationary, rel=3e-15)


def test_joint_pdf_integrates_to_angular_marginal_and_scores() -> None:
    table = xraylib_bound_compton(13)
    energy, cosine = 80.0, 0.3
    normalisation = (
        2
        * math.pi
        * quad(lambda c: impulse_angular_weight(energy, c, table), -1, 1, epsabs=1e-11)[0]
    )
    endpoints = sorted({0.0, energy, *(energy - s.binding_kev for s in table.shells)})
    integral = sum(
        quad(
            lambda e: bound_compton_joint_pdf(
                energy, cosine, e, table, solid_angle_normalisation=normalisation
            ),
            a,
            b,
            epsabs=1e-11,
            limit=200,
        )[0]
        for a, b in pairwise(endpoints)
    )
    assert integral == pytest.approx(
        impulse_angular_weight(energy, cosine, table) / normalisation, rel=3e-10, abs=1e-12
    )
    outgoing = 72.0
    c_score, e_score = bound_compton_joint_scores(energy, cosine, outgoing, table)

    def log_pdf(c: float, e: float) -> float:
        return math.log(
            bound_compton_joint_pdf(energy, c, e, table, solid_angle_normalisation=normalisation)
        )

    step = 1e-5
    difference_c = (
        log_pdf(cosine - 2 * step, outgoing)
        - 8 * log_pdf(cosine - step, outgoing)
        + 8 * log_pdf(cosine + step, outgoing)
        - log_pdf(cosine + 2 * step, outgoing)
    ) / (12 * step)
    difference_e = (
        log_pdf(cosine, outgoing - 2 * step)
        - 8 * log_pdf(cosine, outgoing - step)
        + 8 * log_pdf(cosine, outgoing + step)
        - log_pdf(cosine, outgoing + 2 * step)
    ) / (12 * step)
    assert c_score == pytest.approx(difference_c, rel=1e-8)
    assert e_score == pytest.approx(difference_e, rel=1e-8)


def test_zero_binding_narrow_profile_recovers_free_electron_angular_law() -> None:
    base = xraylib_bound_compton(1)
    table = BoundComptonTable(1, (ComptonShell("K", 1, 0, 1000),), base.provenance)
    for cosine in (-0.9, 0.0, 0.9):
        energy = 80.0
        ratio = 1 / (1 + energy / ELECTRON_REST_ENERGY_KEV * (1 - cosine))
        exact = ratio**2 * (ratio + 1 / ratio - 1 + cosine**2)
        assert shell_accessibility(energy, cosine, table.shells[0]) == 1
        assert impulse_angular_weight(energy, cosine, table) == pytest.approx(exact, rel=1e-15)
    with pytest.raises(TransportError, match="occupancies"):
        replace(table, shells=(ComptonShell("K", 0.5, 0, 1),))


def test_bound_coordinate_layout_is_complete_disjoint_and_versioned() -> None:
    design = SamplingDesign("scrambled-sobol", 4, 8, max_branches=2, layout="bound-compton-v3")
    seen = set(range(3))
    for branch in range(2):
        prefix = branch * 65536
        for event in range(4):
            roles = [
                (0, 4),
                (1, 1),
                (2, 3),
                (3, 1),
                (0x40000000, 2),
                (0x40000001, 1),
                (0x10000000, 2),
                (0x20000000, 2),
            ]
            roles.extend(
                (namespace + t, 3) for namespace in (0x80000000, 0xC0000000) for t in range(8)
            )
            for domain, width in roles:
                for lane in range(width):
                    dimension = design.dimension(event, prefix + domain, lane)
                    assert dimension not in seen
                    seen.add(dimension)
    assert seen == set(range(design.dimensions))
    legacy = SamplingDesign("stratified", 4, 8)
    with pytest.raises(TransportError, match="bound Compton"):
        legacy.dimension(0, 0x10000000, 0)

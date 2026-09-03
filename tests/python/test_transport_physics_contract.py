"""Physical input contracts using actual optional atomic provider outputs."""

from dataclasses import replace

import pytest

from dpt.transport.model import TransportError
from dpt.transport.physical import (
    AtomicScatteringTable,
    PhysicalTransport,
    xraylib_atomic_scattering,
    xraylib_fluorescence_lines,
)
from dpt.transport.source_spectrum import SourceSpectrum


def test_atomic_provider_retains_actual_data_identity_and_sum_rules() -> None:
    provider = pytest.importorskip("xraylib")
    q = (0.0, 0.01, 0.1, 1.0, 10.0)
    table = xraylib_atomic_scattering(13, q)
    assert table.provider is not None and table.provider.version == "4.3.0"
    assert table.form_factor[0] == 13 and table.scattering_function[0] == 0
    assert table.form_factor[1:] == tuple(provider.FF_Rayl(13, node) for node in q[1:])
    assert table.scattering_function[1:] == tuple(provider.SF_Compt(13, node) for node in q[1:])
    assert len(table.provenance.sha256) == 64 and "BSD" in table.provenance.rights
    with pytest.raises(TransportError, match=r"F\(0\)=Z"):
        replace(table, form_factor=(12.0, *table.form_factor[1:]))
    with pytest.raises(TransportError, match="increase strictly"):
        replace(table, momentum_transfer_inverse_angstrom=(0, 0.01, 0.1, 0.1, 10))
    with pytest.raises(TransportError, match="exceeds"):
        replace(table, scattering_function=(0, 14, 0, 0, 0))


def test_single_vacancy_lines_retain_thresholds_and_provider_branch_probabilities() -> None:
    provider = pytest.importorskip("xraylib")
    lines = xraylib_fluorescence_lines(13, (1.0, 2.0, 10.0), lines=("KL2_LINE", "KL3_LINE"))
    for line, name in zip(lines, ("KL2_LINE", "KL3_LINE"), strict=True):
        assert line.energy_kev == provider.LineEnergy(13, getattr(provider, name))
        assert line.threshold_kev == provider.EdgeEnergy(13, provider.K_SHELL)
        assert line.probabilities[0] == 0
        expected = (
            (1 - 1 / provider.JumpFactor(13, provider.K_SHELL))
            * provider.FluorYield(13, provider.K_SHELL)
            * provider.RadRate(13, getattr(provider, name))
        )
        assert line.probabilities[1:] == pytest.approx((expected, expected), rel=2e-14)
    with pytest.raises(TransportError, match="individual"):
        xraylib_fluorescence_lines(13, (10.0, 20.0), lines=("KA_LINE",))


def test_physical_probability_and_majorant_contracts() -> None:
    pytest.importorskip("xraylib")
    table = xraylib_atomic_scattering(13, (0.0, 0.1, 1.0, 10.0))
    physical = PhysicalTransport((table,), (0.1, 0.01), majorant_mm_inverse=(1.0, 0.1))
    with pytest.raises(TransportError, match="positive"):
        replace(physical, majorant_mm_inverse=(1.0, 0.0))
    with pytest.raises(TransportError, match="sum to one"):
        replace(
            physical,
            spectrum=SourceSpectrum((20.0, 40.0), (0.2, 0.3), table.provenance.source),
        )
    assert isinstance(table, AtomicScatteringTable)

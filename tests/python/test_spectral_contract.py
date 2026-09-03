# pytest.approx has intentionally dynamic third-party stubs.
# pyright: reportUnknownMemberType=false
"""Unit, layout and independent limiting-case specifications; CPU-only imports."""

import hashlib
import math
from dataclasses import replace

import pytest

from dpt.contracts import ContractError
from dpt.materials import Provenance
from dpt.spectral import SpectralSpec, Spectrum
from dpt.validation.spectral import spectral_reference

FIXTURE = Provenance(
    source="analytic: equal two-energy mixture; mu=(0.02,0.04)/mm; energies=(40,80)keV",
    sha256=hashlib.sha256(b"E=(40,80)keV; mu=(0.02,0.04)/mm; f=(0.5,0.5)").hexdigest(),
    rights="Original mathematical fixture; Apache-2.0",
    description="Analytic slab, not tissue data",
)


def test_density_and_bin_fluence_have_different_integration_contracts() -> None:
    density = Spectrum((40.0, 80.0), (2.0, 3.0), "density", FIXTURE, (5.0, 10.0))
    bins = Spectrum((40.0, 80.0), (10.0, 30.0), "bin-fluence", FIXTURE)
    assert density.bin_fluence == bins.bin_fluence
    with pytest.raises(ValueError, match="twice"):
        replace(bins, quadrature_weights_kev=(5.0, 10.0))
    with pytest.raises(ValueError, match="quadrature"):
        replace(density, quadrature_weights_kev=None)


def test_zero_fluence_requires_no_invented_normalised_shape() -> None:
    spectrum = Spectrum((40.0, 80.0), (0.0, 0.0), "bin-fluence", FIXTURE)
    assert spectrum.bin_fluence == (0.0, 0.0)


def test_material_layout_requires_each_input_provenance() -> None:
    spec = SpectralSpec(1, 2, (FIXTURE,), FIXTURE, FIXTURE, "counts", "fixed-density analytic slab")
    with pytest.raises(ValueError, match="per material"):
        replace(spec, materials=2)
    with pytest.raises(ValueError, match="boolean"):
        replace(spec, shared_weights=1)


def test_two_energy_reference_and_slab_derivative_are_independent_closed_forms() -> None:
    reference = spectral_reference(
        (50.0,), (0.02, 0.04), (0.5, 0.5), (1.0, 1.0), materials=1, energies=2
    )
    expected = 0.5 * (math.exp(-1) + math.exp(-2))
    derivative = -0.5 * (0.02 * math.exp(-1) + 0.04 * math.exp(-2))
    assert float(reference.mean[0]) == pytest.approx(expected, rel=1e-14)
    assert float(reference.grad_paths[0]) == pytest.approx(derivative, rel=1e-14)
    assert -math.log(float(reference.mean[0])) < 1.5


def test_zero_weight_and_zero_response_derivatives_are_not_divisions() -> None:
    result = spectral_reference(
        (0.0,), (0.02, 0.04), (0.0, 2.0), (3.0, 0.0), materials=1, energies=2
    )
    assert tuple(map(float, result.mean)) == (0.0,)
    assert tuple(map(float, result.grad_weights)) == (3.0, 0.0)
    assert tuple(map(float, result.grad_response)) == (0.0, 2.0)


def test_range_preserving_reference_retains_weighted_underflow_tail() -> None:
    result = spectral_reference((120.0,), (1.0,), (1e30,), (1e10,), materials=1, energies=1)
    assert float(result.mean[0]) == pytest.approx(1e40 * math.exp(-120), rel=1e-14)
    assert float(result.mean[0]) > 1e-13


@pytest.mark.parametrize("values", [(True,), ("1",), (float("nan"),), (-1.0,)])
def test_spectrum_values_use_shared_contract_errors(values: object) -> None:
    from typing import Any, cast

    with pytest.raises(ContractError):
        Spectrum((40.0,), cast(Any, values), "bin-fluence", FIXTURE)


def test_retained_precision_is_explicit_and_defaults_remain_binary32() -> None:
    spec = SpectralSpec(1, 2, (FIXTURE,), FIXTURE, FIXTURE, "counts", "fixed-density analytic slab")
    assert spec.precision == "float32"
    assert replace(spec, precision="float64").precision == "float64"
    with pytest.raises(ContractError, match="precision"):
        replace(spec, precision="float16")  # type: ignore[arg-type]

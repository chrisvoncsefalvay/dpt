"""CPU-safe recovery chart, identifiability and observation-domain contracts."""

import hashlib
import math
from dataclasses import replace
from typing import Any, cast

import pytest

from dpt.contracts import ContractError, TrialDomainError
from dpt.geometry import DetectorGeometry
from dpt.materials import Provenance
from dpt.objectives import ObjectiveSpec
from dpt.spectral import SpectralSpec
from dpt.spectral_recovery import CalibrationBlock, SpectralPoseProblem, SpectralView
from dpt.volumes import GridSpec

# pytest.approx supplies dynamic third-party stubs.
# pyright: reportUnknownMemberType=false
FIXTURE = Provenance(
    source="analytic: calibration chart definitions in this test",
    sha256=hashlib.sha256(b"gain=2;exposure=3;offset=-4;scale_step=.5;offset_step=2").hexdigest(),
    rights="Original analytic test specification; Apache-2.0",
    description="No measured physical data",
)


def view() -> SpectralView:
    geometry = DetectorGeometry(
        (0.0, 0.0, -10.0), (0.0, 0.0, 10.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 1.0), (1, 1)
    )
    spec = SpectralSpec(1, 1, (FIXTURE,), FIXTURE, FIXTURE, "counts", "analytic fixed basis")
    # CPU contracts do not dereference device data. Actual arrays are required
    # only by preparation, tested separately in the CUDA acceptance suite.
    return SpectralView("view", geometry, spec, None, None, None, None, "group", FIXTURE)


def test_positive_scale_chart_and_analytic_chain_rule() -> None:
    block = CalibrationBlock(
        "group",
        gain=2,
        exposure=3,
        offset=-4,
        fit_scale="exposure",
        fit_offset=True,
        scale_step=0.5,
        offset_step=2,
    )
    coordinates = (math.log(2) / 0.5, 3.0)
    assert block.decode(coordinates) == (2.0, 6.0, 2.0)
    assert block.gradient(coordinates, (10.0, 20.0, 30.0)) == (60.0, 60.0)
    assert block.dimension == 2


def test_unrepresentable_trial_is_a_rejectable_trial_domain_error() -> None:
    block = CalibrationBlock("group", fit_scale="gain")
    for value in (-10000.0, 10000.0, float("nan")):
        with pytest.raises(TrialDomainError):
            block.decode((value,))
    with pytest.raises(ContractError):
        CalibrationBlock("group", fit_scale=cast(Any, "gain+exposure"))


def test_chart_priors_are_explicit_and_not_repeated_per_view() -> None:
    block = CalibrationBlock(
        "group",
        fit_scale="exposure",
        fit_offset=True,
        scale_prior_precision=2,
        offset_prior_precision=3,
    )
    prior = block.prior((4.0, -2.0))
    assert prior.loss == 22.0
    assert prior.gradient == (8.0, -6.0)
    with pytest.raises(ContractError, match="active scale"):
        CalibrationBlock("group", scale_prior_precision=1)


def test_named_groups_must_exist_be_used_and_have_consistent_units() -> None:
    first = view()
    second = replace(first, name="second", spectral=replace(first.spectral, output_unit="keV"))
    with pytest.raises(ContractError, match="identical detector output units"):
        SpectralPoseProblem(
            GridSpec((1, 1, 1), (1.0, 1.0, 1.0)),
            1,
            None,
            FIXTURE,
            (first, second),
            (CalibrationBlock("group"),),
        )
    with pytest.raises(ContractError, match="defined and used"):
        SpectralPoseProblem(
            GridSpec((1, 1, 1), (1.0, 1.0, 1.0)),
            1,
            None,
            FIXTURE,
            (first,),
            (CalibrationBlock("another"),),
        )


def test_poisson_likelihood_cannot_be_applied_after_electronic_offset() -> None:
    measured = replace(view(), objective=ObjectiveSpec("poisson", "counts"))
    with pytest.raises(ContractError, match="zero offset"):
        SpectralPoseProblem(
            GridSpec((1, 1, 1), (1.0, 1.0, 1.0)),
            1,
            None,
            FIXTURE,
            (measured,),
            (CalibrationBlock("group", fit_offset=True),),
        )


def test_log_chart_products_preserve_range_before_final_scaling() -> None:
    block = CalibrationBlock("group", gain=1e30, fit_scale="gain", scale_step=1e300)
    derivative = block.gradient((0.0,), (1e-300, 0.0, 0.0))
    assert derivative[0] == pytest.approx(1e30, rel=2e-6)
    small = CalibrationBlock("group", gain=1e30, fit_scale="gain")
    assert small.gradient((0.0,), (1e-60, 0.0, 0.0))[0] == pytest.approx(1e-30, rel=2e-6, abs=0)


@pytest.mark.parametrize(
    "spectral_wide,objective_wide", [(True, False), (False, True), (True, True)]
)
def test_recovery_adapter_rejects_unsupported_precision_before_device_preparation(
    spectral_wide: bool, objective_wide: bool
) -> None:
    original = view()
    with pytest.raises(ContractError, match="requires float32 spectral and objective precision"):
        replace(
            original,
            spectral=replace(
                original.spectral, precision="float64" if spectral_wide else "float32"
            ),
            objective=replace(
                original.objective, precision="float64" if objective_wide else "float32"
            ),
        )
    # No device arrays are needed to reject unsupported compositions, and the
    # existing explicit binary32 configuration retains its constructor contract.
    assert (
        replace(
            original,
            spectral=replace(original.spectral, precision="float32"),
            objective=replace(original.objective, precision="float32"),
        )
        == original
    )

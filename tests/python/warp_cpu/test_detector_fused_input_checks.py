"""Checked detector calls preserve pre-write and sticky diagnostic boundaries."""

import math
from contextlib import nullcontext

import numpy as np
import pytest
import warp as wp

from dpt._runtime import cpu_test_context
from dpt.aperture import (
    ApertureGeometry,
    aperture_projection_vjp,
    prepare_aperture_projection,
)
from dpt.contracts import ContractError
from dpt.detector import ObservationIdentity, prepare_detector, sample_compound_poisson
from dpt.geometry import RayBundleGeometry, RigidTransform
from dpt.projection import ProjectionSpec
from dpt.volumes import GridSpec


@pytest.mark.parametrize("wide", [False, True])
@pytest.mark.parametrize("shared_scores", [False, True])
@pytest.mark.parametrize(
    "domain,invalid",
    [
        ("rates", -1.0),
        ("rates", math.nan),
        ("rates", math.inf),
        ("rates", 2.0**30),
        ("scores", -1.0),
        ("scores", math.nan),
        ("scores", math.inf),
    ],
)
def test_compound_invalid_inputs_leave_realisation_untouched_and_consume_input_status(
    wide: bool, shared_scores: bool, domain: str, invalid: float, warp_device: str
) -> None:
    with cpu_test_context() if warp_device == "cpu" else nullcontext():
        energies, pixels = 3, 5
        rate_dtype = wp.float64 if wide else wp.float32
        rates_host = np.ones(energies * pixels, dtype=np.float64 if wide else np.float32)
        scores_host = np.ones(energies if shared_scores else energies * pixels, dtype=np.float32)
        (rates_host if domain == "rates" else scores_host)[-1] = invalid
        rates = wp.array(rates_host, dtype=rate_dtype, device=warp_device)
        scores = wp.array(scores_host, dtype=wp.float32, device=warp_device)
        sentinel = np.arange(pixels, dtype=np.float32) + 17.0
        output = wp.array(sentinel, dtype=wp.float32, device=warp_device)
        workspace = prepare_detector(max_pixels=pixels, device=warp_device)
        with pytest.raises(ContractError):
            sample_compound_poisson(
                rates,
                scores,
                energies=energies,
                shared_scores=shared_scores,
                out_signal=output,
                identity=ObservationIdentity(71, 93),
                workspace=workspace,
            )
        np.testing.assert_array_equal(output.numpy(), sentinel)
        # Input validation consumes its own failure. It never turns the sentinel into a draw.
        workspace.check_status()


@pytest.mark.parametrize("wide", [False, True])
@pytest.mark.parametrize("invalid", [-1.0, math.nan, math.inf])
def test_zero_pixel_compound_still_checks_supplied_shared_scores(
    wide: bool, invalid: float, warp_device: str
) -> None:
    with cpu_test_context() if warp_device == "cpu" else nullcontext():
        workspace = prepare_detector(max_pixels=0, device=warp_device)
        rates = wp.array([], dtype=wp.float64 if wide else wp.float32, device=warp_device)
        scores = wp.array([1.0, invalid, 1.0], dtype=wp.float32, device=warp_device)
        output = wp.array([], dtype=wp.float32, device=warp_device)
        with pytest.raises(ContractError):
            sample_compound_poisson(
                rates,
                scores,
                energies=3,
                shared_scores=True,
                out_signal=output,
                identity=ObservationIdentity(71, 93),
                workspace=workspace,
            )
        assert output.size == 0
        workspace.check_status()


@pytest.mark.parametrize("wide", [False, True])
@pytest.mark.parametrize("shared_scores", [False, True])
def test_zero_pixel_compound_validates_all_supplied_scores_without_drawing(
    wide: bool, shared_scores: bool, warp_device: str
) -> None:
    with cpu_test_context() if warp_device == "cpu" else nullcontext():
        workspace = prepare_detector(max_pixels=0, device=warp_device)
        rates = wp.array([], dtype=wp.float64 if wide else wp.float32, device=warp_device)
        scores = wp.array(
            [1.0, 2.0, 3.0] if shared_scores else [], dtype=wp.float32, device=warp_device
        )
        output = wp.array([], dtype=wp.float32, device=warp_device)
        sample_compound_poisson(
            rates,
            scores,
            energies=3,
            shared_scores=shared_scores,
            out_signal=output,
            identity=ObservationIdentity(71, 93),
            workspace=workspace,
        )
        assert output.size == 0
        workspace.check_status()


@pytest.mark.parametrize(
    "beam_value,seed_value",
    [
        (-1.0, -0.5),
        (math.nan, -0.5),
        (math.inf, -0.5),
        (2.0, math.nan),
        (2.0, math.inf),
        (-1.0, math.nan),
    ],
)
def test_aperture_invalid_beam_or_cotangent_preserves_all_owned_outputs_and_sticky_status(
    beam_value: float, seed_value: float, warp_device: str
) -> None:
    with cpu_test_context() if warp_device == "cpu" else nullcontext():
        geometry = ApertureGeometry(
            RayBundleGeometry(((0.0, 0.0, -2.0),), ((0.0, 0.0, 2.0),)),
            (0, 1),
            (1.0,),
        )
        depths = wp.array([11.0], dtype=wp.float64, device=warp_device)
        depth_cotangents = wp.array([13.0], dtype=wp.float64, device=warp_device)
        workspace = prepare_aperture_projection(
            GridSpec((1, 1, 1), (1.0, 1.0, 1.0)),
            geometry,
            ProjectionSpec(17, active_pose=False, precision="float64", integration="cell_gauss"),
            depths=depths,
            depth_cotangents=depth_cotangents,
            device=warp_device,
        )
        field = wp.array([1.0], dtype=wp.float32, device=warp_device)
        pose = wp.array(RigidTransform().packed(), dtype=wp.float64, device=warp_device)
        beam = wp.array([beam_value], dtype=wp.float64, device=warp_device)
        cotangent = wp.array([seed_value], dtype=wp.float64, device=warp_device)
        beam_gradient = wp.array([17.0], dtype=wp.float64, device=warp_device)
        with pytest.raises(ContractError):
            aperture_projection_vjp(
                field,
                pose,
                beam,
                cotangent=cotangent,
                workspace=workspace,
                out_beam=beam_gradient,
            )
        for array, expected in ((depths, 11.0), (depth_cotangents, 13.0), (beam_gradient, 17.0)):
            assert array.numpy().tolist() == [expected]
        with pytest.raises(ContractError):
            workspace.check_status()
        with pytest.raises(ContractError):
            workspace.consume_status()
        workspace.check_status()

"""Independent first-order deterministic Torch images, adjoints and scalar objectives."""

import importlib
import math
from decimal import Decimal, localcontext
from typing import Any

import pytest

from dpt.contracts import ContractError, NumericalError
from dpt.detector import (
    BlurSpec,
    CalibrationSpec,
    SeparableBlurSpec,
    prepare_blur,
    prepare_detector,
)
from dpt.geometry import DetectorGeometry, RigidTransform
from dpt.interop.torch import project_optical_depth
from dpt.objectives import ObjectiveSpec, prepare_objective
from dpt.projection import ProjectionSpec, prepare_projection
from dpt.torch.operators import affine_calibrate, blur, objective, transmit
from dpt.transmission import TransmissionSpec, prepare_transmission
from dpt.validation.projection import integrate_sampled_field
from dpt.volumes import GridSpec
from helpers.torch_detector_oracles import blur_matrix, objective_reference

torch: Any = pytest.importorskip("torch", reason="optional Torch runtime is not installed")
wp: Any = importlib.import_module("warp")


@pytest.mark.parametrize("output", ["transmission", "counts", "log_transmission", "removed"])
@pytest.mark.parametrize("beam_mode", ["device-scalar", "per-pixel"])
def test_transmission_selected_outputs_and_active_beam_closed_forms(output, beam_mode, warp_device):
    depth = torch.tensor([[0, 1e-7, 0.3, 2, 100]], dtype=torch.float32, device=warp_device)
    depth.requires_grad_()
    beam = torch.tensor(
        [300] if beam_mode == "device-scalar" else [[300, 20, 100, 0, 30]],
        dtype=torch.float32,
        device=warp_device,
        requires_grad=True,
    )
    workspace = prepare_transmission(
        TransmissionSpec(beam=beam_mode, active_beam=True), max_pixels=5, device=warp_device
    )
    result = transmit(depth, beam, output=output, workspace=workspace)
    assert result.shape == depth.shape and result.dtype == torch.float32
    assert result.data_ptr() != depth.data_ptr() and result.data_ptr() != beam.data_ptr()
    attenuation = torch.exp(-depth.detach().double())
    values = {
        "transmission": attenuation,
        "counts": beam.detach().double() * attenuation,
        "log_transmission": -depth.detach().double(),
        "removed": -torch.expm1(-depth.detach().double()),
    }
    torch.testing.assert_close(result.double(), values[output], rtol=2e-6, atol=1e-44)
    seed = torch.linspace(-0.4, 0.5, 10, device=warp_device).reshape(1, 10)[:, ::2]
    before = seed.clone()
    result.backward(seed)
    derivative = {
        "transmission": -attenuation,
        "counts": -beam.detach().double() * attenuation,
        "log_transmission": -torch.ones_like(attenuation),
        "removed": attenuation,
    }[output]
    torch.testing.assert_close(
        depth.grad.double(), seed.double() * derivative, rtol=3e-6, atol=1e-44
    )
    beam_gradient = (
        seed.double() * attenuation if output == "counts" else torch.zeros_like(attenuation)
    )
    if beam_mode == "device-scalar":
        beam_gradient = beam_gradient.sum().reshape(beam.shape)
    torch.testing.assert_close(beam.grad.double(), beam_gradient, rtol=3e-6, atol=1e-44)
    assert torch.equal(seed, before)


@pytest.mark.parametrize("beam_mode", ["none", "scalar"])
def test_transmission_fixed_python_beam_and_output_policy(beam_mode, warp_device):
    depth = torch.tensor([0.1, 0.8], device=warp_device, requires_grad=True)
    workspace = prepare_transmission(
        TransmissionSpec(beam=beam_mode), max_pixels=2, device=warp_device
    )
    beam = None if beam_mode == "none" else 25.0
    output = "transmission" if beam is None else "counts"
    result = transmit(depth, beam, output=output, workspace=workspace)
    result.sum().backward()
    torch.testing.assert_close(depth.grad, -result.detach())
    if beam is None:
        with pytest.raises(ContractError, match="beam"):
            transmit(depth, output="counts", workspace=workspace)


@pytest.mark.parametrize("active_scale", ["gain", "exposure"])
def test_affine_calibration_shared_parameter_derivatives(active_scale, warp_device):
    mean = torch.arange(1, 7, dtype=torch.float32, device=warp_device).reshape(2, 3)
    mean.requires_grad_()
    gain = torch.tensor(1.7, device=warp_device, requires_grad=active_scale == "gain")
    exposure = torch.tensor([0.4], device=warp_device, requires_grad=active_scale == "exposure")
    offset = torch.tensor(-0.2, device=warp_device, requires_grad=True)
    spec = CalibrationSpec(
        output_unit="detector counts",
        active_gain=active_scale == "gain",
        active_exposure=active_scale == "exposure",
        active_offset=True,
    )
    workspace = prepare_detector(max_pixels=6, device=warp_device)
    result = affine_calibrate(mean, gain, exposure, offset, spec=spec, workspace=workspace)
    expected = (
        gain.detach().double() * exposure.detach().double() * mean.detach().double()
        + offset.detach().double()
    )
    torch.testing.assert_close(result.double(), expected, rtol=2e-6, atol=2e-7)
    seed = torch.tensor([[0.2, -0.3, 0.4], [0.5, 0, -0.7]], device=warp_device)
    result.backward(seed)
    torch.testing.assert_close(mean.grad, seed * gain.detach() * exposure.detach())
    shared = (seed.double() * mean.detach().double()).sum()
    if active_scale == "gain":
        torch.testing.assert_close(
            gain.grad.double(), shared * exposure.detach().double().reshape(())
        )
        assert exposure.grad is None
    else:
        torch.testing.assert_close(
            exposure.grad.double(), (shared * gain.detach().double()).reshape(1)
        )
        assert gain.grad is None
    torch.testing.assert_close(offset.grad.double(), seed.double().sum())


def test_affine_calibration_fixed_pixel_factors_and_zero_exposure(warp_device):
    mean = torch.tensor(
        [[2, 3], [4, 5]], dtype=torch.float32, device=warp_device, requires_grad=True
    )
    gain = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32, device=warp_device)
    exposure = torch.tensor(0.0, device=warp_device, requires_grad=True)
    offset = torch.tensor([[0, -1], [2, -3]], dtype=torch.float32, device=warp_device)
    spec = CalibrationSpec(shared_gain=False, shared_offset=False, active_exposure=True)
    workspace = prepare_detector(max_pixels=4, device=warp_device)
    result = affine_calibrate(mean, gain, exposure, offset, spec=spec, workspace=workspace)
    torch.testing.assert_close(result, offset)
    result.sum().backward()
    torch.testing.assert_close(mean.grad, torch.zeros_like(mean))
    torch.testing.assert_close(exposure.grad, (gain * mean.detach()).sum())


@pytest.mark.parametrize("separable", [False, True])
def test_blur_forward_and_reverse_match_independent_zero_extended_matrix(separable, warp_device):
    spec = (
        SeparableBlurSpec(3, 4, (0.1, 0.3, 0.2), (0.2, 0.4, 0.1))
        if separable
        else BlurSpec(3, 4, 3, 3, (0.01, 0.02, 0.03, 0.05, 0.3, 0.1, 0.02, 0.04, 0.03))
    )
    matrix = torch.tensor(blur_matrix(spec), dtype=torch.float64, device=warp_device)
    signal = torch.linspace(-0.7, 2, 12, device=warp_device).reshape(3, 4).requires_grad_()
    workspace = prepare_blur(spec, device=warp_device)
    result = blur(signal, workspace=workspace)
    torch.testing.assert_close(
        result.double().reshape(-1),
        matrix @ signal.detach().double().reshape(-1),
        rtol=3e-6,
        atol=1e-7,
    )
    seed = torch.linspace(-0.8, 0.3, 24, device=warp_device).reshape(3, 8)[:, ::2]
    result.backward(seed)
    torch.testing.assert_close(
        signal.grad.double().reshape(-1), matrix.T @ seed.double().reshape(-1), rtol=3e-6, atol=1e-7
    )
    assert matrix.sum(dim=1)[0] < matrix.sum(dim=1)[5]


@pytest.mark.parametrize("kind", ["squared_error", "poisson"])
@pytest.mark.parametrize("precision", ["float32", "float64"])
@pytest.mark.parametrize("reduction", ["sum", "mean"])
def test_objective_weight_mask_units_reduction_and_prediction_gradient(
    kind, precision, reduction, warp_device
):
    dtype = getattr(torch, precision)
    prediction = torch.tensor(
        [[2.5, 0, math.nan], [4, 5, 6]], dtype=dtype, device=warp_device, requires_grad=True
    )
    observation = torch.tensor(
        [[2, 0, math.nan], [1, 4, 2]], dtype=torch.float32, device=warp_device
    )
    weights = torch.tensor(
        [[0.5, 0.7, math.nan], [1, 0, 1.2]], dtype=torch.float32, device=warp_device
    )
    valid = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.uint8, device=warp_device)
    workspace = prepare_objective(
        ObjectiveSpec(
            kind=kind,
            domain="counts" if kind == "poisson" else "signal",
            reduction=reduction,
            weighted=True,
            masked=True,
            precision=precision,
        ),
        max_pixels=6,
        device=warp_device,
    )
    expected_loss, expected_gradient = objective_reference(
        prediction.detach().cpu().double().numpy(),
        observation.cpu().double().numpy(),
        weights.cpu().double().numpy(),
        valid.cpu().numpy(),
        kind,
        reduction,
    )
    result = objective(prediction, observation, weights=weights, valid=valid, workspace=workspace)
    assert result.shape == () and result.dtype == torch.float64
    assert result.item() == pytest.approx(expected_loss, rel=2e-12)
    result.backward(torch.tensor(-0.25, dtype=torch.float64, device=warp_device))
    torch.testing.assert_close(
        prediction.grad.double(),
        torch.tensor(-0.25 * expected_gradient, dtype=torch.float64, device=warp_device),
        rtol=3e-6 if precision == "float32" else 2e-12,
        atol=0,
    )
    assert observation.grad is None and weights.grad is None


@pytest.mark.parametrize("prediction,weight,cotangent", [(1e-30, 1e-30, 1e30), (1e20, 1e20, 1e-30)])
def test_fp64_loss_cotangent_rescues_tiny_or_large_fp32_prediction_seed(
    prediction, weight, cotangent, warp_device
):
    value = torch.tensor([prediction], device=warp_device, requires_grad=True)
    observation = torch.zeros(1, device=warp_device)
    weights = torch.tensor([weight], device=warp_device)
    seed = torch.tensor(cotangent, dtype=torch.float64, device=warp_device)
    workspace = prepare_objective(ObjectiveSpec(weighted=True), max_pixels=1, device=warp_device)
    result = objective(value, observation, weights=weights, workspace=workspace)
    result.backward(seed)
    with localcontext() as context:
        context.prec = 150
        expected = (
            Decimal.from_float(value.item())
            * Decimal.from_float(weights.item())
            * Decimal.from_float(seed.item())
        )
    assert value.grad.item() == pytest.approx(float(expected), rel=2e-6, abs=0)


def test_zero_and_invalid_objective_cotangents_and_final_gradient_range(warp_device):
    workspace = prepare_objective(ObjectiveSpec(weighted=True), max_pixels=1, device=warp_device)
    observation = torch.zeros(1, device=warp_device)
    weights = torch.tensor([1e20], device=warp_device)
    value = torch.tensor([1e20], device=warp_device, requires_grad=True)
    objective(value, observation, weights=weights, workspace=workspace).backward(
        torch.tensor(0.0, dtype=torch.float64, device=warp_device)
    )
    assert value.grad.item() == 0 and not torch.signbit(value.grad).item()
    for seed, error in [(math.inf, ContractError), (math.nan, ContractError), (1, NumericalError)]:
        value = value.detach().clone().requires_grad_()
        with pytest.raises(error):
            objective(value, observation, weights=weights, workspace=workspace).backward(
                torch.tensor(seed, dtype=torch.float64, device=warp_device)
            )
        assert value.grad is None


@pytest.mark.parametrize("input_name", ["prediction", "observation", "weights"])
def test_saved_objective_inputs_version_checked_before_backward(input_name, warp_device):
    prediction = torch.tensor([2, 3], dtype=torch.float32, device=warp_device, requires_grad=True)
    observation = torch.tensor([1, 2], dtype=torch.float32, device=warp_device)
    weights = torch.ones(2, device=warp_device)
    workspace = prepare_objective(ObjectiveSpec(weighted=True), max_pixels=2, device=warp_device)
    result = objective(prediction, observation, weights=weights, workspace=workspace)
    with torch.no_grad():
        {"prediction": prediction, "observation": observation, "weights": weights}[input_name].add_(
            0.1
        )
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        result.backward()
    assert prediction.grad is None


@pytest.mark.parametrize(
    "binding", ["optical_depth", "open_beam", "gain", "observation", "weights"]
)
def test_native_active_and_fixed_input_policies_are_preserved(binding, warp_device):
    value = torch.ones(2, device=warp_device, requires_grad=binding == "optical_depth")
    if binding in ("optical_depth", "open_beam"):
        workspace = prepare_transmission(
            TransmissionSpec(beam="per-pixel", active_L=False), max_pixels=2, device=warp_device
        )
        beam = torch.ones(2, device=warp_device, requires_grad=binding == "open_beam")
        with pytest.raises(ContractError, match="fixed"):
            transmit(value, beam, workspace=workspace)
    elif binding == "gain":
        gain = torch.tensor(1.0, device=warp_device, requires_grad=True)
        with pytest.raises(ContractError, match="fixed"):
            affine_calibrate(
                value,
                gain,
                torch.tensor(1.0, device=warp_device),
                torch.tensor(0.0, device=warp_device),
                spec=CalibrationSpec(),
                workspace=prepare_detector(max_pixels=2, device=warp_device),
            )
    else:
        observation = torch.ones(2, device=warp_device, requires_grad=binding == "observation")
        weights = torch.ones(2, device=warp_device, requires_grad=binding == "weights")
        with pytest.raises(ContractError, match="fixed"):
            objective(
                value,
                observation,
                weights=weights,
                workspace=prepare_objective(
                    ObjectiveSpec(weighted=True), max_pixels=2, device=warp_device
                ),
            )


def test_operator_metadata_domain_and_higher_derivative_failures(warp_device):
    workspace = prepare_transmission(max_pixels=2, device=warp_device)
    depth = torch.tensor([0.2, 0.4], device=warp_device, requires_grad=True)
    with pytest.raises(ContractError, match="float32"):
        transmit(depth.double(), workspace=workspace)
    with pytest.raises(ContractError, match="boolean"):
        invalid_validate: Any = 1
        transmit(depth, workspace=workspace, validate=invalid_validate)
    with pytest.raises(ContractError):
        transmit(torch.tensor([-0.2, 0.4], device=warp_device), workspace=workspace)
    workspace.clear_status()
    with pytest.raises(ContractError, match="first-order"):
        torch.autograd.grad(transmit(depth, workspace=workspace).sum(), depth, create_graph=True)


@pytest.mark.parametrize("operation", ["transmission", "calibration", "blur", "objective"])
def test_detector_functions_reject_higher_derivatives(operation, warp_device):
    value = torch.tensor([[0.2, 0.4]], device=warp_device, requires_grad=True)
    if operation == "transmission":
        result = transmit(value, workspace=prepare_transmission(max_pixels=2, device=warp_device))
    elif operation == "calibration":
        result = affine_calibrate(
            value,
            torch.tensor(1.0, device=warp_device),
            torch.tensor(0.5, device=warp_device),
            torch.tensor(0.0, device=warp_device),
            spec=CalibrationSpec(),
            workspace=prepare_detector(max_pixels=2, device=warp_device),
        )
    elif operation == "blur":
        result = blur(
            value,
            workspace=prepare_blur(BlurSpec(1, 2, 1, 1, (1.0,)), device=warp_device),
        )
    else:
        result = objective(
            value,
            torch.zeros_like(value),
            workspace=prepare_objective(ObjectiveSpec(), max_pixels=2, device=warp_device),
        )
    with pytest.raises(ContractError, match="first-order"):
        torch.autograd.grad(result.sum(), value, create_graph=True)


@pytest.mark.parametrize("binding", ["depth", "beam", "mean", "gain", "blur-signal"])
def test_retained_detector_inputs_including_fixed_factors_detect_mutation(binding, warp_device):
    value = torch.tensor([[0.2, 0.4]], device=warp_device, requires_grad=True)
    target = value
    if binding in ("depth", "beam"):
        beam = torch.full_like(value, 10.0)
        if binding == "beam":
            target = beam
        result = transmit(
            value,
            beam,
            output="counts",
            workspace=prepare_transmission(
                TransmissionSpec(beam="per-pixel"), max_pixels=2, device=warp_device
            ),
        )
    elif binding in ("mean", "gain"):
        gain = torch.tensor(1.0, device=warp_device)
        if binding == "gain":
            target = gain
        result = affine_calibrate(
            value,
            gain,
            torch.tensor(0.5, device=warp_device),
            torch.tensor(0.0, device=warp_device),
            spec=CalibrationSpec(),
            workspace=prepare_detector(max_pixels=2, device=warp_device),
        )
    else:
        result = blur(
            value,
            workspace=prepare_blur(BlurSpec(1, 2, 1, 1, (1.0,)), device=warp_device),
        )
    with torch.no_grad():
        target.add_(0.1)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        result.sum().backward()
    assert value.grad is None


@pytest.mark.parametrize("cotangent", [0.0, 1.0])
def test_retained_fp64_seed_workspace_does_not_hide_original_pending_error(cotangent, warp_device):
    prediction = torch.tensor([2.0], device=warp_device, requires_grad=True)
    observation = torch.zeros_like(prediction)
    workspace = prepare_objective(ObjectiveSpec(), max_pixels=1, device=warp_device)
    result = objective(prediction, observation, workspace=workspace)
    invalid = wp.array([math.nan], dtype=wp.float32, device=warp_device)
    with pytest.raises(ContractError):
        workspace.validate_observation(invalid)
    with pytest.raises(ContractError):
        result.backward(torch.tensor(cotangent, dtype=torch.float64, device=warp_device))
    with pytest.raises(ContractError):
        workspace.check_status()
    assert prediction.grad is None


def test_two_outstanding_detector_chains_share_workspaces_without_seed_reuse(warp_device):
    transmission = prepare_transmission(
        TransmissionSpec(beam="per-pixel", active_beam=True), max_pixels=6, device=warp_device
    )
    detector = prepare_detector(max_pixels=6, device=warp_device)
    blur_spec = BlurSpec(2, 3, 1, 3, (0.1, 0.5, 0.2))
    spreading = prepare_blur(blur_spec, device=warp_device)
    loss_workspace = prepare_objective(ObjectiveSpec(), max_pixels=6, device=warp_device)
    depths = [
        torch.tensor([[0.1, 0.3, 0.5], [0.2, 0.7, 1.1]], device=warp_device, requires_grad=True),
        torch.tensor([[0.8, 0.4, 0.2], [1, 0.1, 0.6]], device=warp_device, requires_grad=True),
    ]
    beam = torch.tensor(
        [[10, 20, 30], [40, 50, 60]], dtype=torch.float32, device=warp_device, requires_grad=True
    )
    gain, exposure, offset = (torch.tensor(v, device=warp_device) for v in (1.2, 0.7, 0.1))
    observed = torch.full((2, 3), 5.0, device=warp_device)
    matrix = torch.tensor(blur_matrix(blur_spec), dtype=torch.float64, device=warp_device)
    losses, expected_depths, expected_beam = [], [], torch.zeros_like(beam, dtype=torch.float64)
    for depth in depths:
        counts = transmit(depth, beam, output="counts", workspace=transmission)
        calibrated = affine_calibrate(
            counts,
            gain,
            exposure,
            offset,
            spec=CalibrationSpec(output_unit="detector counts"),
            workspace=detector,
        )
        spread = blur(calibrated, workspace=spreading)
        losses.append(objective(spread, observed, workspace=loss_workspace))
        image_seed = matrix.T @ (spread.detach().double() - observed.double()).reshape(-1)
        mean_seed = image_seed.reshape(2, 3) * gain.double() * exposure.double()
        expected_depths.append(-mean_seed * counts.detach().double())
        expected_beam += mean_seed * torch.exp(-depth.detach().double())
    (losses[1] + losses[0]).backward()
    for depth, expected in zip(depths, expected_depths, strict=True):
        torch.testing.assert_close(depth.grad.double(), expected, rtol=4e-6, atol=2e-6)
    torch.testing.assert_close(beam.grad.double(), expected_beam, rtol=4e-6, atol=2e-6)


def test_projection_to_detector_objective_chain_matches_independent_matrix(warp_device):
    grid = GridSpec((2, 2, 2), (0.8, 1.2, 0.9), (-0.5, -0.5, -0.5))
    geometry = DetectorGeometry(
        (-12, 0.15, 0.11), (12, -0.3, -0.3), (0, 1, 0), (0, 0, 1), (0.13, 0.17), (2, 3)
    )
    projection = prepare_projection(
        grid, geometry, ProjectionSpec(31, active_volume=True), device=warp_device
    )
    volume = torch.linspace(0.1, 0.3, 8, device=warp_device).reshape(grid.shape).requires_grad_()
    pose = torch.tensor(RigidTransform().packed(), dtype=torch.float64, device=warp_device)
    columns = [
        [
            integrate_sampled_field(
                [float(index == voxel) for index in range(8)],
                grid,
                geometry,
                RigidTransform(),
                row,
                column,
                midpoint_samples=31,
            )
            for row in range(2)
            for column in range(3)
        ]
        for voxel in range(8)
    ]
    matrix = torch.tensor(columns, dtype=torch.float64, device=warp_device).T
    depth = project_optical_depth(volume, pose, workspace=projection)
    transmission = prepare_transmission(
        TransmissionSpec(beam="scalar"), max_pixels=6, device=warp_device
    )
    counts = transmit(depth, 20.0, output="counts", workspace=transmission)
    detector = prepare_detector(max_pixels=6, device=warp_device)
    gain, exposure, offset = (torch.tensor(value, device=warp_device) for value in (1.2, 0.7, 0.1))
    calibrated = affine_calibrate(
        counts,
        gain,
        exposure,
        offset,
        spec=CalibrationSpec(output_unit="detector counts"),
        workspace=detector,
    )
    blur_spec = BlurSpec(2, 3, 1, 3, (0.1, 0.5, 0.2))
    spreading = prepare_blur(blur_spec, device=warp_device)
    spread = blur(calibrated, workspace=spreading)
    observation = torch.ones_like(spread)
    loss = objective(
        spread,
        observation,
        workspace=prepare_objective(ObjectiveSpec(), max_pixels=6, device=warp_device),
    )
    loss.backward()
    spread_matrix = torch.tensor(blur_matrix(blur_spec), dtype=torch.float64, device=warp_device)
    image_seed = spread_matrix.T @ (spread.detach().double() - observation.double()).reshape(-1)
    depth_seed = (
        -image_seed * gain.double() * exposure.double() * counts.detach().double().reshape(-1)
    )
    expected = (matrix.T @ depth_seed).reshape(grid.shape)
    torch.testing.assert_close(volume.grad.double(), expected, rtol=1e-5, atol=2e-6)

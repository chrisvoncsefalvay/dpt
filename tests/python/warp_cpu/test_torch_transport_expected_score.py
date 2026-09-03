"""Torch score adapters against analytic expectation and independent batch laws."""

import importlib
import math
from contextlib import nullcontext
from dataclasses import replace
from decimal import Decimal, localcontext

import numpy as np
import pytest

from dpt._runtime import cpu_test_context
from dpt.contracts import ContractError
from dpt.interop.torch_transport import ExpectedTransportEstimator, TransportAutogradSpec
from dpt.transport.model import IncompleteHistoryError
from dpt.transport.rng import HistoryBatch
from helpers.torch_expected_transport_inputs import (
    analytic,
    coordinates,
    evaluate,
    layer_problem,
    point_source_problem,
    point_source_reference,
)

torch = pytest.importorskip("torch", reason="optional Torch runtime is not installed")
wp = importlib.import_module("warp")


@pytest.fixture(autouse=True)
def _native_context(warp_device):
    with cpu_test_context() if warp_device == "cpu" else nullcontext():
        yield


@pytest.mark.parametrize("estimator", ["analogue", "continuous-absorption"])
@pytest.mark.parametrize("scoring", ["photon-count", "energy-kev"])
@pytest.mark.parametrize("amplitude", [0.0, 0.7])
@pytest.mark.parametrize("flat", [False, True])
def test_expected_signal_and_all_scores_match_independent_layers(
    estimator, scoring, amplitude, flat, warp_device
):
    p = layer_problem(
        torch, wp, warp_device, estimator=estimator, scoring=scoring, amplitude=amplitude, flat=flat
    )
    if estimator == "analogue":
        p.estimator = ExpectedTransportEstimator(
            HistoryBatch(7315, 0, 4096), HistoryBatch(8427, 0, 8192)
        )
    expected, gradients = analytic(p, -0.4)
    result = evaluate(p)
    assert result.signal.shape == (1, 1) and result.signal.dtype == torch.float64
    assert result.estimator.forward_batch.source_identity is not None
    result.signal.backward(torch.full_like(result.signal, -0.4))
    if estimator == "analogue":
        unit = p.spec.energy_kev if scoring == "energy-kev" else 1.0
        survival = float(gradients[2]) / (-0.4 * p.source.weight * unit)

        def bernstein(count):
            # Exact independent Bernoulli survival law, with two-sided tail < 1e-9.
            log_tail = math.log(2e9)
            return math.sqrt(2 * survival * (1 - survival) * log_tail / count) + 2 * log_tail / (
                3 * count
            )

        assert abs(result.signal.item() - expected) <= (
            amplitude * p.source.weight * unit * bernstein(p.estimator.forward_batch.count) + 3e-13
        )
    else:
        np.testing.assert_allclose(
            result.signal.detach().cpu().numpy(), [[expected]], rtol=3e-13, atol=3e-13
        )
    for value, target in zip(coordinates(p), gradients, strict=True):
        assert value.grad.dtype == torch.float64 and value.grad.shape == value.shape
        if estimator == "analogue":
            allowed = (
                np.abs(target) * bernstein(p.estimator.derivative_batch.count) / survival + 3e-13
            )
            assert np.all(np.abs(value.grad.cpu().numpy() - target) <= allowed)
        else:
            np.testing.assert_allclose(value.grad.cpu().numpy(), target, rtol=3e-13, atol=3e-13)


@pytest.mark.parametrize(
    "active",
    [
        (),
        ("voxel",),
        ("material",),
        ("amplitude",),
        ("voxel", "material"),
        ("voxel", "amplitude"),
        ("material", "amplitude"),
        ("voxel", "material", "amplitude"),
    ],
)
def test_selected_input_charts_only_receive_their_scores(active, warp_device):
    p = layer_problem(torch, wp, warp_device, active=active)
    expected, gradients = analytic(p)
    result = evaluate(p).signal
    assert result.requires_grad == bool(active)
    if active:
        result.sum().backward()
    for name, value, target in zip(
        ("voxel", "material", "amplitude"), coordinates(p), gradients, strict=True
    ):
        assert (value.grad is not None) == (name in active)
        if value.grad is not None:
            np.testing.assert_allclose(value.grad.cpu().numpy(), target, rtol=3e-13, atol=3e-13)
    assert result.item() == pytest.approx(expected, rel=3e-13)


def test_material_groups_and_masked_voxel_material_entries(warp_device):
    p = layer_problem(
        torch,
        wp,
        warp_device,
        materials=6,
        layers=6,
        parameters=TransportAutogradSpec(active_voxels=(0, 2, 5), active_materials=(5, 1, 4, 0, 2)),
    )
    expected, gradients = analytic(p, 0.6)
    result = evaluate(p).signal
    result.backward(torch.full_like(result, 0.6))
    assert result.item() == pytest.approx(expected, rel=3e-13)
    for value, target in zip(coordinates(p), gradients, strict=True):
        np.testing.assert_allclose(value.grad.cpu().numpy(), target, rtol=3e-13, atol=3e-13)


def test_deep_attenuation_contracts_cotangent_before_rounding_score_columns(warp_device):
    p = layer_problem(torch, wp, warp_device)
    voxel = p.log_voxel_density.detach().exp().cpu().numpy().reshape(-1)
    material = p.log_material_density.detach().exp().cpu().numpy()
    initial_depth = sum(
        p.absorption[m] * material[m] * voxel[index] * 0.6 for index, m in enumerate(p.ids)
    )
    with torch.no_grad():
        p.log_material_density.add_(math.log(1000 / initial_depth))
    material = p.log_material_density.detach().exp().cpu().numpy()
    with localcontext() as context:
        context.prec = 80
        depths = [
            Decimal.from_float(p.absorption[m])
            * Decimal.from_float(float(material[m]))
            * Decimal.from_float(float(voxel[index]))
            * Decimal.from_float(0.6)
            for index, m in enumerate(p.ids)
        ]
        base = (
            Decimal.from_float(p.source.weight)
            * (-sum(depths, Decimal(0))).exp()
            * Decimal("1e300")
        )
        mean = base * Decimal.from_float(float(p.source_amplitude.item()))
        gradients = (
            np.array([float(-depth * mean) for depth in depths]).reshape(p.log_voxel_density.shape),
            np.array(
                [
                    float(
                        -sum(depth for depth, m in zip(depths, p.ids, strict=True) if m == selected)
                        * mean
                    )
                    for selected in range(material.size)
                ]
            ),
            np.array(float(base)),
        )
    signal = evaluate(p).signal
    assert signal.item() == 0.0
    signal.backward(torch.full_like(signal, 1e300))
    for value, target in zip(coordinates(p), gradients, strict=True):
        assert np.all(np.abs(target) > 0)
        np.testing.assert_allclose(value.grad.cpu().numpy(), target, rtol=2e-10, atol=0)


def test_outstanding_forwards_keep_separate_native_field_and_source_snapshots(warp_device):
    p = layer_problem(torch, wp, warp_device)
    first = evaluate(p)
    _, first_grad = analytic(p, -0.3)
    previous = coordinates(p)
    p.log_voxel_density = (p.log_voxel_density.detach() + 0.2).requires_grad_()
    p.log_material_density = (p.log_material_density.detach() - 0.1).requires_grad_()
    p.source_amplitude = torch.tensor(
        1.1, dtype=torch.float64, device=warp_device, requires_grad=True
    )
    p.estimator = ExpectedTransportEstimator(HistoryBatch(7315, 30, 13), HistoryBatch(8427, 40, 19))
    second = evaluate(p)
    _, second_grad = analytic(p, 0.7)
    assert first.signal.data_ptr() != second.signal.data_ptr()
    second.signal.backward(torch.full_like(second.signal, 0.7))
    first.signal.backward(torch.full_like(first.signal, -0.3))
    for value, target in zip(previous, first_grad, strict=True):
        np.testing.assert_allclose(value.grad.cpu().numpy(), target, rtol=3e-13, atol=3e-13)
    for value, target in zip(coordinates(p), second_grad, strict=True):
        np.testing.assert_allclose(value.grad.cpu().numpy(), target, rtol=3e-13, atol=3e-13)


def test_registered_parameters_update_every_new_forward(warp_device):
    from dpt.interop.torch_transport import ExpectedTransport

    p = layer_problem(torch, wp, warp_device)
    parameters = tuple(torch.nn.Parameter(value.detach()) for value in coordinates(p))
    module = ExpectedTransport(*parameters, workspace=p.workspace, estimator=p.estimator)
    assert list(module.parameters()) == list(parameters)
    optimizer = torch.optim.SGD(module.parameters(), lr=0.02)
    seen = []
    for _ in range(3):
        optimizer.zero_grad()
        p.log_voxel_density, p.log_material_density, p.source_amplitude = parameters
        expected, gradients = analytic(p)
        estimate = module()
        seen.append(estimate.signal.item())
        assert seen[-1] == pytest.approx(expected, rel=3e-13)
        estimate.signal.sum().backward()
        for value, target in zip(parameters, gradients, strict=True):
            np.testing.assert_allclose(value.grad.cpu().numpy(), target, rtol=3e-13, atol=3e-13)
        optimizer.step()
    assert len(set(seen)) == 3


@pytest.mark.parametrize("index", [0, 1, 2])
@pytest.mark.parametrize("trainable", [False, True])
def test_mutation_of_any_saved_original_rejects_old_backward(index, trainable, warp_device):
    names = ("voxel", "material", "amplitude")
    p = layer_problem(
        torch,
        wp,
        warp_device,
        active=names if trainable else tuple(name for i, name in enumerate(names) if i != index),
    )
    result = evaluate(p).signal
    with torch.no_grad():
        coordinates(p)[index].add_(0.1)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        result.sum().backward()


@pytest.mark.parametrize("index", [0, 1])
@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf, 1000.0, -1000.0])
def test_nonrepresentable_log_coordinate_is_rejected(index, bad, warp_device):
    p = layer_problem(torch, wp, warp_device)
    with torch.no_grad():
        coordinates(p)[index].reshape(-1)[0] = bad
    with pytest.raises(ContractError, match="finite and positive"):
        evaluate(p)


@pytest.mark.parametrize("bad", [-0.1, math.nan, math.inf])
def test_invalid_direct_source_amplitude_is_rejected(bad, warp_device):
    p = layer_problem(torch, wp, warp_device, amplitude=bad)
    with pytest.raises(ContractError, match="finite and nonnegative"):
        evaluate(p)


@pytest.mark.parametrize("bad", [math.nan, math.inf])
def test_nonfinite_upstream_cotangent_is_rejected(bad, warp_device):
    p = layer_problem(torch, wp, warp_device)
    result = evaluate(p).signal
    with pytest.raises(ContractError, match="cotangent must be finite"):
        result.backward(torch.full_like(result, bad))


def test_first_order_and_native_tape_contracts(warp_device):
    p = layer_problem(torch, wp, warp_device)
    result = evaluate(p).signal
    with pytest.raises(ContractError, match="first-order"):
        result.sum().backward(create_graph=True)
    with wp.Tape(), pytest.raises(ContractError, match="ambient tape"):
        evaluate(p)


@pytest.mark.parametrize("index", [0, 1, 2])
def test_fp64_coordinate_dtype_is_explicit(index, warp_device):
    p = layer_problem(torch, wp, warp_device)
    name = ("log_voxel_density", "log_material_density", "source_amplitude")[index]
    setattr(p, name, coordinates(p)[index].detach().float())
    with pytest.raises(ContractError, match="contiguous FP64"):
        evaluate(p)


@pytest.mark.parametrize("index", [0, 1])
def test_noncontiguous_density_coordinates_are_rejected(index, warp_device):
    p = layer_problem(torch, wp, warp_device, flat=True)
    name = ("log_voxel_density", "log_material_density")[index]
    value = coordinates(p)[index].detach().repeat_interleave(2)[::2].requires_grad_()
    assert not value.is_contiguous()
    setattr(p, name, value)
    with pytest.raises(ContractError, match="contiguous FP64"):
        evaluate(p)


@pytest.mark.parametrize("index", [0, 1, 2])
def test_coordinate_shape_is_explicit(index, warp_device):
    p = layer_problem(torch, wp, warp_device)
    name = ("log_voxel_density", "log_material_density", "source_amplitude")[index]
    shape = ((2, 2), (1, 2), (1, 1))[index]
    setattr(p, name, coordinates(p)[index].detach().reshape(shape))
    with pytest.raises(ContractError, match=r"must have|must be a scalar"):
        evaluate(p)


def test_noncontiguous_image_cotangent_is_packed_without_changing_scores(warp_device):
    p = layer_problem(torch, wp, warp_device, detector_columns=2)
    expected, gradients = analytic(p, -0.7)
    backing = torch.tensor([[0.2, 9.0, -0.7, 11.0]], dtype=torch.float64, device=warp_device)
    seed = backing[:, ::2]
    assert seed.shape == (1, 2) and not seed.is_contiguous()
    result = evaluate(p).signal
    np.testing.assert_allclose(result.detach().cpu().numpy(), [[0.0, expected]], rtol=3e-13)
    result.backward(seed)
    for value, target in zip(coordinates(p), gradients, strict=True):
        np.testing.assert_allclose(value.grad.cpu().numpy(), target, rtol=3e-13, atol=3e-13)


@pytest.mark.parametrize("index", [0, 1, 2])
def test_fixed_input_charts_reject_trainable_coordinates(index, warp_device):
    keywords = (
        {"active_voxels": ()},
        {"active_materials": ()},
        {"active_source_amplitude": False},
    )[index]
    p = layer_problem(torch, wp, warp_device, parameters=TransportAutogradSpec(**keywords))
    with pytest.raises(ContractError, match="is fixed"):
        evaluate(p)


def test_one_entry_amplitude_shape_and_fixed_buffers_preserve_gradients(warp_device):
    p = layer_problem(torch, wp, warp_device)
    p.source_amplitude = p.source_amplitude.detach().reshape(1).requires_grad_()
    _, gradients = analytic(p, -0.3)
    estimate = evaluate(p)
    estimate.signal.backward(torch.full_like(estimate.signal, -0.3))
    assert p.source_amplitude.grad.shape == (1,)
    np.testing.assert_allclose(p.source_amplitude.grad.cpu().numpy(), [gradients[2]], rtol=3e-13)


def test_point_cone_scores_match_independent_chords_for_each_source_batch(warp_device):
    p = point_source_problem(torch, wp, warp_device)
    expected, gradients = point_source_reference(p, wp, -0.4)
    estimate = evaluate(p)
    assert estimate.estimator.forward_batch.source_identity is not None
    assert estimate.estimator.forward_batch.source_identity.sampler == "point-cone-area-v1"
    assert estimate.signal.item() == pytest.approx(expected, rel=3e-13)
    estimate.signal.backward(torch.full_like(estimate.signal, -0.4))
    for value, target in zip(coordinates(p), gradients, strict=True):
        np.testing.assert_allclose(value.grad.cpu().numpy(), target, rtol=3e-13, atol=3e-13)


def test_sparse_slot_and_incomplete_history_failures_propagate(warp_device):
    p = layer_problem(torch, wp, warp_device, slots=1)
    result = evaluate(p).signal
    with pytest.raises(IncompleteHistoryError, match="slots_per_history"):
        result.sum().backward()
    failed = layer_problem(torch, wp, warp_device, max_crossings=1)
    with pytest.raises(IncompleteHistoryError, match="budget"):
        evaluate(failed)


def test_inference_avoids_score_storage_for_active_parameter_inputs(warp_device):
    p = layer_problem(torch, wp, warp_device, max_bytes=3000)
    with torch.no_grad():
        output = evaluate(p).signal
    assert not output.requires_grad and math.isfinite(output.item())
    with pytest.raises(ContractError, match="max_snapshot_bytes"):
        evaluate(p)


def test_fixed_model_tables_are_owned_after_preparation(warp_device):
    p = layer_problem(torch, wp, warp_device)
    expected, gradients = analytic(p)
    with p.native.context.scope():
        p.native.absorption.fill_(8.0)
        p.native.material_ids.fill_(1)
    result = evaluate(p).signal
    result.sum().backward()
    assert result.item() == pytest.approx(expected, rel=3e-13)
    for value, target in zip(coordinates(p), gradients, strict=True):
        np.testing.assert_allclose(value.grad.cpu().numpy(), target, rtol=3e-13, atol=3e-13)


@pytest.mark.parametrize("origin", ["deterministic", "external-batch"])
def test_declared_upstream_origin_keeps_independent_score_identity(origin, warp_device):
    p = layer_problem(torch, wp, warp_device)
    estimator = replace(
        p.estimator,
        cotangent_origin=origin,
        external_cotangent_batch=HistoryBatch(989, 0, 10).with_library_source(
            "parallel-rectangle-v1"
        )
        if origin == "external-batch"
        else None,
    )
    estimate = evaluate(p, estimator=estimator)
    expected, gradients = analytic(p, 0.4)
    estimate.signal.backward(torch.full_like(estimate.signal, 0.4))
    assert estimate.estimator.cotangent_origin == origin
    assert estimate.signal.item() == pytest.approx(expected, rel=3e-13)
    for value, target in zip(coordinates(p), gradients, strict=True):
        np.testing.assert_allclose(value.grad.cpu().numpy(), target, rtol=3e-13, atol=3e-13)

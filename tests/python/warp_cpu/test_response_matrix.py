"""Independent response-bank physics, derivatives and preparation contracts."""

from __future__ import annotations

import importlib
import math
from dataclasses import replace
from decimal import Decimal, localcontext
from fractions import Fraction
from typing import Any, Literal

import numpy as np
import pytest

from dpt.contracts import ContractError, NumericalError
from dpt.geometry import RayBundleGeometry, RigidTransform
from dpt.projection import ProjectionSpec
from dpt.response_matrix import ResponseMatrixInputs, ResponseMatrixPipeline
from dpt.spectral import SpectralSpec
from dpt.volumes import GridSpec

wp: Any = importlib.import_module("warp")


def factors(device: str, *, shared: bool = True) -> tuple[Any, Any, Any]:
    coefficients = wp.array([[0.125, 0.25]], dtype=wp.float32, device=device)
    weights = wp.array([8.0, 16.0], dtype=wp.float32, device=device)
    if not shared:
        weights = wp.array([[8.0, 12.0], [16.0, 20.0]], dtype=wp.float32, device=device)
    response = wp.array([[0.25, 0.75], [1.0, 0.0]], dtype=wp.float32, device=device)
    return coefficients, weights, response


@pytest.mark.parametrize("shared", [False, True])
def test_response_bank_owns_independent_factors_and_preserves_axes(
    shared: bool, warp_device: str
) -> None:
    spec = SpectralSpec(1, 2, shared_weights=shared, output_unit="expected photons")
    source = factors(warp_device, shared=shared)
    record = ResponseMatrixInputs.from_arrays(spec, *source, pixels=2, device=warp_device)
    expected = [values.numpy().copy() for values in source]
    for values in source:
        values.fill_(float("nan"))
    for values, reference in zip(
        (record.coefficients, record.weights, record.response), expected, strict=True
    ):
        np.testing.assert_array_equal(values.numpy(), reference)
    assert record.spec is spec
    assert record.bands == 2 and record.pixels == 2
    assert record.coefficients.shape == (1, 2)
    assert record.weights.shape == ((2,) if shared else (2, 2))
    assert record.response.shape == (2, 2)
    assert record.storage_bytes == (32 if shared else 40)


@pytest.mark.parametrize("invalid", [-1.0, float("nan"), float("inf")])
def test_response_bank_checks_all_bands_before_use(invalid: float, warp_device: str) -> None:
    coefficients, weights, response = factors(warp_device)
    host = response.numpy()
    host[1, 0] = invalid
    response.assign(host)
    with pytest.raises(ContractError, match="invalid fixed device input"):
        ResponseMatrixInputs.from_arrays(
            SpectralSpec(1, 2), coefficients, weights, response, pixels=2, device=warp_device
        )


def test_response_bank_shape_dtype_and_budget_contracts(warp_device: str) -> None:
    spec = SpectralSpec(1, 2)
    coefficients, weights, response = factors(warp_device)
    with pytest.raises(ContractError, match="explicit shape"):
        ResponseMatrixInputs.from_arrays(
            spec, coefficients, weights, response.flatten(), pixels=2, device=warp_device
        )
    with pytest.raises(ContractError, match="shape"):
        ResponseMatrixInputs.from_arrays(
            spec,
            coefficients,
            weights,
            wp.zeros((2, 3), dtype=wp.float32, device=warp_device),
            pixels=2,
            device=warp_device,
        )
    with pytest.raises(ContractError, match="dtype"):
        ResponseMatrixInputs.from_arrays(
            spec,
            coefficients,
            weights,
            wp.zeros((2, 2), dtype=wp.float64, device=warp_device),
            pixels=2,
            device=warp_device,
        )
    with pytest.raises(ContractError, match="max_owned_bytes"):
        ResponseMatrixInputs.from_arrays(
            spec,
            coefficients,
            weights,
            response,
            pixels=2,
            device=warp_device,
            max_owned_bytes=31,
        )
    with pytest.raises(ContractError, match="shared_response"):
        ResponseMatrixInputs.from_arrays(
            replace(spec, shared_response=False),
            coefficients,
            weights,
            response,
            pixels=2,
            device=warp_device,
        )
    with pytest.raises(TypeError, match="from_arrays"):
        ResponseMatrixInputs()


def make_matrix(
    device: str,
    *,
    precision: Literal["float32", "float64"] = "float64",
    shared: bool = True,
    domain: Literal["linear", "log"] = "linear",
    channels: Literal["serial", "fused"] = "serial",
    bands: int = 3,
) -> tuple[ResponseMatrixPipeline, Any, Any, tuple[Any, Any, Any]]:
    grid = GridSpec((3, 3, 3), (1.0, 1.0, 1.0), (-1.0, -1.0, -1.0))
    geometry = RayBundleGeometry(
        ((-10.0, 0.0, 0.0), (-10.0, 0.0, 0.0)),
        ((10.0, 0.0, 0.0), (10.0, 0.0, 0.0)),
    )
    coefficients = wp.array([[0.125, 0.25], [0.5, 0.75]], dtype=wp.float32, device=device)
    weights = wp.array(
        [8.0, 16.0] if shared else [[8.0, 12.0], [16.0, 20.0]],
        dtype=wp.float32,
        device=device,
    )
    response = wp.array(
        [[0.25, 0.75], [1.0, 0.0], [0.5, 0.125]][:bands], dtype=wp.float32, device=device
    )
    spec = SpectralSpec(
        2,
        2,
        shared_weights=shared,
        active_weights=True,
        active_response=True,
        active_coefficients=True,
        precision=precision,
        output_domain=domain,
    )
    inputs = ResponseMatrixInputs.from_arrays(
        spec, coefficients, weights, response, pixels=2, device=device
    )
    pipeline = ResponseMatrixPipeline(
        grid,
        geometry,
        ProjectionSpec(48, active_pose=True, active_volume=True, precision=precision),
        inputs,
        channel_execution=channels,
    )
    host = np.empty((2, *grid.shape), dtype=np.float32)
    host[0], host[1] = 0.25, 0.5
    fields = wp.array(host, dtype=wp.float32, device=device)
    pose = wp.array(RigidTransform().packed(), dtype=wp.float64, device=device)
    return pipeline, fields, pose, (coefficients, weights, response)


@pytest.mark.parametrize("precision", ["float32", "float64"])
@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("domain", ["linear", "log"])
@pytest.mark.parametrize("channels", ["serial", "fused"])
def test_response_images_and_all_active_vjps_match_independent_uniform_slab(
    precision: Literal["float32", "float64"],
    shared: bool,
    domain: Literal["linear", "log"],
    channels: Literal["serial", "fused"],
    warp_device: str,
) -> None:
    pipeline, fields, pose, factors_ = make_matrix(
        warp_device,
        precision=precision,
        shared=shared,
        domain=domain,
        channels=channels,
    )
    coefficients, weights, response = (array.numpy().astype(np.float64) for array in factors_)
    if shared:
        weights = np.broadcast_to(weights[:, None], (2, 2))
    paths = np.broadcast_to(np.array([0.75, 1.5])[:, None], (2, 2))
    transmission = np.exp(-(coefficients.T @ paths))
    components = response[:, :, None] * weights[None, :, :] * transmission[None, :, :]
    mean = components.sum(axis=1)
    expected_images = np.log(mean) if domain == "log" else mean
    image = pipeline.forward(fields, pose)
    np.testing.assert_allclose(image.numpy(), expected_images, rtol=2e-7, atol=1e-7)
    host_seed = np.array([[1.0, -2.0], [-3.0, 4.0], [0.25, -0.5]], dtype=precision)
    seed = wp.array(host_seed, dtype=getattr(wp, precision), device=warp_device)
    q = host_seed.astype(np.float64) / mean if domain == "log" else host_seed.astype(np.float64)
    gradients = pipeline.vjp(seed)
    expected_paths = -np.einsum("bp,bkp,mk->mp", q, components, coefficients)
    expected_coefficients = -np.einsum("bp,bkp,mp->mk", q, components, paths)
    expected_weights = np.einsum("bp,bk,kp->kp", q, response, transmission)
    if shared:
        expected_weights = expected_weights.sum(axis=1)
    expected_response = np.einsum("bp,kp,kp->bk", q, weights, transmission)
    expected_fields = np.zeros((2, 3, 3, 3), dtype=np.float32)
    expected_fields[:, 1, 1, :] = expected_paths.sum(axis=1)[:, None]
    assert gradients.coefficients is not None and gradients.weights is not None
    assert (
        gradients.response is not None
        and gradients.fields is not None
        and gradients.pose is not None
    )
    np.testing.assert_allclose(
        gradients.coefficients.numpy(), expected_coefficients, rtol=3e-7, atol=1e-12
    )
    np.testing.assert_allclose(gradients.weights.numpy(), expected_weights, rtol=3e-7, atol=1e-12)
    np.testing.assert_allclose(gradients.response.numpy(), expected_response, rtol=3e-7, atol=1e-12)
    np.testing.assert_allclose(gradients.fields.numpy(), expected_fields, rtol=6e-7, atol=2e-7)
    np.testing.assert_allclose(gradients.pose.numpy(), 0.0, atol=1e-10)
    np.testing.assert_array_equal(seed.numpy(), host_seed)


@pytest.mark.parametrize("bands", [1, 3])
@pytest.mark.parametrize("channels", ["serial", "fused"])
def test_material_projection_launches_do_not_grow_with_response_bands(
    bands: int,
    channels: Literal["serial", "fused"],
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
    warp_device: str,
) -> None:
    pipeline, fields, pose, _ = make_matrix(warp_device, bands=bands, channels=channels)
    module: Any = importlib.import_module("dpt.response_matrix")
    material = pipeline._material
    if channels == "fused":
        forward_kernel = material.kernels.get_material_forward(2, True, False)
        reverse_kernel = material.kernels.get_material_vjp(2, True, False, True, True)
    else:
        forward_kernel = material.projection._kernels.get_forward(True, False)
        reverse_kernel = material.projection._kernels.get_vjp(
            True,
            False,
            True,
            double_seed=True,
            cell_gauss=False,
            double_volume_gradient=False,
            mixed=False,
            storage_precision="float32",
        )
    counts = {
        "material_forward_calls": 0,
        "projection_forward_launches": 0,
        "material_reverse_calls": 0,
        "projection_reverse_launches": 0,
        "spectral_forward_calls": 0,
        "spectral_reverse_calls": 0,
        "exact_accumulator_updates": 0,
        "exact_accumulator_finishes": 0,
    }
    native_project, native_reverse, native_launch = (
        module.project_material_paths,
        module.material_projection_vjp,
        wp.launch,
    )
    native_tiled = wp.launch_tiled
    native_signal, native_spectral_vjp = module.spectral_signal, module.spectral_vjp
    accumulator_update = pipeline._kernels.get_accumulate(True)
    accumulator_finish = pipeline._kernels.get_finish(True, True)

    def project(*args: Any, **kwargs: Any) -> Any:
        counts["material_forward_calls"] += 1
        return native_project(*args, **kwargs)

    def reverse(*args: Any, **kwargs: Any) -> Any:
        counts["material_reverse_calls"] += 1
        return native_reverse(*args, **kwargs)

    def launch(*args: Any, **kwargs: Any) -> Any:
        kernel = args[0] if args else kwargs.get("kernel")
        if kernel is forward_kernel:
            counts["projection_forward_launches"] += 1
        elif kernel is accumulator_update:
            counts["exact_accumulator_updates"] += 1
        elif kernel is accumulator_finish:
            counts["exact_accumulator_finishes"] += 1
        return native_launch(*args, **kwargs)

    def launch_tiled(*args: Any, **kwargs: Any) -> Any:
        if (args[0] if args else kwargs.get("kernel")) is reverse_kernel:
            counts["projection_reverse_launches"] += 1
        return native_tiled(*args, **kwargs)

    def signal(*args: Any, **kwargs: Any) -> Any:
        counts["spectral_forward_calls"] += 1
        return native_signal(*args, **kwargs)

    def spectral_reverse(*args: Any, **kwargs: Any) -> Any:
        counts["spectral_reverse_calls"] += 1
        return native_spectral_vjp(*args, **kwargs)

    monkeypatch.setattr(module, "project_material_paths", project)
    monkeypatch.setattr(module, "material_projection_vjp", reverse)
    monkeypatch.setattr(wp, "launch", launch)
    monkeypatch.setattr(wp, "launch_tiled", launch_tiled)
    monkeypatch.setattr(module, "spectral_signal", signal)
    monkeypatch.setattr(module, "spectral_vjp", spectral_reverse)
    pipeline.forward(fields, pose)
    pipeline.vjp(wp.ones((bands, 2), dtype=wp.float64, device=warp_device))
    assert counts == {
        "material_forward_calls": 1,
        "projection_forward_launches": 1 if channels == "fused" else 2,
        "material_reverse_calls": 1,
        "projection_reverse_launches": 1 if channels == "fused" else 2,
        "spectral_forward_calls": bands,
        "spectral_reverse_calls": bands,
        "exact_accumulator_updates": 3 * bands,
        "exact_accumulator_finishes": 3,
    }
    for name, count in counts.items():
        request.node.user_properties.append((name, count))
    request.node.user_properties.append(("response_bands", bands))


def scalar_matrix(
    device: str,
    bands: int,
    *,
    precision: Literal["float32", "float64"] = "float64",
    coefficient: float = 1.0,
    weight: float = 1.0,
    field: float = 0.0,
    active_coefficients: bool = False,
    active_response: bool = False,
    active_paths: bool = True,
    active_weights: bool = True,
) -> tuple[ResponseMatrixPipeline, Any, Any]:
    grid = GridSpec((1, 1, 1), (1.0, 1.0, 1.0))
    geometry = RayBundleGeometry(((-2.0, 0.0, 0.0),), ((2.0, 0.0, 0.0),))
    spec = SpectralSpec(
        1,
        1,
        precision=precision,
        active_paths=active_paths,
        active_weights=active_weights,
        active_coefficients=active_coefficients,
        active_response=active_response,
    )
    inputs = ResponseMatrixInputs.from_arrays(
        spec,
        wp.array([[coefficient]], dtype=wp.float32, device=device),
        wp.array([weight], dtype=wp.float32, device=device),
        wp.ones((bands, 1), dtype=wp.float32, device=device),
        pixels=1,
        device=device,
    )
    pipeline = ResponseMatrixPipeline(
        grid,
        geometry,
        ProjectionSpec(16, precision=precision, active_pose=False, active_volume=False),
        inputs,
    )
    fields = wp.full((1, 1, 1, 1), field, dtype=wp.float32, device=device)
    pose = wp.array(RigidTransform().packed(), dtype=wp.float64, device=device)
    pipeline.forward(fields, pose)
    return pipeline, fields, pose


@pytest.mark.parametrize("residual", [0.0, 1.0, math.ldexp(1.0, -1074)])
def test_exact_common_cotangents_preserve_intermediate_overflow_and_deep_residual(
    residual: float,
    warp_device: str,
) -> None:
    high = math.ldexp(1.0, 1023)
    seeds = (high, residual, -high)
    pipeline, _, _ = scalar_matrix(warp_device, 3)
    seed = wp.array(np.array(seeds).reshape(3, 1), dtype=wp.float64, device=warp_device)
    with localcontext() as context:
        context.prec = 1600
        expected = float(sum((Decimal.from_float(value) for value in seeds), Decimal(0)))
    gradients = pipeline.vjp(seed)
    assert gradients.weights is not None
    assert float(gradients.weights.numpy()[0]) == expected
    assert float(pipeline._material.cotangent_paths.numpy()[0]) == -expected
    if expected == 0.0:
        assert not np.signbit(gradients.weights.numpy()[0])
        assert not np.signbit(pipeline._material.cotangent_paths.numpy()[0])


def test_exact_common_gradient_can_exceed_destination_range_then_cancel(warp_device: str) -> None:
    high = math.ldexp(1.0, 1023)
    pipeline, _, _ = scalar_matrix(warp_device, 3)
    seed = wp.array([[high], [high], [-high]], dtype=wp.float64, device=warp_device)
    gradients = pipeline.vjp(seed)
    assert gradients.weights is not None
    assert float(gradients.weights.numpy()[0]) == high
    assert float(pipeline._material.cotangent_paths.numpy()[0]) == -high


def test_exact_binary32_path_sum_rounds_directly_without_binary64_double_rounding(
    warp_device: str,
) -> None:
    pipeline, _, _ = scalar_matrix(warp_device, 3, precision="float32", active_response=True)
    seed = wp.array([[1.0], [2.0**-24], [2.0**-80]], dtype=wp.float32, device=warp_device)
    pipeline.vjp(seed)
    expected = -np.nextafter(np.float32(1.0), np.float32(2.0))
    assert pipeline._material.cotangent_paths.numpy()[0] == expected
    assert np.float32(1.0 + 2.0**-24 + 2.0**-80) == np.float32(1.0)


def test_final_overflow_is_checked_and_valid_seed_retry_recovers(warp_device: str) -> None:
    high = math.ldexp(1.0, 1023)
    pipeline, _, _ = scalar_matrix(warp_device, 2)
    with pytest.raises(NumericalError, match="response matrix"):
        pipeline.vjp(wp.array([[high], [high]], dtype=wp.float64, device=warp_device))
    gradients = pipeline.vjp(wp.array([[1.0], [-1.0]], dtype=wp.float64, device=warp_device))
    assert gradients.weights is not None
    assert gradients.weights.numpy()[0] == 0.0


def test_native_band_derivative_overflow_remains_a_native_error_despite_cancellation(
    warp_device: str,
) -> None:
    pipeline, _, _ = scalar_matrix(
        warp_device,
        2,
        coefficient=0.0,
        weight=2.0,
        field=1.0,
        active_coefficients=True,
        active_paths=False,
        active_weights=False,
    )
    high = math.ldexp(1.0, 1023)
    with pytest.raises(NumericalError, match="spectral"):
        pipeline.vjp(wp.array([[high], [-high]], dtype=wp.float64, device=warp_device))
    gradients = pipeline.vjp(wp.array([[1.0], [-1.0]], dtype=wp.float64, device=warp_device))
    assert gradients.coefficients is not None
    assert gradients.coefficients.numpy()[0, 0] == 0.0


@pytest.mark.parametrize("domain", ["linear", "log"])
def test_zero_response_band_preserves_linear_derivative_and_native_log_domain(
    domain: Literal["linear", "log"],
    warp_device: str,
) -> None:
    pipeline, fields, pose = scalar_matrix(warp_device, 2, active_response=True)
    inputs = ResponseMatrixInputs.from_arrays(
        replace(pipeline.inputs.spec, output_domain=domain),
        pipeline.inputs.coefficients,
        pipeline.inputs.weights,
        wp.array([[0.0], [1.0]], dtype=wp.float32, device=warp_device),
        pixels=1,
        device=warp_device,
    )
    pipeline = ResponseMatrixPipeline(pipeline.grid, pipeline.geometry, pipeline.projection, inputs)
    image = pipeline.forward(fields, pose).numpy()
    if domain == "log":
        assert image[0, 0] == -np.inf and image[1, 0] == 0.0
        with pytest.raises(NumericalError, match="spectral"):
            pipeline.vjp(wp.zeros((2, 1), dtype=wp.float64, device=warp_device))
    else:
        np.testing.assert_array_equal(image, [[0.0], [1.0]])
        gradients = pipeline.vjp(wp.array([[1.0], [0.0]], dtype=wp.float64, device=warp_device))
        assert gradients.response is not None and gradients.weights is not None
        np.testing.assert_array_equal(gradients.response.numpy(), [[1.0], [0.0]])
        np.testing.assert_array_equal(gradients.weights.numpy(), [0.0])


def test_response_matrix_vjp_matches_independent_response_finite_differences(
    warp_device: str,
) -> None:
    pipeline, fields, pose, factors_ = make_matrix(warp_device)
    seed_host = np.array([[1.0, -2.0], [-3.0, 4.0], [0.25, -0.5]])
    pipeline.forward(fields, pose)
    gradients = pipeline.vjp(wp.array(seed_host, dtype=wp.float64, device=warp_device))
    assert gradients.response is not None
    native = gradients.response.numpy().copy()
    initial_response = factors_[2].numpy()
    step = 2.0**-10
    for band, energy in np.ndindex(initial_response.shape):
        outcomes = []
        signs = (0.0, 1.0) if initial_response[band, energy] == 0.0 else (-1.0, 1.0)
        for sign in signs:
            response = initial_response.copy()
            response[band, energy] += sign * step
            inputs = ResponseMatrixInputs.from_arrays(
                pipeline.inputs.spec,
                pipeline.inputs.coefficients,
                pipeline.inputs.weights,
                wp.array(response, dtype=wp.float32, device=warp_device),
                pixels=2,
                device=warp_device,
            )
            changed = ResponseMatrixPipeline(
                pipeline.grid, pipeline.geometry, pipeline.projection, inputs
            )
            outcomes.append(float(np.sum(seed_host * changed.forward(fields, pose).numpy())))
        derivative = (outcomes[1] - outcomes[0]) / ((signs[1] - signs[0]) * step)
        assert native[band, energy] == pytest.approx(derivative, rel=3e-11, abs=3e-11)


def test_response_bank_pose_gradient_matches_affine_slab_and_central_difference(
    warp_device: str,
) -> None:
    pipeline, fields, pose, factors_ = make_matrix(warp_device)
    host = fields.numpy()
    slope = 1.0 / 64
    host[0] += np.array([-slope, 0.0, slope], dtype=np.float32)[None, :, None]
    fields.assign(host)
    seed_host = np.array([[1.0, -2.0], [-3.0, 4.0], [0.25, -0.5]])
    pipeline.forward(fields, pose)
    gradients = pipeline.vjp(wp.array(seed_host, dtype=wp.float64, device=warp_device))
    assert gradients.pose is not None
    gradient = float(gradients.pose.numpy()[1])
    coefficients, weights, response = (array.numpy().astype(np.float64) for array in factors_)
    transmission = np.exp(-(np.array([0.75, 1.5]) @ coefficients))
    expected = (
        3
        * slope
        * np.sum(seed_host * (response @ (coefficients[0] * weights * transmission))[:, None])
    )
    assert gradient == pytest.approx(expected, rel=2e-12, abs=1e-12)
    step = 1e-3
    objectives = []
    for displacement in (-step, step):
        changed_pose = wp.array(
            RigidTransform(translation_mm=(0.0, displacement, 0.0)).packed(),
            dtype=wp.float64,
            device=warp_device,
        )
        objectives.append(float(np.sum(seed_host * pipeline.forward(fields, changed_pose).numpy())))
    assert gradient == pytest.approx(
        (objectives[1] - objectives[0]) / (2 * step), rel=1e-7, abs=1e-10
    )


def test_owned_forward_and_factor_copies_preserve_vjp_after_caller_mutations(
    warp_device: str,
) -> None:
    pipeline, fields, pose, source = make_matrix(warp_device)
    image = pipeline.forward(fields, pose).numpy().copy()
    seed = wp.ones((3, 2), dtype=wp.float64, device=warp_device)
    first = pipeline.vjp(seed)
    assert first.fields is not None and first.response is not None
    expected_fields, expected_response = first.fields.numpy().copy(), first.response.numpy().copy()
    fields.fill_(float("nan"))
    pose.fill_(float("nan"))
    for array in source:
        array.fill_(float("nan"))
    gradients = pipeline.vjp(seed)
    assert gradients.fields is not None and gradients.response is not None
    np.testing.assert_array_equal(gradients.fields.numpy(), expected_fields)
    np.testing.assert_array_equal(gradients.response.numpy(), expected_response)
    np.testing.assert_array_equal(pipeline._images.numpy(), image)
    np.testing.assert_array_equal(seed.numpy(), np.ones((3, 2)))


def test_pipeline_budget_counts_all_retained_and_preparation_buffers_before_allocation(
    monkeypatch: pytest.MonkeyPatch,
    warp_device: str,
) -> None:
    pipeline, _, _, _ = make_matrix(warp_device)
    exact_budget = pipeline.storage_bytes + pipeline.scratch_bytes + pipeline.host_bytes
    rebuilt = ResponseMatrixPipeline(
        pipeline.grid,
        pipeline.geometry,
        pipeline.projection,
        pipeline.inputs,
        max_owned_bytes=exact_budget,
    )
    assert rebuilt.storage_bytes + rebuilt.scratch_bytes + rebuilt.host_bytes == exact_budget

    def allocation_forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("budget rejection happened after allocation")

    monkeypatch.setattr(wp, "empty", allocation_forbidden)
    monkeypatch.setattr(wp, "zeros", allocation_forbidden)
    with pytest.raises(ContractError, match="max_owned_bytes"):
        ResponseMatrixPipeline(
            pipeline.grid,
            pipeline.geometry,
            pipeline.projection,
            pipeline.inputs,
            max_owned_bytes=exact_budget - 1,
        )


@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_exact_band_reduction_matches_fraction_across_all_finite_exponents(
    precision: Literal["float32", "float64"],
    warp_device: str,
) -> None:
    kernel_module: Any = importlib.import_module("dpt.kernels.response_matrix")
    rng = np.random.default_rng(20261002)
    wide = precision == "float64"
    integer_type, fraction_bits, largest_exponent = (
        (np.uint64, 52, 2046) if wide else (np.uint32, 23, 254)
    )
    bits = (
        (rng.integers(0, 2, size=(9, 131), dtype=integer_type) << (63 if wide else 31))
        | (
            rng.integers(0, largest_exponent + 1, size=(9, 131), dtype=integer_type)
            << fraction_bits
        )
        | rng.integers(0, 1 << fraction_bits, size=(9, 131), dtype=integer_type)
    )
    original = bits.view(precision)
    rows = np.concatenate((original[:8], -original[:8], original[8:]), axis=0)
    source = wp.array(rows, dtype=getattr(wp, precision), device=warp_device)
    count = rows.shape[1]
    limbs = wp.zeros(count * (34 if wide else 5), dtype=wp.uint64, device=warp_device)
    status = wp.zeros(1, dtype=wp.int32, device=warp_device)
    result = wp.empty(count, dtype=getattr(wp, precision), device=warp_device)
    flat = source.flatten()
    for band in range(rows.shape[0]):
        wp.launch(
            kernel_module.get_accumulate(wide),
            dim=count,
            inputs=[flat[band * count : (band + 1) * count], count],
            outputs=[limbs, status],
            device=warp_device,
            record_tape=False,
        )
    wp.launch(
        kernel_module.get_finish(wide, wide),
        dim=count,
        inputs=[limbs, count],
        outputs=[result, status],
        device=warp_device,
        record_tape=False,
    )
    expected = np.array(
        [
            float(
                sum((Fraction.from_float(float(value)) for value in rows[:, component]), Fraction())
            )
            for component in range(count)
        ],
        dtype=precision,
    )
    expected[expected == 0] = 0.0
    np.testing.assert_array_equal(result.numpy().view(integer_type), expected.view(integer_type))
    np.testing.assert_array_equal(status.numpy(), [0])


@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("domain", ["linear", "log"])
def test_coefficients_weights_and_fields_vjps_match_independent_central_differences(
    shared: bool,
    domain: Literal["linear", "log"],
    warp_device: str,
) -> None:
    pipeline, fields, pose, _ = make_matrix(warp_device, shared=shared, domain=domain)
    seed_host = np.array([[1.0, -2.0], [-3.0, 4.0], [0.25, -0.5]])
    pipeline.forward(fields, pose)
    gradients = pipeline.vjp(wp.array(seed_host, dtype=wp.float64, device=warp_device))
    assert (
        gradients.coefficients is not None
        and gradients.weights is not None
        and gradients.fields is not None
    )
    native_coefficients = gradients.coefficients.numpy().copy()
    native_weights = gradients.weights.numpy().copy()
    native_fields = gradients.fields.numpy().copy()
    coefficient_host, weight_host = (
        pipeline.inputs.coefficients.numpy(),
        pipeline.inputs.weights.numpy(),
    )
    step = 2.0**-12
    for role, original, expected in (
        ("coefficient", coefficient_host, native_coefficients),
        ("weight", weight_host, native_weights),
    ):
        for index in np.ndindex(original.shape):
            objective = []
            for sign in (-1.0, 1.0):
                values = original.copy()
                values[index] += sign * step
                inputs = ResponseMatrixInputs.from_arrays(
                    pipeline.inputs.spec,
                    wp.array(
                        values if role == "coefficient" else coefficient_host,
                        dtype=wp.float32,
                        device=warp_device,
                    ),
                    wp.array(
                        values if role == "weight" else weight_host,
                        dtype=wp.float32,
                        device=warp_device,
                    ),
                    pipeline.inputs.response,
                    pixels=2,
                    device=warp_device,
                )
                changed = ResponseMatrixPipeline(
                    pipeline.grid, pipeline.geometry, pipeline.projection, inputs
                )
                objective.append(float(np.sum(seed_host * changed.forward(fields, pose).numpy())))
            derivative = (objective[1] - objective[0]) / (2 * step)
            assert expected[index] == pytest.approx(derivative, rel=3e-7, abs=3e-9)
    initial_fields = fields.numpy()
    for material in range(2):
        objective = []
        for sign in (-1.0, 1.0):
            changed_fields = initial_fields.copy()
            changed_fields[material, 1, 1, 1] += sign * step
            objective.append(
                float(
                    np.sum(
                        seed_host
                        * pipeline.forward(
                            wp.array(changed_fields, dtype=wp.float32, device=warp_device), pose
                        ).numpy()
                    )
                )
            )
        derivative = (objective[1] - objective[0]) / (2 * step)
        # The existing material scatter stores and adds binary32 sample
        # contributions, including cancellation between the two seeded rays.
        assert native_fields[material, 1, 1, 1] == pytest.approx(derivative, rel=2e-6, abs=5e-7)


def test_deep_attenuation_and_seed_rescued_derivatives_match_independent_decimal(
    warp_device: str,
) -> None:
    pipeline, fields, pose = scalar_matrix(
        warp_device,
        2,
        coefficient=800.0,
        weight=1e38,
        field=1.0,
        active_coefficients=True,
        active_response=True,
    )
    inputs = ResponseMatrixInputs.from_arrays(
        pipeline.inputs.spec,
        pipeline.inputs.coefficients,
        pipeline.inputs.weights,
        wp.array([[1e38], [5e37]], dtype=wp.float32, device=warp_device),
        pixels=1,
        device=warp_device,
    )
    pipeline = ResponseMatrixPipeline(pipeline.grid, pipeline.geometry, pipeline.projection, inputs)
    image = pipeline.forward(fields, pose)
    seed_values = (1e300, -2.5e299)
    gradients = pipeline.vjp(
        wp.array(np.array(seed_values).reshape(2, 1), dtype=wp.float64, device=warp_device)
    )
    assert (
        gradients.weights is not None
        and gradients.response is not None
        and gradients.coefficients is not None
    )
    with localcontext() as context:
        context.prec = 160
        attenuation = (-Decimal(800)).exp()
        weight = Decimal.from_float(float(inputs.weights.numpy()[0]))
        response = [
            Decimal.from_float(float(value)) for value in inputs.response.numpy().reshape(-1)
        ]
        seeds = [Decimal.from_float(value) for value in seed_values]
        means = [weight * value * attenuation for value in response]
        products = [seed * mean for seed, mean in zip(seeds, means, strict=True)]
        expected_coefficients = -sum(products)
        expected_paths = expected_coefficients * Decimal(800)
        expected_weights = sum(
            seed * value * attenuation for seed, value in zip(seeds, response, strict=True)
        )
        expected_response = [seed * weight * attenuation for seed in seeds]
    assert math.exp(-800.0) == 0.0
    np.testing.assert_allclose(
        image.numpy().reshape(-1), [float(value) for value in means], rtol=2e-12, atol=0
    )
    assert gradients.coefficients.numpy()[0, 0] == pytest.approx(
        float(expected_coefficients), rel=2e-12
    )
    assert pipeline._material.cotangent_paths.numpy()[0] == pytest.approx(
        float(expected_paths), rel=2e-12
    )
    assert gradients.weights.numpy()[0] == pytest.approx(float(expected_weights), rel=2e-12)
    np.testing.assert_allclose(
        gradients.response.numpy().reshape(-1),
        [float(value) for value in expected_response],
        rtol=2e-12,
    )


def test_fixed_factor_policy_and_invalid_forward_seed_recovery(warp_device: str) -> None:
    original, fields, pose, _ = make_matrix(warp_device)
    inputs = ResponseMatrixInputs.from_arrays(
        replace(
            original.inputs.spec,
            active_weights=False,
            active_response=False,
            active_coefficients=False,
        ),
        original.inputs.coefficients,
        original.inputs.weights,
        original.inputs.response,
        pixels=2,
        device=warp_device,
    )
    pipeline = ResponseMatrixPipeline(original.grid, original.geometry, original.projection, inputs)
    seed = wp.ones((3, 2), dtype=wp.float64, device=warp_device)
    with pytest.raises(ContractError, match="successful forward"):
        pipeline.vjp(seed)
    pipeline.forward(fields, pose)
    with pytest.raises(ContractError, match="invalid fixed device input"):
        pipeline.vjp(wp.full((3, 2), float("nan"), dtype=wp.float64, device=warp_device))
    gradients = pipeline.vjp(seed)
    assert gradients.fields is not None and gradients.pose is not None
    assert (
        gradients.coefficients is None and gradients.weights is None and gradients.response is None
    )
    fields.fill_(2.0)
    with pytest.raises(NumericalError):
        pipeline.forward(fields, pose)
    with pytest.raises(ContractError, match="successful forward"):
        pipeline.vjp(seed)
    host = np.empty((2, 3, 3, 3), dtype=np.float32)
    host[0], host[1] = 0.25, 0.5
    fields.assign(host)
    pipeline.forward(fields, pose)
    assert pipeline.vjp(seed).fields is not None


def test_flat_seed_shaped_image_contract_and_inactive_reverse(warp_device: str) -> None:
    pipeline, fields, pose, _ = make_matrix(warp_device)
    assert pipeline.forward(fields, pose).shape == (3, 2)
    assert pipeline.vjp(wp.ones(6, dtype=wp.float64, device=warp_device)).response is not None
    with pytest.raises(ContractError, match="shape"):
        pipeline.vjp(wp.ones((2, 3), dtype=wp.float64, device=warp_device))
    with pytest.raises(ContractError, match="dtype"):
        pipeline.vjp(wp.ones((3, 2), dtype=wp.float32, device=warp_device))
    inactive = ResponseMatrixInputs.from_arrays(
        replace(
            pipeline.inputs.spec,
            active_paths=False,
            active_weights=False,
            active_response=False,
            active_coefficients=False,
        ),
        pipeline.inputs.coefficients,
        pipeline.inputs.weights,
        pipeline.inputs.response,
        pixels=2,
        device=warp_device,
    )
    fixed = ResponseMatrixPipeline(
        pipeline.grid,
        pipeline.geometry,
        replace(pipeline.projection, active_pose=False, active_volume=False),
        inactive,
    )
    fixed.forward(fields, pose)
    with pytest.raises(ContractError, match="active response matrix gradient"):
        fixed.vjp(wp.ones((3, 2), dtype=wp.float64, device=warp_device))


def test_factor_budget_includes_temporary_validation_before_allocating(
    monkeypatch: pytest.MonkeyPatch,
    warp_device: str,
) -> None:
    source = factors(warp_device)
    spec = SpectralSpec(1, 2)
    record = ResponseMatrixInputs.from_arrays(
        spec, *source, pixels=2, device=warp_device, max_owned_bytes=40
    )
    assert record.storage_bytes == 32

    def allocation_forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("temporary validation was allocated before the budget check")

    monkeypatch.setattr(wp, "empty", allocation_forbidden)
    monkeypatch.setattr(wp, "zeros", allocation_forbidden)
    with pytest.raises(ContractError, match="max_owned_bytes"):
        ResponseMatrixInputs.from_arrays(
            spec, *source, pixels=2, device=warp_device, max_owned_bytes=39
        )


def test_pending_aggregate_status_is_delivered_without_hiding_an_unchecked_failure(
    warp_device: str,
) -> None:
    pipeline, fields, pose, _ = make_matrix(warp_device)
    expected = pipeline.forward(fields, pose).numpy().copy()
    pipeline._status.fill_(2)
    with pytest.raises(NumericalError, match="response matrix"):
        pipeline.forward(fields, pose)
    np.testing.assert_array_equal(pipeline._images.numpy(), expected)
    np.testing.assert_array_equal(pipeline._status.numpy(), [2])
    with pytest.raises(NumericalError, match="response matrix"):
        pipeline.consume_status()
    np.testing.assert_array_equal(pipeline._status.numpy(), [0])
    assert pipeline.forward(fields, pose).shape == (3, 2)

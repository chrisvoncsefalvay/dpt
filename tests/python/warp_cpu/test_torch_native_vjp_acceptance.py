"""Joint Torch/native VJPs with independently admitted represented-field oracles.

The gamma term models FP32 cast/scatter arithmetic conditional on the upstream
FP64 contributions. The existing 7e-6/3e-7 oracle allowances remain empirical
regression thresholds, not universal geometry, exp/log or platform guarantees.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import struct
from collections.abc import Iterator
from contextlib import nullcontext
from dataclasses import dataclass
from decimal import ROUND_FLOOR, Decimal, localcontext
from fractions import Fraction
from functools import cache
from itertools import pairwise
from pathlib import Path
from typing import Any

import pytest

from dpt._runtime import cpu_test_context
from dpt.geometry import DetectorGeometry
from dpt.interop.torch import project_spectral_signal
from dpt.material_projection import (
    material_projection_vjp,
    prepare_material_projection,
    project_material_paths,
)
from dpt.projection import ProjectionSpec
from dpt.spectral import SpectralSpec, prepare_spectral, spectral_signal, spectral_vjp
from dpt.volumes import GridSpec

_CASES = ("linear-shared-serial", "log-pixel-fused")
_SHAPE = (3, 3, 4)
_SPACING = (0.7, 0.8, 0.9)
_ORIGIN = (-1.0, -1.0, -1.0)
_SOURCE = (-8.0, 0.1, -0.2)
_DETECTOR_ORIGIN = (8.0, -0.3, -0.4)
_PITCH = (0.2, 0.2)
_COEFFICIENTS = ((0.25, 0.5, 0.75), (0.75, 0.5, 0.25))
_BASE_WEIGHTS = (16.0, 32.0, 8.0)
_BASE_RESPONSE = (0.5, 0.25, 0.75)
_PATTERN = (0.75, 0.8125, 0.875, 0.9375, 1.0, 1.0625)
_SEED = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5)
_FD_COORDINATES = ((0, 1, 1, 1), (0, 1, 1, 2), (1, 1, 1, 1), (1, 1, 1, 2))
_H = Fraction(1, 2**16)
_TERM_BOUND = 6 * 11 * 2 * 8
_GAMMA = Fraction(1057, 16776159)
_LINEAR_TRUNCATION = Fraction(75103875, 281474976710656)
_LOG_TRUNCATION = Fraction(23625, 2199023255552)
_DECIMAL_GAP = Decimal("1e-60")
_EMPIRICAL_RTOL = Decimal("7e-6")
_EMPIRICAL_ATOL = Decimal("3e-7")
_MIN_MEAN = Decimal("1.8")
_POSE = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)

_Matrix = tuple[tuple[Decimal, ...], ...]


@dataclass(frozen=True)
class _Reference:
    matrix: _Matrix
    paths: _Matrix
    means: tuple[Decimal, ...]
    output: tuple[Decimal, ...]
    fields: _Matrix
    coefficients: _Matrix
    weights: tuple[Decimal, ...]
    response: tuple[Decimal, ...]
    objective: Decimal


def _decimal_fraction(value: Fraction) -> Decimal:
    return Decimal(value.numerator) / Decimal(value.denominator)


def _fields() -> tuple[tuple[float, ...], ...]:
    return tuple(
        tuple(float(Fraction(material + 1, 16) + Fraction(voxel % 8, 256)) for voxel in range(36))
        for material in range(2)
    )


def _factors(case: str) -> tuple[tuple[tuple[float, ...], ...], tuple[tuple[float, ...], ...]]:
    assert case in _CASES
    shared = case == _CASES[0]
    return (
        tuple(tuple(w * (1.0 if shared else _PATTERN[p]) for p in range(6)) for w in _BASE_WEIGHTS),
        tuple(
            tuple(r * (1.0 if shared else _PATTERN[5 - p]) for p in range(6))
            for r in _BASE_RESPONSE
        ),
    )


def _represented_step(value: float) -> tuple[float, float]:
    def binary32(number: float) -> float:
        return struct.unpack("<f", struct.pack("<f", number))[0]

    high, low = binary32(value + float(_H)), binary32(value - float(_H))
    assert binary32(value) == value and high == value + float(_H) and low == value - float(_H)
    assert high - low == 2 * float(_H) and 0 < low < high < 0.25
    return high, low


def _basis(voxel: int, point: tuple[Decimal, ...]) -> Decimal:
    """Nested one-hot trilinear interpolation, independently of device corners."""
    extent = (4, 3, 3)
    q = tuple(min(max(x, Decimal(0)), Decimal(n - 1)) for x, n in zip(point, extent, strict=True))
    lo = tuple(int(x.to_integral_value(rounding=ROUND_FLOOR)) for x in q)
    hi = tuple(min(i + 1, n - 1) for i, n in zip(lo, extent, strict=True))
    fraction = tuple(x - i for x, i in zip(q, lo, strict=True))
    planes = []
    for z in (lo[2], hi[2]):
        rows = []
        for y in (lo[1], hi[1]):
            a = Decimal(int((z * 3 + y) * 4 + lo[0] == voxel))
            b = Decimal(int((z * 3 + y) * 4 + hi[0] == voxel))
            rows.append(a + (b - a) * fraction[0])
        planes.append(rows[0] + (rows[1] - rows[0]) * fraction[1])
    return planes[0] + (planes[1] - planes[0]) * fraction[2]


@cache
def _decimal_matrix(precision: int) -> _Matrix:
    """Clip represented physical rays; sort interpolation planes; integrate cubics."""
    with localcontext() as context:
        context.prec = precision
        d = Decimal.from_float
        spacing, origin, source = (
            tuple(map(d, _SPACING)),
            tuple(map(d, _ORIGIN)),
            tuple(map(d, _SOURCE)),
        )
        detector, pitch = tuple(map(d, _DETECTOR_ORIGIN)), tuple(map(d, _PITCH))
        extent = (4, 3, 3)
        rows = []
        for pixel in range(6):
            row, column = divmod(pixel, 3)
            endpoint = (detector[0], detector[1] + column * pitch[0], detector[2] + row * pitch[1])
            start = tuple((source[i] - origin[i]) / spacing[i] for i in range(3))
            direction = tuple((endpoint[i] - source[i]) / spacing[i] for i in range(3))
            lower, upper = Decimal(0), Decimal(1)
            for axis, n in enumerate(extent):
                if not direction[axis]:
                    assert Decimal("-0.5") <= start[axis] <= Decimal(n) - Decimal("0.5")
                else:
                    faces = (
                        (Decimal("-0.5") - start[axis]) / direction[axis],
                        (Decimal(n) - Decimal("0.5") - start[axis]) / direction[axis],
                    )
                    lower, upper = max(lower, min(faces)), min(upper, max(faces))
            assert lower < upper
            length = sum(((endpoint[i] - source[i]) ** 2 for i in range(3)), Decimal(0)).sqrt()
            assert length * (upper - lower) < 5
            knots = {lower, upper}
            for axis, n in enumerate(extent):
                if direction[axis]:
                    for centre in range(n):
                        t = (Decimal(centre) - start[axis]) / direction[axis]
                        if lower < t < upper:
                            knots.add(t)
            ordered = sorted(knots)
            assert len(ordered) - 1 <= 11
            nodes = []
            for left, right in pairwise(ordered):
                centre, radius = (left + right) / 2, (right - left) / 2
                offset = radius / Decimal(3).sqrt()
                for t in (centre - offset, centre + offset):
                    nodes.append(
                        (tuple(start[i] + t * direction[i] for i in range(3)), radius * length)
                    )
            integrals = tuple(
                sum((weight * _basis(voxel, point) for point, weight in nodes), Decimal(0))
                for voxel in range(36)
            )
            assert all(value >= 0 for value in integrals)
            assert abs(sum(integrals, Decimal(0)) - length * (upper - lower)) <= Decimal("1e-60")
            rows.append(integrals)
        return tuple(rows)


def _reference(
    case: str, precision: int, perturbation: tuple[int, int, Decimal] | None = None
) -> _Reference:
    assert case in _CASES
    with localcontext() as context:
        context.prec = precision
        matrix = _decimal_matrix(precision)
        field = [list(map(Decimal.from_float, row)) for row in _fields()]
        if perturbation is not None:
            material, voxel, displacement = perturbation
            field[material][voxel] += displacement
        coefficients = tuple(tuple(map(Decimal.from_float, row)) for row in _COEFFICIENTS)
        weight_values, response_values = _factors(case)
        weights = tuple(tuple(map(Decimal.from_float, row)) for row in weight_values)
        response = tuple(tuple(map(Decimal.from_float, row)) for row in response_values)
        paths = tuple(
            tuple(
                sum((field[m][v] * matrix[p][v] for v in range(36)), Decimal(0)) for p in range(6)
            )
            for m in range(2)
        )
        attenuation = tuple(
            tuple(
                (-sum((coefficients[m][k] * paths[m][p] for m in range(2)), Decimal(0))).exp()
                for p in range(6)
            )
            for k in range(3)
        )
        bins = tuple(
            tuple(weights[k][p] * response[k][p] * attenuation[k][p] for p in range(6))
            for k in range(3)
        )
        means = tuple(sum((bins[k][p] for k in range(3)), Decimal(0)) for p in range(6))
        assert min(means) > _MIN_MEAN
        logarithmic = case == _CASES[1]
        output = tuple(value.ln() if logarithmic else value for value in means)
        seed = tuple(map(Decimal.from_float, _SEED))
        assert all(
            value > 0
            for value in (
                *seed,
                *(x for row in coefficients for x in row),
                *(x for row in weights for x in row),
                *(x for row in response for x in row),
            )
        )
        scale = tuple(seed[p] / means[p] if logarithmic else seed[p] for p in range(6))
        bin_seed = tuple(tuple(bins[k][p] * scale[p] for p in range(6)) for k in range(3))
        path_gradient = tuple(
            tuple(
                -sum((coefficients[m][k] * bin_seed[k][p] for k in range(3)), Decimal(0))
                for p in range(6)
            )
            for m in range(2)
        )
        assert all(value < 0 for row in path_gradient for value in row)
        fields = tuple(
            tuple(
                sum((path_gradient[m][p] * matrix[p][v] for p in range(6)), Decimal(0))
                for v in range(36)
            )
            for m in range(2)
        )
        absolute_contributions = tuple(
            tuple(
                sum((abs(path_gradient[m][p] * matrix[p][v]) for p in range(6)), Decimal(0))
                for v in range(36)
            )
            for m in range(2)
        )
        assert absolute_contributions == tuple(tuple(abs(value) for value in row) for row in fields)
        coefficient_gradient = tuple(
            tuple(
                -sum((paths[m][p] * bin_seed[k][p] for p in range(6)), Decimal(0)) for k in range(3)
            )
            for m in range(2)
        )
        weight_gradient = tuple(
            tuple(response[k][p] * attenuation[k][p] * scale[p] for p in range(6)) for k in range(3)
        )
        response_gradient = tuple(
            tuple(weights[k][p] * attenuation[k][p] * scale[p] for p in range(6)) for k in range(3)
        )
        shared = case == _CASES[0]
        return _Reference(
            matrix,
            paths,
            means,
            output,
            fields,
            coefficient_gradient,
            tuple(sum(row, Decimal(0)) for row in weight_gradient)
            if shared
            else tuple(x for row in weight_gradient for x in row),
            tuple(sum(row, Decimal(0)) for row in response_gradient)
            if shared
            else tuple(x for row in response_gradient for x in row),
            sum((seed[p] * output[p] for p in range(6)), Decimal(0)),
        )


def _reference_values(reference: _Reference) -> tuple[Decimal, ...]:
    matrices = (reference.matrix, reference.paths, reference.fields, reference.coefficients)
    return (
        *tuple(x for matrix in matrices for row in matrix for x in row),
        *reference.means,
        *reference.output,
        *reference.weights,
        *reference.response,
        reference.objective,
    )


def _field_allowance(value: Decimal, *, bridge: bool = False) -> Decimal:
    """Conditional scatter model plus the separately declared empirical allowance."""
    with localcontext() as context:
        context.prec = 96
        gamma = _decimal_fraction(_GAMMA)
        scatter = (2 if bridge else 1) * gamma * abs(value)
        return scatter + _EMPIRICAL_RTOL * abs(value) + _EMPIRICAL_ATOL


def _admit(case: str) -> tuple[_Reference, tuple[Decimal, ...], dict[str, str]]:
    references = (_reference(case, 80), _reference(case, 96))
    gaps = [
        abs(a - b)
        for a, b in zip(
            _reference_values(references[0]), _reference_values(references[1]), strict=True
        )
    ]
    h = _decimal_fraction(_H)
    central = []
    support = []
    budgets = []
    for precision, reference in zip((80, 96), references, strict=True):
        with localcontext() as context:
            context.prec = precision
            derivatives = []
            for material, z, y, x in _FD_COORDINATES:
                voxel = (z * 3 + y) * 4 + x
                high, low = _represented_step(_fields()[material][voxel])
                denominator = Decimal.from_float(high) - Decimal.from_float(low)
                assert denominator == 2 * h
                nonzero_support = sum((row[voxel] for row in reference.matrix), Decimal(0))
                assert nonzero_support > Decimal("0.01")
                plus = _reference(case, precision, (material, voxel, h)).objective
                minus = _reference(case, precision, (material, voxel, -h)).objective
                derivative = (plus - minus) / denominator
                truncation = _decimal_fraction(
                    _LINEAR_TRUNCATION if case == _CASES[0] else _LOG_TRUNCATION
                )
                assert (
                    abs(derivative - reference.fields[material][voxel]) <= truncation + _DECIMAL_GAP
                )
                assert derivative < 0 and reference.fields[material][voxel] < 0
                derivatives.append(derivative)
                support.append(nonzero_support)
                budgets.append(_field_allowance(reference.fields[material][voxel]) + truncation)
            central.append(tuple(derivatives))
    gaps.extend(abs(a - b) for a, b in zip(central[0], central[1], strict=True))
    gaps.extend(abs(a - b) for a, b in zip(support[:4], support[4:], strict=True))
    gaps.extend(abs(a - b) for a, b in zip(budgets[:4], budgets[4:], strict=True))
    assert max(gaps) <= _DECIMAL_GAP
    return (
        references[1],
        central[1],
        {
            "case": case,
            "Decimal80_96_max_gap": str(max(gaps)),
            "frozen_gap_limit": str(_DECIMAL_GAP),
            "minimum_nonzero_FD_support_mm": str(min(support)),
            "minimum_mean": str(min(references[1].means)),
            "FD_denominator": str(2 * h),
            "precision_comparisons": str(len(gaps)),
            "gamma_scope": "conditional modeled FP32 cast/scatter only",
            "backend_scope": (
                "existing empirical7e-6/3e-7 regression allowance, no universal exp/log bound"
            ),
            "B_j_scope": (
                "positive seeds/factors and nonnegative basis: all per-ray contributions "
                "nonpositive; absolute sum equals abs(gradient), checked independently"
            ),
        },
    )


@pytest.fixture(autouse=True)
def _native_context(warp_device: str) -> Iterator[None]:
    with cpu_test_context() if warp_device == "cpu" else nullcontext():
        yield


def _device_guard(torch: Any, wp: Any, warp_device: str, record_property: Any) -> Any:
    device = wp.get_device(warp_device)
    assert device.alias == warp_device and bool(device.is_cuda) == warp_device.startswith("cuda:")
    record_property("t01_actual_device_alias", device.alias)
    if device.is_cuda:
        uuid = device.uuid
        assert isinstance(uuid, str) and uuid.startswith("GPU-")
        properties = torch.cuda.get_device_properties(device.ordinal)
        assert (
            properties.name == device.name
            and device.arch == properties.major * 10 + properties.minor
        )
        record_property("t01_actual_device_uuid", uuid)
        record_property("t01_torch_device_name", properties.name)
    else:
        record_property("t01_actual_device_uuid", "cpu")
    return device


def _record_sources(record_property: Any) -> None:
    module = importlib.import_module("dpt.material_projection")
    filename = module.__file__
    assert isinstance(filename, str)
    root = Path(filename).resolve().parent
    paths = (
        "material_projection.py",
        "spectral.py",
        "interop/torch.py",
        "projection.py",
        "kernels/material_projection.py",
        "kernels/spectral_common.py",
        "kernels/projection.py",
        "kernels/spectral.py",
    )
    record_property(
        "t01_actual_source_hashes",
        json.dumps(
            {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in paths}, sort_keys=True
        ),
    )


@pytest.mark.parametrize("case", _CASES, ids=_CASES)
def test_joint_torch_native_vjp_and_represented_field_fd(
    case: str, warp_device: str, monkeypatch: pytest.MonkeyPatch, record_property: Any
) -> None:
    reference, finite_difference, admission = _admit(case)
    record_property("t01_scalar_admission", json.dumps(admission, sort_keys=True))
    torch = pytest.importorskip("torch", reason="optional Torch runtime is not installed")
    wp = importlib.import_module("warp")
    np = importlib.import_module("numpy")
    device = _device_guard(torch, wp, warp_device, record_property)
    _record_sources(record_property)
    stage = {"value": "torch"}
    depth = {"value": 0}
    calls = []
    # launch_tiled forwards the launch arguments through a generic signature.
    # Bind the installed launch contract even when an outer API witness is active.
    signature = inspect.signature(importlib.import_module("warp._src.context").launch)
    assert {"kernel", "device", "stream"} <= signature.parameters.keys()
    for api in ("launch", "launch_tiled"):
        original = getattr(wp, api)

        def observed(
            *args: Any,
            _api: str = api,
            _original: Any = original,
            _signature: inspect.Signature = signature,
            **kwargs: Any,
        ) -> Any:
            bound = _signature.bind_partial(*args, **kwargs).arguments
            stream = bound.get("stream")
            selected = wp.get_device(stream.device if stream is not None else bound.get("device"))
            if bound.get("device") is not None:
                assert wp.get_device(bound["device"]).alias == selected.alias
            assert selected.alias == device.alias and selected.is_cuda == device.is_cuda
            if device.is_cuda:
                assert selected.uuid == device.uuid
            outer = depth["value"] == 0
            depth["value"] += 1
            try:
                result = _original(*args, **kwargs)
            finally:
                depth["value"] -= 1
            if outer:
                kernel = bound["kernel"]
                calls.append(
                    {
                        "stage": stage["value"],
                        "api": _api,
                        "device": selected.alias,
                        "kernel": str(kernel.key),
                        "module": str(kernel.module.name),
                    }
                )
            return result

        monkeypatch.setattr(wp, api, observed)
    shared = case == _CASES[0]
    domain, execution = ("linear", "serial") if shared else ("log", "fused")
    grid = GridSpec(_SHAPE, _SPACING, _ORIGIN)
    geometry = DetectorGeometry(
        _SOURCE, _DETECTOR_ORIGIN, (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), _PITCH, (2, 3)
    )
    projection = ProjectionSpec(
        47, active_pose=True, active_volume=True, precision="float64", integration="cell_gauss"
    )
    specification = SpectralSpec(
        2,
        3,
        precision="float64",
        output_domain=domain,
        shared_weights=shared,
        shared_response=shared,
        active_paths=True,
        active_coefficients=True,
        active_weights=True,
        active_response=True,
    )
    weight_rows, response_rows = _factors(case)
    host = {
        "fields": np.asarray(_fields(), dtype=np.float32).reshape(2, *_SHAPE),
        "pose": np.asarray(_POSE, dtype=np.float64),
        "coefficients": np.asarray(_COEFFICIENTS, dtype=np.float32),
        "weights": np.asarray(_BASE_WEIGHTS if shared else weight_rows, dtype=np.float32),
        "response": np.asarray(_BASE_RESPONSE if shared else response_rows, dtype=np.float32),
    }
    tensors = {
        name: torch.tensor(value, device=warp_device, requires_grad=True)
        for name, value in host.items()
    }
    seed = torch.tensor(_SEED, dtype=torch.float64, device=warp_device).reshape(2, 3)
    saved_inputs = {name: value.detach().clone() for name, value in tensors.items()}
    saved_seed = seed.clone()
    torch_material = prepare_material_projection(
        grid,
        geometry,
        projection,
        materials=2,
        fields=wp.from_torch(tensors["fields"].reshape(-1), requires_grad=False),
        out_paths=wp.empty((2, 6), dtype=wp.float64, device=warp_device),
        cotangent_paths=wp.empty((2, 6), dtype=wp.float64, device=warp_device),
        device=warp_device,
        channel_execution=execution,
    )
    torch_spectral = prepare_spectral(
        specification, max_pixels=6, device=warp_device, stream=torch_material.context.stream
    )
    output = project_spectral_signal(
        tensors["fields"],
        tensors["pose"],
        tensors["coefficients"],
        tensors["weights"],
        tensors["response"],
        material_workspace=torch_material,
        spectral_workspace=torch_spectral,
    )
    output.backward(seed)
    assert output.shape == (2, 3) and output.dtype == torch.float64
    assert output.device.type == ("cuda" if device.is_cuda else "cpu")
    for name, value in tensors.items():
        assert (
            value.grad is not None
            and value.grad.shape == value.shape
            and value.grad.dtype == value.dtype
        )
        assert value.grad.device == value.device and torch.isfinite(value.grad).all().item()
        assert torch.equal(value.detach(), saved_inputs[name])
    assert torch.equal(seed, saved_seed)
    wp.synchronize_device(device)
    stage["value"] = "native"
    native_inputs = {
        name: wp.array(
            value, dtype=wp.float64 if name == "pose" else wp.float32, device=warp_device
        )
        for name, value in host.items()
    }
    paths = wp.empty((2, 6), dtype=wp.float64, device=warp_device)
    path_seed = wp.empty((2, 6), dtype=wp.float64, device=warp_device)
    native_seed = wp.array(_SEED, dtype=wp.float64, device=warp_device)
    native_mean = wp.empty(6, dtype=wp.float64, device=warp_device)
    gradients = {
        name: wp.empty(
            value.shape, dtype=wp.float64 if name == "pose" else wp.float32, device=warp_device
        )
        for name, value in host.items()
    }
    native_arrays = [
        *native_inputs.values(),
        paths,
        path_seed,
        native_seed,
        native_mean,
        *gradients.values(),
    ]
    spans = [
        (array.ptr, array.ptr + array.size * wp.types.type_size_in_bytes(array.dtype))
        for array in native_arrays
    ]
    torch_paths = [torch_material.paths, torch_material.cotangent_paths]
    spans.extend(
        (array.ptr, array.ptr + array.size * wp.types.type_size_in_bytes(array.dtype))
        for array in torch_paths
    )
    torch_arrays = [
        *tensors.values(),
        *[v.grad for v in tensors.values()],
        *saved_inputs.values(),
        seed,
        saved_seed,
        output,
    ]
    spans.extend(
        (value.data_ptr(), value.data_ptr() + value.numel() * value.element_size())
        for value in torch_arrays
    )
    assert all(
        right <= other_left or other_right <= left
        for index, (left, right) in enumerate(spans)
        for other_left, other_right in spans[index + 1 :]
    )
    assert all(array.device.alias == device.alias for array in native_arrays)
    native_material = prepare_material_projection(
        grid,
        geometry,
        projection,
        materials=2,
        fields=native_inputs["fields"],
        out_paths=paths,
        cotangent_paths=path_seed,
        out_fields=gradients["fields"],
        device=warp_device,
        channel_execution=execution,
    )
    native_spectral = prepare_spectral(
        specification, max_pixels=6, device=warp_device, stream=native_material.context.stream
    )
    project_material_paths(native_inputs["pose"], workspace=native_material)
    spectral_signal(
        paths,
        native_inputs["coefficients"],
        native_inputs["weights"],
        native_inputs["response"],
        out_mean=native_mean,
        workspace=native_spectral,
    )
    spectral_vjp(
        paths,
        native_inputs["coefficients"],
        native_inputs["weights"],
        native_inputs["response"],
        cotangent=native_seed,
        out_grad_paths=path_seed,
        out_grad_coefficients=gradients["coefficients"],
        out_grad_weights=gradients["weights"],
        out_grad_response=gradients["response"],
        workspace=native_spectral,
    )
    material_projection_vjp(
        native_inputs["pose"], workspace=native_material, out_pose_matrix=gradients["pose"]
    )
    native_material.consume_status()
    native_spectral.consume_status()
    wp.synchronize_device(device)
    required_reverse = [
        native_material.projection._kernels.get_vjp(
            shared,
            True,
            True,
            double_seed=True,
            cell_gauss=True,
            double_volume_gradient=False,
            mixed=False,
            storage_precision="float32",
        )
    ]
    if not shared:
        required_reverse.append(
            native_material.kernels.get_material_vjp(2, True, True, True, False)
        )
    required_reverse_ids = {
        (str(kernel.module.name), str(kernel.key)) for kernel in required_reverse
    }
    torch_gradient = {name: value.grad.detach().cpu().numpy() for name, value in tensors.items()}
    native_gradient = {name: value.numpy() for name, value in gradients.items()}
    for name in ("pose", "coefficients", "weights", "response"):
        np.testing.assert_array_equal(native_gradient[name], torch_gradient[name])
    assert torch_gradient["pose"].shape == (12,) and np.count_nonzero(torch_gradient["pose"]) > 0
    field_oracle = np.asarray([[float(x) for x in row] for row in reference.fields]).reshape(
        2, *_SHAPE
    )
    oracle_allowance = np.asarray(
        [[float(_field_allowance(x)) for x in row] for row in reference.fields]
    ).reshape(2, *_SHAPE)
    bridge_allowance = np.asarray(
        [[float(_field_allowance(x, bridge=True)) for x in row] for row in reference.fields]
    ).reshape(2, *_SHAPE)
    assert bool(
        np.all(np.abs(native_gradient["fields"] - torch_gradient["fields"]) <= bridge_allowance)
    )
    for values in (native_gradient["fields"], torch_gradient["fields"]):
        assert bool(np.all(np.abs(values - field_oracle) <= oracle_allowance))
    truncation = float(_LINEAR_TRUNCATION if shared else _LOG_TRUNCATION)
    for coordinate, expected in zip(_FD_COORDINATES, finite_difference, strict=True):
        for values in (native_gradient["fields"], torch_gradient["fields"]):
            assert (
                abs(float(values[coordinate]) - float(expected))
                <= float(oracle_allowance[coordinate]) + truncation
            )
    np.testing.assert_allclose(
        output.detach().cpu().numpy().reshape(-1),
        [float(x) for x in reference.output],
        rtol=7e-6,
        atol=3e-7,
    )
    np.testing.assert_array_equal(native_mean.numpy(), output.detach().cpu().numpy().reshape(-1))
    for name, expected in (
        ("coefficients", reference.coefficients),
        ("weights", reference.weights),
        ("response", reference.response),
    ):
        oracle = np.asarray(expected, dtype=np.float64).reshape(host[name].shape)
        np.testing.assert_allclose(torch_gradient[name], oracle, rtol=7e-6, atol=3e-7)
    for name, value in native_inputs.items():
        np.testing.assert_array_equal(value.numpy(), host[name])
    np.testing.assert_array_equal(native_seed.numpy(), _SEED)
    for completed_stage in ("torch", "native"):
        assert any(call["stage"] == completed_stage and call["api"] == "launch" for call in calls)
        assert any(
            call["stage"] == completed_stage and call["api"] == "launch_tiled" for call in calls
        )
        actual_reverse_ids = {
            (call["module"], call["kernel"])
            for call in calls
            if call["stage"] == completed_stage and call["api"] == "launch_tiled"
        }
        assert required_reverse_ids <= actual_reverse_ids
    record_property("t01_successful_target_API_calls", json.dumps(calls, sort_keys=True))
    record_property(
        "t01_active_gradient_inputs", "fields,pose_storage12,coefficients,weights,response"
    )
    record_property("t01_fixed_field_FD_coordinates", json.dumps(_FD_COORDINATES))

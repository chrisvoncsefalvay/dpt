"""Shared CPU/CUDA spectral tail contracts against independent Decimal arithmetic."""

# pytest.approx has intentionally dynamic third-party stubs.
# pyright: reportUnknownMemberType=false

import hashlib
import importlib
from collections.abc import Callable
from contextlib import nullcontext
from decimal import Decimal, localcontext
from math import isfinite
from typing import Any

import pytest

from dpt._runtime import cpu_test_context
from dpt.materials import Provenance
from dpt.spectral import (
    SpectralSpec,
    prepare_spectral,
    spectral_bin_counts,
    spectral_signal,
    spectral_vjp,
)

wp: Any = importlib.import_module("warp")
FIXTURE = Provenance(
    source="analytic:test_spectral explicit arrays below",
    sha256=hashlib.sha256(b"A=(0,50,120;20,5,0); mu=(.02,.04;0,.1)").hexdigest(),
    rights="Original mathematical fixture; Apache-2.0",
    description="Analytic material-path fixture",
)


@pytest.fixture(autouse=True)
def execution_context(warp_device: str):
    with cpu_test_context() if warp_device == "cpu" else nullcontext():
        yield


@pytest.fixture
def array(warp_device: str) -> Callable[..., Any]:
    def make(values: tuple[float, ...], dtype: Any = wp.float32) -> Any:
        return wp.array(values, dtype=dtype, device=warp_device)

    return make


def exact_values(device: Any) -> tuple[float, ...]:
    return tuple(float(value) for value in device.numpy())


def tail_policy() -> SpectralSpec:
    return SpectralSpec(
        1,
        1,
        (FIXTURE,),
        FIXTURE,
        FIXTURE,
        "keV",
        "analytic fixed basis",
        active_weights=True,
        active_response=True,
        active_coefficients=True,
        precision="float64",
    )


@pytest.mark.parametrize(
    "destination", ["mean", "paths", "weights", "response", "coefficients", "bins"]
)
def test_wide_retains_deep_weighted_tail(
    destination: str, array: Callable[..., Any], warp_device: str
) -> None:
    workspace = prepare_spectral(tail_policy(), max_pixels=1, device=warp_device)
    paths, coefficients = array((750.0,), wp.float64), array((1.0,))
    weights, response = array((1e30,)), array((1e10,))
    output = array((0.0,), wp.float64)
    if destination == "mean":
        spectral_signal(
            paths, coefficients, weights, response, out_mean=output, workspace=workspace
        )
    elif destination == "bins":
        spectral_bin_counts(
            paths, coefficients, weights, array((1.0,)), out_bin_counts=output, workspace=workspace
        )
    else:
        spectral_vjp(
            paths,
            coefficients,
            weights,
            response,
            cotangent=array((1.0,), wp.float64),
            workspace=workspace,
            **{f"out_grad_{destination}": output},
        )
    workspace.check_status()
    assert all(isfinite(value) for value in exact_values(output))
    with localcontext() as ctx:
        ctx.prec = 100
        w, r = Decimal(float(weights.numpy()[0])), Decimal(float(response.numpy()[0]))
        factor = {
            "mean": w * r,
            "paths": w * r,
            "weights": r,
            "response": w,
            "coefficients": Decimal(750) * w * r,
            "bins": w,
        }[destination]
        reference = float((-Decimal(750)).exp() * factor)
    assert exact_values(output)[0] != 0
    assert abs(exact_values(output)[0]) == pytest.approx(reference, rel=2e-13, abs=5e-324)


@pytest.mark.parametrize("seed", [1e-140, 1e300])
@pytest.mark.parametrize("depth", [500.0, 750.0, 800.0])
def test_wide_gradient_has_independent_exponent_range(
    seed: float, depth: float, array: Callable[..., Any], warp_device: str
) -> None:
    workspace = prepare_spectral(tail_policy(), max_pixels=1, device=warp_device)
    paths, coefficients = array((depth,), wp.float64), array((1.0,))
    weights, response = array((1e30,)), array((1e10,))
    output = array((0.0,), wp.float64)
    spectral_vjp(
        paths,
        coefficients,
        weights,
        response,
        cotangent=array((seed,), wp.float64),
        out_grad_paths=output,
        workspace=workspace,
    )
    with localcontext() as ctx:
        ctx.prec = 100
        reference = float(
            -Decimal(seed)
            * (-Decimal(depth)).exp()
            * Decimal(float(weights.numpy()[0]))
            * Decimal(float(response.numpy()[0]))
        )
    assert exact_values(output)[0] == pytest.approx(reference, rel=2e-13, abs=5e-324)

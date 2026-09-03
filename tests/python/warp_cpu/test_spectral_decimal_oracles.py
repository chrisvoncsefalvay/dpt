"""Portable exact-input spectral oracles and retained nuisance-gradient tails."""

import importlib
from decimal import Decimal, localcontext
from itertools import product
from typing import Any

import numpy as np
import pytest

from dpt._runtime import cpu_test_context
from dpt.contracts import ContractError
from dpt.materials import Provenance
from dpt.spectral import (
    SpectralSpec,
    prepare_spectral,
    spectral_signal,
    spectral_vjp,
)
from dpt.validation.spectral import spectral_reference

FIXTURE = Provenance("analytic:spectral-bin-inputs", "0" * 64, "Apache-2.0", "Exact arrays")
wp: Any = importlib.import_module("warp")


@pytest.mark.parametrize("shared_weights,shared_response", list(product((True, False), repeat=2)))
@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_all_layouts_and_cotangents_match_decimal(
    shared_weights, shared_response, precision, warp_device
):
    with cpu_test_context():
        materials, energies, pixels = 3, 7, 5
        policy = SpectralSpec(
            materials,
            energies,
            (FIXTURE,) * materials,
            FIXTURE,
            FIXTURE,
            "ADU",
            "analytic",
            shared_weights=shared_weights,
            shared_response=shared_response,
            active_weights=True,
            active_response=True,
            active_coefficients=True,
            precision=precision,
            reduction_groups=2,
        )
        dtype = wp.float64 if precision == "float64" else wp.float32

        def array(data: Any, scalar: Any = wp.float32) -> Any:
            return wp.array(data, dtype=scalar, device=warp_device)

        paths = array([0, 1, 50, 120, 500] * materials, dtype)
        coefficients = array([0.005 + i * 0.004 for i in range(materials * energies)])
        weights = array(
            [
                0 if i % 4 == 0 else 3 + i % 3
                for i in range(energies * (1 if shared_weights else pixels))
            ]
        )
        response = array(
            [
                0 if i % 5 == 0 else 0.1 + i % 7
                for i in range(energies * (1 if shared_response else pixels))
            ]
        )
        seed = array([1, -3, 0, 0.1, 2], dtype)
        workspace = prepare_spectral(policy, max_pixels=pixels, device=warp_device)
        mean = wp.empty(pixels, dtype=dtype, device=warp_device)
        gp = wp.empty_like(paths)
        gw = wp.empty(
            weights.size, dtype=wp.float64 if shared_weights else wp.float32, device=warp_device
        )
        gr = wp.empty(
            response.size, dtype=wp.float64 if shared_response else wp.float32, device=warp_device
        )
        gc = wp.empty(coefficients.size, dtype=wp.float64, device=warp_device)

        def exact(x):
            return tuple(float(v) for v in x.numpy())

        reference = spectral_reference(
            *(exact(x) for x in (paths, coefficients, weights, response)),
            materials=materials,
            energies=energies,
            seed=exact(seed),
            shared_weights=shared_weights,
            shared_response=shared_response,
        )
        spectral_signal(paths, coefficients, weights, response, out_mean=mean, workspace=workspace)
        spectral_vjp(
            paths,
            coefficients,
            weights,
            response,
            cotangent=seed,
            out_grad_paths=gp,
            out_grad_weights=gw,
            out_grad_response=gr,
            out_grad_coefficients=gc,
            workspace=workspace,
        )
        for actual, expected in zip(
            (mean, gp, gw, gr, gc),
            (
                reference.mean,
                reference.grad_paths,
                reference.grad_weights,
                reference.grad_response,
                reference.grad_coefficients,
            ),
            strict=True,
        ):
            assert exact(actual) == pytest.approx(
                tuple(map(float, expected)),
                rel=2e-13 if actual.dtype == wp.float64 else 2e-6,
                abs=0,
            )
        assert exact(seed) == (
            1.0,
            -3.0,
            0.0,
            float(np.float64(0.1) if dtype == wp.float64 else np.float32(0.1)),
            2.0,
        )


@pytest.mark.parametrize("destination", ["weights", "response", "coefficients"])
@pytest.mark.parametrize("output_dtype", [wp.float32, wp.float64])
def test_shared_tail_has_nonvacuous_destination_precision(destination, output_dtype, warp_device):
    with cpu_test_context():
        policy = SpectralSpec(
            1,
            1,
            (FIXTURE,),
            FIXTURE,
            FIXTURE,
            "ADU",
            "analytic",
            precision="float64",
            active_weights=True,
            active_response=True,
            active_coefficients=True,
        )
        workspace = prepare_spectral(policy, max_pixels=1, device=warp_device)
        paths = wp.array([750.0], dtype=wp.float64, device=warp_device)
        mu = wp.array([1.0], dtype=wp.float32, device=warp_device)
        weights = wp.array([1e30], dtype=wp.float32, device=warp_device)
        response = wp.array([1e10], dtype=wp.float32, device=warp_device)
        seed = wp.array([1.0], dtype=wp.float64, device=warp_device)
        output = wp.empty(1, dtype=output_dtype, device=warp_device)
        spectral_vjp(
            paths,
            mu,
            weights,
            response,
            cotangent=seed,
            workspace=workspace,
            **{f"out_grad_{destination}": output},
        )
        with localcontext() as context:
            context.prec = 100
            tail = (-Decimal(750)).exp()
            w, r = Decimal(float(weights.numpy()[0])), Decimal(float(response.numpy()[0]))
            reference = {
                "weights": r * tail,
                "response": w * tail,
                "coefficients": -Decimal(750) * w * r * tail,
            }[destination]
        actual = float(output.numpy()[0])
        if output_dtype == wp.float32:
            assert actual == float(np.float32(float(reference))) == 0.0
        else:
            assert actual != 0
            assert actual == pytest.approx(float(reference), rel=2e-13, abs=5e-324)


def test_per_pixel_nuisance_destinations_still_require_binary32(warp_device):
    with cpu_test_context():
        policy = SpectralSpec(
            1,
            1,
            (FIXTURE,),
            FIXTURE,
            FIXTURE,
            "ADU",
            "analytic",
            shared_weights=False,
            active_weights=True,
        )
        workspace = prepare_spectral(policy, max_pixels=1, device=warp_device)
        value = wp.array([1], dtype=wp.float32, device=warp_device)
        with pytest.raises(ContractError, match="binary32"):
            spectral_vjp(
                value,
                value,
                value,
                value,
                cotangent=value,
                out_grad_weights=wp.empty(1, dtype=wp.float64, device=warp_device),
                workspace=workspace,
            )


@pytest.mark.parametrize(
    "destinations",
    [
        ("weights", "response"),
        ("weights", "coefficients"),
        ("response", "coefficients"),
        ("weights", "response", "coefficients"),
    ],
)
def test_fused_shared_products_preserve_rescued_range_and_bounded_scratch(
    destinations, warp_device
):
    with cpu_test_context():
        pixels = 513
        policy = SpectralSpec(
            1,
            1,
            (FIXTURE,),
            FIXTURE,
            FIXTURE,
            "ADU",
            "analytic",
            precision="float64",
            active_weights=True,
            active_response=True,
            active_coefficients=True,
            reduction_groups=2,
        )
        workspace = prepare_spectral(policy, max_pixels=pixels, device=warp_device)
        assert workspace.scratch_bytes == 4 + 8 * 2 * (1 + 1 + 1)
        paths = wp.full(pixels, 1000, dtype=wp.float64, device=warp_device)
        mu = wp.array([0.001], dtype=wp.float32, device=warp_device)
        weights = wp.array([1e-30], dtype=wp.float32, device=warp_device)
        response = wp.array([1e-30], dtype=wp.float32, device=warp_device)
        seed = wp.full(pixels, 1e308, dtype=wp.float64, device=warp_device)
        outputs = {name: wp.empty(1, dtype=wp.float64, device=warp_device) for name in destinations}
        reference = spectral_reference(
            tuple(paths.numpy()),
            tuple(mu.numpy()),
            tuple(weights.numpy()),
            tuple(response.numpy()),
            materials=1,
            energies=1,
            seed=tuple(seed.numpy()),
        )
        spectral_vjp(
            paths,
            mu,
            weights,
            response,
            cotangent=seed,
            workspace=workspace,
            **{f"out_grad_{name}": value for name, value in outputs.items()},
        )
        # sum(seed*exp(-tau)) itself overflows here; multiplying each original
        # response/weight before reduction preserves every finite derivative.
        for name, output in outputs.items():
            expected = float(getattr(reference, f"grad_{name}")[0])
            assert expected != 0
            assert float(output.numpy()[0]) == pytest.approx(expected, rel=2e-13, abs=0)

"""Independent indexing, compensated-sum, and distinct material-row contracts."""

# Warp annotations are executable DSL expressions.
# pyright: reportInvalidTypeForm=false

import importlib
from decimal import Decimal, localcontext
from itertools import product
from math import fsum
from typing import Any, Literal

import numpy as np
import pytest

from dpt._runtime import cpu_test_context
from dpt.kernels.spectral_common import (
    STRICT,
    compensated_sum_update,
    energy_pixel_index,
    material_major_index,
)
from dpt.spectral import SpectralSpec, prepare_spectral, spectral_signal, spectral_vjp
from dpt.validation.spectral import spectral_reference

wp: Any = importlib.import_module("warp")


@wp.kernel(module="unique", module_options=STRICT)
def _indices(
    coordinates: wp.array2d(dtype=wp.int32),
    output: wp.array2d(dtype=wp.int32),
):
    case = wp.tid()
    outer, inner, samples = coordinates[case, 0], coordinates[case, 1], coordinates[case, 2]
    output[case, 0] = material_major_index(outer, inner, samples)
    output[case, 1] = energy_pixel_index(outer, inner, samples)


@wp.kernel(module="unique", module_options=STRICT)
def _sum_prefixes(
    values: wp.array(dtype=wp.float64),
    prefixes: wp.array2d(dtype=wp.float64),
):
    total, compensation = wp.float64(0.0), wp.float64(0.0)
    for index in range(values.shape[0]):
        total, compensation = compensated_sum_update(total, compensation, values[index])
        prefixes[index, 0] = total
        prefixes[index, 1] = compensation


def test_flat_indices_match_independent_numpy_axes_and_int32_boundary(warp_device: str) -> None:
    coordinates = [
        (outer, inner, samples)
        for outers, samples in ((3, 5), (3, 7), (7, 5), (1, 1))
        for outer, inner in product(range(outers), range(samples))
    ]
    coordinates.extend(((2, 715827881, 715827882), (1, 1073741823, 1073741824)))
    # NumPy's multidimensional axis mapping does not use the production helpers.
    expected = np.array(
        [
            np.ravel_multi_index((outer, inner), (outer + 1, samples))
            for outer, inner, samples in coordinates
        ],
        dtype=np.int32,
    )
    inputs = wp.array(np.array(coordinates, dtype=np.int32), device=warp_device)
    output = wp.empty((len(coordinates), 2), dtype=wp.int32, device=warp_device)
    wp.launch(_indices, dim=len(coordinates), inputs=[inputs], outputs=[output], device=warp_device)
    np.testing.assert_array_equal(output.numpy(), np.column_stack((expected, expected)))
    assert int(output.numpy()[-1, 0]) == 2147483647


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_compensated_prefixes_retain_small_bins_and_state(sign: float, warp_device: str) -> None:
    # Four bins together are representable at the large term's precision;
    # ordinary sequential summation loses all four. fsum is an independent oracle.
    sequence = sign * np.array([2.0**54, 1.0, 1.0, 1.0, 1.0, -(2.0**54)])
    values = wp.array(sequence, dtype=wp.float64, device=warp_device)
    prefixes = wp.empty((len(sequence), 2), dtype=wp.float64, device=warp_device)
    wp.launch(_sum_prefixes, dim=1, inputs=[values], outputs=[prefixes], device=warp_device)
    actual = prefixes.numpy()
    expected = [fsum(sequence[:index]) for index in range(1, len(sequence) + 1)]
    np.testing.assert_array_equal(actual[:, 0], expected)
    assert actual[1, 1] == -sign
    assert actual[-1, 0] == 4.0 * sign
    assert sum(sequence) == 0.0


@pytest.mark.parametrize("shared_weights,shared_response", list(product((True, False), repeat=2)))
@pytest.mark.parametrize("domain", ["linear", "log"])
@pytest.mark.parametrize("requested", ["paths", "coefficients", "all"])
def test_distinct_material_rows_and_requested_products_match_decimal(
    shared_weights: bool,
    shared_response: bool,
    domain: Literal["linear", "log"],
    requested: str,
    warp_device: str,
) -> None:
    with cpu_test_context():
        materials, energies, pixels = 3, 4, 5
        paths_host = np.array(
            [
                [0.0, 1.0, 3.0, 7.0, 13.0],
                [2.0, 5.0, 11.0, 17.0, 23.0],
                [19.0, 29.0, 31.0, 37.0, 41.0],
            ],
            dtype=np.float64,
        )
        coefficients_host = np.array(
            [[0.0, 0.01, 0.02, 0.03], [0.04, 0.05, 0.06, 0.07], [0.08, 0.09, 0.10, 0.11]],
            dtype=np.float32,
        )
        weight_grid = np.arange(1, energies * pixels + 1, dtype=np.float32).reshape(
            energies, pixels
        )
        response_grid = np.arange(2, energies * pixels + 2, dtype=np.float32).reshape(
            energies, pixels
        )
        weights_host = weight_grid[:, 0] if shared_weights else weight_grid
        response_host = response_grid[:, -1] if shared_response else response_grid
        paths = wp.array(paths_host.ravel(), dtype=wp.float64, device=warp_device)
        coefficients = wp.array(coefficients_host.ravel(), dtype=wp.float32, device=warp_device)
        weights = wp.array(weights_host.ravel().copy(), dtype=wp.float32, device=warp_device)
        response = wp.array(response_host.ravel().copy(), dtype=wp.float32, device=warp_device)
        seed_values = (1.0, -2.0, 0.0, 0.5, 3.0)
        seed = wp.array(seed_values, dtype=wp.float64, device=warp_device)
        inputs = (paths, coefficients, weights, response)
        exact = tuple(tuple(float(v) for v in value.numpy()) for value in inputs)
        reference = spectral_reference(
            *exact,
            materials=materials,
            energies=energies,
            seed=seed_values,
            shared_weights=shared_weights,
            shared_response=shared_response,
        )
        expected_mean = reference.mean
        if domain == "log":
            # The log VJP is the linear VJP seeded by seed / mean, at Decimal
            # precision before conversion to any destination storage type.
            with localcontext() as context:
                context.prec = 100
                expected_mean = tuple(value.ln() for value in reference.mean)
                zero_seed = (0.0,) * pixels
                per_pixel = [
                    spectral_reference(
                        *exact,
                        materials=materials,
                        energies=energies,
                        seed=tuple(1.0 if p == pixel else zero_seed[p] for p in range(pixels)),
                        shared_weights=shared_weights,
                        shared_response=shared_response,
                    )
                    for pixel in range(pixels)
                ]
                expected = {
                    name: tuple(
                        sum(
                            (
                                getattr(per_pixel[p], f"grad_{name}")[index]
                                * Decimal.from_float(seed_values[p])
                                / reference.mean[p]
                                for p in range(pixels)
                            ),
                            Decimal(0),
                        )
                        for index in range(len(getattr(reference, f"grad_{name}")))
                    )
                    for name in ("paths", "coefficients", "weights", "response")
                }
        else:
            expected = {
                name: getattr(reference, f"grad_{name}")
                for name in ("paths", "coefficients", "weights", "response")
            }
        policy = SpectralSpec(
            materials,
            energies,
            shared_weights=shared_weights,
            shared_response=shared_response,
            active_weights=True,
            active_response=True,
            active_coefficients=True,
            precision="float64",
            output_domain=domain,
            reduction_groups=2,
        )
        workspace = prepare_spectral(policy, max_pixels=pixels, device=warp_device)
        mean = wp.empty(pixels, dtype=wp.float64, device=warp_device)
        names = (
            ("paths", "coefficients", "weights", "response") if requested == "all" else (requested,)
        )
        outputs = {
            name: wp.empty(
                len(expected[name]),
                dtype=wp.float32
                if (name == "weights" and not shared_weights)
                or (name == "response" and not shared_response)
                else wp.float64,
                device=warp_device,
            )
            for name in names
        }
        spectral_signal(*inputs, out_mean=mean, workspace=workspace)
        spectral_vjp(
            *inputs,
            cotangent=seed,
            workspace=workspace,
            **{f"out_grad_{name}": value for name, value in outputs.items()},
        )
        workspace.check_status()
        np.testing.assert_allclose(
            mean.numpy(), tuple(map(float, expected_mean)), rtol=2e-13, atol=0
        )
        for name, output in outputs.items():
            np.testing.assert_allclose(
                output.numpy(),
                tuple(map(float, expected[name])),
                rtol=2e-6 if output.dtype == wp.float32 else 2e-13,
                atol=0,
            )

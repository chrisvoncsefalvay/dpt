"""Six material precision modes share an independently conditioned ideal MLE."""

from __future__ import annotations

import importlib
import json
import math
import struct
from contextlib import nullcontext
from dataclasses import replace
from decimal import Decimal, localcontext
from fractions import Fraction
from typing import Any, Literal

import numpy as np
import pytest

from dpt._runtime import cpu_test_context
from dpt.material_reconstruction import (
    MaterialReconstruction,
    MaterialReconstructionSettings,
)
from helpers.gradient_references import (
    COUNTS,
    HESSIAN_MINIMUM,
    INCIDENT,
    TAU,
    gamma,
    poisson_reference,
    solver_gradient_budget,
)
from helpers.material_solver_inputs import _inputs
from helpers.solver_backend_admission import (
    get_vjp_attenuation_admission,
    primitive32,
    primitive64,
)

wp: Any = importlib.import_module("warp")


def _binary32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _ulp(value: float, narrow: bool) -> Decimal:
    if narrow:
        values = np.asarray([value], dtype=np.float32)
        gap = float(np.nextafter(values, np.float32(math.inf))[0]) - value
    else:
        gap = math.ulp(value)
    return Decimal.from_float(gap)


def _fraction_decimal(value: Fraction) -> Decimal:
    return Decimal(value.numerator) / Decimal(value.denominator)


@pytest.mark.parametrize("mode", ["float32_strict", "float64_strict", "float32_arithmetic32"])
@pytest.mark.parametrize("step_selection", ["geometric", "bb"])
def test_material_modes_match_conditioned_ideal_mle(
    mode: str,
    step_selection: Literal["geometric", "bb"],
    warp_device: str,
    monkeypatch: pytest.MonkeyPatch,
    record_property: Any,
) -> None:
    with cpu_test_context() if warp_device == "cpu" else nullcontext():
        inputs = _inputs(precision="float64" if mode == "float64_strict" else "float32")
        inputs["coefficients"] = np.asarray([[0.5, 0.125], [0.125, 0.5]], dtype=np.float32)
        view = inputs["views"][0]
        inputs["views"] = (
            replace(
                view,
                counts=np.asarray(COUNTS, dtype=np.float32).reshape(2, 1, 1),
                weights=INCIDENT * np.eye(2, dtype=np.float32),
            ),
        )
        inputs["spectral_spec"] = replace(
            inputs["spectral_spec"], arithmetic="float32" if "arithmetic32" in mode else "strict"
        )
        inputs["initial_fractions"] = np.full((2, 1, 1, 1), 0.125, dtype=np.float32)
        inputs["settings"] = MaterialReconstructionSettings(
            iterations=150,
            step_selection=step_selection,
            gradient_mapping_tolerance=float(TAU),
            relative_gradient_mapping_tolerance=0,
            acceleration="none",
        )
        solver = MaterialReconstruction(**inputs, device=warp_device)
        states: list[dict[str, Any]] = []
        original = solver._launch_evaluation

        def observed(**kwargs: Any) -> None:
            fields = kwargs["fields"].numpy().reshape(-1).copy()
            original(**kwargs)
            view = solver.views[0]
            paths = (view.trial_paths if kwargs["trial"] else view.paths).numpy().reshape(-1).copy()
            states.append(
                {
                    "fields": fields,
                    "paths": paths,
                    "means": tuple(float(c.mean.numpy()[0]) for c in view.channels),
                    "gradient": kwargs["gradient"],
                    "trial": kwargs["trial"],
                    "reuse": kwargs["reuse_forward"],
                }
            )

        monkeypatch.setattr(solver, "_launch_evaluation", observed)
        report = solver.solve()
        assert states and len(states) == solver.evaluations
        narrow = mode != "float64_strict"
        arithmetic32 = "arithmetic32" in mode
        depths: list[float] = []
        primitive_receipts: list[dict[str, Any]] = []
        vjp_receipts: list[dict[str, Any]] = []
        with localcontext() as ctx:
            ctx.prec = 100
            for state in states:
                assert np.isfinite(state["fields"]).all() and np.isfinite(state["paths"]).all()
                assert (
                    np.all(state["fields"] >= 0) and sum(map(float, state["fields"])) <= 1 + 2**-23
                )
                for field, path in zip(state["fields"], state["paths"], strict=True):
                    exact = Decimal.from_float(2 * float(field))
                    allowance = abs(exact) * _fraction_decimal(gamma(64, 53)) + (
                        _ulp(float(path), True) / 2 if narrow else Decimal(0)
                    )
                    assert abs(Decimal.from_float(float(path)) - exact) <= allowance
                for energy in range(2):
                    terms = [
                        float(state["paths"][m]) * float(inputs["coefficients"][m, energy])
                        for m in range(2)
                    ]
                    depth = (
                        _binary32(_binary32(terms[0]) + _binary32(terms[1]))
                        if arithmetic32
                        else math.fsum(terms)
                    )
                    depths.append(depth)
                    exact_mean = Decimal(INCIDENT) * (-Decimal.from_float(depth)).exp()
                    actual = state["means"][energy]
                    # This independently admits every compound forward value actually
                    # used by Armijo/gradient evaluations, including rejected trials.
                    assert abs(Decimal.from_float(actual) - exact_mean) <= 4 * _ulp(actual, narrow)
            native_depths = wp.array(depths, dtype=wp.float64, device=warp_device)
            factors = wp.empty(len(depths), dtype=wp.float64, device=warp_device)
            wp.launch(
                get_vjp_attenuation_admission(not narrow, arithmetic32),
                dim=len(depths),
                inputs=[native_depths, factors],
                device=warp_device,
                record_tape=False,
            )
            for depth, factor in zip(depths, factors.numpy(), strict=True):
                expected_factor = Decimal(INCIDENT) * (-Decimal.from_float(depth)).exp()
                ulp = (
                    _ulp(_binary32(float(factor)), True)
                    if arithmetic32
                    else _ulp(float(factor), False)
                )
                assert abs(Decimal.from_float(float(factor)) - expected_factor) <= 4 * ulp
                vjp_receipts.append({"depth": depth, "native_factor": float(factor)})
            # Native exp/log are admitted at all exercised positive depths/counts.
            # Zero depths need exp(0)=1; log inputs are positive by construction.
            arguments = sorted(
                set(
                    [
                        *depths,
                        *map(float, COUNTS),
                        *[mu for s in states for mu in s["means"]],
                        *[y / mu for s in states for y, mu in zip(COUNTS, s["means"], strict=True)],
                    ]
                )
            )
            positive = [x for x in arguments if x > 0]
            for is32, kernel in (
                ((False, primitive64), (True, primitive32))
                if arithmetic32
                else ((False, primitive64),)
            ):
                dtype = wp.float32 if is32 else wp.float64
                values = [_binary32(x) if is32 else x for x in positive]
                native_input = wp.array(values, dtype=dtype, device=warp_device)
                output = wp.empty(2 * len(values), dtype=dtype, device=warp_device)
                wp.launch(
                    kernel,
                    dim=len(values),
                    inputs=[native_input, output],
                    device=warp_device,
                    record_tape=False,
                )
                observed_values = output.numpy().reshape(-1, 2)
                for value, row in zip(values, observed_values, strict=True):
                    exact = Decimal.from_float(value)
                    expected = ((-exact).exp(), exact.ln())
                    primitive_receipts.append(
                        {
                            "precision": "float32" if is32 else "float64",
                            "input": value,
                            "native_exp": float(row[0]),
                            "native_log": float(row[1]),
                        }
                    )
                    for actual, target in zip(row, expected, strict=True):
                        if target == 0:
                            assert float(actual) == 0
                        elif 0 < target < Decimal("1e-300"):
                            # exp(-counts) underflows in both formats; this input
                            # exercises log only, not a normal-range exp contract.
                            assert float(actual) == 0
                        else:
                            assert abs(Decimal.from_float(float(actual)) - target) <= 4 * _ulp(
                                float(actual), is32
                            )
            final = tuple(map(float, solver.fractions.numpy().reshape(-1)))
            assert all(0 < x < 0.5 for x in final)
            reference = poisson_reference(final)
            budget = _fraction_decimal(solver_gradient_budget(mode))
            gradient = reference["gradient"]
            assert max(map(abs, gradient)) <= _fraction_decimal(TAU) + budget
            distance = sum(
                (
                    (Decimal.from_float(x) - target) ** 2
                    for x, target in zip(final, reference["optimum"], strict=True)
                ),
                Decimal(0),
            ).sqrt()
            allowance = (
                Decimal(2).sqrt()
                * (_fraction_decimal(TAU) + budget)
                / _fraction_decimal(HESSIAN_MINIMUM)
                + Decimal(2).sqrt() * Decimal(2) ** -26
            )
            assert distance <= allowance
            assert report["termination"] == "projected_gradient_tolerance"
            assert report["final_gradient_mapping"] <= float(TAU)
            assert report["gradient_mapping_threshold"] == float(TAU)
            assert all(
                row["objective"] <= row["objective_before"]
                for row in report["history"]
                if row.get("accepted", False)
            )
            # At an admitted mean, the loss is Lipschitz with |1-y/mu|<=1
            # on the final box; each mean is within four retained-value ULPs.
            series_tail = (
                Decimal(2 * INCIDENT) * (Decimal(1) / 7) ** 25 / (25 * (1 - (Decimal(1) / 7) ** 2))
            )
            loss_bound = (
                sum(
                    4 * _ulp(float(channel.mean.numpy()[0]), narrow)
                    for channel in solver.views[0].channels
                )
                + _fraction_decimal(2 * INCIDENT * gamma(128, 53))
                + series_tail
            )
            assert (
                abs(Decimal.from_float(report["final_objective"]) - reference["loss"]) <= loss_bound
            )
        np.testing.assert_array_equal(solver._status.numpy(), 0)
        record_property(
            "ideal_mle_acceptance",
            json.dumps(
                {
                    "mode": mode,
                    "step": step_selection,
                    "evaluations": len(states),
                    "gradient_budget": str(budget),
                    "distance": str(distance),
                    "distance_bound": str(allowance),
                    "fields": final,
                    "backend_admitted_states": len(states),
                    "actual_states": [
                        {
                            **state,
                            "fields": list(map(float, state["fields"])),
                            "paths": list(map(float, state["paths"])),
                        }
                        for state in states
                    ],
                    "native_primitive_receipts": primitive_receipts,
                    "native_VJP_attenuation_receipts": vjp_receipts,
                },
                sort_keys=True,
            ),
        )

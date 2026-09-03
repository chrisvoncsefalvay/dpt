"""Joint factor/rigid-chart checks and statistical saved-state checks."""

from __future__ import annotations

import gc
import importlib
import json
import math
from contextlib import nullcontext
from copy import copy
from typing import Any, Literal

import numpy as np

from dpt.geometry import RayBundleGeometry, RigidTransform
from dpt.interop.torch import project_spectral_signal
from dpt.interop.torch_transport import ExpectedTransportEstimator
from dpt.material_projection import prepare_material_projection
from dpt.projection import ProjectionSpec
from dpt.spectral import SpectralSpec, prepare_spectral
from dpt.transport.rng import HistoryBatch
from dpt.volumes import GridSpec
from helpers.gradient_references import (
    COEFFICIENTS,
    ENDS,
    INTERCEPTS,
    RESPONSE,
    SEED,
    SLOPES,
    SOURCES,
    WEIGHTS,
    affine_reference,
)
from helpers.torch_expected_transport_inputs import analytic, coordinates, evaluate, layer_problem


def _world(point: tuple[float, ...]) -> tuple[float, float, float]:
    # Exact quarter turn and translation, independent of production pose code.
    return -point[1] + 0.25, point[0] - 0.5, point[2] + 0.75


def run_joint_pose_factor_saved_state(
    device: str,
    domain: Literal["linear", "log"],
    shared: bool,
    execution: Literal["serial", "fused"],
    reverse_order: bool,
    monkeypatch: Any,
    record_property: Any,
) -> None:
    torch, wp = importlib.import_module("torch"), importlib.import_module("warp")
    from dpt.torch import compose_pose

    cuda = wp.get_device(device).is_cuda
    producer = torch.cuda.Stream(device=device) if cuda else None
    owner = torch.cuda.Stream(device=device) if cuda else None
    caller = torch.cuda.Stream(device=device) if cuda else None
    backward = torch.cuda.Stream(device=device) if cuda else None
    guard = torch.cuda.stream(producer) if cuda else nullcontext()
    with guard:
        grid = GridSpec((7, 7, 7), (1.0, 1.0, 1.0), (-3.0, -3.0, -3.0))
        geometry = RayBundleGeometry(tuple(map(_world, SOURCES)), tuple(map(_world, ENDS)))
        host = np.asarray(
            [
                [
                    intercept + slope[0] * (x - 3) + slope[1] * (y - 3) + slope[2] * (z - 3)
                    for z in range(7)
                    for y in range(7)
                    for x in range(7)
                ]
                for intercept, slope in zip(INTERCEPTS, SLOPES, strict=True)
            ],
            dtype=np.float32,
        ).reshape(2, *grid.shape)
        tensors: list[dict[str, Any]] = []
        for amplitude, displacement in ((1.0, (0.0, 0.0, 0.0)), (0.75, (0.125, -0.0625, 0.03125))):
            fields = torch.tensor(
                amplitude * host, dtype=torch.float32, device=device, requires_grad=True
            )
            local = torch.zeros(6, dtype=torch.float64, device=device, requires_grad=True)
            anchor = RigidTransform(
                (0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
                (0.25 - displacement[1], -0.5 + displacement[0], 0.75 + displacement[2]),
            )
            pose = compose_pose(
                torch.tensor(anchor.packed(), dtype=torch.float64, device=device), local
            )
            coeff = torch.tensor(
                COEFFICIENTS, dtype=torch.float32, device=device, requires_grad=True
            )
            pattern = np.asarray((0.75, 1.0, 1.25), dtype=np.float32)
            weights = torch.tensor(
                np.asarray(WEIGHTS, dtype=np.float32)
                if shared
                else np.asarray(WEIGHTS, dtype=np.float32)[:, None] * pattern,
                device=device,
                requires_grad=True,
            )
            response = torch.tensor(
                np.asarray(RESPONSE, dtype=np.float32)
                if shared
                else np.asarray(RESPONSE, dtype=np.float32)[:, None] * pattern[::-1].copy(),
                device=device,
                requires_grad=True,
            )
            tensors.append(
                {
                    "fields": fields,
                    "local": local,
                    "pose": pose,
                    "coefficients": coeff,
                    "weights": weights,
                    "response": response,
                    "amplitude": amplitude,
                    "displacement": displacement,
                }
            )
        material = prepare_material_projection(
            grid,
            geometry,
            ProjectionSpec(
                13,
                active_pose=True,
                active_volume=True,
                precision="float64",
                integration="cell_gauss",
            ),
            materials=2,
            fields=wp.from_torch(tensors[0]["fields"].reshape(-1), requires_grad=False),
            out_paths=wp.empty(6, dtype=wp.float64, device=device),
            cotangent_paths=wp.empty(6, dtype=wp.float64, device=device),
            device=device,
            stream=wp.stream_from_torch(owner) if cuda else None,
            channel_execution=execution,
        )
        spectral = prepare_spectral(
            SpectralSpec(
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
            ),
            max_pixels=3,
            device=device,
            stream=material.context.stream,
        )
    if cuda:
        assert caller is not None and producer is not None
        caller.wait_stream(producer)

        def forbid_wait(*args: Any, **kwargs: Any) -> None:
            raise AssertionError(
                "joint pose/factor autograd must preserve stream dependencies without a global wait"
            )

        monkeypatch.setattr(torch.cuda, "synchronize", forbid_wait)
    with torch.cuda.stream(caller) if cuda else nullcontext():
        outputs = [
            project_spectral_signal(
                t["fields"],
                t["pose"],
                t["coefficients"],
                t["weights"],
                t["response"],
                material_workspace=material,
                spectral_workspace=spectral,
            )
            for t in tensors
        ]
        preserved = [x.detach().clone() for x in outputs]
        seed = torch.tensor(SEED, dtype=torch.float64, device=device).reshape(geometry.pixels)
    if cuda:
        assert backward is not None and caller is not None
        backward.wait_stream(caller)
    with torch.cuda.stream(backward) if cuda else nullcontext():
        for index in (1, 0) if reverse_order else (0, 1):
            outputs[index].backward(seed)
        retained = [
            [t[name].grad for name in ("fields", "local", "coefficients", "weights", "response")]
            for t in tensors
        ]
    if cuda:
        torch.cuda.current_stream(device).wait_stream(backward)
    del seed
    gc.collect()
    with torch.cuda.stream(caller) if cuda else nullcontext():
        for _ in range(16):
            churn = torch.empty(4096, dtype=torch.float32, device=device)
            churn.fill_(17)
    if cuda:
        torch.cuda.current_stream(device).wait_stream(caller)
    assert outputs[0].data_ptr() != outputs[1].data_ptr()
    for t, output, old, gradients in zip(tensors, outputs, preserved, retained, strict=True):
        reference = affine_reference(shared, domain, t["amplitude"], t["displacement"])
        assert torch.equal(output.detach(), old)
        np.testing.assert_allclose(
            output.detach().cpu().numpy().reshape(-1), reference["output"], rtol=3e-12, atol=3e-12
        )
        np.testing.assert_allclose(
            t["local"].grad.cpu().numpy(), reference["pose"], rtol=3e-10, atol=3e-11
        )
        for name in ("coefficients", "weights", "response"):
            np.testing.assert_allclose(
                t[name].grad.cpu().numpy(), reference[name], rtol=6e-6, atol=2e-7
            )
        contraction = float(
            np.sum(t["fields"].grad.cpu().numpy().astype(np.float64) * host.astype(np.float64))
        )
        assert (
            abs(contraction - reference["field_amplitude"])
            <= 6e-6 * abs(reference["field_amplitude"]) + 2e-7
        )
        assert all(g is not None and torch.isfinite(g).all().item() for g in gradients)
    record_property(
        "joint_saved_state_reference",
        json.dumps(
            {
                "domain": domain,
                "shared": shared,
                "execution": execution,
                "reverse_order": reverse_order,
                "oracle": (
                    "affine midpoint integrals, explicit inverse-coordinate generators "
                    "and independent scalar spectral chain"
                ),
                "device": device,
            },
            sort_keys=True,
        ),
    )


def run_statistical_transport_saved_state(
    device: str,
    scoring: Literal["photon-count", "energy-kev"],
    reverse_order: bool,
    monkeypatch: Any,
    record_property: Any,
) -> None:
    torch, wp = importlib.import_module("torch"), importlib.import_module("warp")
    cuda = wp.get_device(device).is_cuda
    owner = torch.cuda.Stream(device=device) if cuda else None
    caller = torch.cuda.Stream(device=device) if cuda else None
    backward = torch.cuda.Stream(device=device) if cuda else None
    with torch.cuda.stream(caller) if cuda else nullcontext():
        problem = layer_problem(
            torch,
            wp,
            device,
            estimator="analogue",
            scoring=scoring,
            amplitude=0.7,
            stream=wp.stream_from_torch(owner) if cuda else None,
        )
        sets = []
        for shift, amplitude in ((0.0, 0.7), (0.25, 1.1)):
            local = copy(problem)
            local.log_voxel_density = (problem.log_voxel_density.detach() + shift).requires_grad_()
            local.log_material_density = (
                problem.log_material_density.detach() - shift / 2
            ).requires_grad_()
            local.source_amplitude = torch.tensor(
                amplitude, dtype=torch.float64, device=device, requires_grad=True
            )
            sets.append(local)
        outputs: list[list[Any]] = [[], []]
        # Independent entire seed keys, not slices of one scrambled replicate.
        for replicate in range(4):
            for index, p in enumerate(sets):
                p.estimator = ExpectedTransportEstimator(
                    HistoryBatch(19001 + 37 * replicate + 7 * index, 0, 2048),
                    HistoryBatch(29003 + 41 * replicate + 11 * index, 0, 2048),
                )
                outputs[index].append(evaluate(p).signal)
        old = [[value.detach().clone() for value in row] for row in outputs]
    if cuda:
        assert backward is not None and caller is not None
        backward.wait_stream(caller)

        def forbid_wait(*args: Any, **kwargs: Any) -> None:
            raise AssertionError(
                "statistical transport must preserve stream dependencies without a global wait"
            )

        monkeypatch.setattr(torch.cuda, "synchronize", forbid_wait)
    with torch.cuda.stream(backward) if cuda else nullcontext():
        order = (1, 0) if reverse_order else (0, 1)
        for index in order:
            for output in reversed(outputs[index]) if reverse_order else outputs[index]:
                output.backward(torch.full_like(output, -0.4 / 4))
        retained = [[value.grad for value in coordinates(p)] for p in sets]
    if cuda:
        torch.cuda.current_stream(device).wait_stream(backward)
    gc.collect()
    with torch.cuda.stream(caller) if cuda else nullcontext():
        for _ in range(16):
            churn = torch.empty(4096, dtype=torch.float64, device=device)
            churn.fill_(19)
    if cuda:
        torch.cuda.current_stream(device).wait_stream(caller)
    evidence = []
    for p, row, saved, gradients in zip(sets, outputs, old, retained, strict=True):
        mean, target = analytic(p, -0.4)
        unit = p.spec.energy_kev if scoring == "energy-kev" else 1.0
        survival = float(target[2]) / (-0.4 * p.source.weight * unit)
        total = 4 * 2048
        log_tail = math.log(2e10)
        bernstein = math.sqrt(2 * survival * (1 - survival) * log_tail / total) + 2 * log_tail / (
            3 * total
        )
        assert all(
            torch.equal(value.detach(), kept) for value, kept in zip(row, saved, strict=True)
        )
        assert len({value.data_ptr() for value in row}) == 4
        observed_mean = sum(value.item() for value in row) / 4
        assert (
            abs(observed_mean - mean)
            <= p.source_amplitude.item() * p.source.weight * unit * bernstein + 3e-12
        )
        for gradient, expected in zip(gradients, target, strict=True):
            assert (
                gradient is not None
                and gradient.dtype == torch.float64
                and torch.isfinite(gradient).all().item()
            )
            allowance = np.abs(expected) * bernstein / survival + 3e-12
            assert np.all(np.abs(gradient.cpu().numpy() - expected) <= allowance)
        # All density scores use one independent derivative survival sample.
        # Their rank-one ratio is an exact absorption-only score-law control.
        direct = gradients[2].item() / float(target[2])
        for actual, expected in zip(gradients[:2], target[:2], strict=True):
            np.testing.assert_allclose(
                actual.cpu().numpy() / expected, direct, rtol=2e-12, atol=2e-12
            )
        evidence.append(
            {
                "survival": survival,
                "histories": total,
                "mean": observed_mean,
                "reference": mean,
                "bernstein": bernstein,
            }
        )
    record_property(
        "independent_expected_score_saved_state",
        json.dumps(
            {
                "scoring": scoring,
                "reverse_order": reverse_order,
                "independent_seed_replicates": 4,
                "oracle": (
                    "Bernoulli survival law and analytic log-density/source-amplitude derivatives"
                ),
                "cases": evidence,
            },
            sort_keys=True,
        ),
    )

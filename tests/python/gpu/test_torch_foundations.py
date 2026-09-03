"""Rigid-chart, batch/view and cross-stream Torch behaviour with native VJPs."""

from __future__ import annotations

import gc
import importlib
from collections.abc import Iterator
from contextlib import nullcontext
from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from dpt._runtime import cpu_test_context
from dpt.contracts import ContractError
from dpt.geometry import DetectorGeometry, RigidTransform
from dpt.geometry import compose_pose as host_compose
from dpt.projection import ProjectionSpec, prepare_projection
from dpt.torch import Projector, compose_pose, project_optical_depth
from dpt.volumes import GridSpec

torch: Any = importlib.import_module("torch")
wp: Any = importlib.import_module("warp")
pytestmark = pytest.mark.gpu


@pytest.fixture(autouse=True)
def _kernel_context(warp_device: str) -> Iterator[None]:
    with cpu_test_context() if warp_device == "cpu" else nullcontext():
        yield


def _problem(
    device: str, stream: Any = None, *, active_volume: bool = True
) -> tuple[Any, Any, Any]:
    grid = GridSpec((3, 3, 4), (0.7, 0.8, 0.9), (-1.0, -1.0, -1.0))
    detector = DetectorGeometry(
        (-8.0, 0.1, -0.2),
        (8.0, -0.3, -0.4),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.2, 0.2),
        (2, 3),
    )
    workspace = prepare_projection(
        grid,
        detector,
        ProjectionSpec(
            47,
            active_volume=active_volume,
            active_pose=True,
            precision="float64",
            integration="cell_gauss",
        ),
        device=device,
        stream=stream,
    )
    fields = torch.linspace(0.1, 0.3, grid.voxels, dtype=torch.float32, device=device).reshape(
        grid.shape
    )
    anchor = torch.tensor(RigidTransform().packed(), dtype=torch.float64, device=device)
    return workspace, fields, anchor


@pytest.mark.parametrize(
    "coordinates",
    [(0.0,) * 6, (0.03, -0.02, 0.01, 1e-10, -2e-10, 3e-10), (0.03, -0.02, 0.01, 0.04, -0.03, 0.02)],
)
def test_se3_chart_matches_independent_host_formula_and_torch_gradcheck(
    coordinates: tuple[float, ...], warp_device: str
) -> None:
    _, _, anchor = _problem(warp_device)
    tangent = torch.tensor(coordinates, dtype=torch.float64, device=warp_device, requires_grad=True)
    expected = host_compose(RigidTransform(), coordinates)
    actual = compose_pose(anchor, tangent)
    np.testing.assert_allclose(
        actual.detach().cpu().numpy(), expected.packed(), rtol=3e-15, atol=3e-15
    )
    assert torch.autograd.gradcheck(lambda x: compose_pose(anchor, x), (tangent,), eps=1e-6)


def test_active_rigid_pose_gradcheck_uses_twelve_storage_partials(warp_device: str) -> None:
    workspace, fields, anchor = _problem(warp_device, active_volume=False)
    tangent = torch.tensor(
        [0.03, -0.02, 0.01, 0.04, -0.03, 0.02],
        dtype=torch.float64,
        device=warp_device,
        requires_grad=True,
    )

    def evaluate(value: Any) -> Any:
        return project_optical_depth(fields, compose_pose(anchor, value), workspace=workspace)

    assert torch.autograd.gradcheck(evaluate, (tangent,), eps=2e-5, atol=2e-6, rtol=2e-5)
    image = evaluate(tangent)
    gradient = torch.autograd.grad(image.sum(), tangent)[0]
    assert gradient.shape == (6,) and torch.isfinite(gradient).all().item()
    assert torch.count_nonzero(gradient[3:]).item() == 3


def test_saved_pose_and_field_mutations_are_detected_and_outputs_are_owned(
    warp_device: str,
) -> None:
    workspace, fields, anchor = _problem(warp_device)
    fields.requires_grad_()
    first = project_optical_depth(fields, anchor, workspace=workspace)
    retained = first.detach().clone()
    second = project_optical_depth(0.5 * fields, anchor, workspace=workspace)
    assert first.data_ptr() != second.data_ptr()
    torch.testing.assert_close(first, retained)
    with torch.no_grad():
        anchor[9] = 0.02
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        first.sum().backward()


@pytest.mark.parametrize("layout", ["shared", "view"])
def test_projector_keeps_pose_broadcast_gradient(layout: str, warp_device: str) -> None:
    first, fields, anchor = _problem(warp_device)
    panel = replace(first.geometry, origin_mm=(8.0, -0.24, -0.31))
    shape = (6,) if layout == "shared" else (2, 6)
    tangent = torch.nn.Parameter(torch.zeros(shape, dtype=torch.float64, device=warp_device))
    module = Projector(
        grid=first.grid,
        geometry=(first.geometry, panel),
        anchor=anchor,
        precision="fp64",
        samples_per_ray=47,
    )
    images = module(fields, tangent)
    assert images.shape == (2, 2, 3)
    assert torch.autograd.gradcheck(
        lambda chart: module(fields, chart), (tangent,), eps=2e-5, atol=2e-6, rtol=2e-5
    )
    images.sum().backward()
    assert tangent.grad.shape == shape
    assert torch.count_nonzero(tangent.grad).item() > 0
    with pytest.raises(ContractError, match="grid shape"):
        module(fields[:0])
    with pytest.raises(ContractError, match="pose must have shape"):
        module(fields, torch.zeros((3, 6), dtype=torch.float64, device=warp_device))


def test_cuda_bridge_orders_distinct_streams_and_preserves_temporary_lifetimes(
    warp_device: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    if warp_device == "cpu":
        workspace, fields, anchor = _problem(warp_device)
        fields.requires_grad_()
        image = project_optical_depth(fields, anchor, workspace=workspace)
        second = project_optical_depth(0.5 * fields, anchor, workspace=workspace)
        retained = image.detach().clone()
        del workspace, anchor
        gc.collect()
        gradient = torch.autograd.grad(image.sum() + second.sum(), fields)[0]
        torch.testing.assert_close(image, retained)
        assert torch.isfinite(gradient).all().item() and gradient.abs().sum().item() > 0
        return
    producer = torch.cuda.Stream(device=warp_device)
    caller = torch.cuda.Stream(device=warp_device)
    owner = torch.cuda.Stream(device=warp_device)
    consumer = torch.cuda.Stream(device=warp_device)

    def global_wait(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("adapter used a global CUDA synchronization")

    monkeypatch.setattr(torch.cuda, "synchronize", global_wait)
    monkeypatch.setattr(wp, "synchronize", global_wait)
    with torch.cuda.stream(producer):
        workspace, fields, anchor = _problem(warp_device, wp.stream_from_torch(owner))
        fields.requires_grad_()
        tangent = torch.zeros(6, dtype=torch.float64, device=warp_device, requires_grad=True)
    caller.wait_stream(producer)
    with torch.cuda.stream(caller):
        pose = compose_pose(anchor, tangent)
        image = project_optical_depth(fields, pose, workspace=workspace)
        second = project_optical_depth(0.5 * fields, pose, workspace=workspace)
        expected = image.detach().clone()
    del workspace, anchor, pose
    gc.collect()
    consumer.wait_stream(caller)
    with torch.cuda.stream(consumer):
        loss = image.sum() + second.sum()
        gradients = torch.autograd.grad(loss, (fields, tangent))
        consumer.wait_stream(caller)
        # Exercise allocator reuse while the returned images remain live.
        for _ in range(8):
            temporary = torch.empty(1024, dtype=torch.float64, device=warp_device)
            temporary.fill_(float("nan"))
        torch.testing.assert_close(image, expected)
        assert all(torch.isfinite(value).all().item() for value in gradients)
        assert gradients[0].abs().sum().item() > 0
        assert gradients[1].abs().sum().item() > 0

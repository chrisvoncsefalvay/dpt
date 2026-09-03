"""Check CUDA dependencies, device execution and Torch/Warp interoperability."""

from __future__ import annotations

import argparse
import importlib
import math
import platform
import sys
from collections.abc import Sequence
from types import ModuleType
from typing import Any


class PreflightError(RuntimeError):
    """A clear, expected GPU preflight failure."""


def _load_dependency(name: str) -> ModuleType:
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as error:
        if error.name == name:
            raise PreflightError(
                f"optional dependency {name!r} is not installed; install "
                "`pip install 'differentiable-photon-transport[warp]'` "
                "for Warp operators or the [gpu] extra for Torch examples"
            ) from error
        raise PreflightError(
            f"optional dependency {name!r} has a missing transitive dependency: {error.name!r}"
        ) from error
    except Exception as error:
        raise PreflightError(f"could not import optional dependency {name!r}: {error}") from error


def _check_torch(torch: Any, wp: Any = None) -> str:
    cuda = getattr(torch, "cuda", None)
    if cuda is None or not callable(getattr(cuda, "is_available", None)):
        raise PreflightError("PyTorch was imported but does not expose the torch.cuda API")

    try:
        is_available = bool(cuda.is_available())
        device_count = int(cuda.device_count())
    except Exception as error:
        raise PreflightError(f"PyTorch could not query CUDA devices: {error}") from error
    if not is_available:
        build_cuda = getattr(getattr(torch, "version", None), "cuda", None)
        raise PreflightError(
            "PyTorch cannot access CUDA "
            f"(torch.version.cuda={build_cuda!r}, visible_device_count={device_count})"
        )

    try:
        device = torch.device("cuda:0")
        stream = torch.cuda.Stream(device=device)
        with torch.cuda.stream(stream):
            left = torch.tensor((1.0, 2.0), dtype=torch.float32, device=device)
            right = torch.tensor((3.0, 4.0), dtype=torch.float32, device=device)
            observed = float(torch.dot(left, right).item())
            if wp is not None:
                values = torch.tensor((1, 2, 3), dtype=torch.int32, device=device)
                shared = wp.from_torch(values)
                owning = wp.stream_from_torch(stream)
                if int(shared.ptr) != int(values.data_ptr()):
                    raise PreflightError("Torch/Warp interop did not preserve shared storage")
                if int(owning.cuda_stream) != int(stream.cuda_stream):
                    raise PreflightError("Torch/Warp interop did not preserve stream ownership")
                kernel = importlib.import_module("dpt._gpu_preflight_kernel").probe_add_one
                wp.launch(kernel, dim=3, inputs=[shared], stream=owning, record_tape=False)
                wp.synchronize_stream(owning)
                if values.tolist() != [2, 3, 4]:
                    raise PreflightError("Torch/Warp shared-stream kernel returned wrong values")
    except Exception as error:
        raise PreflightError(f"PyTorch CUDA operation failed: {error}") from error
    if observed != 11.0:
        raise PreflightError(f"PyTorch CUDA dot product returned {observed!r}, expected 11.0")

    build_cuda = getattr(getattr(torch, "version", None), "cuda", "unknown")
    return (
        f"PyTorch {getattr(torch, '__version__', 'unknown')}: PASS — "
        f"{cuda.get_device_name(device)}; build CUDA {build_cuda}; CUDA dot=11.0"
        + ("; shared-storage/stream interop PASS" if wp is not None else "")
    )


def _probe_projection(wp: Any, device: Any) -> None:
    """Exercise the FP64 projector, tiled pose VJP and volume adjoint on 4 rays."""
    from dpt.geometry import DetectorGeometry, RigidTransform
    from dpt.projection import (
        ProjectionSpec,
        prepare_projection,
        project_optical_depth,
        projection_vjp,
    )
    from dpt.volumes import GridSpec

    grid = GridSpec((3, 3, 3), (1.0, 1.0, 1.0), (-1.0, -1.0, -1.0))
    geometry = DetectorGeometry(
        (0.0, 0.0, -5.0),
        (-0.25, -0.25, 5.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.5, 0.5),
        (2, 2),
    )
    stream = wp.Stream(device=device) if device.is_cuda else None
    workspace = prepare_projection(
        grid,
        geometry,
        ProjectionSpec(16, active_volume=True, precision="float64"),
        device=device,
        stream=stream,
    )
    with workspace.context.scope():
        field = wp.array([0.25] * grid.voxels, dtype=wp.float32, device=device)
        pose = wp.array(RigidTransform().packed(), dtype=wp.float64, device=device)
        depth = wp.empty(geometry.pixels, dtype=wp.float64, device=device)
        seed = wp.array((0.2, -0.4, 0.3, 0.5), dtype=wp.float64, device=device)
        gradient = wp.empty(grid.voxels, dtype=wp.float32, device=device)
        pose_gradient = wp.empty(6, dtype=wp.float64, device=device)
        project_optical_depth(field, pose, out_L=depth, workspace=workspace, stream=stream)
        projection_vjp(
            field,
            pose,
            cotangent=seed,
            out_mu=gradient,
            out_pose_tangent=pose_gradient,
            workspace=workspace,
            stream=stream,
        )
        workspace.check_status()
    observed = [float(value) for value in depth.numpy()]
    expected = 0.75 * math.sqrt(1.0 + 2 * (0.25 / 10.0) ** 2)
    if any(not math.isclose(value, expected, rel_tol=1e-12, abs_tol=1e-12) for value in observed):
        raise PreflightError("DPT FP64 projection disagrees with the analytic uniform-field result")
    lhs = math.fsum(
        value * weight for value, weight in zip(observed, (0.2, -0.4, 0.3, 0.5), strict=True)
    )
    rhs = 0.25 * math.fsum(float(value) for value in gradient.numpy())
    if not math.isclose(lhs, rhs, rel_tol=3e-6, abs_tol=1e-9):
        raise PreflightError(f"DPT volume adjoint dot product failed: {lhs!r} != {rhs!r}")
    if any(not math.isfinite(float(value)) for value in pose_gradient.numpy()):
        raise PreflightError("DPT tiled FP64 pose VJP returned a nonfinite gradient")


def _check_warp(wp: Any) -> str:
    try:
        wp.init()
        device_count = int(wp.get_cuda_device_count())
    except Exception as error:
        raise PreflightError(f"Warp could not initialise CUDA: {error}") from error
    if device_count < 1:
        raise PreflightError("Warp reports no CUDA devices")

    try:
        device = wp.get_device("cuda:0")
        _probe_projection(wp, device)
    except Exception as error:
        raise PreflightError(f"Warp CUDA kernel execution failed: {error}") from error
    device_name = getattr(device, "name", "cuda:0")
    architecture = getattr(device, "arch", "unknown architecture")
    return (
        f"Warp {getattr(wp, '__version__', 'unknown')}: PASS — "
        f"{device_name}; {architecture}; FP64 projection, tiled pose VJP and volume adjoint PASS"
    )


def run_preflight() -> tuple[str, ...]:
    """Require production Warp CUDA execution; report optional Torch interop separately."""
    wp = _load_dependency("warp")
    warp_result = _check_warp(wp)
    try:
        torch = importlib.import_module("torch")
    except ModuleNotFoundError as error:
        if error.name != "torch":
            raise PreflightError(
                f"PyTorch has a missing transitive dependency: {error.name!r}"
            ) from error
        torch_result = "PyTorch interop: SKIP — optional Torch dependency is not installed"
    except Exception as error:
        raise PreflightError(f"could not import optional PyTorch dependency: {error}") from error
    else:
        cuda = getattr(torch, "cuda", None)
        available = getattr(cuda, "is_available", None)
        if not callable(available):
            raise PreflightError("PyTorch was imported but does not expose the torch.cuda API")
        try:
            torch_cuda = bool(available())
        except Exception as error:
            raise PreflightError(f"PyTorch could not query CUDA devices: {error}") from error
        if not torch_cuda:
            torch_result = "PyTorch interop: SKIP — installed Torch build cannot access CUDA"
        else:
            torch_result = _check_torch(torch, wp)
    return (warp_result, torch_result)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the local GPU preflight and return a process exit status."""
    parser = argparse.ArgumentParser(
        description="Verify DPT GPU dependencies with real tiny CUDA operations."
    )
    parser.parse_args(argv)

    print(f"DPT GPU preflight — Python {platform.python_version()}", flush=True)
    try:
        results = run_preflight()
    except PreflightError as error:
        print(f"GPU preflight FAILED: {error}", file=sys.stderr)
        return 1

    for result in results:
        print(result)
    print("GPU preflight PASSED — DPT Warp CUDA projection and gradient checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

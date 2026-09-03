"""Optional-dependency and CUDA failure paths without a GPU or Torch import."""

from types import SimpleNamespace
from typing import Any

import pytest

from dpt import gpu_check


def test_warp_only_preflight_reports_missing_torch_as_skip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    warp = object()

    def modules(name: str) -> Any:
        if name == "warp":
            return warp
        raise ModuleNotFoundError(name=name)

    monkeypatch.setattr(gpu_check.importlib, "import_module", modules)
    monkeypatch.setattr(
        gpu_check, "_check_warp", lambda observed: "Warp: PASS" if observed is warp else "wrong"
    )
    results = gpu_check.run_preflight()
    assert results[0] == "Warp: PASS"
    assert "SKIP" in results[1] and "not installed" in results[1]


def test_cpu_torch_build_does_not_block_warp_cuda_preflight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    monkeypatch.setattr(
        gpu_check.importlib, "import_module", lambda name: torch if name == "torch" else object()
    )
    monkeypatch.setattr(gpu_check, "_check_warp", lambda _: "Warp: PASS")
    results = gpu_check.run_preflight()
    assert results[0] == "Warp: PASS"
    assert "SKIP" in results[1] and "cannot access CUDA" in results[1]


def test_missing_warp_error_targets_installed_package(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing(name: str) -> Any:
        raise ModuleNotFoundError(name=name)

    monkeypatch.setattr(gpu_check.importlib, "import_module", missing)
    with pytest.raises(gpu_check.PreflightError, match=r"pip install.*\[warp\]"):
        gpu_check.run_preflight()


def test_no_cuda_devices_is_a_failure() -> None:
    wp = SimpleNamespace(init=lambda: None, get_cuda_device_count=lambda: 0)
    with pytest.raises(gpu_check.PreflightError, match="no CUDA devices"):
        gpu_check._check_warp(wp)


def test_real_probe_failure_is_preserved(monkeypatch: pytest.MonkeyPatch) -> None:
    def failed(*_: object) -> None:
        raise ArithmeticError("adjoint mismatch fixture")

    wp = SimpleNamespace(
        init=lambda: None, get_cuda_device_count=lambda: 1, get_device=lambda _: object()
    )
    monkeypatch.setattr(gpu_check, "_probe_projection", failed)
    with pytest.raises(gpu_check.PreflightError, match="adjoint mismatch fixture"):
        gpu_check._check_warp(wp)

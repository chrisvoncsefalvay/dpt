"""Environment and prepared-device details remain available to benchmark scripts."""

from types import SimpleNamespace
from typing import Any

import pytest

from dpt import experiments
from dpt.experiments import describe_device, describe_environment


def test_environment_reads_installed_versions_without_importing_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    installed = [
        SimpleNamespace(metadata={"Name": name}, version=number)
        for name, number in (("torch", "test-torch"), ("scipy", "test-scipy"))
    ]
    monkeypatch.setattr(experiments, "distributions", lambda: installed)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    environment = describe_environment()
    assert environment["distributions"] == {"torch": "test-torch", "scipy": "test-scipy"}
    assert environment["CUDA_VISIBLE_DEVICES"] == "2"


def test_device_metadata_uses_prepared_context_and_public_warp_queries() -> None:
    device = SimpleNamespace(is_cuda=True, name="fixture GPU", arch=121)
    wp = SimpleNamespace(
        __version__="test-warp",
        get_cuda_driver_version=lambda: (13, 0),
        get_cuda_toolkit_version=lambda: (13, 0),
        config=SimpleNamespace(mode="release", verify_cuda=False, ptx_target_arch=121),
    )
    context: Any = SimpleNamespace(wp=wp, device=device, stream=SimpleNamespace(cuda_stream=17))
    metadata = describe_device(context)
    assert metadata["device_class"] == "cuda"
    assert metadata["device_name"] == "fixture GPU"
    assert metadata["cuda_arch"] == 121
    assert metadata["cuda_driver_version"] == (13, 0)
    assert metadata["cuda_toolkit_version"] == (13, 0)
    assert metadata["cuda_stream"] == 17
    assert metadata["warp_config"] == {
        "mode": "release",
        "verify_cuda": False,
        "ptx_target_arch": 121,
    }

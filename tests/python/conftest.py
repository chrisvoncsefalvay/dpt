"""Keep CPU CI independent of the optional CUDA runtime, including collection."""

from pathlib import Path

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--run-gpu", action="store_true", help="require and run the real CUDA suite")


def pytest_ignore_collect(collection_path: Path, config: pytest.Config) -> bool | None:
    if collection_path.is_relative_to(Path(__file__).parent / "gpu") and not config.getoption(
        "--run-gpu"
    ):
        return True
    return None


def pytest_report_header(config: pytest.Config) -> str:
    return "CUDA suite: required" if config.getoption("--run-gpu") else "CUDA suite: not requested"


def pytest_sessionstart(session: pytest.Session) -> None:
    """Select the kernel cache and initialise the explicitly requested CUDA runtime."""
    import importlib.util

    if importlib.util.find_spec("warp") is not None:
        import warp as wp

        wp.config.kernel_cache_dir = str(session.config.rootpath / ".warp-cache")
        if session.config.getoption("--run-gpu"):
            wp.init()


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "warp_device" in metafunc.fixturenames:
        devices = ["cpu"]
        if metafunc.config.getoption("--run-gpu"):
            devices.append("cuda:0")
        metafunc.parametrize("warp_device", devices)

"""Small output and environment helpers for example and benchmark scripts."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from importlib.metadata import distributions
from pathlib import Path
from typing import Any


def experiment_parser(
    entrypoint: str | Path,
    description: str | None,
    *,
    configuration: bool = True,
    cuda: bool = True,
) -> argparse.ArgumentParser:
    """Build the common experiment command-line arguments."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--output", type=Path, required=True)
    if configuration:
        parser.add_argument(
            "--config", type=Path, default=Path(entrypoint).with_name("config.json")
        )
    if cuda:
        parser.add_argument("--device", default="cuda:0")
    return parser


def write_json(path: Path, payload: object) -> None:
    """Save settings or numerical results, rejecting nonfinite JSON numbers."""
    text = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def prepare_output(
    path: Path,
    *,
    configuration: dict[str, Any],
    metadata: dict[str, object] | None = None,
) -> Path:
    """Create an ordinary output directory and save the supplied settings."""
    output = path.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "configuration.json", configuration)
    if metadata is not None:
        write_json(output / "metadata.json", metadata)
    return output


def write_metadata(output: Path, **fields: object) -> None:
    """Add execution details to an example's small metadata file."""
    path = output / "metadata.json"
    metadata = json.loads(path.read_text()) if path.exists() else {}
    metadata.update(fields)
    write_json(path, metadata)


def describe_environment() -> dict[str, object]:
    """Read Python and installed-package versions without importing CUDA."""
    installed = {
        str(item.metadata["Name"]): item.version
        for item in distributions()
        if item.metadata["Name"]
    }
    environment: dict[str, object] = {
        "python": sys.version,
        "platform": platform.platform(),
        "distributions": dict(sorted(installed.items())),
    }
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        environment["CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"]
    return environment


def describe_device(context: Any) -> dict[str, object]:
    """Read device and Warp configuration from an already prepared context."""
    wp, device = context.wp, context.device
    info: dict[str, object] = {
        "device": str(device),
        "device_class": "cuda" if device.is_cuda else "cpu",
        "device_name": str(getattr(device, "name", device)),
        "warp_version": str(getattr(wp, "__version__", "unknown")),
    }
    if device.is_cuda:
        info["cuda_arch"] = getattr(device, "arch", None)
        for name in ("cuda_driver_version", "cuda_toolkit_version"):
            query = getattr(wp, f"get_{name}", None)
            info[name] = query() if callable(query) else None
        if context.stream is not None:
            info["cuda_stream"] = int(context.stream.cuda_stream)
    config = getattr(wp, "config", None)
    info["warp_config"] = {
        name: getattr(config, name)
        for name in (
            "mode",
            "verify_cuda",
            "verify_fp",
            "enable_backward",
            "fast_math",
            "ptx_target_arch",
        )
        if isinstance(getattr(config, name, None), (str, bool, int, float))
    }
    return info

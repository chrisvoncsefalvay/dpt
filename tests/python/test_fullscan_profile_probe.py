"""CPU checks for the public profiling command's requested operation."""

from __future__ import annotations

import ast
import importlib.util
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SOURCE = Path(__file__).resolve().parents[2] / "benchmarks/fullscan_profile.py"
SPEC = importlib.util.spec_from_file_location("_fullscan_profile_cli_test", SOURCE)
assert SPEC is not None and SPEC.loader is not None
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)


def arguments(*extra):
    return PROBE.parse_args(
        ["--data", "/input", "--output", "/output", "--cotangent", "/seed.npy", *extra]
    )


def test_existing_command_keeps_joint_default_and_checkout_paths():
    args = arguments()
    assert args.gradients == "both"
    assert args.source == SOURCE.parents[1]
    assert args.inputs_module == SOURCE.parent
    assert args.views == 16
    assert args.precision == "fp64"


@pytest.mark.parametrize("gradients", ["volume", "both"])
def test_requested_gradient_selection(gradients):
    assert arguments("--gradients", gradients).gradients == gradients


@pytest.mark.parametrize("value", ["none", "pose", "all"])
def test_unsupported_gradient_request_fails(value):
    with pytest.raises(SystemExit):
        arguments("--gradients", value)


@pytest.mark.parametrize("flag,value", [("--views", "0"), ("--views", "361"), ("--repeats", "0")])
def test_invalid_capture_size_fails(flag, value):
    with pytest.raises(SystemExit):
        arguments(flag, value)


@pytest.mark.parametrize("gradients", ["volume", "both"])
def test_warmup_status_and_measured_forward_use_requested_pose_route(gradients):
    """Evaluate the actual call expressions without importing CUDA dependencies."""
    tree = ast.parse(SOURCE.read_text())
    expressions = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "image" for target in node.targets)
    ]
    assert len(expressions) == 3
    field, pose = object(), object()
    received = []
    environment = {
        "args": SimpleNamespace(gradients=gradients),
        "field": field,
        "pose": pose,
        "projector": lambda *args: received.append(args),
    }
    for expression in expressions:
        eval(compile(ast.Expression(expression), str(SOURCE), "eval"), environment)
    assert received == ([(field, pose)] * 3 if gradients == "both" else [(field,)] * 3)
    selected = next(
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "gradient_inputs"
            for target in node.targets
        )
    )
    targets = eval(compile(ast.Expression(selected), str(SOURCE), "eval"), environment)
    assert targets == ((field, pose) if gradients == "both" else field)


def test_help_and_import_do_not_require_torch_or_warp():
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            "import runpy, sys\nrunpy.run_path(sys.argv[1])\n"
            "assert 'torch' not in sys.modules and 'warp' not in sys.modules",
            str(SOURCE),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    help_result = subprocess.run(
        [sys.executable, str(SOURCE), "--help"], capture_output=True, text=True, check=False
    )
    assert help_result.returncode == 0, help_result.stderr
    assert "--gradients {volume,both}" in help_result.stdout

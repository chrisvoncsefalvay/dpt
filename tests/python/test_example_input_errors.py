"""Missing case fields fail at the input boundary before optional GPU imports."""

import importlib
from types import SimpleNamespace
from typing import Any

import pytest

from dpt.contracts import ContractError
from dpt.examples._common import geometry_from_json, grid_from_json, pose_from_json
from dpt.examples.reconstruction import Reconstruction, SolverSettings


def test_solver_missing_field_is_a_named_contract_error() -> None:
    with pytest.raises(ContractError, match="iterations"):
        SolverSettings.read({})
    with pytest.raises(ContractError, match="reconstruction"):
        Reconstruction(SimpleNamespace(config={}), device="cuda:0", torch=None, wp=None)


@pytest.mark.parametrize(
    "name,configuration,missing",
    [
        ("reconstruction", {"reconstruction": {"solver": {}}}, "iterations"),
        ("registration", {"registration": {}}, "chart"),
        ("acquisition_design", {"acquisition_design": {}}, "grid"),
    ],
)
def test_all_example_clis_report_contract_errors_without_tracebacks(
    name: str,
    configuration: dict[str, Any],
    missing: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module: Any = importlib.import_module(f"dpt.examples.{name}")
    monkeypatch.setattr(module, "load_case", lambda _: SimpleNamespace(config=configuration))
    monkeypatch.setattr("sys.argv", [name, "--case", "case.json", "--output", "run"])
    with pytest.raises(SystemExit) as caught:
        module.main()
    assert caught.value.code == 2
    error = capsys.readouterr().err
    assert missing in error and "error:" in error and "Traceback" not in error


def test_shared_json_geometry_decoders_preserve_axis_contracts() -> None:
    grid = grid_from_json({"shape": [2, 3, 4], "spacing_mm": [0.5, 0.7, 2.0]})
    assert grid.shape == (2, 3, 4) and grid.spacing_mm == (0.5, 0.7, 2.0)
    assert pose_from_json({"translation_mm": [1, 2, 3]}).translation_mm == (1.0, 2.0, 3.0)
    with pytest.raises(ContractError, match="spacing_mm"):
        grid_from_json({"shape": [2, 3, 4]})
    with pytest.raises(ContractError, match="invalid fields"):
        grid_from_json({"shape": [2, 3, 4], "spacing_mm": [1, 1, 1], "unknown": 3})
    with pytest.raises(ContractError, match="source_mm"):
        geometry_from_json({})

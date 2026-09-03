"""Exercise the actual worker launcher beside a conflicting profile module."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace


def test_worker_launch_keeps_standard_library_profile(tmp_path, monkeypatch):
    original = Path(__file__).resolve().parents[2] / "benchmarks/material_campaign.py"
    script = tmp_path / "runner/benchmarks/material_campaign.py"
    script.parent.mkdir(parents=True)
    shutil.copyfile(original, script)
    script.with_name("profile.py").write_text(
        'raise RuntimeError("benchmark directory shadowed stdlib profile")\n'
    )
    source = tmp_path / "source"
    package = source / "python/dpt/__init__.py"
    package.parent.mkdir(parents=True)
    package.write_text("")
    helper = tmp_path / "helpers"
    benchmarks = helper / "benchmarks"
    benchmarks.mkdir(parents=True)
    (benchmarks / "__init__.py").write_text("")
    marker = tmp_path / "resolved-profile.json"
    (benchmarks / "common.py").write_text(
        "def load_runtime():\n"
        "    import cProfile, json, profile, sys\n"
        "    from pathlib import Path\n"
        "    assert callable(cProfile.run) and callable(profile.run)\n"
        f"    Path({str(marker)!r}).write_text(json.dumps("
        "{'profile': profile.__file__, 'safe_path': sys.flags.safe_path}))\n"
    )
    (benchmarks / "real_ct.py").write_text("class ReleaseBenchmark:\n    pass\n")
    spec = importlib.util.spec_from_file_location("worker_import_campaign", script)
    campaign = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(campaign)
    output = tmp_path / "output"
    output.mkdir()
    run_file = output / "campaign.json"
    run = {
        "sources": {"baseline": {"path": str(source), "digest": campaign.source_digest(source)}},
        "harness_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
        "helper_hashes": {},
        "input_manifest": {},
        "gradient_contract": {},
        "helper_source": str(helper),
        "output": str(output),
        "cases": [],
        "commands": [],
    }
    campaign.write_json(run_file, run)
    # Execute the real nested launcher with a runtime-free worker fixture.
    tree = ast.parse(script.read_text())
    launch = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "launch"
    )
    module = ast.fix_missing_locations(ast.Module(body=[launch], type_ignores=[]))
    namespace = vars(campaign) | {
        "args": SimpleNamespace(output=output),
        "run_file": run_file,
        "run": run,
    }
    monkeypatch.delenv("PYTHONSAFEPATH", raising=False)
    monkeypatch.delenv("PYTHONPATH", raising=False)
    exec(compile(module, str(script), "exec"), namespace)
    namespace["launch"]("baseline", "gate")
    result = json.loads(marker.read_text())
    assert result["safe_path"] is True
    assert Path(result["profile"]).resolve() != script.with_name("profile.py")
    assert json.loads((output / "gate/baseline/worker.json").read_text())["passed"] is True
    assert run["commands"][0]["command"][0] == sys.executable

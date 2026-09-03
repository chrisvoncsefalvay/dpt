"""CPU fixtures for source-bound documentation and derivative-aware findings."""

import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

TOOLS = Path(__file__).resolve().parents[2] / "tools"
sys.path.insert(0, str(TOOLS))
SPEC = importlib.util.spec_from_file_location(
    "update_fullscan_docs", TOOLS / "update_fullscan_docs.py"
)
docs = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(docs)


def profile_fixture():
    signature = {"requested_gradients": ["volume"], "views": 1}
    workload = {
        "precision": "fp32",
        "operation": "backward",
        "views": 1,
        "requested_gradients": ["volume"],
    }
    return {
        "operations": [
            {
                "id": "arbitrary-control",
                "role": "before",
                "workload": copy.deepcopy(workload),
                "comparison_signature": signature,
            },
            {
                "id": "name-does-not-define-gradients",
                "role": "final",
                "workload": workload,
                "comparison_signature": copy.deepcopy(signature),
            },
        ],
        "profiles": [
            {
                "operation_id": "name-does-not-define-gradients",
                "ncu_kernel_id": 7,
                "plotted_hot_kernel": True,
                "requested_gradients": ["volume"],
                "registers_per_thread": 42,
                "register_spill_instructions": None,
                "additional_counters": {"local_memory_sector_bytes": None},
                "throughput_percent_of_peak": {"L2_elapsed": None, "DRAM_elapsed": 2.0},
                "atomic_thread_instructions": {"global_reduction_RED_REDG": 100},
            },
        ],
        "comparisons": [
            {
                "before": "arbitrary-control",
                "after": "name-does-not-define-gradients",
                "scope": "projector_and_vjp_kernels",
                "metric": "REDG_thread_instructions",
                "before_value": 200,
                "after_value": 100,
                "change_percent": -50.0,
            },
        ],
        "timelines": [],
    }


def test_profile_findings_use_metadata_without_fixed_count_or_label():
    result = docs.profiling_findings(profile_fixture())
    assert "| fp32 | volume-only VJP | 1 | 42 | unavailable | unavailable |" in result
    assert "| REDG_thread_instructions | 200 | 100 | -50 |" not in result
    assert "Reference" not in result
    assert "Optimised" not in result
    assert "reciprocal" not in result
    assert "five" not in result
    assert ";" not in result
    tables = [block for block in result.split("\n\n") if block.startswith("|")]
    assert len(tables) == 1
    assert all(line.startswith("|") for block in tables for line in block.splitlines())


def test_kernel_derivative_metadata_mismatch_rejects():
    profile = profile_fixture()
    profile["profiles"][0]["requested_gradients"] = ["volume", "pose"]
    with pytest.raises(ValueError, match="gradient metadata"):
        docs.profiling_findings(profile)


def test_unmatched_volume_counter_comparison_rejects():
    profile = profile_fixture()
    profile["operations"][0]["comparison_signature"] = {"views": 16}
    with pytest.raises(ValueError, match="does not match"):
        docs.profiling_findings(profile)


def test_joint_and_forward_are_separate_optional_scopes():
    profile = profile_fixture()
    profile["comparisons"] = []
    for operation, gradients, name in (
        ("backward", ["volume", "pose"], "joint"),
        ("forward", ["volume"], "forward"),
    ):
        row = copy.deepcopy(profile["profiles"][0])
        row.update(operation_id=name, requested_gradients=gradients)
        profile["profiles"].append(row)
        profile["operations"].append(
            {
                "id": name,
                "role": "final",
                "workload": {
                    "precision": "fp64",
                    "operation": operation,
                    "views": 16,
                    "requested_gradients": gradients,
                },
            }
        )
    result = docs.profiling_findings(profile)
    assert "| fp64 | joint volume/pose VJP | 16 |" in result
    assert "| fp64 | forward | 16 |" in result


def test_timeline_counts_do_not_reuse_old_status_claim():
    profile = profile_fixture()
    profile["timelines"] = [
        {
            "id": "trace",
            "role": "final",
            "precision": "fp32",
            "views": 360,
            "requested_gradients": ["volume"],
            "ranges": [
                {
                    "range": "probe_backward",
                    "inclusive_nvtx_ms": 10.0,
                    "kernel_count": 12,
                    "full_volume_validation_count": 2,
                    "four_byte_d2h_count": 0,
                }
            ],
        }
    ]
    result = docs.profiling_findings(profile)
    assert "| trace | fp32 | 360 | volume-only VJP | 10 | 12 | 2 | 0 |" in result
    assert "status reads fall" not in result


@pytest.fixture
def volume_documents(tmp_path, monkeypatch):
    source = {"sha256": "candidate", "files": {"module.py": "file"}}
    inputs = {
        "metadata_sha256": "metadata",
        "metadata": {"physical": "grid"},
        "selected_view_indices": [0],
        "selected_geometry": [{"source": [0, -600, 0]}],
        "cotangent": {"sha256": "seed", "raw_sha256": "seed values"},
    }
    main = {"dpt": {"candidate": {"fp32": {"source": source}}}, "input": inputs}
    volume = {
        "dpt": {
            "candidate": {
                "1024": {"source": copy.deepcopy(source)},
                "256": {"source": copy.deepcopy(source)},
            }
        },
        "input": copy.deepcopy(inputs),
    }
    (tmp_path / "results.json").write_text(json.dumps(volume))
    (tmp_path / "summary.md").write_text("# Volume report\n\n## Sampling\n")
    monkeypatch.setitem(
        sys.modules,
        "fullscan_volume_report",
        SimpleNamespace(markdown=lambda _: "# Volume report\n\n## Sampling\n"),
    )
    return tmp_path, main, volume


def test_volume_summary_binds_candidate_and_demotes_headings(volume_documents):
    path, main, _ = volume_documents
    text = docs.volume_report_text(path, main)
    assert text.startswith("## Volume report\n\n### Sampling\n")
    assert "volume_iteration.json" in text


@pytest.mark.parametrize("change", ["source", "geometry", "seed", "summary"])
def test_volume_summary_mismatches_reject_before_writing(volume_documents, change):
    path, main, volume = volume_documents
    if change == "source":
        volume["dpt"]["candidate"]["256"]["source"]["sha256"] = "other"
    elif change == "geometry":
        volume["input"]["selected_geometry"][0]["source"] = [0, -601, 0]
    elif change == "seed":
        volume["input"]["cotangent"]["raw_sha256"] = "different values"
    else:
        (path / "summary.md").write_text("hand-edited numbers")
    (path / "results.json").write_text(json.dumps(volume))
    with pytest.raises(ValueError):
        docs.volume_report_text(path, main)


@pytest.fixture
def profile_binding():
    data = {
        "dpt": {
            "candidate": {mode: {"source": {"sha256": "selected"}} for mode in ("fp32", "fp64")}
        },
        "input": {
            "metadata_sha256": "metadata",
            "metadata": {"sampling": {"samples_per_ray": 1024}},
            "cotangent": {"sha256": "seed"},
        },
    }
    profiles = {
        "operations": [
            {
                "role": "final",
                "source_sha256": "selected",
                "workload": {
                    "input_metadata_sha256": "metadata",
                    "cotangent_sha256": "seed",
                    "samples_per_ray": 1024,
                    "views": 1,
                    "view_indices": [0],
                },
            }
        ],
        "timelines": [
            {
                "role": "final",
                "source_sha256": "selected",
                "input_metadata": copy.deepcopy(data["input"]["metadata"]),
                "cotangent": {"sha256": "seed"},
                "views": 360,
                "view_indices": list(range(360)),
            }
        ],
    }
    return profiles, data


def test_profile_binding_preserves_one_view_ncu_and_fullscan_nsys(profile_binding):
    docs.validate_profile_binding(*profile_binding)


@pytest.mark.parametrize("field", ["input_metadata_sha256", "cotangent_sha256", "samples_per_ray"])
def test_ncu_input_or_sampling_mismatch_rejects(profile_binding, field):
    profiles, data = profile_binding
    profiles["operations"][0]["workload"][field] = "changed"
    with pytest.raises(ValueError, match="Final NCU workload"):
        docs.validate_profile_binding(profiles, data)


@pytest.mark.parametrize("field", ["input_metadata", "cotangent"])
def test_nsys_input_or_seed_mismatch_rejects(profile_binding, field):
    profiles, data = profile_binding
    profiles["timelines"][0][field] = {"sha256": "changed"}
    with pytest.raises(ValueError, match="Final NSYS workload"):
        docs.validate_profile_binding(profiles, data)

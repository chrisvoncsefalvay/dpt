"""CPU regression checks using the packaged measured-report schema."""

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
SPEC = importlib.util.spec_from_file_location("fullscan_report", ROOT / "tools/fullscan_report.py")
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


@pytest.fixture
def data():
    measured = json.loads((ROOT / "docs/assets/results/fullscan/measurements.json").read_text())
    measured["dpt"]["baseline"]["fp32"]["checks"]["forward"]["images_vs_reference"] = {
        "status": "reference run without a prior output"
    }
    return measured


@pytest.fixture
def quality_data(data):
    """Synthetic optional-sweep payload for one recorded, bound 1024-sample row.

    This fixture exercises the optional quality schema without claiming that the
    packaged report includes a sampling sweep. It never writes measured assets.
    """
    measured = copy.deepcopy(data)
    errors = copy.deepcopy(measured["precision_comparison"]["errors"])
    measured["precision_comparison"] = None
    fp32 = measured["dpt"]["candidate"]["fp32"]
    fp64 = measured["dpt"]["candidate"]["fp64"]
    evidence = measured["evidence"]["candidate_fp64"]
    row = copy.deepcopy(next(row for row in fp32["measurements"] if row["label"] == "forward"))
    row["label"] = "fp32/1024-samples/forward"
    row["error_vs_fp64_1024"] = copy.deepcopy(errors)
    row["configuration"]["error_vs_fp64_1024"] = copy.deepcopy(errors)
    image_hash = fp32["outputs"]["arrays"]["images"]["raw_sha256"]
    measured["quality"] = {
        "status": "complete",
        "source_unchanged": True,
        "source": copy.deepcopy(fp32["source"]),
        "source_after": copy.deepcopy(fp32["source"]),
        "input": copy.deepcopy(measured["input"]),
        "started_at": fp32["started_at"],
        "completed_at": fp32["completed_at"],
        "reference": {
            "results_sha256": evidence["sha256"],
            "outputs_sha256": evidence["outputs_sha256"],
            "source_sha256": fp64["source"]["sha256"],
            "completed_at": fp64["completed_at"],
            "samples_per_ray": 1024,
            "images": copy.deepcopy(fp64["outputs"]["arrays"]["images"]),
        },
        "measurements": [row],
        "checks": {
            row["label"]: {
                "error_vs_fp64_1024": copy.deepcopy(errors),
                "raw_sha256": image_hash,
                "after_timing": {"bitwise_unchanged": True, "raw_sha256": image_hash},
                "arithmetic_accuracy_gate": {
                    "passed": True,
                    "thresholds": copy.deepcopy(
                        fp32["contract"]["settings"]["quality"]["image_fp32_vs_fp64"]
                    ),
                },
            }
        },
    }
    return measured


def test_table_uses_bound_quality_error_without_baseline_error(quality_data):
    expected = quality_data["quality"]["checks"]["fp32/1024-samples/forward"]["error_vs_fp64_1024"]
    assert report.fp32_fp64_error(quality_data) == expected
    text = report.tables(quality_data)
    assert f"maximum optical-depth error {expected['max_abs']:.6g}" in text
    assert f"relative L2 {expected['relative_l2']:.6g}" in text


@pytest.mark.parametrize(
    "changed", ["source", "image", "reference", "reference_image", "input", "plotted_error"]
)
def test_changed_quality_bindings_reject(quality_data, changed):
    quality = quality_data["quality"]
    if changed == "source":
        quality["source"] = copy.deepcopy(quality["source"])
        quality["source"]["sha256"] = "different"
    elif changed == "image":
        quality["checks"]["fp32/1024-samples/forward"]["raw_sha256"] = "different"
    elif changed == "reference":
        quality["reference"]["results_sha256"] = "different"
    elif changed == "reference_image":
        quality["reference"]["images"]["raw_sha256"] = "different"
    elif changed == "input":
        quality["input"]["metadata_sha256"] = "different"
    else:
        row = next(
            row
            for row in quality["measurements"]
            if row["configuration"]["samples_per_ray"] == 1024
        )
        row["error_vs_fp64_1024"]["relative_l2"] *= 2
    with pytest.raises(ValueError, match="precision error"):
        report.fp32_fp64_error(quality_data)


def test_no_precision_evidence_does_not_invent_cross_precision_error(data, monkeypatch, tmp_path):
    data["quality"] = None
    data["precision_comparison"] = None
    assert report.fp32_fp64_error(data) is None
    assert "maximum optical-depth error" not in report.tables(data)
    plotted = {}

    def save(figure, _output, name):
        plotted[name] = figure.axes[0].get_legend_handles_labels()[1]

    monkeypatch.setattr(report, "save_figure", save)
    report.figures(data, tmp_path)
    labels = plotted["fullscan-latency-error"]
    assert labels
    assert all(not label.startswith("DPT") for label in labels)


def test_public_latency_uses_selected_measurements_only(data, monkeypatch, tmp_path):
    before = copy.deepcopy(data)
    rendered = report.tables(data)
    assert "Speedup" not in rendered
    assert "Optimised" not in rendered
    saved = {}

    def capture(figure, _output, name):
        if name == "fullscan-latency":
            saved[name] = [[bar.get_height() for bar in axis.patches] for axis in figure.axes]

    monkeypatch.setattr(report, "save_figure", capture)
    report.figures(data, tmp_path)
    for values, mode in zip(saved["fullscan-latency"], ("fp64", "fp32"), strict=True):
        measured = report.rows_by_operation(data, "candidate", mode)
        np.testing.assert_array_equal(
            values, [measured[name]["wall_median_ms"] / 1000 for name in report.OPERATIONS]
        )
    assert data == before


def test_packaged_saved_images_supply_bound_error_without_a_sweep(data):
    assert data["quality"] is None
    expected = data["precision_comparison"]["errors"]
    before = copy.deepcopy(data)
    assert report.fp32_fp64_error(data) == expected
    text = report.tables(data)
    assert f"maximum optical-depth error {expected['max_abs']:.6g}" in text
    assert f"relative L2 {expected['relative_l2']:.6g}" in text
    assert "untimed comparison" in text
    assert "FP32 sampling quality versus time" not in text
    assert data == before


@pytest.fixture
def saved_comparison(data, monkeypatch):
    """Small host arrays isolate error arithmetic from the checked full-scan loader."""
    data["quality"] = None
    arrays = {
        "fp32": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "fp64": np.array([[1.0, 2.0], [3.0, 4.25]], dtype=np.float64),
    }

    def load(directory, mode, metadata, metadata_hash):
        assert directory == Path(mode)
        assert metadata == data["input"]["metadata"]
        assert metadata_hash == data["input"]["metadata_sha256"]
        return (
            arrays[mode],
            report.precision_run_binding(data, mode),
            data["dpt"]["candidate"][mode]["outputs"],
        )

    monkeypatch.setattr(report, "load_run", load)
    before = copy.deepcopy(data)
    comparison = report.collect_precision_comparison({mode: Path(mode) for mode in arrays}, data)
    assert data == before
    assert comparison["errors"]["max_abs"] == 0.25
    assert comparison["errors"]["rmse"] == 0.125
    assert comparison["errors"]["relative_l2"] == pytest.approx(0.25 / np.sqrt(32.0625))
    data["precision_comparison"] = copy.deepcopy(comparison)
    return data


def test_saved_images_supply_one_point_without_a_sweep(saved_comparison, monkeypatch, tmp_path):
    measured = copy.deepcopy(saved_comparison["dpt"])
    errors = report.fp32_fp64_error(saved_comparison)
    assert errors == saved_comparison["precision_comparison"]["errors"]
    assert "untimed comparison" in report.tables(saved_comparison)
    assert saved_comparison["quality"] is None
    plotted = {}

    def save(figure, _output, name):
        axis = figure.axes[0]
        plotted[name] = [
            (collection.get_label(), np.asarray(collection.get_offsets()))
            for collection in axis.collections
        ]

    monkeypatch.setattr(report, "save_figure", save)
    report.figures(saved_comparison, tmp_path)
    points = [item for item in plotted["fullscan-latency-error"] if item[0].startswith("DPT")]
    assert len(points) == 1
    forward = report.rows_by_operation(saved_comparison, "candidate", "fp32")["forward"]
    np.testing.assert_array_equal(
        points[0][1], [[forward["wall_median_ms"], errors["relative_l2"]]]
    )
    assert saved_comparison["dpt"] == measured


@pytest.mark.parametrize(
    "changed",
    [
        "source",
        "input",
        "image",
        "result",
        "manifest",
        "sampling",
        "nonfinite",
        "negative",
        "ambiguous",
    ],
)
def test_changed_saved_comparison_rejects(saved_comparison, changed):
    value = saved_comparison["precision_comparison"]
    if changed == "source":
        value["source"]["files"]["python/dpt/__init__.py"] = "0" * 64
    elif changed == "input":
        value["input"]["selected_geometry"][0]["source_mm"][0] += 1
    elif changed == "image":
        value["runs"]["fp32"]["images"]["raw_sha256"] = "0" * 64
    elif changed == "result":
        value["runs"]["fp64"]["results_sha256"] = "0" * 64
    elif changed == "manifest":
        value["runs"]["fp32"]["outputs_sha256"] = "0" * 64
    elif changed == "sampling":
        value["samples_per_ray"] = 256
    elif changed == "nonfinite":
        value["errors"]["relative_l2"] = float("nan")
    elif changed == "negative":
        value["errors"]["rmse"] = -1.0
    else:
        saved_comparison["quality"] = {}
    with pytest.raises(ValueError, match="precision error"):
        report.fp32_fp64_error(saved_comparison)


def test_loaded_files_must_match_the_collected_run(saved_comparison, monkeypatch):
    original = report.load_run

    def wrong_run(*args):
        values, binding, outputs = original(*args)
        binding["results_sha256"] = "0" * 64
        return values, binding, outputs

    monkeypatch.setattr(report, "load_run", wrong_run)
    with pytest.raises(ValueError, match="saved files differ"):
        report.collect_precision_comparison(
            {mode: Path(mode) for mode in ("fp32", "fp64")}, saved_comparison
        )


@pytest.mark.parametrize("changed", ["source_map", "dtype", "shape", "input", "sampling"])
def test_different_selected_run_rejects_before_loading(data, monkeypatch, changed):
    data["quality"] = None
    run = data["dpt"]["candidate"]["fp32"]
    if changed == "source_map":
        run["source"]["files"]["python/dpt/__init__.py"] = "0" * 64
    elif changed == "dtype":
        run["outputs"]["arrays"]["images"]["dtype"] = "float64"
    elif changed == "shape":
        run["outputs"]["arrays"]["images"]["shape"] = [1, 512, 512]
    elif changed == "input":
        run["outputs"]["input_metadata_sha256"] = "0" * 64
    else:
        run["measurements"][0]["configuration"]["samples_per_ray"] = 256

    def forbidden(*_args):
        raise AssertionError("invalid metadata must fail before array reads")

    monkeypatch.setattr(report, "load_run", forbidden)
    with pytest.raises(ValueError, match="precision error"):
        report.collect_precision_comparison({mode: Path(mode) for mode in ("fp32", "fp64")}, data)


@pytest.mark.parametrize("changed", ["file_hash", "raw_hash", "dtype", "shape"])
def test_actual_npy_loader_rejects_changed_array_binding(tmp_path, changed):
    from plot_fullscan_images import checked_array

    path = tmp_path / "images.npy"
    values = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
    np.save(path, values, allow_pickle=False)
    record = {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "raw_sha256": hashlib.sha256(values.tobytes()).hexdigest(),
        "shape": list(values.shape),
        "dtype": str(values.dtype),
    }
    np.testing.assert_array_equal(checked_array(path, record, values.shape, values.dtype), values)
    if changed == "file_hash":
        with path.open("ab") as stream:
            stream.write(b"changed")
    elif changed == "raw_hash":
        record["raw_sha256"] = "0" * 64
    elif changed == "dtype":
        record["dtype"] = "float64"
    else:
        record["shape"] = [4]
    with pytest.raises(ValueError):
        checked_array(path, record, values.shape, values.dtype)

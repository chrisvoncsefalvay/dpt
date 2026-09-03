"""Input and assessment boundaries for the two-view agent tools, without CUDA."""

import importlib.util
import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from dpt.contracts import ContractError
from dpt.examples.make_case import CaseArray, write_case
from dpt.geometry import DetectorGeometry, RigidTransform
from dpt.recovery import PoseChart
from dpt.volumes import GridSpec

_PATH = Path(__file__).resolve().parents[2] / "examples" / "agentic_imaging.py"
_SPEC = importlib.util.spec_from_file_location("agentic_imaging", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
example = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(example)


def _case(tmp_path, *, counts=None, mask=None, number_of_views=2):
    grid = GridSpec((3, 3, 3), (1, 1, 1), (-1, -1, -1))
    panel = DetectorGeometry((-10, 0, 0), (10, -1, -1), (0, 1, 0), (0, 0, 1), (1, 1), (2, 2))
    observed = np.full(panel.shape, 80, dtype=np.float32) if counts is None else counts
    included = np.ones(panel.shape, dtype=np.float32) if mask is None else mask
    arrays = {
        "attenuation": CaseArray(
            np.full(grid.shape, 0.02, dtype=np.float32), "mm^-1", "test", "Apache-2.0"
        ),
        "observed": CaseArray(observed, "counts", "test", "Apache-2.0"),
        "mask": CaseArray(included, "dimensionless", "test", "Apache-2.0"),
    }
    return write_case(
        tmp_path / "case",
        {
            "registration": {
                "grid": asdict(grid),
                "attenuation": "attenuation",
                "chart": asdict(PoseChart(anchor=RigidTransform(translation_mm=(0.2, 0, 0)))),
                "samples_per_ray": 32,
                "policy": {},
                "views": [
                    {
                        "id": f"v{index}",
                        "geometry": asdict(panel),
                        "observation": "observed",
                        "mask": "mask",
                        "open_beam_counts": 100,
                    }
                    for index in range(number_of_views)
                ],
            }
        },
        arrays,
    )


def _nifti(path):
    nib = pytest.importorskip("nibabel")
    image = nib.Nifti1Image(np.arange(60, dtype=np.float32).reshape(3, 4, 5), np.eye(4))
    image.header.set_xyzt_units("mm")
    nib.save(image, path)
    return path


def _dicom_series(directory):
    pydicom = pytest.importorskip("pydicom")
    from dicom_fixtures import pixels, record, save

    directory.mkdir()
    uid, frame_uid = pydicom.uid.generate_uid(), pydicom.uid.generate_uid()
    paths = []
    for index in range(2):
        path = directory / f"slice-{1 - index}.dcm"
        image = record(path, enhanced=False)
        image.SeriesInstanceUID, image.FrameOfReferenceUID = uid, frame_uid
        image.ImagePositionPatient = [31.0, -9.0, 15.0 + 4 * index]
        image.RescaleSlope, image.RescaleIntercept, image.RescaleType = 2, -1000, "HU"
        image.PixelData = pixels()[index].astype("<i2").tobytes()
        paths.append(save(image, path))
    return uid, paths


def _host_views(values, grid, panels, truth, samples, device, source, rights):
    """Build valid case records with constant test counts; no projection is measured."""
    arrays = {"attenuation": CaseArray(values, "mm^-1", source, rights)}
    views = []
    for index, panel in enumerate(panels):
        arrays[f"counts{index}"] = CaseArray(
            np.ones(panel.shape, dtype=np.float32), "counts", "host schema test", rights
        )
        arrays[f"mask{index}"] = CaseArray(
            np.ones(panel.shape, dtype=np.float32), "dimensionless", "host schema test", rights
        )
        views.append(
            {
                "id": f"view{index}",
                "geometry": asdict(panel),
                "observation": f"counts{index}",
                "mask": f"mask{index}",
                "open_beam_counts": 1000000,
            }
        )
    return arrays, views


def test_inspection_keeps_pose_and_inputs_unchanged(tmp_path):
    case = _case(tmp_path)
    original = {path.name: path.read_bytes() for path in case.parent.iterdir()}
    report = example.inspect_case(case)
    assert report["status"] == "inputs_valid"
    assert report["initial_pose_object_to_world"]["translation_mm"] == (0.2, 0, 0)
    assert [row["included_pixels"] for row in report["views"]] == [4, 4]
    assert report["evaluation_reference_used_for_fit"] is False
    assert {path.name: path.read_bytes() for path in case.parent.iterdir()} == original


@pytest.mark.parametrize("number", [1, 3])
def test_two_view_example_refuses_other_view_counts(tmp_path, number):
    with pytest.raises(ContractError, match="exactly two"):
        example.inspect_case(_case(tmp_path, number_of_views=number))


@pytest.mark.parametrize("value", [-1.0, 0.5, float(2**24 + 2)])
def test_display_intensities_and_out_of_range_counts_are_rejected(tmp_path, value):
    counts = np.full((2, 2), value, dtype=np.float32)
    with pytest.raises(ContractError, match="integer counts"):
        example.inspect_case(_case(tmp_path, counts=counts))


def test_mask_cannot_silently_exclude_nonzero_observations(tmp_path):
    mask = np.ones((2, 2), dtype=np.float32)
    mask[0, 0] = 0
    with pytest.raises(ContractError, match="already be zero"):
        example.inspect_case(_case(tmp_path, mask=mask))


def test_inspection_refuses_changed_array_bytes(tmp_path):
    case = _case(tmp_path)
    np.save(case.parent / "observed.npy", np.zeros((2, 2), dtype=np.float32), allow_pickle=False)
    with pytest.raises(ContractError, match="sha256"):
        example.inspect_case(case)


def test_failed_or_budget_limited_fit_stays_unresolved_even_with_matching_truth(tmp_path):
    summary = {
        "status": "unresolved",
        "termination": "iteration_budget",
        "case_sha256": "a" * 64,
        "initial_pose_object_to_world": asdict(RigidTransform(translation_mm=(1, 0, 0))),
        "pose_object_to_world": asdict(RigidTransform()),
    }
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    reference = tmp_path / "reference.json"
    reference.write_text(
        json.dumps(
            {
                "case_sha256": "a" * 64,
                "pose_object_to_world": asdict(RigidTransform()),
                "landmarks_object_mm": [[0, 0, 0], [1, 0, 0]],
                "source": "synthetic truth",
                "uncertainty": "not real anatomy",
            }
        )
    )
    scored = example.assess_run(tmp_path, reference)
    assert scored["status"] == "unresolved"
    assert scored["geometric_evaluation"]["initial"]["translation_mm"] == 1
    assert scored["geometric_evaluation"]["final"]["translation_mm"] == 0
    assert (tmp_path / "summary.json").read_text() == json.dumps(summary)
    wrong = json.loads(reference.read_text())
    wrong["case_sha256"] = "b" * 64
    reference.write_text(json.dumps(wrong))
    with pytest.raises(ContractError, match="different case"):
        example.assess_run(tmp_path, reference)


def test_existing_outputs_are_preserved_before_cuda_is_needed(tmp_path):
    case = _case(tmp_path)
    output = tmp_path / "result"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("existing result")
    with pytest.raises(ContractError, match="already exists"):
        example.prepare_demo(output, "cuda:0")
    with pytest.raises(ContractError, match="already exists"):
        example.fit_case(case, output, "cuda:0", 10, 40)
    with pytest.raises(ContractError, match="already exists"):
        example.prepare_ct(
            tmp_path / "not-read.nii.gz", output, "cuda:0", units="HU", source="test", rights="test"
        )
    with pytest.raises(ContractError, match="already exists"):
        example.run_ct(tmp_path / "not-read.nii.gz", output, "cuda:0")
    assert sentinel.read_text() == "existing result"
    assert list(output.iterdir()) == [sentinel]


def test_ct_initial_rotation_uses_physical_centre_on_an_oblique_grid():
    angle = math.radians(30)
    grid = GridSpec(
        (7, 9, 11),
        (0.8, 1.2, 2.5),
        (1000, -500, 300),
        (math.cos(angle), -math.sin(angle), 0, math.sin(angle), math.cos(angle), 0, 0, 0, 1),
    )
    displacement = (3, -2, 1.5)
    anchor, centre, points = example._centred_pose(grid, displacement, (0, 0, 2))
    np.testing.assert_allclose(anchor.point(centre), np.array(centre) + displacement, atol=2e-13)
    point = grid.grid_to_object((2, 3, 4))
    offset = np.array(point) - centre
    phi = math.radians(2)
    independently_rotated = (
        math.cos(phi) * offset[0] - math.sin(phi) * offset[1],
        math.sin(phi) * offset[0] + math.cos(phi) * offset[1],
        offset[2],
    )
    expected = np.array(centre) + displacement + independently_rotated
    np.testing.assert_allclose(anchor.point(point), expected, atol=2e-13)
    assert len(points) == 8
    for physical_point in points:
        index = grid.object_to_grid(physical_point)
        assert index[0] == pytest.approx(2.5) or index[0] == pytest.approx(7.5)
        assert index[1] == pytest.approx(2) or index[1] == pytest.approx(6)
        assert index[2] == pytest.approx(1.5) or index[2] == pytest.approx(4.5)


@pytest.mark.parametrize(
    "field,value", [("units", "unknown"), ("source", " "), ("rights", ""), ("energy_kev", 80)]
)
def test_ct_rejects_invalid_explicit_fields_before_loading(tmp_path, field, value):
    options = {"units": "HU", "source": "supplied CT", "rights": "declared use terms"}
    options[field] = value
    with pytest.raises(ContractError):
        example.prepare_ct(tmp_path / "not-read.nii.gz", tmp_path / "output", "cuda:0", **options)
    assert not (tmp_path / "output").exists()


def test_ct_preparation_keeps_native_voxels_geometry_and_source_identity(tmp_path, monkeypatch):
    nib = pytest.importorskip("nibabel")
    hu = np.arange(4 * 5 * 6, dtype=np.float32).reshape(4, 5, 6) * 10 - 1000
    affine = np.array([[0, -1.2, 0, 100], [0.8, 0, 0, -200], [0, 0, 2.5, 300], [0, 0, 0, 1]])
    input_path = tmp_path / "host-io-fixture.nii.gz"
    image = nib.Nifti1Image(hu, affine)
    image.header.set_xyzt_units("mm")
    nib.save(image, input_path)
    original = input_path.read_bytes()
    captured = {}

    def no_cuda(values, grid, panels, truth, samples, device, source, rights):
        captured.update(grid=grid, panels=panels, truth=truth, samples=samples)
        expected = np.maximum(1 + hu.transpose(2, 1, 0).astype(np.float64) / 1000, 0) * 0.02059
        np.testing.assert_array_equal(values, expected.astype(np.float32))
        arrays = {"attenuation": CaseArray(values, "mm^-1", source, rights)}
        views = []
        for index, panel in enumerate(panels):
            arrays[f"counts{index}"] = CaseArray(
                np.ones(panel.shape, dtype=np.float32), "counts", "host schema test only", rights
            )
            arrays[f"mask{index}"] = CaseArray(
                np.ones(panel.shape, dtype=np.float32),
                "dimensionless",
                "host schema test only",
                rights,
            )
            views.append(
                {
                    "id": f"view{index}",
                    "geometry": asdict(panel),
                    "observation": f"counts{index}",
                    "mask": f"mask{index}",
                    "open_beam_counts": 1000000,
                }
            )
        return arrays, views

    monkeypatch.setattr(example, "_simulated_views", no_cuda)
    report = example.prepare_ct(
        input_path,
        tmp_path / "output",
        "cuda:0",
        units="HU",
        source="host-only CT fixture",
        rights="test fixture",
    )
    config = json.loads(Path(report["case"]).read_text())
    assert captured["grid"].shape == (6, 5, 4)
    assert captured["grid"].spacing_mm == pytest.approx((0.8, 1.2, 2.5))
    np.testing.assert_allclose(
        captured["grid"].grid_to_object((1, 2, 3)), (affine @ [1, 2, 3, 1])[:3], atol=1e-6
    )
    assert all(panel.shape == (128, 128) for panel in captured["panels"])
    assert captured["samples"] == 768
    assert captured["truth"] == RigidTransform()
    assert config["ct_input"]["resampled"] is False
    assert config["ct_input"]["sha256"] == example._sha256(input_path)
    assert config["ct_input"]["rights"] == "test fixture"
    assert "evaluation" not in config["registration"]
    assert example.inspect_case(Path(report["case"]))["status"] == "inputs_valid"
    reference = json.loads(Path(report["reference"]).read_text())
    assert reference["landmark_kind"] == "eight geometric points derived from the CT grid"
    assert input_path.read_bytes() == original


@pytest.mark.parametrize("suffix", [".nii", ".nii.gz"])
def test_ct_defaults_accept_nifti_and_keep_single_file_hashes(tmp_path, monkeypatch, suffix):
    input_path = _nifti(tmp_path / f"ct{suffix}")
    monkeypatch.setattr(example, "_simulated_views", _host_views)
    report = example.prepare_ct(input_path, tmp_path / "output", "cuda:0", size=4)
    config = json.loads(Path(report["case"]).read_text())
    digest = example._sha256(input_path)
    assert report["source_sha256"] == digest
    assert report["source_file_sha256"] == [digest]
    assert config["ct_input"]["sha256"] == digest
    assert config["ct_input"]["source_file_sha256"] == [digest]
    assert config["ct_input"]["input_units"] == "HU"
    assert config["ct_input"]["source"] == "User-supplied CT"
    assert config["ct_input"]["rights"] == "Not specified"
    assert config["ct_input"]["input_metadata"]["coordinate_frame"] == "RAS"
    assert example.inspect_case(Path(report["case"]))["status"] == "inputs_valid"


def test_dicom_preparation_preserves_native_grid_and_selected_file_hashes(tmp_path, monkeypatch):
    pytest.importorskip("pydicom")
    from dicom_fixtures import pixels

    _, paths = _dicom_series(tmp_path / "selected")
    unrelated = tmp_path / "selected" / "readme.txt"
    unrelated.write_text("auxiliary file")
    original = [path.read_bytes() for path in paths]
    captured = {}

    def host_views(values, grid, *args):
        captured["grid"] = grid
        expected = np.maximum(1 + (pixels()[:2].astype(np.float64) * 2 - 1000) / 1000, 0) * 0.02059
        np.testing.assert_array_equal(values, expected.astype(np.float32))
        unrelated.write_text("changed auxiliary file")
        return _host_views(values, grid, *args)

    monkeypatch.setattr(example, "_simulated_views", host_views)
    report = example.prepare_ct(tmp_path / "selected", tmp_path / "output", "cuda:0", size=4)
    config = json.loads(Path(report["case"]).read_text())
    hashes = [example._sha256(path) for path in paths]
    assert report["source_sha256"] == hashes
    assert report["source_file_sha256"] == hashes
    assert config["ct_input"]["source_file_sha256"] == hashes
    assert "sha256" not in config["ct_input"]
    assert config["ct_input"]["input_metadata"]["source_format"] == "dicom"
    assert config["ct_input"]["resampled"] is False
    assert captured["grid"].shape == (2, 2, 3)
    assert captured["grid"].spacing_mm == (1, 1, 4)
    np.testing.assert_allclose(captured["grid"].grid_to_object((1, 1, 1)), (-30.8, 7.6, 19))
    assert [path.read_bytes() for path in paths] == original
    assert example.inspect_case(Path(report["case"]))["status"] == "inputs_valid"


def test_dicom_multiple_series_requires_and_honours_uid(tmp_path, monkeypatch):
    uid_a, paths_a = _dicom_series(tmp_path / "a")
    uid_b, paths_b = _dicom_series(tmp_path / "b")
    monkeypatch.setattr(example, "_simulated_views", _host_views)
    with pytest.raises(ContractError, match="multiple image series") as error:
        example.prepare_ct(tmp_path, tmp_path / "ambiguous", "cuda:0", size=4)
    assert uid_a in str(error.value) and uid_b in str(error.value)
    assert not (tmp_path / "ambiguous").exists()
    report = example.prepare_ct(
        tmp_path, tmp_path / "output", "cuda:0", series_uid=uid_b, size=4
    )
    selected = [example._sha256(path) for path in paths_b]
    assert report["source_file_sha256"] == selected
    assert not set(selected).intersection(example._sha256(path) for path in paths_a)
    config = json.loads(Path(report["case"]).read_text())
    assert config["ct_input"]["series_uid"] == uid_b


@pytest.mark.parametrize("format", ["nifti", "dicom"])
def test_ct_preparation_detects_input_changes(tmp_path, monkeypatch, format):
    if format == "nifti":
        input_path = _nifti(tmp_path / "ct.nii.gz")

        def change_input():
            import nibabel as nib

            image = nib.load(input_path)
            changed = nib.Nifti1Image(image.get_fdata() + 1, image.affine, image.header)
            nib.save(changed, input_path)

    else:
        _, paths = _dicom_series(tmp_path / "dicom")
        input_path = tmp_path / "dicom"

        def change_input():
            import pydicom

            image = pydicom.dcmread(paths[0])
            image.RescaleIntercept = -999
            image.save_as(paths[0], enforce_file_format=True)

    def changed_views(*args):
        result = _host_views(*args)
        change_input()
        return result

    monkeypatch.setattr(example, "_simulated_views", changed_views)
    with pytest.raises(ContractError, match="source CT changed"):
        example.prepare_ct(input_path, tmp_path / "output", "cuda:0", size=4)
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("status", ["stationary", "unresolved"])
@pytest.mark.parametrize("input_kind", ["nifti", "dicom"])
def test_run_prepares_inspects_fits_and_assesses_with_bounded_solver(
    tmp_path, monkeypatch, status, input_kind
):
    preparation_options = {}
    output = tmp_path / "run"
    if input_kind == "nifti":
        input_path = _nifti(tmp_path / "ct.nii.gz")
    else:
        uid, _ = _dicom_series(tmp_path / "selected")
        _dicom_series(tmp_path / "other")
        input_path = tmp_path
        with pytest.raises(ContractError, match="multiple image series"):
            example.run_ct(input_path, output, "cuda:0", size=4)
        assert not output.exists()
        preparation_options["series_uid"] = uid
    monkeypatch.setattr(example, "_simulated_views", _host_views)
    calls = []

    def host_fit(case_path, output, device, iterations, evaluations):
        calls.append((output, device, iterations, evaluations))
        inspection = json.loads((output.parent / "inspection.json").read_text())
        assert inspection["status"] == "inputs_valid"
        config = json.loads(case_path.read_text())
        output.mkdir()
        summary = {
            "status": status,
            "termination": "gradient_tolerance" if status == "stationary" else "iteration_budget",
            "case_sha256": example._sha256(case_path),
            "initial_pose_object_to_world": config["registration"]["chart"]["anchor"],
            "pose_object_to_world": asdict(RigidTransform()),
        }
        (output / "summary.json").write_text(json.dumps(summary))
        return summary

    monkeypatch.setattr(example, "fit_case", host_fit)
    report = example.run_ct(
        input_path,
        output,
        "cuda:0",
        size=4,
        max_iterations=7,
        max_evaluations=23,
        **preparation_options,
    )
    assert calls == [(output / "fit", "cuda:0", 7, 23)]
    assert report["status"] == status
    assert Path(report["case"]) == output / "case" / "case.json"
    assert json.loads((output / "run.json").read_text()) == report
    for name in ("case", "reference", "preparation", "inspection", "fit", "summary", "assessment"):
        assert Path(report[name]).exists()
    assessment = json.loads(Path(report["assessment"]).read_text())
    assert assessment["status"] == status
    assert assessment["geometric_evaluation"]["final"]["translation_mm"] == 0


def test_run_preserves_partial_preparation_evidence_on_failure(tmp_path, monkeypatch):
    def failed_preparation(volume_path, output, device, **options):
        output.mkdir()
        (output / "evidence.txt").write_text("partial preparation")
        raise ContractError("preparation interrupted")

    monkeypatch.setattr(example, "prepare_ct", failed_preparation)
    output = tmp_path / "run"
    with pytest.raises(ContractError, match="preparation interrupted"):
        example.run_ct(tmp_path / "ct.nii.gz", output, "cuda:0")
    assert (output / "case" / "evidence.txt").read_text() == "partial preparation"


@pytest.mark.parametrize("field", ["max_iterations", "max_evaluations"])
def test_run_rejects_invalid_budget_before_input_or_cuda(tmp_path, field):
    with pytest.raises(ContractError, match=field):
        example.run_ct(tmp_path / "not-read.nii", tmp_path / "output", "cuda:0", **{field: 0})
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("command,function", [("prepare-ct", "prepare_ct"), ("run", "run_ct")])
@pytest.mark.parametrize("series_uid", [None, "1.2.826.0.1.99"])
def test_ct_cli_needs_only_input_output_and_forwards_series(
    tmp_path, monkeypatch, capsys, command, function, series_uid
):
    captured = {}

    def operation(volume, output, device, **options):
        captured.update(volume=volume, output=output, device=device, **options)
        return {"status": "host_dispatch_test"}

    arguments = ["agentic_imaging.py", command, "ct.nii.gz", "--output", str(tmp_path / "output")]
    if series_uid is not None:
        arguments += ["--series-uid", series_uid]
    monkeypatch.setattr(example, function, operation)
    monkeypatch.setattr("sys.argv", arguments)
    example.main()
    assert json.loads(capsys.readouterr().out) == {"status": "host_dispatch_test"}
    assert captured["units"] == "HU"
    assert captured["source"] == "User-supplied CT"
    assert captured["rights"] == "Not specified"
    assert captured["series_uid"] == series_uid
    if command == "run":
        assert captured["max_iterations"] == 120
        assert captured["max_evaluations"] == 800

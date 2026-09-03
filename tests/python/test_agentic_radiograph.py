"""Real image ingestion, physical pose conventions and numerical fit contracts."""

import hashlib
import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from dpt.contracts import ContractError
from dpt.geometry import RigidTransform, compose_pose
from dpt.volumes import GridSpec

_PATH = Path(__file__).resolve().parents[2] / "examples" / "agentic_radiograph.py"
_SPEC = importlib.util.spec_from_file_location("agentic_radiograph", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
example = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(example)


def _ct(tmp_path):
    nib = pytest.importorskip("nibabel")
    values = (np.arange(4 * 5 * 6).reshape(4, 5, 6) * 10 - 1000).astype(np.int16)
    affine = np.array([[0, -1.2, 0, 100], [0.8, 0, 0, -200], [0, 0, 2.5, 300], [0, 0, 0, 1]])
    path = tmp_path / "ct.nii.gz"
    image = nib.Nifti1Image(values, affine)
    image.header.set_xyzt_units("mm")
    nib.save(image, path)
    return path, affine


def _image(tmp_path, name="radiograph.npy"):
    path = tmp_path / name
    values = np.arange(48, dtype=np.float32).reshape(6, 8)
    np.save(path, values, allow_pickle=False)
    return path


@pytest.mark.parametrize("extension", [".png", ".tiff", ".npy"])
def test_image_input_preserves_sixteen_bit_grayscale_contrast(tmp_path, extension):
    image_module = pytest.importorskip("PIL.Image")
    values = np.array([[30001, 30002, 30005], [30000, 30004, 30003]], dtype=np.uint16)
    path = tmp_path / f"radiograph{extension}"
    if extension == ".npy":
        np.save(path, values, allow_pickle=False)
    else:
        image_module.fromarray(values).save(path)
    decoded, metadata = example.load_radiograph(path)
    np.testing.assert_allclose(decoded, (values.astype(float) - 30000) / 5, atol=1e-7)
    assert len(np.unique(decoded)) == 6
    assert metadata["input_range"] == [30000, 30005]
    assert metadata["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert decoded.flags.c_contiguous and decoded.dtype == np.float32


@pytest.mark.parametrize("extension", ["png", "jpg", "tiff"])
def test_rgb_images_use_luminance_and_keep_raster_shape(tmp_path, extension):
    image_module = pytest.importorskip("PIL.Image")
    values = np.arange(6 * 8 * 3, dtype=np.uint8).reshape(6, 8, 3)
    path = tmp_path / f"radiograph.{extension}"
    image_module.fromarray(values).save(path)
    decoded, metadata = example.load_radiograph(path)
    assert decoded.shape == (6, 8)
    assert "luminance" in metadata["conversion"]
    assert np.isfinite(decoded).all() and decoded.min() == 0 and decoded.max() == 1


@pytest.mark.parametrize(
    "values,match",
    [
        (np.ones((4, 5)), "variation"),
        (np.array([[1.0, np.nan], [2, 3]]), "finite"),
        (np.array([[1.0, 2], [np.inf, 3]]), "finite"),
        (np.ones((2, 2, 2)), "two-dimensional"),
        (np.ones((1, 5)), "at least 2"),
        (np.array([[1 + 1j, 2], [3, 4]]), "real numeric"),
    ],
)
def test_invalid_radiographs_fail_before_any_cuda_use(tmp_path, values, match):
    path = tmp_path / "invalid.npy"
    np.save(path, values, allow_pickle=False)
    with pytest.raises(ContractError, match=match):
        example.load_radiograph(path)


def test_finite_extreme_image_range_is_normalised_without_overflow(tmp_path):
    path = tmp_path / "extreme.npy"
    np.save(path, np.array([[-1e308, 1e308], [0.0, 5e307]]), allow_pickle=False)
    decoded, _ = example.load_radiograph(path)
    np.testing.assert_allclose(decoded, [[0, 1], [0.5, 0.75]])


def test_resampling_uses_one_pixel_scale_and_preserves_physical_centre():
    yy, xx = np.indices((301, 500))
    values = (2 * xx + 3 * yy).astype(np.float32)
    resized, scale = example.resize_radiograph(values, 160)
    assert resized.shape == (96, 160)
    assert scale == 3.125
    np.testing.assert_allclose(np.diff(resized, axis=1), 2 * scale, atol=1e-4)
    np.testing.assert_allclose(np.diff(resized, axis=0), 3 * scale, atol=1e-4)
    assert resized.mean() == pytest.approx(values.mean())
    unchanged, unit_scale = example.resize_radiograph(values, 0)
    np.testing.assert_array_equal(unchanged, values)
    assert unit_scale == 1


def test_extreme_aspect_ratio_cannot_be_stretched_into_two_rows():
    values = np.arange(2000, dtype=np.float32).reshape(2, 1000)
    with pytest.raises(ContractError, match="aspect ratio"):
        example.resize_radiograph(values, 192)


def test_approximate_non_square_camera_has_square_pixels_and_top_left_rows():
    grid = GridSpec((7, 9, 11), (0.8, 1.2, 2.5), (100, -300, 200))
    geometry, settings = example.approximate_geometry(
        grid, (96, 160), raster_scale=2, sod_mm=1000, sdd_mm=1500, pixel_spacing_mm=0.8
    )
    panel = geometry.views[0]
    assert panel.shape == (96, 160)
    assert panel.spacing_mm == (1.6, 1.6)
    assert settings["pixel_spacing_original_raster_mm"] == 0.8
    np.testing.assert_allclose(panel.v, [0, 0, -1], atol=1e-12)
    np.testing.assert_allclose(panel.u, [-1, 0, 0], atol=1e-12)
    centre = np.array(example.physical_centre(grid))
    detector_centre = (
        np.asarray(panel.origin_mm)
        + 0.5 * 159 * 1.6 * np.asarray(panel.u)
        + 0.5 * 95 * 1.6 * np.asarray(panel.v)
    )
    np.testing.assert_allclose(detector_centre, centre + np.array([0, -500, 0]), atol=1e-10)
    np.testing.assert_allclose(panel.source_mm, centre + np.array([0, 1000, 0]), atol=1e-10)
    assert panel.origin_mm[2] > centre[2]


def test_default_camera_covers_axially_rotated_ct_support():
    grid = GridSpec((45, 70, 85), (2, 1.5, 2), (20, -40, 50))
    geometry, _ = example.approximate_geometry(grid, (96, 160))
    panel = geometry.views[0]
    source = np.array(panel.source_mm)
    origin = np.array(panel.origin_mm)
    u, v = np.array(panel.u), np.array(panel.v)
    normal = np.cross(u, v)
    distance = np.dot(origin - source, normal)
    for candidate in example.seed_candidates(grid):
        pose = RigidTransform(**candidate["seed_pose_object_to_world"])
        for x in grid.support[0][0], grid.support[1][0]:
            for y in grid.support[0][1], grid.support[1][1]:
                for z in grid.support[0][2], grid.support[1][2]:
                    point = np.array(pose.point(grid.grid_to_object((x, y, z))))
                    ray = point - source
                    detector_point = source + ray * distance / np.dot(ray, normal)
                    column = np.dot(detector_point - origin, u) / panel.spacing_mm[0]
                    row = np.dot(detector_point - origin, v) / panel.spacing_mm[1]
                    assert -0.5 <= column <= panel.shape[1] - 0.5
                    assert -0.5 <= row <= panel.shape[0] - 0.5


def test_pitch_search_is_bounded_and_keeps_detector_centre_and_square_pixels():
    grid = GridSpec((45, 70, 85), (2, 1.5, 2), (20, -40, 50))
    geometry, settings = example.approximate_geometry(grid, (96, 160), raster_scale=2)
    cameras = example.camera_hypotheses(geometry, settings, search_pitch=True)
    assert [camera["camera_scale"] for _, camera in cameras] == [1, 0.5, 0.65, 0.85, 1.2, 1.5]
    baseline = geometry.views[0]
    detector_centre = (
        np.asarray(baseline.origin_mm)
        + 0.5 * 159 * baseline.spacing_mm[0] * np.asarray(baseline.u)
        + 0.5 * 95 * baseline.spacing_mm[1] * np.asarray(baseline.v)
    )
    for hypothesis, camera in cameras:
        panel = hypothesis.views[0]
        scale = camera["camera_scale"]
        assert panel.shape == baseline.shape
        assert panel.source_mm == baseline.source_mm
        assert panel.spacing_mm[0] == panel.spacing_mm[1]
        assert panel.spacing_mm[0] == baseline.spacing_mm[0] * scale
        assert camera["pixel_spacing_original_raster_mm"] == panel.spacing_mm[0] / 2
        actual_centre = (
            np.asarray(panel.origin_mm)
            + 0.5 * 159 * panel.spacing_mm[0] * np.asarray(panel.u)
            + 0.5 * 95 * panel.spacing_mm[1] * np.asarray(panel.v)
        )
        np.testing.assert_allclose(actual_centre, detector_centre, atol=1e-12)
    fixed = example.camera_hypotheses(geometry, settings, search_pitch=False)
    assert len(fixed) == 1
    assert fixed[0][0] == geometry


def test_joint_camera_pose_candidates_preserve_baseline_and_exact_budget():
    grid = GridSpec((7, 9, 11), (0.8, 1.2, 2.5), (1000, -500, 300))
    geometry, settings = example.approximate_geometry(grid, (96, 160))
    cameras = [
        camera for _, camera in example.camera_hypotheses(geometry, settings, search_pitch=True)
    ]
    candidates = example.pose_camera_candidates(grid, cameras, 45)
    assert len(candidates) == len({candidate["id"] for candidate in candidates}) == 48
    assert candidates[0]["view"] == "AP"
    assert candidates[0]["camera_scale"] == 1.0
    np.testing.assert_allclose(candidates[0]["seed_matrix_object_to_world"], np.eye(4))
    for camera in cameras:
        matching = [candidate for candidate in candidates if candidate["camera_id"] == camera["id"]]
        assert len(matching) == 8
        assert all(candidate["camera_scale"] == camera["camera_scale"] for candidate in matching)
    budget = example.fit_budget(3, 80, 45, len(cameras))
    assert budget["seed_evaluations"] == 48
    assert budget["maximum_projection_evaluations"] == 291
    assert budget["maximum_backward_evaluations"] == 240
    fixed = example.pose_camera_candidates(grid, cameras[:1], 45)
    assert [candidate["id"] for candidate in fixed] == [
        f"axial_{angle:03d}" for angle in range(0, 360, 45)
    ]
    assert example.fit_budget(3, 80, 45, 1)["maximum_projection_evaluations"] == 251


def test_best_pose_keeps_its_camera_after_worse_later_evaluations():
    grid = GridSpec((7, 9, 11), (0.8, 1.2, 2.5), (1000, -500, 300))
    geometry, settings = example.approximate_geometry(grid, (96, 160))
    cameras = [
        camera for _, camera in example.camera_hypotheses(geometry, settings, search_pitch=True)
    ]
    trajectory = [
        {"score": 0.7, "camera_id": cameras[0]["id"], "pose": "initial"},
        {"score": 0.95, "camera_id": cameras[2]["id"], "pose": "best"},
        {"score": 0.8, "camera_id": cameras[5]["id"], "pose": "last"},
    ]
    best, camera = example.best_pose_and_camera(trajectory, cameras)
    assert best["pose"] == "best"
    assert camera["camera_scale"] == 0.65
    assert camera["pixel_spacing_fit_raster_mm"] == settings["pixel_spacing_fit_raster_mm"] * 0.65
    with pytest.raises(ContractError, match="exactly one camera"):
        example.best_pose_and_camera(trajectory, cameras[:1])


def test_seed_transforms_rotate_about_native_ct_centre_and_round_trip():
    grid = GridSpec((7, 9, 11), (0.8, 1.2, 2.5), (1000, -500, 300))
    centre = np.array(example.physical_centre(grid))
    candidates = example.seed_candidates(grid)
    assert [candidate["id"] for candidate in candidates] == [
        f"axial_{angle:03d}" for angle in range(0, 360, 45)
    ]
    for candidate in candidates:
        pose = RigidTransform(**candidate["seed_pose_object_to_world"])
        np.testing.assert_allclose(pose.point(tuple(centre)), centre, atol=2e-12)
        point = grid.grid_to_object((1, 2, 3))
        np.testing.assert_allclose(pose.inverse().point(pose.point(point)), point, atol=2e-12)
        np.testing.assert_allclose(
            np.array(candidate["seed_matrix_object_to_world"]) @ [*point, 1],
            [*pose.point(point), 1],
            atol=2e-12,
        )


def test_differentiable_centred_chart_matches_independent_se3_composition():
    torch = pytest.importorskip("torch")
    seed = compose_pose(RigidTransform(), (0, 0, 0, 0, 0, math.pi / 2))
    centre = np.array([1000.0, -500, 300])
    xi = (3, -2, 1.5, 0.03, -0.02, 0.01)
    local = compose_pose(seed, xi)
    expected = (
        RigidTransform(translation_mm=tuple(centre))
        .compose(local)
        .compose(RigidTransform(translation_mm=tuple(-centre)))
    )
    tangent = torch.tensor(xi, dtype=torch.float64, requires_grad=True)
    packed = example.centred_torch_pose(
        torch.tensor(seed.rotation, dtype=torch.float64),
        tangent,
        torch.tensor(centre, dtype=torch.float64),
    )
    np.testing.assert_allclose(packed.detach().numpy(), expected.packed(), atol=3e-12)
    point = torch.tensor([1001, -499, 301], dtype=torch.float64)
    transformed = packed[:9].reshape(3, 3) @ point + packed[9:]
    transformed.square().sum().backward()
    assert tangent.grad is not None and torch.isfinite(tangent.grad).all()
    assert bool((tangent.grad != 0).all())


def test_ncc_is_gain_offset_and_polarity_invariant_with_finite_derivatives():
    torch = pytest.importorskip("torch")
    target = torch.tensor([[0.0, 0.3, 0.7], [1.0, 0.8, 0.2]], dtype=torch.float64)
    for gain in (3.0, -7.0):
        prediction = (gain * target + 17).requires_grad_()
        correlation = example.normalised_cross_correlation(prediction, target)
        assert float(correlation.detach()) == pytest.approx(math.copysign(1, gain), abs=1e-12)
        correlation.backward()
        assert prediction.grad is not None and torch.isfinite(prediction.grad).all()
    constant = torch.ones_like(target, requires_grad=True)
    score = example.normalised_cross_correlation(constant, target)
    score.backward()
    assert score.item() == 0
    assert constant.grad is not None and torch.isfinite(constant.grad).all()


def test_ncc_derivative_matches_finite_difference_for_mismatched_structure():
    torch = pytest.importorskip("torch")
    prediction = torch.tensor([[0.1, 0.4, 0.8], [0.9, 0.6, 0.2]], dtype=torch.float64)
    target = torch.tensor([[0.2, 0.1, 0.7], [1.0, 0.8, 0.4]], dtype=torch.float64)
    prediction.requires_grad_()
    score = example.normalised_cross_correlation(prediction, target)
    score.backward()
    assert 0 < score.item() < 1
    for index in range(prediction.numel()):
        plus, minus = prediction.detach().clone(), prediction.detach().clone()
        plus.flatten()[index] += 1e-6
        minus.flatten()[index] -= 1e-6
        difference = (
            example.normalised_cross_correlation(plus, target)
            - example.normalised_cross_correlation(minus, target)
        ) / 2e-6
        assert prediction.grad.flatten()[index].item() == pytest.approx(difference.item(), abs=1e-9)


def test_best_visited_pose_survives_a_worse_final_iterate_and_ties():
    trajectory = [
        {"score": score, "evaluation": index} for index, score in enumerate([0.6, 0.9, 0.9, 0.7])
    ]
    assert example.best_evaluation(trajectory) is trajectory[1]
    assert example.best_evaluation([{"score": 0.8}, {"score": 0.7}])["score"] == 0.8
    with pytest.raises(ContractError, match="at least one"):
        example.best_evaluation([])


def test_budget_covers_every_seed_and_post_update_evaluation():
    budget = example.fit_budget(3, 80, 45)
    assert budget["maximum_projection_evaluations"] == 8 + 3 * 81
    assert budget["maximum_backward_evaluations"] == 240
    for starts, iterations, step in [(0, 80, 45), (3, 0, 45), (3, 80, 47), (9, 80, 45)]:
        with pytest.raises(ContractError):
            example.fit_budget(starts, iterations, step)


def test_existing_output_is_rejected_before_loading_input(tmp_path):
    output = tmp_path / "fit"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("existing result")
    with pytest.raises(ContractError, match="already exists"):
        example.fit_radiograph(tmp_path / "absent.nii.gz", tmp_path / "absent.png", output)
    assert list(output.iterdir()) == [sentinel]
    assert sentinel.read_text() == "existing result"


def test_inspection_uses_real_nifti_bytes_and_retains_ras_grid(tmp_path):
    ct, affine = _ct(tmp_path)
    radiograph = _image(tmp_path)
    original = {path: path.read_bytes() for path in (ct, radiograph)}
    attenuation, image, geometry, report = example._load_inputs(ct, radiograph)
    assert attenuation.grid.shape == (6, 5, 4)
    np.testing.assert_allclose(
        attenuation.grid.grid_to_object((1, 2, 3)), (affine @ [1, 2, 3, 1])[:3], atol=1e-6
    )
    assert image.shape == geometry.views[0].shape == (6, 8)
    assert report["ct"]["source_file_sha256"] == [hashlib.sha256(ct.read_bytes()).hexdigest()]
    assert report["ct"]["resampled"] is False
    assert report["camera"]["coordinate_frame"] == "RAS"
    assert report["camera_search"]["pitch_scales"] == [1, 0.5, 0.65, 0.85, 1.2, 1.5]
    assert len(report["camera_hypotheses"]) == 6
    supplied = example.inspect_inputs(ct, radiograph, pixel_spacing_mm=0.8)
    assert supplied["camera_search"]["pitch_scales"] == [1]
    assert len(supplied["camera_hypotheses"]) == 1
    assert supplied["camera_hypotheses"][0]["pixel_spacing_original_raster_mm"] == 0.8
    assert {path: path.read_bytes() for path in original} == original
    process = subprocess.run(
        [sys.executable, str(_PATH), "inspect", str(ct), str(radiograph)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(process.stdout)["status"] == "inputs_valid"
    assert "Warp" not in process.stdout


def test_comparison_polarity_and_exact_optical_depth_are_saved(tmp_path):
    image_module = pytest.importorskip("PIL.Image")
    depth = np.arange(24, dtype=np.float32).reshape(4, 6) / 5
    target = 1 - depth / depth.max()
    display = example.save_comparison(tmp_path, target, depth)
    assert display["gain"] < 0
    np.testing.assert_array_equal(np.load(tmp_path / "optical_depth.npy"), depth)
    np.testing.assert_allclose(np.load(tmp_path / "display_fit.npy"), target, atol=1e-7)
    with image_module.open(tmp_path / "comparison.png") as saved:
        assert saved.size == (12, 4)
        pixels = np.array(saved)
    np.testing.assert_array_equal(pixels[:, :6], pixels[:, 6:])
    with image_module.open(tmp_path / "overlay.png") as saved:
        overlay = np.array(saved)
    np.testing.assert_array_equal(overlay[:, :, 0], overlay[:, :, 1])


def _dicom(path, *, uid, frame_uid, slice_index=0, frames=1):
    pydicom = pytest.importorskip("pydicom")
    meta = pydicom.dataset.FileMetaDataset()
    meta.MediaStorageSOPClassUID = pydicom.uid.CTImageStorage
    meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    image = pydicom.dataset.FileDataset(str(path), {}, file_meta=meta, preamble=bytes(128))
    image.SOPClassUID, image.SOPInstanceUID = (
        meta.MediaStorageSOPClassUID,
        meta.MediaStorageSOPInstanceUID,
    )
    image.SeriesInstanceUID, image.FrameOfReferenceUID = uid, frame_uid
    image.Modality = "CT"
    image.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    image.ImagePositionPatient = [10, 20, 30 + 2 * slice_index]
    image.PixelSpacing, image.SpacingBetweenSlices = [1.5, 0.8], 2
    image.Rows, image.Columns = 3, 4
    image.SamplesPerPixel, image.PhotometricInterpretation = 1, "MONOCHROME2"
    image.BitsAllocated, image.BitsStored, image.HighBit, image.PixelRepresentation = 16, 16, 15, 1
    image.RescaleSlope, image.RescaleIntercept, image.RescaleType = 2, -1000, "HU"
    values = np.arange(frames * 12, dtype=np.int16).reshape(frames, 3, 4) + 20 * slice_index
    image.PixelData = values.tobytes()
    if frames != 1:
        image.NumberOfFrames = frames
    image.save_as(path, enforce_file_format=True)
    return values[0]


def test_dicom_ct_series_and_single_frame_radiograph_are_read_from_files(tmp_path):
    pydicom = pytest.importorskip("pydicom")
    uid, frame = pydicom.uid.generate_uid(), pydicom.uid.generate_uid()
    directory = tmp_path / "ct"
    directory.mkdir()
    paths = [directory / "slice-1.dcm", directory / "slice-0.dcm"]
    for index, path in enumerate(paths):
        _dicom(path, uid=uid, frame_uid=frame, slice_index=index)
    radiograph = tmp_path / "radiograph.dcm"
    expected = _dicom(radiograph, uid=pydicom.uid.generate_uid(), frame_uid=frame)
    image, metadata = example.load_radiograph(radiograph)
    np.testing.assert_allclose(image, expected / expected.max(), atol=1e-7)
    assert "modality LUT" in metadata["conversion"]
    attenuation, _, _, report = example._load_inputs(directory, radiograph, series_uid=uid)
    assert attenuation.grid.shape == (2, 3, 4)
    np.testing.assert_allclose(attenuation.grid.grid_to_object((1, 1, 1)), [-10.8, -21.5, 32])
    assert report["ct"]["series_uid"] == uid
    assert sorted(report["ct"]["source_file_sha256"]) == sorted(
        hashlib.sha256(path.read_bytes()).hexdigest() for path in paths
    )


def test_multiframe_radiograph_is_rejected(tmp_path):
    pydicom = pytest.importorskip("pydicom")
    path = tmp_path / "frames.dcm"
    _dicom(path, uid=pydicom.uid.generate_uid(), frame_uid=pydicom.uid.generate_uid(), frames=2)
    with pytest.raises(ContractError, match="one frame"):
        example.load_radiograph(path)

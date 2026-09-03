"""Hand-derived geometry checks, independent of image optimisation."""

import math

import pytest

from dpt.contracts import ContractError, TrialDomainError
from dpt.geometry import DetectorGeometry, RigidTransform
from dpt.recovery import PoseChart
from dpt.validation.recovery import detector_coordinate, pose_error, reprojection_rms_pixels


def test_rotation_translation_and_target_errors_remain_separate() -> None:
    quarter_turn = RigidTransform((0, -1, 0, 1, 0, 0, 0, 0, 1), (0, 0, 3))
    error = pose_error(quarter_turn, RigidTransform(), [(1, 0, 0), (0, 1, 0)])
    assert error.translation_mm == 3
    assert math.isclose(error.rotation_radians, math.pi / 2)
    assert math.isclose(error.landmark_rms_mm, math.sqrt(11))
    assert math.isclose(error.landmark_max_mm, math.sqrt(11))


def test_half_turn_and_small_angle_use_stable_angle_metric() -> None:
    for angle in (1e-12, math.pi):
        c, s = math.cos(angle), math.sin(angle)
        pose = RigidTransform((c, -s, 0, s, c, 0, 0, 0, 1))
        error = pose_error(pose, RigidTransform(), [(0, 0, 0)])
        assert math.isclose(error.rotation_radians, angle, rel_tol=1e-12, abs_tol=1e-16)


def test_perspective_pixel_metric_keeps_outside_detector_coordinates() -> None:
    geometry = DetectorGeometry((0, 0, -10), (-2, -3, 10), (1, 0, 0), (0, 1, 0), (1, 2), (4, 5))
    assert detector_coordinate(geometry, (2, 0, 0)) == (6, 1.5)
    error = reprojection_rms_pixels(
        RigidTransform(translation_mm=(1, 0, 0)), RigidTransform(), geometry, [(0, 0, 0)]
    )
    assert error == 2
    with pytest.raises(ContractError):
        detector_coordinate(geometry, (0, 0, -10))
    with pytest.raises(ContractError):
        detector_coordinate(geometry, (0, 0, -20))


def test_chart_domain_is_a_rejectable_trial_error() -> None:
    chart = PoseChart()
    with pytest.raises(TrialDomainError):
        chart.pose((0, 0, 0, 400, 0, 0))
    with pytest.raises(TrialDomainError):
        PoseChart(anchor=RigidTransform(translation_mm=(1e308, 0, 0))).pose((1e308, 0, 0, 0, 0, 0))
    with pytest.raises(ContractError):
        chart.pose((0, 0))

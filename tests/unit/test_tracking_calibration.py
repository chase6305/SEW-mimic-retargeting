import numpy as np
import pytest

from sew_mimic import rot
from sew_mimic.tracking import (
    CalibrationFitError,
    fit_hand_tool_rotation,
    fit_rigid_transform,
)


@pytest.fixture
def references():
    source = np.random.default_rng(501).uniform(-0.5, 0.5, (20, 3))
    rotation = rot([1, 2, -3], 1.3)
    translation = np.array([0.3, -0.8, 1.2])
    return source, source @ rotation.T + translation, rotation, translation


def test_rigid_fit_recovers_world_transform_and_owned_results(references):
    source, target, rotation, translation = references
    fit = fit_rigid_transform(source, target)
    np.testing.assert_allclose(fit.rotation, rotation, atol=1e-14)
    np.testing.assert_allclose(fit.translation, translation, atol=1e-14)
    fit.require_accuracy(max_rms_error=1e-13, max_point_error=1e-13)
    assert fit.rms_error < 1e-14
    assert fit.max_error < 1e-14
    source[:] = target[:] = 0
    np.testing.assert_allclose(fit.rotation, rotation, atol=1e-14)
    for value in (fit.rotation, fit.translation, fit.residuals):
        assert not value.flags.writeable


def test_three_noncollinear_planar_references_are_sufficient():
    points = np.array([[0, 0, 0], [0.5, 0, 0], [0, 0.3, 0]])
    rotation = rot([1, 2, 3], 2.5)
    fit = fit_rigid_transform(points, points @ rotation.T + [1, 2, 3])
    np.testing.assert_allclose(fit.rotation, rotation, atol=1e-14)
    np.testing.assert_allclose(fit.translation, [1, 2, 3], atol=1e-14)


def test_rigid_fit_handles_large_coordinate_origin(references):
    source, _, rotation, translation = references
    source = source + [1e8, -2e8, 1e8]
    target = source @ rotation.T + translation
    fit = fit_rigid_transform(source, target)
    np.testing.assert_allclose(fit.rotation, rotation, atol=1e-7)
    # At a far origin, tiny angular uncertainty can amplify translation error;
    # assess predictions within the observed reference volume instead.
    np.testing.assert_allclose(source @ fit.rotation.T + fit.translation, target, atol=1e-6, rtol=0)


def test_noise_and_outlier_residuals_are_reported_without_discarding_samples(references):
    source, target, rotation, _ = references
    noisy = target + np.random.default_rng(502).normal(0, 0.0004, target.shape)
    fit = fit_rigid_transform(source, noisy)
    fit.require_accuracy(max_rms_error=0.001, max_point_error=0.002)
    np.testing.assert_allclose(fit.rotation, rotation, atol=0.002)
    np.testing.assert_allclose(
        fit.residuals, np.linalg.norm(source @ fit.rotation.T + fit.translation - noisy, axis=1)
    )
    assert len(fit.residuals) == len(source)
    noisy[-1] += [0.1, 0, 0]
    bad = fit_rigid_transform(source, noisy)
    with pytest.raises(CalibrationFitError, match="RMS=.*max=.*m"):
        bad.require_accuracy(max_rms_error=0.01)
    with pytest.raises(CalibrationFitError):
        bad.require_accuracy(max_rms_error=1.0, max_point_error=0.01)


@pytest.mark.parametrize(
    "points",
    [
        np.zeros((4, 3)),
        np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]]),
        np.array([[0, 0, 0], [1, 1e-8, 0], [2, 0, 0]]),
        np.array([[0, 0, 0], [1e-6, 0, 0], [0, 1e-6, 0]]),
    ],
)
def test_uninformative_reference_geometry_is_rejected(points):
    with pytest.raises(CalibrationFitError, match="spread|collinear"):
        fit_rigid_transform(points, points + [1, 2, 3])


def test_scale_and_reflection_are_not_silently_fitted_as_rigid_motion():
    source = np.array([[0, 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 3]])
    for target in (source * 100, source * [-1, 1, 1]):
        fit = fit_rigid_transform(source, target)
        assert np.linalg.det(fit.rotation) == pytest.approx(1.0)
        with pytest.raises(CalibrationFitError):
            fit.require_accuracy(max_rms_error=0.001)


@pytest.mark.parametrize(
    "source,target",
    [
        (np.zeros((2, 3)), np.zeros((2, 3))),
        (np.zeros((3, 2)), np.zeros((3, 2))),
        (np.zeros((3, 3)), np.zeros((4, 3))),
        (np.full((3, 3), np.nan), np.zeros((3, 3))),
        (np.zeros((3, 3)), np.full((3, 3), np.inf)),
    ],
)
def test_point_fit_rejects_invalid_input(source, target):
    with pytest.raises(ValueError):
        fit_rigid_transform(source, target)


@pytest.mark.parametrize("threshold", [0, -1, np.nan, np.inf, True])
def test_point_fit_requires_valid_spread_limit(references, threshold):
    with pytest.raises(ValueError):
        fit_rigid_transform(*references[:2], min_spread=threshold)


def test_point_fit_rejects_coordinate_overflow():
    points = np.array([[-1e308, 0, 0], [1e308, 0, 0], [0, 1e308, 0]])
    with pytest.raises(CalibrationFitError, match="numerically"):
        fit_rigid_transform(points, points)


@pytest.mark.parametrize("angle", [0.5, np.pi - 1e-10, np.pi])
def test_hand_fit_preserves_noncommuting_world_and_local_transform_order(angle):
    world = rot([0, 0, 1], 0.6)
    offset = rot([1, 2, 0], angle)
    hands = np.stack([rot([1, 0, 0], t) @ rot([0, 1, 0], 0.3 * t) for t in (0.0, 0.4, 1.2)])
    tools = world @ hands @ offset
    fit = fit_hand_tool_rotation(hands, tools, tracking_to_robot_rotation=world)
    np.testing.assert_allclose(fit.rotation, offset, atol=1e-14)
    assert fit.rms_angle < 1e-14
    assert fit.max_angle < 1e-14
    fit.require_accuracy(max_rms_angle=1e-13, max_angle=1e-13)
    assert not fit.rotation.flags.writeable
    assert not fit.angular_residuals.flags.writeable


def test_known_world_rotation_allows_one_complete_orientation_pair():
    world = rot([0, 1, 0], 0.7)
    hand = rot([1, 0, 0], -0.2)
    offset = rot([0, 0, 1], 0.3)
    fit = fit_hand_tool_rotation([hand], [world @ hand @ offset], tracking_to_robot_rotation=world)
    np.testing.assert_allclose(fit.rotation, offset, atol=1e-14)


def test_hand_fit_reports_dispersion_in_radians_and_rejects_large_error():
    hands = np.stack([np.eye(3)] * 3)
    tools = np.stack([rot([0, 0, 1], t) for t in (-0.01, 0, 0.01)])
    fit = fit_hand_tool_rotation(hands, tools, tracking_to_robot_rotation=np.eye(3))
    np.testing.assert_allclose(fit.angular_residuals, [0.01, 0, 0.01], atol=1e-15)
    fit.require_accuracy(max_rms_angle=0.01, max_angle=0.011)
    with pytest.raises(CalibrationFitError, match="rad"):
        fit.require_accuracy(max_rms_angle=0.001)
    with pytest.raises(CalibrationFitError):
        fit.require_accuracy(max_rms_angle=0.1, max_angle=0.005)


def test_conflicting_orientation_offsets_are_rejected_as_ambiguous():
    hands = [np.eye(3), np.eye(3)]
    tools = [np.eye(3), rot([0, 0, 1], np.pi)]
    with pytest.raises(CalibrationFitError, match="stable rotation"):
        fit_hand_tool_rotation(hands, tools, tracking_to_robot_rotation=np.eye(3))


def test_hand_offset_cannot_hide_wrong_world_calibration_across_distinct_poses():
    hands = np.stack([rot([1, 0, 0], t) for t in (-1, 0, 1)])
    tools = rot([0, 0, 1], 0.8) @ hands @ rot([0, 1, 0], 0.5)
    fit = fit_hand_tool_rotation(hands, tools, tracking_to_robot_rotation=np.eye(3))
    with pytest.raises(CalibrationFitError):
        fit.require_accuracy(max_rms_angle=0.01)


@pytest.mark.parametrize(
    "hands,tools,world",
    [
        ([], [], np.eye(3)),
        ([np.eye(3)], [np.eye(3), np.eye(3)], np.eye(3)),
        ([np.eye(3)], [np.zeros((3, 3))], np.eye(3)),
        ([np.eye(3)], [np.eye(3)], np.diag([-1, 1, 1])),
        ([np.full((3, 3), np.nan)], [np.eye(3)], np.eye(3)),
    ],
)
def test_hand_fit_rejects_invalid_input(hands, tools, world):
    with pytest.raises(ValueError):
        fit_hand_tool_rotation(hands, tools, tracking_to_robot_rotation=world)


@pytest.mark.parametrize("value", [-1, np.nan, np.inf, True])
def test_fit_accuracy_requires_valid_limits(references, value):
    fit = fit_rigid_transform(*references[:2])
    with pytest.raises(ValueError):
        fit.require_accuracy(max_rms_error=value)

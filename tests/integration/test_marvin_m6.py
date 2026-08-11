from pathlib import Path

import numpy as np
import pytest

from examples.demo_marvin_viser import (
    humanlike_reference_trajectory,
    minimum_jerk_trajectory,
    reference_keypoint_trajectory,
    retarget_reference_trajectory,
)
from sew_mimic import alignment_diagnostics, sew_mimic
from sew_mimic.robots import DEFAULT_MARVIN_URDF, load_marvin_arm


def test_minimum_jerk_trajectory_endpoints():
    q0 = np.array([0.0, -0.2, 0.3])
    q1 = np.array([1.0, 0.4, -0.5])
    times, positions = minimum_jerk_trajectory(q0, q1, duration=2.0, fps=20.0)
    assert len(times) == 41
    assert np.allclose(positions[0], q0)
    assert np.allclose(positions[-1], q1)
    assert np.all(np.diff(times) > 0.0)
    # The analytic minimum-jerk blend has zero endpoint velocity.
    assert np.linalg.norm(positions[1] - positions[0]) < 1e-3
    assert np.linalg.norm(positions[-1] - positions[-2]) < 1e-3


def test_humanlike_trajectory_is_smooth_and_closed():
    times, positions = humanlike_reference_trajectory("left", duration=16.0, fps=20.0)
    assert len(times) == 321
    assert np.all(np.diff(times) > 0.0)
    assert np.allclose(positions[0], positions[-1])
    assert np.max(np.linalg.norm(np.diff(positions, axis=0), axis=1)) < 0.5
    # A closed cubic curve should also have matching endpoint velocity.
    assert np.allclose(
        positions[1] - positions[0],
        positions[-1] - positions[-2],
        atol=4e-2,
    )


@pytest.mark.skipif(not Path(DEFAULT_MARVIN_URDF).is_file(), reason="Marvin asset is not installed")
def test_reference_keypoints_have_human_limb_lengths():
    robot = load_marvin_arm(side="left").robot
    _, reference = humanlike_reference_trajectory("left", duration=2.0, fps=10.0)
    points = reference_keypoint_trajectory(robot, reference)
    assert points.shape == (21, 3, 3)
    assert np.allclose(np.linalg.norm(points[:, 1] - points[:, 0], axis=1), 0.287)
    assert np.allclose(np.linalg.norm(points[:, 2] - points[:, 1], axis=1), 0.314)


@pytest.mark.skipif(not Path(DEFAULT_MARVIN_URDF).is_file(), reason="Marvin asset is not installed")
@pytest.mark.parametrize("side", ["left", "right"])
def test_marvin_urdf_fk_to_sew(side):
    arm = load_marvin_arm(side=side)
    robot = arm.robot
    q_gt = np.array([0.35, -0.45, 0.40, -0.75, 0.30, 0.25, -0.20])

    upper = robot.axis_world(q_gt, 3)
    lower = robot.axis_world(q_gt, 5)
    hand = robot.tool_orientation(q_gt)
    shoulder = np.zeros(3)
    elbow = shoulder + 0.287 * upper
    wrist = elbow + 0.314 * lower

    q = sew_mimic(robot, np.zeros(7), shoulder, elbow, wrist, hand)
    errors = alignment_diagnostics(robot, q, shoulder, elbow, wrist, hand)
    assert errors["upper_vector_l2"] < 1e-8
    assert errors["lower_vector_l2"] < 1e-8
    assert errors["tool_rotation_fro"] < 1e-8
    assert np.max(np.abs(robot.consecutive_axis_dot_products())) < 1e-5


@pytest.mark.skipif(not Path(DEFAULT_MARVIN_URDF).is_file(), reason="Marvin asset is not installed")
@pytest.mark.parametrize("side", ["left", "right"])
def test_humanlike_trajectory_retargets_both_arms(side):
    robot = load_marvin_arm(side=side).robot
    _, reference = humanlike_reference_trajectory(side, duration=1.0, fps=10.0)
    solved, max_error = retarget_reference_trajectory(robot, reference)
    assert solved.shape == reference.shape
    assert max_error < 1e-8
    assert np.all(solved >= robot.q_min - 1e-10)
    assert np.all(solved <= robot.q_max + 1e-10)

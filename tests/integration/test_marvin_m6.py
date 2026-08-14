from pathlib import Path

import numpy as np
import pytest

from examples.demo_robot_collision_avoidance import (
    collision_test_trajectory,
    plan_filtered_trajectory,
)
from examples.demo_robot_viser import (
    humanlike_reference_trajectory,
    minimum_jerk_trajectory,
    openarm_reference_trajectory,
    reference_keypoint_trajectory,
    retarget_reference_trajectory,
    urdf_reference_keypoint_trajectory,
)
from sew_mimic import alignment_diagnostics, cpp_backend_available, sew_mimic
from sew_mimic.robots import (
    DEFAULT_MARVIN_URDF,
    DEFAULT_OPENARM_URDF,
    MarvinSafetyFilter,
    OpenArmSafetyFilter,
    RobotSafetyFilter,
    SerialArmSpec,
    URDFBimanualPoseEvaluator,
    URDFKinematics,
    available_robots,
    estimate_marvin_capsule_config,
    load_marvin_arm,
    load_openarm_arm,
    load_robot_arm,
    load_serial_7dof_arm,
    marvin_bimanual_pose,
    urdf_bimanual_pose,
    validate_robot_adapter,
)


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


@pytest.mark.skipif(not Path(DEFAULT_MARVIN_URDF).is_file(), reason="Marvin asset is not installed")
def test_marvin_position_fk_and_oobb_capsules():
    left = load_marvin_arm(side="left")
    right = load_marvin_arm(side="right")
    kinematics = URDFKinematics(DEFAULT_MARVIN_URDF)
    pose = marvin_bimanual_pose(kinematics, left, right, np.zeros(7), np.zeros(7))
    generic_pose = urdf_bimanual_pose(kinematics, left, right, np.zeros(7), np.zeros(7))
    points = pose.points()
    assert generic_pose.points() == pytest.approx(points)
    evaluator = URDFBimanualPoseEvaluator(kinematics, left, right)
    assert evaluator.evaluate(np.zeros(7), np.zeros(7)).points() == pytest.approx(points)
    assert points.shape == (8, 3)
    assert np.all(np.isfinite(points))
    assert np.all(
        np.linalg.norm(points[[1, 2, 3, 5, 6, 7]] - points[[0, 1, 2, 4, 5, 6]], axis=1) > 0.05
    )

    config = estimate_marvin_capsule_config()
    assert 0.04 < config.radii.upper_arm < 0.12
    assert 0.04 < config.radii.lower_arm < 0.12
    assert 0.04 < config.radii.hand < 0.12
    assert 0.05 < config.radii.torso < 0.20

    with pytest.raises(ValueError, match="matching side labels"):
        urdf_bimanual_pose(kinematics, right, left, np.zeros(7), np.zeros(7))


@pytest.mark.skipif(not Path(DEFAULT_MARVIN_URDF).is_file(), reason="Marvin asset is not installed")
def test_collision_demo_blocks_unsafe_trajectory():
    safety_filter = MarvinSafetyFilter()
    _, desired_left, desired_right = collision_test_trajectory(duration=2.0, fps=10.0)
    solve_timings = []
    safe_left, safe_right, target_distances, command_distances, accepted = plan_filtered_trajectory(
        safety_filter, desired_left, desired_right, solve_timings
    )
    assert len(solve_timings) == len(desired_left)
    assert np.all(np.asarray(solve_timings) > 0.0)
    assert np.all(np.linalg.norm(np.diff(desired_left, axis=0), axis=1) > 0.0)
    assert np.min(target_distances) < 0.0
    assert np.min(command_distances) >= safety_filter.config.minimum_distance
    assert np.any(~accepted)
    assert accepted[-1]
    assert np.allclose(safe_left[-1], desired_left[-1])
    assert np.allclose(safe_right[-1], desired_right[-1])

    middle_pose = safety_filter.forward_kinematics(
        desired_left[len(desired_left) // 2], desired_right[len(desired_right) // 2]
    )
    middle_points = middle_pose.points()
    moving_heights = middle_points[[1, 2, 3, 5, 6, 7], 2]
    assert np.min(moving_heights) > 0.85
    assert np.max(moving_heights) < 1.20
    # Elbows remain on their natural sides while the forearms cross visibly.
    assert middle_points[1, 1] > 0.15
    assert middle_points[5, 1] < -0.15
    assert middle_points[3, 1] < 0.0 < middle_points[7, 1]


def test_cpp_and_python_safety_paths_enforce_same_clearance():
    if not cpp_backend_available():
        pytest.skip("native extension is not built")
    python_filter = MarvinSafetyFilter(backend="python")
    cpp_filter = MarvinSafetyFilter(backend="cpp")
    _, desired_left, desired_right = collision_test_trajectory(duration=2.0, fps=10.0)
    python_plan = plan_filtered_trajectory(python_filter, desired_left, desired_right)
    cpp_plan = plan_filtered_trajectory(cpp_filter, desired_left, desired_right)
    assert np.min(python_plan[3]) >= python_filter.config.minimum_distance
    assert np.min(cpp_plan[3]) >= cpp_filter.config.minimum_distance
    assert np.array_equal(python_plan[4], cpp_plan[4])


@pytest.mark.skipif(
    not Path(DEFAULT_OPENARM_URDF).is_file(), reason="OpenArm asset is not installed"
)
@pytest.mark.parametrize("side", ["left", "right"])
def test_openarm_urdf_and_demo_trajectory(side):
    arm = load_openarm_arm(side=side)
    assert arm.ee_link == f"{side}_hand_base"
    assert np.max(np.abs(arm.robot.consecutive_axis_dot_products())) < 1e-5
    _, reference = openarm_reference_trajectory(side, duration=2.0, fps=20.0)
    target_orientation = arm.robot.tool_orientation(reference[0])
    orientation_errors = []
    for q_reference in reference:
        relative = target_orientation.T @ arm.robot.tool_orientation(q_reference)
        orientation_errors.append(np.arccos(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)))
    assert np.max(orientation_errors) < np.deg2rad(15.0)
    solved, max_error = retarget_reference_trajectory(arm.robot, reference)
    assert solved.shape == reference.shape
    assert max_error < 1e-8
    assert np.all(solved >= arm.robot.q_min - 1e-10)
    assert np.all(solved <= arm.robot.q_max + 1e-10)

    kinematics = URDFKinematics(DEFAULT_OPENARM_URDF)
    keypoints = urdf_reference_keypoint_trajectory(kinematics, arm, reference)
    solved_keypoints = urdf_reference_keypoint_trajectory(kinematics, arm, solved)
    assert keypoints.shape == (len(reference), 3, 3)
    assert np.allclose(solved_keypoints, keypoints, atol=1e-10)
    # The visual target uses the physical J1/J4/hand-base landmarks, including the
    # real OpenArm link offsets instead of generic human limb lengths.
    upper_lengths = np.linalg.norm(keypoints[:, 1] - keypoints[:, 0], axis=1)
    lower_lengths = np.linalg.norm(keypoints[:, 2] - keypoints[:, 1], axis=1)
    assert np.all((upper_lengths > 0.20) & (upper_lengths < 0.30))
    assert np.all((lower_lengths > 0.27) & (lower_lengths < 0.33))


def test_unified_robot_registry_loads_bundled_adapters():
    assert available_robots() == ("marvin", "openarm")
    marvin = load_robot_arm("m6", "left")
    openarm = load_robot_arm("open-arm", "right")
    assert marvin.side == "left"
    assert openarm.side == "right"
    assert len(marvin.joint_names) == len(openarm.joint_names) == 7


@pytest.mark.parametrize("name", ["marvin", "openarm"])
def test_bundled_robot_adapters_pass_generic_validation(name):
    report = validate_robot_adapter(name)
    assert report.name == name
    assert report.keypoints_finite
    assert report.minimum_joint_range > 0.0
    assert report.maximum_consecutive_axis_dot < 1e-4


def test_generic_arm_loader_rejects_invalid_contracts():
    arm = load_marvin_arm(side="left")
    with pytest.raises(ValueError, match="seven unique joints"):
        load_serial_7dof_arm(DEFAULT_MARVIN_URDF, arm.joint_names[:-1], arm.ee_link)
    with pytest.raises(ValueError, match="No fixed tool chain"):
        load_serial_7dof_arm(DEFAULT_MARVIN_URDF, arm.joint_names, "missing_tool_link")


def test_declarative_arm_spec_loads_both_sides():
    left = load_marvin_arm(side="left")
    right = load_marvin_arm(side="right")
    spec = SerialArmSpec(
        left_joint_names=left.joint_names,
        right_joint_names=right.joint_names,
        left_ee_link=left.ee_link,
        right_ee_link=right.ee_link,
    )
    assert spec.load(DEFAULT_MARVIN_URDF, "left").joint_names == left.joint_names
    assert spec.load(DEFAULT_MARVIN_URDF, "right").joint_names == right.joint_names


def test_declarative_arm_spec_validates_contract():
    names = tuple(f"joint_{index}" for index in range(7))
    with pytest.raises(ValueError, match="seven unique joints"):
        SerialArmSpec(names[:-1], names, "left_tool", "right_tool")
    spec = SerialArmSpec(names, names, "left_tool", "right_tool")
    with pytest.raises(ValueError, match="side must"):
        spec.load(DEFAULT_MARVIN_URDF, "centre")


def test_bundled_safety_filters_share_generic_orchestration():
    assert issubclass(MarvinSafetyFilter, RobotSafetyFilter)
    assert issubclass(OpenArmSafetyFilter, RobotSafetyFilter)


@pytest.mark.skipif(
    not Path(DEFAULT_OPENARM_URDF).is_file(), reason="OpenArm asset is not installed"
)
def test_openarm_collision_demo_blocks_unsafe_target():
    safety_filter = OpenArmSafetyFilter()
    _, desired_left, desired_right = collision_test_trajectory(
        duration=2.0, fps=10.0, robot="openarm"
    )
    _, _, target_distances, command_distances, accepted = plan_filtered_trajectory(
        safety_filter, desired_left, desired_right
    )
    assert np.min(target_distances) < 0.0
    assert np.min(command_distances) >= safety_filter.config.minimum_distance
    assert np.any(~accepted)

    middle_pose = safety_filter.forward_kinematics(
        desired_left[len(desired_left) // 2], desired_right[len(desired_right) // 2]
    )
    middle_points = middle_pose.points()
    assert middle_points[1, 1] > 0.15
    assert middle_points[5, 1] < -0.15
    # The tracked hand bases pass each other at the torso centreline.
    assert middle_points[3, 1] < 0.0 < middle_points[7, 1]
    assert np.max(np.abs(middle_points[[3, 7], 1])) < 0.03
    assert np.all((middle_points[[1, 2, 3, 5, 6, 7], 2] > 0.55))

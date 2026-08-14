import logging

import numpy as np
import pytest

from sew_mimic import (
    ArmPose,
    BimanualPose,
    CapsuleRadii,
    SafetyFilterConfig,
    SafetyFilterStatus,
    cpp_backend_available,
    find_first_collision,
    minimum_capsule_distance,
    recover_tool_orientation,
    sew_safety_filter,
)
from sew_mimic.backends import get_cpp_collision_backend
from sew_mimic.safety import _xpbd_iteration


def _pose(elbow_offset: float) -> BimanualPose:
    return BimanualPose(
        ArmPose(
            np.array([0.0, 0.5, 1.0]),
            np.array([0.3, elbow_offset, 1.0]),
            np.array([0.6, elbow_offset, 1.0]),
            np.array([0.8, elbow_offset, 1.0]),
            np.eye(3),
        ),
        ArmPose(
            np.array([0.0, -0.5, 1.0]),
            np.array([0.3, -elbow_offset, 1.0]),
            np.array([0.6, -elbow_offset, 1.0]),
            np.array([0.8, -elbow_offset, 1.0]),
            np.eye(3),
        ),
    )


def _config() -> SafetyFilterConfig:
    return SafetyFilterConfig(
        CapsuleRadii(torso=0.1, upper_arm=0.08, lower_arm=0.07, hand=0.06),
        torso_start=np.array([0.0, 0.0, -2.0]),
        torso_end=np.array([0.0, 0.0, -1.0]),
        iterations=100,
    )


def test_continuous_check_stops_before_deep_collision():
    config = _config()
    current = _pose(0.3).points()
    desired = _pose(0.0).points()
    first_collision = find_first_collision(current, desired, config)
    assert not np.shares_memory(first_collision, current)
    assert not np.shares_memory(first_collision, desired)
    assert minimum_capsule_distance(desired, config) < 0.0
    assert minimum_capsule_distance(first_collision, config) > minimum_capsule_distance(
        desired, config
    )


@pytest.mark.skipif(not cpp_backend_available(), reason="native extension is not built")
def test_cpp_continuous_check_matches_python_sampling():
    config = _config()
    current = _pose(0.3).keypoints()
    desired = _pose(0.0).keypoints()
    python_points = find_first_collision(current, desired, config, backend="python")
    cpp_points = find_first_collision(current, desired, config, backend="cpp")
    assert cpp_points == pytest.approx(python_points, abs=1e-15)


def test_safety_filter_resolves_cross_arm_collision():
    config = _config()

    def fk(q_left, q_right):
        return _pose(0.0 if q_left[0] > 0.5 else 0.3)

    result = sew_safety_filter(
        np.zeros(7),
        np.zeros(7),
        np.ones(7),
        np.ones(7),
        forward_kinematics=fk,
        solve_left=lambda q, pose: q,
        solve_right=lambda q, pose: q,
        config=config,
    )
    assert result.safe
    assert result.minimum_distance >= config.minimum_distance - config.tolerance
    assert result.status is SafetyFilterStatus.CORRECTED


def test_recover_tool_orientation_aligns_x_axis():
    recovered = recover_tool_orientation(np.eye(3), np.array([0.0, 1.0, 0.0]))
    assert np.allclose(recovered[:, 0], [0.0, 1.0, 0.0])


def test_keypoint_extraction_skips_unneeded_orientation_validation():
    pose = _pose(0.3)
    invalid_left = ArmPose(
        pose.left.shoulder,
        pose.left.elbow,
        pose.left.wrist,
        pose.left.tool,
        np.zeros((3, 3)),
    )
    bimanual = BimanualPose(invalid_left, pose.right)

    assert bimanual.keypoints().shape == (8, 3)
    with pytest.raises(ValueError, match=r"SO\(3\)"):
        bimanual.points()


def test_safety_filter_reports_stage_logs(caplog):
    caplog.set_level(logging.DEBUG, logger="sew_mimic.safety")

    def fk(q_left, q_right):
        return _pose(0.3)

    result = sew_safety_filter(
        np.zeros(7),
        np.zeros(7),
        np.zeros(7),
        np.zeros(7),
        forward_kinematics=fk,
        solve_left=lambda q, pose: q,
        solve_right=lambda q, pose: q,
        config=_config(),
    )
    assert result.safe
    assert result.status is SafetyFilterStatus.ACCEPTED
    assert result.desired_minimum_distance == pytest.approx(result.minimum_distance)
    assert result.command_minimum_distance == pytest.approx(result.minimum_distance)
    assert "Safety stage 1/6" in caplog.text
    assert "accepted desired pose on fast path" in caplog.text


def test_safety_config_accepts_robot_specific_collision_pairs():
    config = SafetyFilterConfig(
        CapsuleRadii(torso=0.1, upper_arm=0.08, lower_arm=0.07, hand=0.06),
        torso_start=np.array([0.0, 0.0, -2.0]),
        torso_end=np.array([0.0, 0.0, -1.0]),
        collision_pairs=((1, 4),),
    )
    assert config.collision_pairs == ((1, 4),)


def test_safety_config_caches_read_only_geometry_arrays():
    config = _config()
    assert config.collision_pair_indices.dtype == np.int32
    assert config.link_radii == pytest.approx([0.1, 0.08, 0.07, 0.06])
    assert config.capsule_radii == pytest.approx([0.1, 0.08, 0.07, 0.06, 0.08, 0.07, 0.06])
    assert config.interpolation_point_radii == pytest.approx(
        [0.08, 0.08, 0.07, 0.06, 0.08, 0.08, 0.07, 0.06]
    )
    assert not config.link_radii.flags.writeable
    assert not config.capsule_radii.flags.writeable
    assert not config.interpolation_point_radii.flags.writeable
    assert not config.collision_pair_indices.flags.writeable


@pytest.mark.skipif(not cpp_backend_available(), reason="native extension is not built")
def test_cpp_xpbd_projection_matches_python_reference():
    config = _config()
    points = find_first_collision(_pose(0.3).points(), _pose(0.0).points(), config)
    python_points = points.copy()
    lengths = np.linalg.norm(points[[1, 2, 3, 5, 6, 7]] - points[[0, 1, 2, 4, 5, 6]], axis=1)
    multipliers = np.zeros(len(config.collision_pairs))
    python_iterations = 0
    python_distance = minimum_capsule_distance(python_points, config)
    for python_iterations in range(1, config.iterations + 1):
        if python_distance >= config.minimum_distance - config.tolerance:
            break
        _xpbd_iteration(python_points, config, multipliers, lengths)
        python_distance = minimum_capsule_distance(python_points, config)

    radii = np.array(
        [
            config.radii.torso,
            config.radii.upper_arm,
            config.radii.lower_arm,
            config.radii.hand,
        ]
    )
    cpp_points, cpp_iterations, cpp_distance = get_cpp_collision_backend().project_xpbd(
        points,
        config.torso_start,
        config.torso_end,
        radii,
        config.collision_pair_indices,
        minimum_distance=config.minimum_distance,
        activation_distance=config.activation_distance,
        release_distance=config.release_distance,
        compliance=config.compliance,
        tolerance=config.tolerance,
        iterations=config.iterations,
    )
    assert cpp_iterations == python_iterations
    assert cpp_distance == pytest.approx(python_distance, abs=1e-12)
    assert np.allclose(cpp_points, python_points, atol=1e-12)

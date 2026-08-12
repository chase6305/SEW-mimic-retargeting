import logging

import numpy as np

from sew_mimic import (
    ArmPose,
    BimanualPose,
    CapsuleRadii,
    SafetyFilterConfig,
    SafetyFilterStatus,
    find_first_collision,
    minimum_capsule_distance,
    recover_tool_orientation,
    sew_safety_filter,
)


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
    assert minimum_capsule_distance(desired, config) < 0.0
    assert minimum_capsule_distance(first_collision, config) > minimum_capsule_distance(
        desired, config
    )


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

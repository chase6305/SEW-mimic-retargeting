from dataclasses import replace

import numpy as np
import pytest

from sew_mimic import (
    ArmPose,
    BimanualPose,
    BimanualPoseFilter,
    JointRateLimiter,
    OneEuroConfig,
    OneEuroFilter,
    OneEuroRotationFilter,
    is_rotation_matrix,
    rot,
)


def _pose(offset: float = 0.0, angle: float = 0.0) -> BimanualPose:
    orientation = rot(np.array([0.0, 0.0, 1.0]), angle)
    left = np.array([[0.0, 0.2, 1.0], [0.2, 0.3, 0.9], [0.4, 0.3, 0.8], [0.5, 0.3, 0.8]])
    right = left * np.array([1.0, -1.0, 1.0])
    translation = np.array([offset, 0.0, 0.0])
    return BimanualPose(
        ArmPose(*left + translation, orientation),
        ArmPose(*right + translation, orientation),
    )


def test_one_euro_preserves_shape_and_smooths_step_input():
    filter_ = OneEuroFilter(min_cutoff=1.0, beta=0.0)
    initial = filter_.update(0.0, np.zeros((2, 3)))
    filtered = filter_.update(0.01, np.ones((2, 3)))
    assert initial.shape == filtered.shape == (2, 3)
    assert np.all((filtered > 0.0) & (filtered < 1.0))
    assert not np.shares_memory(filtered, filter_._value)


def test_one_euro_adapts_more_to_fast_motion():
    fixed = OneEuroFilter(min_cutoff=1.0, beta=0.0)
    adaptive = OneEuroFilter(min_cutoff=1.0, beta=1.0)
    fixed.update(0.0, 0.0)
    adaptive.update(0.0, 0.0)
    assert adaptive.update(0.01, 1.0) > fixed.update(0.01, 1.0)


def test_one_euro_validates_time_shape_and_reset():
    filter_ = OneEuroFilter()
    filter_.update(1.0, np.zeros(3))
    with pytest.raises(ValueError, match="increase strictly"):
        filter_.update(1.0, np.zeros(3))
    with pytest.raises(ValueError, match="shape changed"):
        filter_.update(2.0, np.zeros(4))
    filter_.reset()
    assert filter_.update(0.0, np.ones(4)) == pytest.approx(np.ones(4))


def test_joint_rate_limiter_enforces_velocity_for_variable_dt():
    limiter = JointRateLimiter(np.array([1.0, 2.0]))
    limiter.reset(np.zeros(2), timestamp=0.0)
    first = limiter.update(0.1, np.array([1.0, -1.0]))
    second = limiter.update(0.3, np.array([1.0, -1.0]))
    assert first == pytest.approx([0.1, -0.2])
    assert second == pytest.approx([0.3, -0.6])
    assert np.all(np.abs((second - first) / 0.2) <= np.array([1.0, 2.0]) + 1e-12)


def test_joint_rate_limiter_requires_reset_and_monotonic_time():
    limiter = JointRateLimiter(1.0)
    with pytest.raises(RuntimeError, match="reset"):
        limiter.update(0.1, np.zeros(2))
    limiter.reset(np.zeros(2), 0.0)
    with pytest.raises(ValueError, match="increase strictly"):
        limiter.update(0.0, np.ones(2))
    with pytest.raises(ValueError, match="preserve command shape"):
        limiter.update(0.1, np.ones(3))


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: OneEuroFilter(min_cutoff=0.0), "min_cutoff"),
        (lambda: OneEuroFilter(beta=-1.0), "beta"),
        (lambda: JointRateLimiter([1.0, 0.0]), "max_velocity"),
    ],
)
def test_filters_reject_invalid_parameters(factory, message):
    with pytest.raises(ValueError, match=message):
        factory()


def test_rotation_filter_smooths_on_so3_and_adapts_to_motion():
    target = rot(np.array([0.0, 0.0, 1.0]), np.pi / 2.0)
    fixed = OneEuroRotationFilter(min_cutoff=1.0, beta=0.0)
    adaptive = OneEuroRotationFilter(min_cutoff=1.0, beta=1.0)
    for filter_ in (fixed, adaptive):
        assert filter_.update(0.0, np.eye(3)) == pytest.approx(np.eye(3))
    fixed_result = fixed.update(0.01, target)
    adaptive_result = adaptive.update(0.01, target)
    assert is_rotation_matrix(fixed_result)
    assert is_rotation_matrix(adaptive_result)
    fixed_angle = np.arccos(np.clip((np.trace(fixed_result) - 1.0) / 2.0, -1.0, 1.0))
    adaptive_angle = np.arccos(np.clip((np.trace(adaptive_result) - 1.0) / 2.0, -1.0, 1.0))
    assert 0.0 < fixed_angle < adaptive_angle < np.pi / 2.0


def test_rotation_filter_stays_on_so3_over_long_motion_sequence():
    filter_ = OneEuroRotationFilter(validate_input=False)
    for index in range(2000):
        axis = np.array([1.0, 0.5 + 0.1 * np.sin(index), -0.25])
        result = filter_.update(index / 120.0, rot(axis, 0.8 * np.sin(index / 31.0)))
        assert is_rotation_matrix(result, tol=1e-6)


def test_rotation_filter_validates_input_time_and_reset():
    filter_ = OneEuroRotationFilter()
    filter_.update(1.0, np.eye(3))
    with pytest.raises(ValueError, match=r"SO\(3\)"):
        filter_.update(2.0, np.ones((3, 3)))
    with pytest.raises(ValueError, match="increase strictly"):
        filter_.update(1.0, np.eye(3))
    filter_.reset()
    assert filter_.update(0.0, np.eye(3)) == pytest.approx(np.eye(3))


def test_rotation_validation_can_be_disabled_for_trusted_fk():
    filter_ = OneEuroRotationFilter(validate_input=False)
    result = filter_.update(0.0, np.eye(3))
    assert is_rotation_matrix(result)
    pose_filter = BimanualPoseFilter(validate_rotations=False)
    assert pose_filter.update(0.0, _pose()).points() == pytest.approx(_pose().points())


def test_bimanual_pose_filter_preserves_structure_and_orientation():
    filter_ = BimanualPoseFilter(min_cutoff=1.0, beta=0.0)
    initial = filter_.update(0.0, _pose())
    result = filter_.update(0.01, _pose(offset=1.0, angle=np.pi / 2.0))
    assert initial.points() == pytest.approx(_pose().points())
    assert np.all(
        (result.points()[:, 0] > initial.points()[:, 0])
        & (result.points()[:, 0] < initial.points()[:, 0] + 1.0)
    )
    assert is_rotation_matrix(result.left.tool_orientation)
    assert is_rotation_matrix(result.right.tool_orientation)


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("initialized", [False, True])
def test_rejected_pose_leaves_all_filter_histories_unchanged(side, initialized):
    filter_ = BimanualPoseFilter()
    reference = BimanualPoseFilter()
    if initialized:
        filter_.update(0.0, _pose())
        reference.update(0.0, _pose())
    target = _pose(offset=1.0, angle=0.5)
    invalid = replace(
        target, **{side: replace(getattr(target, side), tool_orientation=np.zeros((3, 3)))}
    )
    with pytest.raises(ValueError, match=r"SO\(3\)"):
        filter_.update(0.01, invalid)
    # Retrying at the rejected timestamp must match a stream without that frame.
    actual = filter_.update(0.01, target)
    expected = reference.update(0.01, target)
    np.testing.assert_allclose(actual.points(), expected.points())
    np.testing.assert_allclose(actual.left.tool_orientation, expected.left.tool_orientation)
    np.testing.assert_allclose(actual.right.tool_orientation, expected.right.tool_orientation)


def test_one_euro_preserves_small_time_steps():
    filter_ = OneEuroFilter(beta=0.0)
    filter_.update(0.0, 0.0)
    result = float(filter_.update(1e-18, 1.0))
    assert result > 0.0
    assert result / (2.0 * np.pi * 1e-18) == pytest.approx(1.0)


@pytest.mark.parametrize("rate", [72, 90, 120])
def test_pose_filter_reduces_stationary_vr_noise_with_irregular_sampling(rate):
    rng = np.random.default_rng(42)
    filter_ = BimanualPoseFilter(
        min_cutoff=1.0,
        beta=0.0,
        rotation_config=OneEuroConfig(min_cutoff=1.5, beta=0.0),
        tool_length=0.1,
    )
    raw_positions, filtered_positions, raw_angles, filtered_angles = [], [], [], []
    timestamp = 0.0
    for index in range(rate * 4):
        timestamp += rng.uniform(0.8, 1.2) / rate
        offset = rng.normal(0.0, 0.003)
        angle = rng.normal(0.0, np.deg2rad(0.8))
        result = filter_.update(timestamp, _pose(offset, angle))
        if index >= rate:  # Exclude the cold-start transient from RMS measurements.
            raw_positions.append(offset)
            filtered_positions.append(result.left.shoulder[0])
            raw_angles.append(angle)
            filtered_angles.append(
                np.arctan2(result.left.tool_orientation[1, 0], result.left.tool_orientation[0, 0])
            )
    for raw, filtered in ((raw_positions, filtered_positions), (raw_angles, filtered_angles)):
        assert np.linalg.norm(filtered) < 0.4 * np.linalg.norm(raw)


def test_pose_filter_tunes_rotation_without_changing_position_response():
    fixed = BimanualPoseFilter(min_cutoff=1.0, beta=0.0)
    adaptive_rotation = BimanualPoseFilter(
        min_cutoff=1.0,
        beta=0.0,
        rotation_config=OneEuroConfig(min_cutoff=1.0, beta=1.0),
    )
    for filter_ in (fixed, adaptive_rotation):
        filter_.update(0.0, _pose())
    target = _pose(1.0, np.pi / 2)
    first = fixed.update(0.01, target)
    second = adaptive_rotation.update(0.01, target)
    np.testing.assert_array_equal(first.points(), second.points())
    # Larger rotation beta follows a fast turn more closely with identical positions.
    assert np.linalg.norm(
        second.left.tool_orientation - target.left.tool_orientation
    ) < np.linalg.norm(first.left.tool_orientation - target.left.tool_orientation)


def test_rotation_filter_takes_short_arc_across_half_turn():
    filter_ = OneEuroRotationFilter(beta=0.0)
    axis = np.array([0.0, 0.0, 1.0])
    filter_.update(0.0, rot(axis, np.deg2rad(179)))
    result = filter_.update(0.01, rot(axis, np.deg2rad(-179)))
    angle = np.rad2deg(np.arctan2(result[1, 0], result[0, 0]))
    assert 179 < angle < 180
    assert is_rotation_matrix(result)


def test_virtual_tool_markers_follow_filtered_wrist_and_orientation():
    filter_ = BimanualPoseFilter(
        beta=0.0, tool_length=0.17, rotation_config=OneEuroConfig(min_cutoff=8.0, beta=0.0)
    )
    for timestamp, pose in ((0.0, _pose()), (0.01, _pose(0.1, 1.0))):
        result = filter_.update(timestamp, pose)
        for arm in (result.left, result.right):
            np.testing.assert_allclose(arm.tool - arm.wrist, 0.17 * arm.tool_orientation[:, 0])
            assert np.linalg.norm(arm.tool - arm.wrist) == pytest.approx(0.17)


@pytest.mark.parametrize("timestamp,value", [(1e-320, 1.0), (0.01, 1e308)])
def test_position_arithmetic_failure_does_not_poison_recovery(timestamp, value):
    filter_, reference = OneEuroFilter(beta=0.0), OneEuroFilter(beta=0.0)
    for item in (filter_, reference):
        item.update(0.0, 0.0)
    with pytest.raises(ValueError, match="finite range"):
        filter_.update(timestamp, value)
    np.testing.assert_array_equal(filter_.update(0.01, 0.1), reference.update(0.01, 0.1))


def test_rotation_arithmetic_failure_does_not_poison_recovery():
    filter_, reference = OneEuroRotationFilter(), OneEuroRotationFilter()
    for item in (filter_, reference):
        item.update(0.0, np.eye(3))
    target = _pose(angle=1.0).left.tool_orientation
    with pytest.raises(ValueError, match="finite range"):
        filter_.update(1e-320, target)
    np.testing.assert_array_equal(filter_.update(0.01, target), reference.update(0.01, target))


@pytest.mark.parametrize("side", ["left", "right"])
def test_rotation_arithmetic_failure_does_not_advance_any_pose_history(side):
    filter_, reference = BimanualPoseFilter(), BimanualPoseFilter()
    initial = _pose()
    for item in (filter_, reference):
        item.update(0.0, initial)
    target = replace(
        initial,
        **{
            side: replace(
                getattr(initial, side), tool_orientation=_pose(angle=1.0).left.tool_orientation
            )
        },
    )
    with pytest.raises(ValueError, match="finite range"):
        filter_.update(1e-320, target)
    actual, expected = filter_.update(0.01, target), reference.update(0.01, target)
    np.testing.assert_array_equal(actual.points(), expected.points())
    np.testing.assert_array_equal(actual.left.tool_orientation, expected.left.tool_orientation)
    np.testing.assert_array_equal(actual.right.tool_orientation, expected.right.tool_orientation)


@pytest.mark.parametrize(
    "factory,initial,target",
    [
        (OneEuroFilter, 0.0, 1.0),
        (OneEuroRotationFilter, np.eye(3), _pose(angle=1.0).left.tool_orientation),
    ],
)
def test_finite_timestamps_cannot_overflow_filter_interval(factory, initial, target):
    filter_ = factory()
    filter_.update(-1e308, initial)
    with pytest.raises(ValueError, match="finite interval"):
        filter_.update(1e308, target)


@pytest.mark.parametrize("field", ["min_cutoff", "beta", "derivative_cutoff"])
@pytest.mark.parametrize("value", [np.nan, np.inf, -1.0, [1.0, 2.0], True])
def test_one_euro_config_rejects_invalid_scalar_parameters(field, value):
    with pytest.raises(ValueError, match=field):
        OneEuroConfig(**{field: value})

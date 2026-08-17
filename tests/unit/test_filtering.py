import numpy as np
import pytest

from sew_mimic import (
    ArmPose,
    BimanualPose,
    BimanualPoseFilter,
    JointRateLimiter,
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

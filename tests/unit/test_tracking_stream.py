from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import numpy as np
import pytest

from sew_mimic import OneEuroConfig, rot
from sew_mimic.tracking import (
    CalibrationRequired,
    FramePolicy,
    JointSample,
    LatestFrameBuffer,
    TrackingCalibration,
    TrackingFrame,
    TrackingIdentity,
    TrackingPoseStream,
    TrackingUnavailable,
)


@pytest.fixture
def frame():
    joints = {}
    for side, y in (("left", 0.3), ("right", -0.3)):
        for name, position in (
            ("shoulder", [0, y, 1.4]),
            ("elbow", [0.2, y, 1.1]),
            ("wrist", [0.5, y, 1.2]),
        ):
            joints[f"{side}_{name}"] = JointSample(position, np.eye(3), True, True)
    return TrackingFrame(TrackingIdentity("tracker", "session-1", "local"), 1, 10, 10.01, joints)


def _advance(frame, *, sequence=2, sample_time=10.02, **changes):
    return replace(
        frame,
        sequence=sequence,
        sample_time=sample_time,
        received_time=sample_time + 0.001,
        **changes,
    )


def test_stream_checks_freshness_on_duplicate_ticks_without_retransforming(frame, monkeypatch):
    calibration = TrackingCalibration(frame.identity)
    stream = TrackingPoseStream(calibration)
    with pytest.raises(TrackingUnavailable, match="No frame"):
        stream.poll(now=10.01)
    assert stream.publish(frame)
    expected = calibration.to_pose(frame, now=10.02)
    np.testing.assert_array_equal(stream.poll(now=10.02).points(), expected.points())

    def unexpected_transform(*args):
        raise AssertionError("duplicate ticks must not transform or refilter an old pose")

    monkeypatch.setattr(TrackingCalibration, "_transform_frame", unexpected_transform)
    assert stream.poll(now=10.03) is None
    with pytest.raises(TrackingUnavailable, match="stale"):
        stream.poll(now=10.2)


def test_loss_recovery_between_control_ticks_is_reported_and_resets_smoothing(frame):
    stream = TrackingPoseStream(TrackingCalibration(frame.identity), min_cutoff=0.1)
    stream.publish(frame)
    stream.poll(now=10.01)
    lost = _advance(frame, active=False, joints={})
    joints = {
        name: replace(joint, position=joint.position + [1, 0, 0])
        for name, joint in frame.joints.items()
    }
    recovered = _advance(frame, sequence=3, sample_time=10.03, joints=joints)
    # A producer delivers both snapshots before the consumer's next tick.
    with ThreadPoolExecutor(max_workers=1) as worker:
        assert worker.submit(stream.publish, lost).result(timeout=5)
        assert worker.submit(stream.publish, recovered).result(timeout=5)
    with pytest.raises(TrackingUnavailable, match="inactive"):
        stream.poll(now=10.04)
    actual = stream.poll(now=10.05)
    expected = TrackingCalibration(frame.identity).to_pose(recovered, now=10.05)
    np.testing.assert_array_equal(actual.points(), expected.points())


@pytest.mark.parametrize("fault", ["missing", "quality", "untracked", "degenerate"])
def test_transient_unusable_body_data_is_not_hidden_by_newer_valid_frame(frame, fault):
    stream = TrackingPoseStream(
        TrackingCalibration(frame.identity),
        policy=FramePolicy(min_confidence=0.5),
        require_tracked=True,
    )
    frame = replace(frame, confidence=0.8)
    stream.publish(frame)
    stream.poll(now=10.01)
    joints = dict(frame.joints)
    if fault == "missing":
        del joints["left_elbow"]
    elif fault == "untracked":
        joints["left_wrist"] = replace(joints["left_wrist"], orientation_tracked=False)
    elif fault == "degenerate":
        joints["left_elbow"] = replace(
            joints["left_elbow"], position=joints["left_shoulder"].position
        )
    bad = _advance(frame, joints=joints, confidence=None if fault == "quality" else 0.8)
    assert stream.publish(bad)
    assert stream.publish(_advance(frame, sequence=3, sample_time=10.03))
    with pytest.raises(TrackingUnavailable):
        stream.poll(now=10.04)
    assert stream.poll(now=10.05) is not None


def test_decoder_or_solver_fault_cannot_retry_the_same_sample(frame):
    stream = TrackingPoseStream(TrackingCalibration(frame.identity))
    stream.publish(frame)
    stream.poll(now=10.01)
    stream.invalidate("decoder failed")
    stream.invalidate("another failure")
    with pytest.raises(TrackingUnavailable, match="decoder failed"):
        stream.poll(now=10.02)
    with pytest.raises(TrackingUnavailable, match="newer tracking sample"):
        stream.poll(now=10.03)
    # A new packet ID is insufficient if it still contains the old sensor sample.
    stream.publish(replace(frame, sequence=2, received_time=10.031))
    with pytest.raises(TrackingUnavailable, match="newer tracking sample"):
        stream.poll(now=10.04)
    stream.publish(_advance(frame, sequence=3, sample_time=10.05))
    assert stream.poll(now=10.052) is not None


def test_timeout_recovery_starts_with_fresh_history(frame):
    stream = TrackingPoseStream(TrackingCalibration(frame.identity), min_cutoff=0.1)
    stream.publish(frame)
    stream.poll(now=10.01)
    with pytest.raises(TrackingUnavailable, match="stale"):
        stream.poll(now=10.2)
    fresh = _advance(frame, sample_time=10.21)
    stream.publish(fresh)
    expected = TrackingCalibration(frame.identity).to_pose(fresh, now=10.22)
    np.testing.assert_array_equal(stream.poll(now=10.22).points(), expected.points())


def test_same_connection_recalibration_preserves_ordering_and_requires_new_sample(frame):
    calibration = TrackingCalibration(frame.identity)
    stream = TrackingPoseStream(calibration)
    stream.publish(frame)
    stream.poll(now=10.01)
    changed = replace(calibration, translation=[1, 0, 0])
    stream.set_calibration(changed)
    assert not stream.publish(frame)
    with pytest.raises(TrackingUnavailable, match="No frame"):
        stream.poll(now=10.02)
    # Another recalibration must not erase the ordering retained across clear().
    stream.set_calibration(changed)
    assert not stream.publish(frame)
    assert stream.publish(replace(frame, sequence=2, received_time=10.021))
    with pytest.raises(TrackingUnavailable, match="newer tracking sample"):
        stream.poll(now=10.03)
    fresh = _advance(frame, sequence=3, sample_time=10.04)
    stream.publish(fresh)
    expected = changed.to_pose(fresh, now=10.05)
    np.testing.assert_array_equal(stream.poll(now=10.05).points(), expected.points())


def test_reference_change_stays_latched_until_current_calibration_is_applied(frame):
    calibration = TrackingCalibration(frame.identity)
    stream = TrackingPoseStream(calibration)
    stream.publish(frame)
    stream.poll(now=10.01)
    identity = replace(frame.identity, space_revision=1)
    changed = _advance(frame, identity=identity)
    assert stream.publish(changed)
    for now in (10.03, 10.04):
        with pytest.raises(CalibrationRequired):
            stream.poll(now=now)
    with pytest.raises(CalibrationRequired, match="obsolete"):
        stream.set_calibration(calibration)
    stream.set_calibration(replace(calibration, identity=identity))
    assert not stream.publish(changed)
    assert not stream.publish(_advance(frame, sequence=3, sample_time=10.045))
    assert stream.publish(_advance(changed, sequence=4, sample_time=10.05))
    assert stream.poll(now=10.06) is not None


@pytest.mark.parametrize("field", ["source_id", "session_id"])
def test_only_explicit_session_selection_restarts_ordering(frame, field):
    stream = TrackingPoseStream(TrackingCalibration(frame.identity))
    stream.publish(frame)
    stream.poll(now=10.01)
    identity = replace(frame.identity, **{field: "new"})
    new = replace(frame, identity=identity, sequence=0)
    assert not stream.publish(new)
    stream.set_calibration(TrackingCalibration(identity))
    assert not stream.publish(_advance(frame))
    assert stream.publish(new)
    assert stream.poll(now=10.02) is not None


def test_rejected_old_fault_packet_cannot_invalidate_the_current_pose(frame):
    stream = TrackingPoseStream(TrackingCalibration(frame.identity))
    stream.publish(frame)
    stream.poll(now=10.01)
    assert not stream.publish(replace(frame, sequence=0, active=False, joints={}))
    assert stream.poll(now=10.02) is None


@pytest.mark.parametrize("now", [np.nan, np.inf, 9.0])
def test_invalid_control_clock_does_not_consume_a_pending_fault(frame, now):
    stream = TrackingPoseStream(TrackingCalibration(frame.identity))
    stream.publish(frame)
    stream.poll(now=10.01)
    stream.invalidate("pending fault")
    with pytest.raises(ValueError):
        stream.poll(now=now)
    with pytest.raises(TrackingUnavailable, match="pending fault"):
        stream.poll(now=10.02)


def test_buffer_clear_retains_sequence_space_and_time_watermarks(frame):
    buffer = LatestFrameBuffer(frame.identity.source_id, frame.identity.session_id)
    assert buffer.publish(frame)
    buffer.clear()
    assert buffer.latest() is None
    assert not buffer.publish(frame)
    assert not buffer.publish(_advance(frame, sample_time=9.0))
    assert buffer.publish(_advance(frame))
    changed = _advance(
        frame, sequence=3, sample_time=10.03, identity=replace(frame.identity, space_revision=1)
    )
    assert buffer.publish(changed)
    buffer.clear()
    assert not buffer.publish(_advance(frame, sequence=4, sample_time=10.04))
    assert buffer.publish(_advance(changed, sequence=5, sample_time=10.05))


def test_oversized_calibration_transform_is_rejected_before_emitting_pose(frame):
    calibration = TrackingCalibration(
        frame.identity, translation=[0, 0, 1.7e308], tool_length=1.7e308
    )
    # Align virtual tool X with world Z so adding its marker overflows.
    calibration = replace(
        calibration, left_hand_to_tool=np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
    )
    with pytest.raises(TrackingUnavailable, match="Calibrated pose"):
        calibration.to_pose(frame, now=10.02)


@pytest.mark.parametrize("active", [True, False])
@pytest.mark.parametrize("initialized", [True, False])
def test_bad_mapped_timestamp_reports_fault_without_blocking_clock_recovery(
    frame, active, initialized
):
    calibration = TrackingCalibration(frame.identity)
    stream = TrackingPoseStream(calibration)
    if initialized:
        stream.publish(frame)
        stream.poll(now=10.01)
    bad = replace(frame, sequence=2, sample_time=1000.0, received_time=10.02, active=active)
    assert stream.publish(bad)
    recovered = _advance(frame, sequence=3, sample_time=10.03)
    assert stream.publish(recovered)
    with pytest.raises(TrackingUnavailable):
        stream.poll(now=10.04)
    np.testing.assert_array_equal(
        stream.poll(now=10.05).points(), calibration.to_pose(recovered, now=10.05).points()
    )
    assert not stream.publish(bad)
    assert stream.poll(now=10.06) is None


def test_bad_clock_payload_cannot_become_usable_when_time_catches_up(frame):
    stream = TrackingPoseStream(
        TrackingCalibration(frame.identity), policy=FramePolicy(max_receive_age=2000)
    )
    assert stream.publish(replace(frame, sample_time=1000))
    with pytest.raises(TrackingUnavailable, match="clock/prediction"):
        stream.poll(now=10.02)
    with pytest.raises(TrackingUnavailable, match="No frame"):
        stream.poll(now=1000)


def test_clock_fault_retains_newest_valid_sample_boundary_before_poll(frame):
    stream = TrackingPoseStream(TrackingCalibration(frame.identity))
    stream.publish(frame)
    assert stream.publish(replace(frame, sequence=2, sample_time=1000))
    with pytest.raises(TrackingUnavailable):
        stream.poll(now=10.02)
    assert stream.publish(replace(frame, sequence=3, received_time=10.03))
    with pytest.raises(TrackingUnavailable, match="newer tracking sample"):
        stream.poll(now=10.04)
    assert stream.publish(_advance(frame, sequence=4, sample_time=10.05))
    assert stream.poll(now=10.06) is not None


def test_clock_fault_does_not_hide_recenter_or_poison_recalibration(frame):
    stream = TrackingPoseStream(TrackingCalibration(frame.identity))
    stream.publish(frame)
    stream.poll(now=10.01)
    identity = replace(frame.identity, space_revision=1)
    assert stream.publish(replace(frame, identity=identity, sequence=2, sample_time=1000))
    with pytest.raises(CalibrationRequired):
        stream.poll(now=10.02)
    with pytest.raises(CalibrationRequired, match="obsolete"):
        stream.set_calibration(TrackingCalibration(frame.identity))
    stream.set_calibration(TrackingCalibration(identity))
    assert stream.publish(_advance(frame, sequence=3, identity=identity))
    assert stream.poll(now=10.03) is not None


def test_allowed_prediction_still_advances_sample_ordering(frame):
    stream = TrackingPoseStream(
        TrackingCalibration(frame.identity), policy=FramePolicy(max_prediction=0.04)
    )
    assert stream.publish(replace(frame, sample_time=10.03))
    assert stream.poll(now=10.02) is not None
    assert not stream.publish(_advance(frame, sample_time=10.025))
    assert stream.poll(now=10.03) is None


def test_stream_applies_rotation_tuning_and_new_calibrated_tool_length(frame):
    calibration = TrackingCalibration(frame.identity)
    stream = TrackingPoseStream(calibration, beta=0.0, rotation_config=OneEuroConfig(8.0, 0.0))
    stream.publish(frame)
    stream.poll(now=10.01)
    joints = {
        name: replace(joint, orientation=rot(np.array([0.0, 0.0, 1.0]), 1.0))
        for name, joint in frame.joints.items()
    }
    rotated = _advance(frame, joints=joints)
    assert stream.publish(rotated)
    result = stream.poll(now=10.03)
    # A fixed 8 Hz rotation filter has an analytic step response at dt = 20 ms.
    expected_angle = -np.expm1(-2 * np.pi * 8 * 0.02)
    np.testing.assert_allclose(
        result.left.tool_orientation, rot(np.array([0.0, 0.0, 1.0]), expected_angle)
    )
    for arm in (result.left, result.right):
        np.testing.assert_allclose(arm.tool - arm.wrist, 0.1 * arm.tool_orientation[:, 0])
    stream.set_calibration(replace(calibration, tool_length=0.23))
    stream.publish(_advance(rotated, sequence=3, sample_time=10.04))
    result = stream.poll(now=10.05)
    for arm in (result.left, result.right):
        np.testing.assert_allclose(arm.tool - arm.wrist, 0.23 * arm.tool_orientation[:, 0])

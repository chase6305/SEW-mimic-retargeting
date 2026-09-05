from dataclasses import replace

import numpy as np
import pytest

from sew_mimic import BimanualPoseFilter, rot
from sew_mimic.tracking import (
    JointSample,
    TrackingCalibration,
    TrackingFrame,
    TrackingIdentity,
    TrackingMotionLimits,
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
    return TrackingFrame(TrackingIdentity("tracker", "session", "local"), 1, 10, 10.001, joints)


def _next(frame, *, sequence=2, dt=0.02, key="left_wrist", delta=None, angle=None):
    joints = dict(frame.joints)
    joint = joints[key]
    if delta is not None:
        joint = replace(joint, position=joint.position + np.asarray(delta))
    if angle is not None:
        joint = replace(joint, orientation=rot(np.array([0.0, 0.0, 1.0]), angle))
    joints[key] = joint
    return replace(
        frame,
        sequence=sequence,
        sample_time=frame.sample_time + dt,
        received_time=frame.sample_time + dt + 0.001,
        joints=joints,
    )


def _stream(frame, **limits):
    stream = TrackingPoseStream(
        TrackingCalibration(frame.identity), motion_limits=TrackingMotionLimits(**limits), beta=5.0
    )
    assert stream.publish(frame)
    assert stream.poll(now=frame.received_time) is not None
    return stream


@pytest.mark.parametrize(
    "key",
    [f"{side}_{name}" for side in ("left", "right") for name in ("shoulder", "elbow", "wrist")],
)
def test_position_spike_between_ticks_never_reaches_filter_and_recovery_is_cold(
    frame, key, monkeypatch
):
    stream = _stream(frame, max_joint_speed=1.0)
    filtered_samples = []
    original = BimanualPoseFilter.update

    def record_update(self, timestamp, pose):
        filtered_samples.append(timestamp)
        return original(self, timestamp, pose)

    monkeypatch.setattr(BimanualPoseFilter, "update", record_update)
    bad = _next(frame, key=key, delta=[1.0, 0, 0])
    good = _next(frame, sequence=3, dt=0.03, key=key, delta=[0.01, 0, 0])
    assert stream.publish(bad)
    assert stream.publish(good)
    with pytest.raises(TrackingUnavailable, match=f"position jump at {key}"):
        stream.poll(now=10.04)
    result = stream.poll(now=10.05)
    expected = TrackingCalibration(frame.identity).to_pose(good, now=10.05)
    np.testing.assert_array_equal(result.points(), expected.points())
    assert filtered_samples == [good.sample_time]


@pytest.mark.parametrize("key", ["left_wrist", "right_wrist"])
@pytest.mark.parametrize("angle,limit", [(np.pi, 2.0), (1e-9, 1e-8)])
def test_rotation_spikes_are_checked_including_half_turns_and_tiny_angles(frame, key, angle, limit):
    stream = _stream(frame, max_wrist_angular_speed=limit)
    assert stream.publish(_next(frame, key=key, angle=angle))
    with pytest.raises(TrackingUnavailable, match=f"rotation jump at {key}"):
        stream.poll(now=10.03)
    assert stream.publish(_next(frame, sequence=3, dt=0.04))
    assert stream.poll(now=10.05) is not None


def test_rotation_motion_check_follows_short_arc_across_half_turn(frame):
    initial = _next(frame, sequence=1, dt=0, angle=np.deg2rad(179))
    stream = _stream(initial, max_wrist_angular_speed=2.0)
    assert stream.publish(_next(initial, angle=np.deg2rad(-179)))
    assert stream.poll(now=10.03) is not None


@pytest.mark.parametrize("rate", [72, 90, 120])
def test_valid_motion_uses_actual_sample_interval(frame, rate):
    stream = _stream(frame, max_joint_speed=1.0, max_wrist_angular_speed=2.0)
    for index in range(1, 10):
        candidate = _next(
            frame,
            sequence=index + 1,
            dt=index / rate,
            delta=[0.9 * index / rate, 0, 0],
            angle=1.8 * index / rate,
        )
        assert stream.publish(candidate)
        assert stream.poll(now=candidate.received_time) is not None


def test_arrival_delay_cannot_relax_sample_speed_limit(frame):
    stream = _stream(frame, max_joint_speed=1.0)
    delayed = replace(_next(frame, dt=0.01, delta=[0.05, 0, 0]), received_time=10.09)
    assert stream.publish(delayed)
    with pytest.raises(TrackingUnavailable, match="position jump"):
        stream.poll(now=10.09)


def test_position_limit_uses_vector_speed_instead_of_separate_axis_bounds(frame):
    stream = _stream(frame, max_joint_speed=1.0)
    assert stream.publish(_next(frame, dt=0.01, delta=[0.008, 0.008, 0.008]))
    with pytest.raises(TrackingUnavailable, match="position jump"):
        stream.poll(now=10.02)


@pytest.mark.parametrize("change", [{"delta": [0.001, 0, 0]}, {"angle": 0.001}])
def test_same_timestamp_payload_cannot_move_monitored_components(frame, change):
    stream = _stream(frame, max_joint_speed=1.0, max_wrist_angular_speed=2.0)
    assert stream.publish(_next(frame, dt=0.0))
    assert stream.poll(now=10.002) is None
    assert stream.publish(_next(frame, sequence=3, dt=0.0, **change))
    with pytest.raises(TrackingUnavailable, match="jump"):
        stream.poll(now=10.003)


def test_outliers_never_replace_reference_or_gain_unbounded_recovery_time(frame):
    stream = _stream(frame, max_joint_speed=1.0, max_sample_gap=0.05)
    for sequence, dt, delta, message in (
        (2, 0.02, 0.2, "position jump"),
        (3, 0.04, 0.205, "position jump"),
        (4, 0.06, 0.205, "reference expired"),
        (5, 0.07, 0.0, "reference expired"),
    ):
        assert stream.publish(_next(frame, sequence=sequence, dt=dt, delta=[delta, 0, 0]))
        with pytest.raises(TrackingUnavailable, match=message):
            stream.poll(now=10 + dt + 0.001)
    # Re-anchoring is explicit and still requires a newer observation.
    stream.set_calibration(TrackingCalibration(frame.identity))
    assert stream.publish(_next(frame, sequence=6, dt=0.07, delta=[0.5, 0, 0]))
    with pytest.raises(TrackingUnavailable, match="newer tracking sample"):
        stream.poll(now=10.072)
    assert stream.publish(_next(frame, sequence=7, dt=0.08, delta=[1.0, 0, 0]))
    assert stream.poll(now=10.09) is not None


@pytest.mark.parametrize("fault", ["inactive", "decoder", "timeout", "clock"])
def test_faults_do_not_silently_disable_motion_check_on_recovery(frame, fault):
    stream = _stream(frame, max_joint_speed=1.0, max_sample_gap=0.5)
    if fault == "inactive":
        stream.publish(replace(_next(frame), active=False, joints={}))
    elif fault == "decoder":
        stream.invalidate("decoder error")
    elif fault == "clock":
        stream.publish(replace(_next(frame), sample_time=1000))
    with pytest.raises(TrackingUnavailable):
        stream.poll(now=10.2)
    assert stream.publish(_next(frame, sequence=3, dt=0.25, delta=[1, 0, 0]))
    with pytest.raises(TrackingUnavailable, match="position jump"):
        stream.poll(now=10.26)
    assert stream.publish(_next(frame, sequence=4, dt=0.27))
    assert stream.poll(now=10.28) is not None


def test_foreign_and_reordered_motion_cannot_invalidate_current_target(frame):
    stream = _stream(frame, max_joint_speed=1.0)
    bad = _next(frame, delta=[1, 0, 0])
    assert not stream.publish(replace(bad, sequence=0))
    assert not stream.publish(replace(bad, identity=replace(frame.identity, session_id="old")))
    assert stream.poll(now=10.03) is None


@pytest.mark.parametrize("name", ["max_joint_speed", "max_wrist_angular_speed", "max_sample_gap"])
@pytest.mark.parametrize("value", [0, -1, np.nan, np.inf, True, [1.0], "1.0"])
def test_motion_limits_require_finite_positive_scalars(name, value):
    arguments = {"max_joint_speed": 1.0, name: value}
    with pytest.raises(ValueError, match=name):
        TrackingMotionLimits(**arguments)


def test_motion_limits_require_at_least_one_speed_limit():
    with pytest.raises(ValueError, match="At least one"):
        TrackingMotionLimits()

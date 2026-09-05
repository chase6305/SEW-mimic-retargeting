from dataclasses import replace

import numpy as np
import pytest

from sew_mimic import rot
from sew_mimic.tracking import (
    FB_BODY_JOINT_MAP,
    OPENXR_TO_FLU,
    UNITY_TO_FLU,
    CalibrationRequired,
    FramePolicy,
    JointSample,
    LatestFrameBuffer,
    LocationFlags,
    OpenXRAdapter,
    OpenXRJoint,
    TrackingCalibration,
    TrackingFrame,
    TrackingIdentity,
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
    return TrackingFrame(TrackingIdentity("quest", "session-1", "local"), 1, 10, 10.01, joints)


def test_tracking_snapshots_own_input_arrays_and_mapping(frame):
    position = np.array([1.0, 2.0, 3.0])
    orientation = np.eye(3)
    joint = JointSample(position, orientation)
    joints = {"left_wrist": joint}
    snapshot = replace(frame, joints=joints)
    position[:] = 0
    orientation[:] = 0
    joints.clear()
    np.testing.assert_array_equal(snapshot.joints["left_wrist"].position, [1, 2, 3])
    np.testing.assert_array_equal(snapshot.joints["left_wrist"].orientation, np.eye(3))
    with pytest.raises(ValueError):
        snapshot.joints["left_wrist"].position[0] = 5
    with pytest.raises(TypeError):
        snapshot.joints["new"] = joint


@pytest.mark.parametrize("basis", [OPENXR_TO_FLU, UNITY_TO_FLU])
def test_coordinate_conversion_preserves_physical_rotation_even_for_reflection(frame, basis):
    adapter = OpenXRAdapter({"left_wrist": "wrist"}, basis=basis, metres_per_unit=0.001)
    sample = OpenXRJoint([0, 1000, 0], [0, 0, 2**-0.5, 2**-0.5], 15)
    result = adapter.decode(
        {"wrist": sample},
        identity=frame.identity,
        sequence=2,
        sample_time=10,
        received_time=10.01,
    ).joints["left_wrist"]
    np.testing.assert_allclose(result.position, [0, 0, 1])
    # Raw +X rotates to raw +Y: canonical left rotates to canonical up.
    np.testing.assert_allclose(result.orientation @ [0, -1, 0], [0, 0, 1], atol=1e-14)
    np.testing.assert_allclose(result.orientation.T @ result.orientation, np.eye(3), atol=1e-14)
    assert np.linalg.det(result.orientation) == pytest.approx(1)
    assert result.position_tracked and result.orientation_tracked


def test_openxr_forward_maps_to_canonical_forward(frame):
    result = OpenXRAdapter({"wrist": "wrist"}).decode(
        {"wrist": OpenXRJoint([0, 0, -1], None, LocationFlags.POSITION_VALID)},
        identity=frame.identity,
        sequence=2,
        sample_time=10,
        received_time=10.01,
    )
    np.testing.assert_array_equal(result.joints["wrist"].position, [1, 0, 0])
    assert result.joints["wrist"].orientation is None
    assert not result.joints["wrist"].position_tracked


def test_decoder_does_not_read_invalid_sdk_components(frame):
    result = OpenXRAdapter({"left_wrist": "wrist", "left_elbow": "elbow"}).decode(
        {"wrist": OpenXRJoint([np.nan] * 3, [np.nan] * 4, 12)},
        identity=frame.identity,
        sequence=2,
        sample_time=10,
        received_time=10.01,
    )
    assert result.joints["left_wrist"] == JointSample()
    assert "left_elbow" not in result.joints
    assert result.confidence is None


def test_inactive_body_does_not_read_stale_valid_bits_or_confidence(frame):
    result = OpenXRAdapter({"wrist": "wrist"}).decode(
        {"wrist": OpenXRJoint([np.nan] * 3, [np.nan] * 4, 15)},
        identity=frame.identity,
        sequence=2,
        sample_time=10,
        received_time=10.01,
        active=False,
        confidence=np.nan,
    )
    assert not result.active
    assert not result.joints
    assert result.confidence is None


@pytest.mark.parametrize("quaternion", [[0, 0, 0, 0], [0, 0, np.nan, 1], [0, 0, 1]])
def test_decoder_rejects_malformed_valid_rotations(frame, quaternion):
    with pytest.raises(ValueError):
        OpenXRAdapter({"wrist": "wrist"}).decode(
            {"wrist": OpenXRJoint(None, quaternion, LocationFlags.ORIENTATION_VALID)},
            identity=frame.identity,
            sequence=2,
            sample_time=10,
            received_time=10.01,
        )


def test_meta_map_uses_arm_origins_and_hand_wrist():
    assert FB_BODY_JOINT_MAP["left_shoulder"] == "XR_BODY_JOINT_LEFT_ARM_UPPER_FB"
    assert FB_BODY_JOINT_MAP["right_elbow"] == "XR_BODY_JOINT_RIGHT_ARM_LOWER_FB"
    assert FB_BODY_JOINT_MAP["left_wrist"] == "XR_BODY_JOINT_LEFT_HAND_WRIST_FB"


def test_calibration_orders_world_and_local_rotations(frame):
    world_rotation = rot([0, 0, 1], np.pi / 2)
    wrist_rotation = rot([1, 0, 0], np.pi / 2)
    local_rotation = rot([0, 1, 0], np.pi / 2)
    joints = dict(frame.joints)
    joints["left_wrist"] = replace(joints["left_wrist"], orientation=wrist_rotation)
    frame = replace(frame, joints=joints)
    calibration = TrackingCalibration(
        frame.identity,
        rotation=world_rotation,
        translation=[1, 2, 3],
        left_hand_to_tool=local_rotation,
    )
    pose = calibration.to_pose(frame, now=10.02)
    np.testing.assert_allclose(pose.left.wrist, [0.7, 2.5, 4.2])
    np.testing.assert_allclose(
        pose.left.tool_orientation, world_rotation @ wrist_rotation @ local_rotation
    )
    np.testing.assert_allclose(pose.right.tool_orientation, world_rotation)
    np.testing.assert_allclose(
        pose.left.tool - pose.left.wrist, 0.1 * pose.left.tool_orientation[:, 0], atol=1e-14
    )


@pytest.mark.parametrize("field", ["source_id", "session_id", "space_id", "space_revision"])
def test_reconnect_or_recenter_requires_new_calibration(frame, field):
    changed = replace(frame.identity, **{field: 1 if field == "space_revision" else "new"})
    with pytest.raises(CalibrationRequired):
        TrackingCalibration(frame.identity).to_pose(replace(frame, identity=changed), now=10.02)


def test_missing_elbow_is_not_invented_from_wrist(frame):
    joints = dict(frame.joints)
    del joints["left_elbow"]
    with pytest.raises(TrackingUnavailable, match="left_elbow"):
        TrackingCalibration(frame.identity).to_pose(replace(frame, joints=joints), now=10.02)


def test_inferred_and_tracked_are_separate_policies(frame):
    joints = dict(frame.joints)
    joints["left_elbow"] = replace(joints["left_elbow"], position_tracked=False)
    frame = replace(frame, joints=joints)
    calibration = TrackingCalibration(frame.identity)
    calibration.to_pose(frame, now=10.02)
    with pytest.raises(TrackingUnavailable, match="not currently tracked"):
        calibration.to_pose(frame, now=10.02, require_tracked=True)


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"sample_time": 9.0}, "stale"),
        ({"received_time": 9.0}, "stale"),
        ({"sample_time": 10.1}, "clock/prediction"),
        ({"received_time": 10.1}, "clock/prediction"),
        ({"active": False}, "inactive"),
    ],
)
def test_input_policy_rejects_unusable_frames(frame, changes, message):
    with pytest.raises(TrackingUnavailable, match=message):
        FramePolicy().check(replace(frame, **changes), now=10.02)


def test_future_prediction_must_be_explicit(frame):
    FramePolicy(max_prediction=0.02).check(replace(frame, sample_time=10.03), now=10.02)


@pytest.mark.parametrize("confidence", [None, 0.2])
def test_unknown_or_low_confidence_does_not_pass_required_threshold(frame, confidence):
    with pytest.raises(TrackingUnavailable, match="confidence"):
        FramePolicy(min_confidence=0.5).check(replace(frame, confidence=confidence), now=10.02)
    FramePolicy(min_confidence=0.5).check(replace(frame, confidence=0.8), now=10.02)


def test_latest_buffer_rejects_old_packets_and_foreign_sessions(frame):
    buffer = LatestFrameBuffer("quest", "session-1")
    assert buffer.latest() is None
    assert buffer.publish(frame)
    assert not buffer.publish(frame)
    assert not buffer.publish(replace(frame, sequence=0))
    assert not buffer.publish(replace(frame, sequence=2, sample_time=9.0))
    for field in ("source_id", "session_id"):
        foreign = replace(frame.identity, **{field: "foreign"})
        assert not buffer.publish(replace(frame, identity=foreign, sequence=2))
    assert buffer.latest() is frame
    # Loss must invalidate the previous good pose even if its sample time repeats.
    lost = replace(frame, sequence=2, active=False, joints={})
    assert buffer.publish(lost)
    with pytest.raises(TrackingUnavailable, match="inactive"):
        FramePolicy().check(buffer.latest(), now=10.02)


def test_buffer_latches_new_space_revision(frame):
    buffer = LatestFrameBuffer("quest", "session-1")
    assert buffer.publish(frame)
    changed = replace(frame.identity, space_id="stage")
    assert not buffer.publish(replace(frame, identity=changed, sequence=2))
    changed = replace(changed, space_revision=1)
    assert buffer.publish(replace(frame, identity=changed, sequence=3))
    assert not buffer.publish(replace(frame, sequence=4))


def test_old_pose_time_cannot_hide_new_loss_or_recenter_event(frame):
    buffer = LatestFrameBuffer("quest", "session-1")
    assert buffer.publish(frame)
    lost = replace(frame, sequence=2, sample_time=9.9, active=False)
    assert buffer.publish(lost)
    assert buffer.latest() is lost
    # Keep the time high-water mark across loss so old poses cannot reactivate it.
    assert not buffer.publish(replace(frame, sequence=3, sample_time=9.95))
    recentered = replace(
        frame, sequence=4, sample_time=9.99, identity=replace(frame.identity, space_revision=1)
    )
    assert buffer.publish(recentered)
    with pytest.raises(CalibrationRequired):
        TrackingCalibration(frame.identity).to_pose(buffer.latest(), now=10.02)


@pytest.mark.parametrize("sequence", [-1, 1.5, True])
def test_frame_rejects_invalid_sequence(frame, sequence):
    with pytest.raises(ValueError, match="sequence"):
        replace(frame, sequence=sequence)


@pytest.mark.parametrize("changes", [{"sample_time": np.nan}, {"confidence": 1.1}])
def test_frame_rejects_nonfinite_time_and_invalid_confidence(frame, changes):
    with pytest.raises(ValueError):
        replace(frame, **changes)


def test_calibration_and_adapter_reject_invalid_geometry(frame):
    with pytest.raises(ValueError, match="SO\\(3\\)"):
        TrackingCalibration(frame.identity, rotation=UNITY_TO_FLU)
    with pytest.raises(ValueError, match="orthogonal"):
        OpenXRAdapter(basis=2 * np.eye(3))
    with pytest.raises(ValueError, match="positive"):
        OpenXRAdapter(metres_per_unit=0)
    with pytest.raises(ValueError, match="must be valid"):
        JointSample(position_tracked=True)

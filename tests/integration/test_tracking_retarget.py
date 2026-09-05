import numpy as np
import pytest

from sew_mimic import BimanualPoseFilter, cpp_backend_available, minimum_capsule_distance
from sew_mimic.filtering import _rotation_quaternion
from sew_mimic.robots import create_robot_safety_filter, get_robot_adapter
from sew_mimic.tracking import (
    FB_BODY_JOINT_MAP,
    OPENXR_TO_FLU,
    LatestFrameBuffer,
    OpenXRAdapter,
    OpenXRJoint,
    TrackingCalibration,
    TrackingIdentity,
)


@pytest.mark.parametrize("robot", ["marvin", "openarm"])
@pytest.mark.parametrize("backend", ["python", "cpp"])
def test_openxr_snapshot_retargets_known_robot_pose(robot, backend):
    if backend == "cpp" and not cpp_backend_available():
        pytest.skip("native extension is not built")
    safety = create_robot_safety_filter(robot, backend=backend)
    profile = get_robot_adapter(robot).collision_profile
    left, right = profile.neutral_left, profile.neutral_right
    expected = safety.forward_kinematics(left, right)
    identity = TrackingIdentity("recorded-body", "test-session", "local")
    samples = {}
    bases = safety.kinematics.link_transforms({}, (safety.left.base_link, safety.right.base_link))
    for side in ("left", "right"):
        arm = getattr(expected, side)
        adapter = getattr(safety, side)
        q = left if side == "left" else right
        base = bases[adapter.base_link]
        # SEW matches model axes, which need not coincide with physical link offsets.
        shoulder = base[:3, 3]
        elbow = shoulder + 0.287 * (base[:3, :3] @ adapter.robot.axis_world(q, 3))
        wrist = elbow + 0.314 * (base[:3, :3] @ adapter.robot.axis_world(q, 5))
        rotation = OPENXR_TO_FLU.T @ arm.tool_orientation @ OPENXR_TO_FLU
        wxyz = _rotation_quaternion(rotation)
        for joint, position in zip(("shoulder", "elbow", "wrist"), (shoulder, elbow, wrist)):
            samples[FB_BODY_JOINT_MAP[f"{side}_{joint}"]] = OpenXRJoint(
                OPENXR_TO_FLU.T @ position, wxyz[[1, 2, 3, 0]], 15
            )
    frame = OpenXRAdapter().decode(
        samples, identity=identity, sequence=1, sample_time=1.0, received_time=1.001
    )
    buffer = LatestFrameBuffer(identity.source_id, identity.session_id)
    assert buffer.publish(frame)
    pose = TrackingCalibration(identity).to_pose(buffer.latest(), now=1.002)
    target = BimanualPoseFilter().update(frame.sample_time, pose)
    result = safety.retarget(target, left, right)
    assert result.safe
    actual = safety.forward_kinematics(result.q_left, result.q_right)
    for side in ("left", "right"):
        np.testing.assert_allclose(
            getattr(actual, side).tool_orientation,
            getattr(expected, side).tool_orientation,
            atol=1e-7,
        )
        for joint in ("shoulder", "elbow", "wrist"):
            np.testing.assert_allclose(
                getattr(getattr(actual, side), joint),
                getattr(getattr(expected, side), joint),
                atol=1e-7,
            )
    assert minimum_capsule_distance(actual.keypoints(), safety.config) >= (
        safety.config.minimum_distance - safety.config.tolerance
    )

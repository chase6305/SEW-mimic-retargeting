import io
from dataclasses import replace

import numpy as np
import pytest

from examples.demo_tracking_input import run_tracking_events, synthetic_tracking_events
from sew_mimic import cpp_backend_available, rot
from sew_mimic.robots import create_robot_safety_filter
from sew_mimic.tracking import (
    TrackingCalibration,
    TrackingFrame,
    TrackingRecorder,
    TrackingRecordingReader,
    fit_hand_tool_rotation,
    fit_rigid_transform,
)


@pytest.mark.parametrize("robot", ["marvin", "openarm"])
@pytest.mark.parametrize("backend", ["python", "cpp"])
def test_fitted_world_and_tool_calibration_preserves_replayed_robot_targets(robot, backend):
    if backend == "cpp" and not cpp_backend_available():
        pytest.skip("native extension is not built")
    safety = create_robot_safety_filter(robot, backend=backend)
    original = list(synthetic_tracking_events(safety, robot))
    original_calibration = original[0]
    identity = replace(original_calibration.identity, space_id="tracker-local")
    known_rotation = rot([1, 2, -1], 0.7)
    known_translation = np.array([0.3, -0.8, 0.2])
    offsets = {"left": rot([1, 0, 1], 0.4), "right": rot([0, 1, 0], -0.6)}

    # Reference correspondences represent physical landmarks, separate from body joints.
    robot_references = np.random.default_rng(504).uniform(-0.5, 0.5, (12, 3))
    tracking_references = (robot_references - known_translation) @ known_rotation
    world = fit_rigid_transform(tracking_references, robot_references)
    world.require_accuracy(max_rms_error=1e-12, max_point_error=1e-12)
    wrist_frames = [
        event for event in original if isinstance(event, TrackingFrame) and event.active
    ][:6]
    local_rotations = {}
    for side in ("left", "right"):
        tools = np.stack([frame.joints[f"{side}_wrist"].orientation for frame in wrist_frames])
        hands = known_rotation.T @ tools @ offsets[side].T
        fit = fit_hand_tool_rotation(hands, tools, tracking_to_robot_rotation=world.rotation)
        fit.require_accuracy(max_rms_angle=1e-12, max_angle=1e-12)
        local_rotations[side] = fit.rotation
    fitted = TrackingCalibration(
        identity,
        rotation=world.rotation,
        translation=world.translation,
        left_hand_to_tool=local_rotations["left"],
        right_hand_to_tool=local_rotations["right"],
    )

    stream = io.StringIO()
    with TrackingRecorder(stream, metadata={"robot": robot}) as recorder:
        recorder.write(fitted)
        for event in original[1:]:
            if isinstance(event, TrackingFrame):
                joints = {
                    name: replace(
                        joint,
                        position=known_rotation.T @ (joint.position - known_translation),
                        orientation=known_rotation.T
                        @ joint.orientation
                        @ offsets[name.split("_")[0]].T,
                    )
                    for name, joint in event.joints.items()
                }
                converted = replace(event, identity=identity, joints=joints)
                if event.active:
                    expected = original_calibration.to_pose(event, now=event.received_time)
                    actual = fitted.to_pose(converted, now=converted.received_time)
                    np.testing.assert_allclose(actual.points(), expected.points(), atol=1e-12)
                    for side in ("left", "right"):
                        np.testing.assert_allclose(
                            getattr(actual, side).tool_orientation,
                            getattr(expected, side).tool_orientation,
                            atol=1e-12,
                        )
                recorder.write(converted)
            else:
                recorder.write(event)
    replay = TrackingRecordingReader(io.StringIO(stream.getvalue()))
    actual = run_tracking_events(create_robot_safety_filter(robot, backend=backend), robot, replay)
    expected = run_tracking_events(safety, robot, original)
    assert actual.solved == expected.solved == 50
    assert actual.held == expected.held == 5
    assert actual.unchanged == expected.unchanged == 5
    np.testing.assert_allclose(actual.q_left, expected.q_left, atol=1e-10, rtol=0)
    np.testing.assert_allclose(actual.q_right, expected.q_right, atol=1e-10, rtol=0)

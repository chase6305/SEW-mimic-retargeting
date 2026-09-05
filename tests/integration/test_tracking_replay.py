import io
import json
from dataclasses import asdict, replace

import numpy as np
import pytest

from examples.demo_tracking_input import run_tracking_events, synthetic_tracking_events
from sew_mimic import SEWMimicError, cpp_backend_available
from sew_mimic.robots import create_robot_safety_filter
from sew_mimic.tracking import (
    TrackingMotionLimits,
    TrackingRecorder,
    TrackingRecordingReader,
    TrackingTick,
)


@pytest.mark.parametrize("robot", ["marvin", "openarm"])
@pytest.mark.parametrize("backend", ["python", "cpp"])
def test_recorded_control_ticks_reproduce_retargeting_and_timeout_holds(robot, backend):
    if backend == "cpp" and not cpp_backend_available():
        pytest.skip("native extension is not built")
    safety = create_robot_safety_filter(robot, backend=backend)
    stream = io.StringIO()
    with TrackingRecorder(stream, metadata={"robot": robot}) as recorder:

        def recording_source():
            for event in synthetic_tracking_events(safety, robot):
                recorder.write(event)
                yield event

        original = run_tracking_events(safety, robot, recording_source())
    reader = TrackingRecordingReader(io.StringIO(stream.getvalue()))
    replayed = run_tracking_events(
        create_robot_safety_filter(robot, backend=backend), robot, reader
    )
    assert original.solved == replayed.solved == 50
    # One inactive frame and four timed-out ticks during the packet gap.
    assert original.held == replayed.held == 5
    assert original.unchanged == replayed.unchanged == 5
    assert original.rejected_frames == replayed.rejected_frames == 0
    assert original.command_digest == replayed.command_digest
    np.testing.assert_array_equal(original.q_left, replayed.q_left)
    np.testing.assert_array_equal(original.q_right, replayed.q_right)


def test_replay_requires_new_calibration_after_reference_space_changes():
    safety = create_robot_safety_filter("marvin")
    source = synthetic_tracking_events(safety, "marvin")
    calibration, frame, tick = next(source), next(source), next(source)
    changed_identity = replace(frame.identity, space_revision=1)
    changed_frame = replace(
        frame,
        identity=changed_identity,
        sequence=frame.sequence + 1,
        sample_time=frame.sample_time + 0.01,
        received_time=frame.received_time + 0.01,
    )
    events = [
        calibration,
        frame,
        tick,
        changed_frame,
        TrackingTick(tick.timestamp + 0.01),  # New reference, old calibration: hold.
        replace(calibration, identity=changed_identity),
        TrackingTick(tick.timestamp + 0.011),  # No frame after recalibration yet: hold.
        replace(
            changed_frame,
            sequence=changed_frame.sequence + 1,
            sample_time=changed_frame.sample_time + 0.001,
            received_time=changed_frame.received_time + 0.001,
        ),
        TrackingTick(tick.timestamp + 0.012),
    ]
    stream = io.StringIO()
    with TrackingRecorder(stream) as recorder:
        for event in events:
            recorder.write(event)
    reader = TrackingRecordingReader(io.StringIO(stream.getvalue()))
    result = run_tracking_events(safety, "marvin", reader)
    assert result.solved == 2
    assert result.held == 2
    assert result.rejected_frames == 0


def test_replay_preserves_tracking_loss_that_occurs_between_control_ticks():
    safety = create_robot_safety_filter("marvin")
    source = synthetic_tracking_events(safety, "marvin")
    calibration, frame, tick = next(source), next(source), next(source)
    lost = replace(
        frame, sequence=1, sample_time=1.01, received_time=1.012, active=False, joints={}
    )
    recovered = replace(frame, sequence=2, sample_time=1.02, received_time=1.022)
    events = [calibration, frame, tick, lost, recovered, TrackingTick(1.023), TrackingTick(1.03)]
    recording = io.StringIO()
    with TrackingRecorder(recording) as recorder:
        for event in events:
            recorder.write(event)
    replay = TrackingRecordingReader(io.StringIO(recording.getvalue()))
    result = run_tracking_events(safety, "marvin", replay)
    assert result.solved == 2
    assert result.held == 1


@pytest.mark.parametrize("robot", ["marvin", "openarm"])
def test_replay_recovers_after_bad_mapped_sample_clock(robot):
    safety = create_robot_safety_filter(robot)
    source = synthetic_tracking_events(safety, robot)
    calibration, frame, tick = next(source), next(source), next(source)
    bad = replace(frame, sequence=1, sample_time=1000, received_time=1.012)
    recovered = replace(frame, sequence=2, sample_time=1.02, received_time=1.022)
    events = [calibration, frame, tick, bad, recovered, TrackingTick(1.023), TrackingTick(1.03)]
    original = run_tracking_events(safety, robot, events)
    recording = io.StringIO()
    with TrackingRecorder(recording) as recorder:
        for event in events:
            recorder.write(event)
    replay = TrackingRecordingReader(io.StringIO(recording.getvalue()))
    result = run_tracking_events(create_robot_safety_filter(robot), robot, replay)
    assert result.solved == original.solved == 2
    assert result.held == original.held == 1
    assert result.rejected_frames == original.rejected_frames == 0
    assert result.command_digest == original.command_digest


@pytest.mark.parametrize("robot", ["marvin", "openarm"])
@pytest.mark.parametrize("backend", ["python", "cpp"])
def test_recorded_motion_limits_keep_spikes_out_of_retargeting(robot, backend, monkeypatch):
    if backend == "cpp" and not cpp_backend_available():
        pytest.skip("native extension is not built")
    safety = create_robot_safety_filter(robot, backend=backend)
    source = synthetic_tracking_events(safety, robot)
    calibration, frame, tick = next(source), next(source), next(source)
    limits = TrackingMotionLimits(max_joint_speed=1.0, max_wrist_angular_speed=2.0)
    joints = dict(frame.joints)
    joints["left_wrist"] = replace(
        joints["left_wrist"], position=joints["left_wrist"].position + [1, 0, 0]
    )
    spike = replace(frame, sequence=1, sample_time=1.01, received_time=1.012, joints=joints)
    recovered = replace(frame, sequence=2, sample_time=1.02, received_time=1.022)
    events = [calibration, frame, tick, spike, TrackingTick(1.013), recovered, TrackingTick(1.023)]
    solver_targets = []
    original_solve = safety.retarget

    def capture_target(pose, q_left, q_right):
        solver_targets.append(pose)
        return original_solve(pose, q_left, q_right)

    monkeypatch.setattr(safety, "retarget", capture_target)
    original = run_tracking_events(safety, robot, events, motion_limits=limits)
    assert len(solver_targets) == 2
    for pose in solver_targets:
        np.testing.assert_allclose(pose.left.wrist, frame.joints["left_wrist"].position)

    recording = io.StringIO()
    with TrackingRecorder(
        recording, metadata={"robot": robot, "motion_limits": json.dumps(asdict(limits))}
    ) as recorder:
        for event in events:
            recorder.write(event)
    replay = TrackingRecordingReader(io.StringIO(recording.getvalue()))
    result = run_tracking_events(
        create_robot_safety_filter(robot, backend=backend),
        robot,
        replay,
        motion_limits=TrackingMotionLimits(**json.loads(replay.metadata["motion_limits"])),
    )
    assert result.solved == original.solved == 2
    assert result.held == original.held == 1
    assert result.rejected_frames == original.rejected_frames == 0
    assert result.command_digest == original.command_digest


def test_recalibration_cannot_replay_the_previous_packet():
    safety = create_robot_safety_filter("marvin")
    source = synthetic_tracking_events(safety, "marvin")
    calibration, frame, tick = next(source), next(source), next(source)
    result = run_tracking_events(
        safety, "marvin", [calibration, frame, tick, calibration, frame, TrackingTick(1.01)]
    )
    assert result.solved == 1
    assert result.held == 1
    assert result.rejected_frames == 1


@pytest.mark.parametrize("failure", ["exception", "unsafe"])
def test_failed_robot_solve_does_not_retry_unchanged_tracking_input(monkeypatch, failure):
    safety = create_robot_safety_filter("marvin")
    source = synthetic_tracking_events(safety, "marvin")
    calibration, frame, tick = next(source), next(source), next(source)
    calls = 0
    original = safety.retarget

    def failed_solve(pose, q_left, q_right):
        nonlocal calls
        calls += 1
        if failure == "exception":
            raise SEWMimicError("injected IK failure")
        return replace(original(pose, q_left, q_right), safe=False)

    monkeypatch.setattr(safety, "retarget", failed_solve)
    result = run_tracking_events(
        safety, "marvin", [calibration, frame, tick, TrackingTick(1.01), TrackingTick(1.02)]
    )
    assert calls == 1
    assert result.solved == 0
    assert result.held == 3

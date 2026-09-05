import io
from dataclasses import replace

import pytest

from examples.demo_tracking_input import run_tracking_events, synthetic_tracking_events
from sew_mimic import OneEuroConfig, cpp_backend_available
from sew_mimic.robots import create_robot_safety_filter
from sew_mimic.tracking import (
    FramePolicy,
    TrackingFrame,
    TrackingMotionLimits,
    TrackingRecorder,
    TrackingRecordingReader,
    TrackingStreamConfig,
)


@pytest.mark.parametrize("robot", ["marvin", "openarm"])
@pytest.mark.parametrize("backend", ["python", "cpp"])
def test_complete_nondefault_settings_reproduce_command_sequence(robot, backend):
    if backend == "cpp" and not cpp_backend_available():
        pytest.skip("native extension is not built")
    safety = create_robot_safety_filter(robot, backend=backend)
    events = [
        replace(event, confidence=0.9) if isinstance(event, TrackingFrame) else event
        for event in synthetic_tracking_events(safety, robot)
    ]
    config = TrackingStreamConfig(
        policy=FramePolicy(0.06, 0.05, 0.004, 0.5),
        position_filter=OneEuroConfig(4.0, 0.5, 2.0),
        rotation_filter=OneEuroConfig(8.0, 0.7, 3.0),
        require_tracked=True,
        motion_limits=TrackingMotionLimits(2.0, 6.0, 0.25),
    )
    original = run_tracking_events(safety, robot, events, config=config)
    recording = io.StringIO()
    with TrackingRecorder(recording, metadata={"robot": robot, **config.to_metadata()}) as recorder:
        for event in events:
            recorder.write(event)
    reader = TrackingRecordingReader(io.StringIO(recording.getvalue()))
    restored = TrackingStreamConfig.from_metadata(reader.metadata)
    replayed = run_tracking_events(
        create_robot_safety_filter(robot, backend=backend), robot, reader, config=restored
    )
    assert replayed.command_digest == original.command_digest
    assert replayed.solved == original.solved > 0
    # The shorter receive-age limit causes seven timeout holds plus tracking
    # loss. Changed filter targets can additionally trigger normal IK holds.
    assert replayed.held == original.held >= 8
    assert replayed.unchanged == original.unchanged == 2
    defaults = run_tracking_events(safety, robot, events)
    assert defaults.held == 5
    assert defaults.command_digest != original.command_digest


def test_offline_runner_rejects_ambiguous_configuration_before_consuming_events():
    def unexpected_events():
        raise AssertionError("configuration must be validated before consuming events")
        yield

    with pytest.raises(ValueError, match="config.motion_limits"):
        run_tracking_events(
            None,
            "marvin",
            unexpected_events(),
            config=TrackingStreamConfig(),
            motion_limits=TrackingMotionLimits(max_joint_speed=2),
        )

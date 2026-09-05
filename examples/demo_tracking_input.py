"""Record or replay device-neutral tracking and retargeting without hardware.

The source generates synthetic human-length SEW chains from model axes. Real
providers supply TrackingFrame objects through OpenXRAdapter or their own SDK
adapter. No robot driver or network connection is opened by this example.
"""

from __future__ import annotations

import argparse
import hashlib
from collections.abc import Iterable, Iterator
from contextlib import ExitStack
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from sew_mimic import SEWMimicError
from sew_mimic.robots import RobotSafetyFilter, create_robot_safety_filter, get_robot_adapter
from sew_mimic.tracking import (
    JointSample,
    RecordingError,
    TrackingCalibration,
    TrackingEvent,
    TrackingFrame,
    TrackingIdentity,
    TrackingMotionLimits,
    TrackingPoseStream,
    TrackingRecorder,
    TrackingRecordingReader,
    TrackingStreamConfig,
    TrackingTick,
    TrackingUnavailable,
)


def synthetic_tracking_events(safety: RobotSafetyFilter, robot: str) -> Iterator[TrackingEvent]:
    """Generate motion, tracking loss and a packet gap on a deterministic clock."""
    profile = get_robot_adapter(robot).collision_profile
    bases = safety.kinematics.link_transforms({}, (safety.left.base_link, safety.right.base_link))
    identity = TrackingIdentity("synthetic-body", "demo-session", "robot-root")
    # Synthetic poses already share the URDF root frame. Real inputs need fitted values.
    yield TrackingCalibration(identity)

    for sequence in range(60):
        sample_time = 1.0 + sequence / 60
        if 40 <= sequence < 49:
            # The controller keeps ticking while the input stream stalls.
            yield TrackingTick(sample_time + 0.003)
            continue
        reference_left = profile.neutral_left.copy()
        reference_right = profile.neutral_right.copy()
        reference_left[0] += 0.01 * np.sin(sequence / 60 * 2 * np.pi)
        reference_right[0] -= 0.01 * np.sin(sequence / 60 * 2 * np.pi)
        joints = {}
        for side, q in (("left", reference_left), ("right", reference_right)):
            arm = getattr(safety, side)
            base = bases[arm.base_link]
            shoulder = base[:3, 3]
            elbow = shoulder + 0.287 * (base[:3, :3] @ arm.robot.axis_world(q, 3))
            wrist = elbow + 0.314 * (base[:3, :3] @ arm.robot.axis_world(q, 5))
            orientation = base[:3, :3] @ arm.robot.tool_orientation(q)
            for name, position in zip(("shoulder", "elbow", "wrist"), (shoulder, elbow, wrist)):
                joints[f"{side}_{name}"] = JointSample(position, orientation, True, True)
        frame = TrackingFrame(identity, sequence, sample_time, sample_time + 0.002, joints)
        if sequence == 30:
            frame = replace(frame, active=False, joints={})
        yield frame
        yield TrackingTick(sample_time + 0.003)


@dataclass(frozen=True)
class TrackingRunResult:
    solved: int
    held: int
    unchanged: int
    rejected_frames: int
    q_left: np.ndarray
    q_right: np.ndarray
    command_digest: str


def run_tracking_events(
    safety: RobotSafetyFilter,
    robot: str,
    events: Iterable[TrackingEvent],
    *,
    motion_limits: TrackingMotionLimits | None = None,
    config: TrackingStreamConfig | None = None,
) -> TrackingRunResult:
    """Use one offline processing path for synthetic and recorded canonical input.

    Every control tick contributes to the digest, including holds and unchanged
    commands. No per-frame history is accumulated. This simulation uses its last
    command as feedback; a physical driver needs measured joint state instead.
    """
    if config is not None and motion_limits is not None:
        raise ValueError("Use config.motion_limits when supplying a tracking config")
    if config is None:
        config = TrackingStreamConfig(motion_limits=motion_limits)
    elif not isinstance(config, TrackingStreamConfig):
        raise ValueError("config must be TrackingStreamConfig")
    profile = get_robot_adapter(robot).collision_profile
    q_left, q_right = profile.neutral_left.copy(), profile.neutral_right.copy()
    poses = None
    solved = held = unchanged = rejected_frames = 0
    digest = hashlib.sha256()
    for event in events:
        if isinstance(event, TrackingCalibration):
            if poses is None:
                poses = TrackingPoseStream.from_config(event, config)
            else:
                poses.set_calibration(event)
            continue
        if poses is None:
            raise ValueError("a calibration event is required before frames or control ticks")
        if isinstance(event, TrackingFrame):
            if not poses.publish(event):
                rejected_frames += 1
            continue
        if not isinstance(event, TrackingTick):
            raise TypeError("unsupported tracking event")
        try:
            pose = poses.poll(now=event.timestamp)
            if pose is None:
                unchanged += 1
            else:
                result = safety.retarget(pose, q_left, q_right)
                if not result.safe:
                    poses.invalidate(f"Retarget rejected: {result.status.value}")
                    held += 1
                else:
                    q_left, q_right = result.q_left, result.q_right
                    solved += 1
        except TrackingUnavailable:
            held += 1
            # Simulation resumes on the next usable sample. Hardware must additionally
            # require explicit re-engagement, fresh robot feedback and a watchdog.
        except SEWMimicError as exc:
            poses.invalidate(f"Retarget failed: {exc}")
            held += 1
        digest.update(np.asarray([event.timestamp], dtype="<f8").tobytes())
        digest.update(np.asarray([q_left, q_right], dtype="<f8").tobytes())
    return TrackingRunResult(
        solved, held, unchanged, rejected_frames, q_left, q_right, digest.hexdigest()
    )


def _recorded_events(
    events: Iterable[TrackingEvent], recorder: TrackingRecorder
) -> Iterator[TrackingEvent]:
    for event in events:
        recorder.write(event)
        yield event


def _configured_stream(
    args: argparse.Namespace, metadata: dict[str, str] | None = None
) -> TrackingStreamConfig:
    if args.config is not None:
        with args.config.open(encoding="utf-8") as stream:
            config = TrackingStreamConfig.from_json(
                stream.read(TrackingStreamConfig.MAX_JSON_CHARS + 1)
            )
    else:
        config = TrackingStreamConfig.from_metadata({} if metadata is None else metadata)
    names = ("max_joint_speed", "max_wrist_angular_speed", "max_sample_gap")
    overrides = {
        name: value
        for name, value in zip(
            names, (args.max_joint_speed, args.max_wrist_angular_speed, args.max_motion_gap)
        )
        if value is not None
    }
    if args.no_motion_limits:
        if overrides:
            raise ValueError("--no-motion-limits cannot be combined with motion limit overrides")
        return replace(config, motion_limits=None)
    if not overrides:
        return config
    parameters = {} if config.motion_limits is None else asdict(config.motion_limits)
    parameters.update(overrides)
    return replace(config, motion_limits=TrackingMotionLimits(**parameters))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", choices=("marvin", "openarm"))
    parser.add_argument("--backend", choices=("python", "cpp"), default="python")
    parser.add_argument(
        "--config",
        type=Path,
        help="load complete tracking settings from JSON, overriding recorded settings",
    )
    parser.add_argument("--max-joint-speed", type=float, help="input joint speed limit in m/s")
    parser.add_argument(
        "--max-wrist-angular-speed", type=float, help="input wrist speed limit in rad/s"
    )
    parser.add_argument(
        "--max-motion-gap",
        type=float,
        help="motion reference gap in seconds (default 0.1 when enabled)",
    )
    parser.add_argument(
        "--no-motion-limits",
        action="store_true",
        help="explicitly disable recorded motion checks for comparison",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--record", type=Path, help="write a new canonical JSONL recording")
    mode.add_argument("--replay", type=Path, help="replay JSONL using its recorded control clock")
    args = parser.parse_args()
    try:
        with ExitStack() as stack:
            if args.replay:
                stream = stack.enter_context(args.replay.open(encoding="utf-8"))
                events = TrackingRecordingReader(stream)
                config = _configured_stream(args, events.metadata)
                recorded_robot = events.metadata.get("robot")
                if args.robot and recorded_robot and args.robot != recorded_robot:
                    raise ValueError("--robot does not match the recording's robot metadata")
                robot = args.robot or recorded_robot
                if robot not in ("marvin", "openarm"):
                    raise ValueError(
                        "select --robot when recording metadata has no supported robot"
                    )
                safety = create_robot_safety_filter(robot, backend=args.backend)
            else:
                config = _configured_stream(args)
                robot = args.robot or "marvin"
                safety = create_robot_safety_filter(robot, backend=args.backend)
                events = synthetic_tracking_events(safety, robot)
                if args.record:
                    stream = stack.enter_context(args.record.open("x", encoding="utf-8"))
                    recorder = stack.enter_context(
                        TrackingRecorder(
                            stream,
                            metadata={
                                "robot": robot,
                                "input": "synthetic",
                                **config.to_metadata(),
                            },
                        )
                    )
                    events = _recorded_events(events, recorder)
            result = run_tracking_events(safety, robot, events, config=config)
    except (RecordingError, ValueError, OSError) as exc:
        parser.exit(2, f"tracking demo: {exc}\n")

    print(
        f"robot={robot} backend={args.backend} solved={result.solved} held={result.held} "
        f"unchanged={result.unchanged} rejected_frames={result.rejected_frames}"
    )
    print(f"command_digest={result.command_digest}")
    print(f"motion_limits={None if config.motion_limits is None else asdict(config.motion_limits)}")
    print(f"tracking_config={config.to_json()}")
    print("Offline simulation only; no actuator commands were sent.")


if __name__ == "__main__":
    main()

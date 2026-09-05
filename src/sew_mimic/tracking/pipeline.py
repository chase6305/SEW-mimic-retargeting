"""Bounded input delivery, freshness checks, and explicit robot calibration."""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock

import numpy as np

from ..safety import ArmPose, BimanualPose
from ..utility import EPS
from .model import TrackingFrame, TrackingIdentity, _array, _rotation


class TrackingUnavailable(ValueError):
    """Input cannot currently supply a usable target; the caller should hold."""


class CalibrationRequired(TrackingUnavailable):
    """Source/session/reference space no longer matches the calibration."""


class LatestFrameBuffer:
    """One slot for one explicitly selected source/session, safe across threads.

    New invalid frames replace old valid frames too. Recreate the buffer when
    selecting another source or reconnecting; old packets cannot take it over.
    """

    def __init__(
        self, source_id: str, session_id: str, *, policy: FramePolicy | None = None
    ) -> None:
        identity = TrackingIdentity(source_id, session_id, "buffer")
        if policy is not None and not isinstance(policy, FramePolicy):
            raise ValueError("policy must be FramePolicy or None")
        self._policy = FramePolicy() if policy is None else policy
        self._source_id = identity.source_id
        self._session_id = identity.session_id
        self._latest: TrackingFrame | None = None
        self._last_sequence = -1
        self._last_identity: TrackingIdentity | None = None
        self._sample_high_water = -np.inf
        self._lock = Lock()

    def publish(self, frame: TrackingFrame) -> bool:
        """Reject duplicates, reordered data, old spaces and foreign sessions."""
        with self._lock:
            if (frame.identity.source_id, frame.identity.session_id) != (
                self._source_id,
                self._session_id,
            ):
                return False
            previous = self._last_identity
            if previous is not None:
                if frame.sequence <= self._last_sequence:
                    return False
                revision = frame.identity.space_revision
                old_revision = previous.space_revision
                if revision < old_revision or (
                    revision == old_revision and frame.identity.space_id != previous.space_id
                ):
                    return False
                # Loss and recenter events must invalidate old data immediately,
                # even if the provider attaches a cached/earlier pose timestamp.
                if (
                    frame.active
                    and revision == old_revision
                    and frame.sample_time < self._sample_high_water
                ):
                    return False
            self._latest = frame
            self._last_sequence = frame.sequence
            self._last_identity = frame.identity
            try:
                self._policy.check_timestamps(frame, now=frame.received_time)
            except TrackingUnavailable:
                # Keep packet/reference ordering and the fault payload, but do
                # not let an invalid mapped clock poison later normal samples.
                pass
            else:
                self._sample_high_water = max(self._sample_high_water, frame.sample_time)
            return True

    def latest(self) -> TrackingFrame | None:
        """Return the newest snapshot; consumers must still check its freshness."""
        with self._lock:
            return self._latest

    def clear(self) -> None:
        """Discard the payload while retaining sequence, time and space ordering.

        Use when recalibrating within a connection. Only a new buffer for an
        explicitly selected source/session starts a new ordering history.
        """
        with self._lock:
            self._latest = None


@dataclass(frozen=True)
class FramePolicy:
    """Application-tunable input limits, in seconds; no actuator authorization.

    Unknown body confidence fails a configured minimum. A future pose requires
    an explicit prediction allowance; network arrival does not refresh its age.
    """

    max_sample_age: float = 0.1
    max_receive_age: float = 0.1
    max_prediction: float = 0.0
    min_confidence: float | None = None

    def __post_init__(self) -> None:
        for name in ("max_sample_age", "max_receive_age", "max_prediction", "min_confidence"):
            value = getattr(self, name)
            if name == "min_confidence" and value is None:
                continue
            scalar = np.asarray(value)
            if scalar.ndim != 0 or scalar.dtype.kind not in "fiu":
                raise ValueError(f"{name} must be a finite nonnegative scalar")
            value = float(scalar)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
            if name == "min_confidence" and value > 1:
                raise ValueError("min_confidence must be between zero and one, or None")
            object.__setattr__(self, name, value)

    def check(self, frame: TrackingFrame, *, now: float) -> None:
        """Raise TrackingUnavailable on inactive, stale, or insufficient input."""
        if not np.isfinite(now):
            raise ValueError("now must be finite and in the receiver's monotonic clock domain")
        if not frame.active:
            raise TrackingUnavailable("Body tracking is inactive")
        self.check_timestamps(frame, now=now)
        if self.min_confidence is not None and (
            frame.confidence is None or frame.confidence < self.min_confidence
        ):
            raise TrackingUnavailable("Body tracking confidence is unknown or below the minimum")

    def check_timestamps(self, frame: TrackingFrame, *, now: float) -> None:
        """Check clock bounds independently of body status, including loss frames."""
        if not np.isfinite(now):
            raise ValueError("now must be finite and in the receiver's monotonic clock domain")
        sample_age = now - frame.sample_time
        receive_age = now - frame.received_time
        if sample_age > self.max_sample_age or receive_age > self.max_receive_age:
            raise TrackingUnavailable("Tracking frame is stale")
        if receive_age < 0 or sample_age < -self.max_prediction:
            raise TrackingUnavailable(
                "Tracking timestamps exceed the allowed clock/prediction range"
            )


@dataclass(frozen=True)
class TrackingCalibration:
    """Transform canonical tracking poses into the robot's shared URDF root frame.

    Hand-to-tool rotations apply on the right, after the world-frame extrinsic.
    tool_length creates a virtual +X tool marker, not robot collision geometry.
    Values are supplied by the application, optionally using the offline
    fit_rigid_transform and fit_hand_tool_rotation helpers.
    """

    identity: TrackingIdentity
    rotation: np.ndarray = field(default_factory=lambda: np.eye(3))
    translation: np.ndarray = field(default_factory=lambda: np.zeros(3))
    left_hand_to_tool: np.ndarray = field(default_factory=lambda: np.eye(3))
    right_hand_to_tool: np.ndarray = field(default_factory=lambda: np.eye(3))
    tool_length: float = 0.1

    def __post_init__(self) -> None:
        if not isinstance(self.identity, TrackingIdentity):
            raise ValueError("identity must be a TrackingIdentity")
        for name in ("rotation", "left_hand_to_tool", "right_hand_to_tool"):
            object.__setattr__(self, name, _rotation(getattr(self, name), name))
        object.__setattr__(self, "translation", _array(self.translation, (3,), "translation"))
        if not np.isfinite(self.tool_length) or self.tool_length <= 0:
            raise ValueError("tool_length must be finite and positive")

    def check_frame(
        self,
        frame: TrackingFrame,
        *,
        now: float,
        policy: FramePolicy = FramePolicy(),
        require_tracked: bool = False,
    ) -> None:
        """Validate identity, age, quality and SEW data without transforming a pose.

        Valid inferred joints are accepted by default; require_tracked additionally
        requires runtime TRACKED bits for every consumed component. Neither mode
        estimates missing shoulder/elbow joints from controllers.
        """
        if frame.identity != self.identity:
            raise CalibrationRequired(
                "Tracking source/session/reference space changed; recalibrate"
            )
        policy.check(frame, now=now)
        for side in ("left", "right"):
            positions = []
            for name in ("shoulder", "elbow", "wrist"):
                key = f"{side}_{name}"
                joint = frame.joints.get(key)
                if joint is None or joint.position is None:
                    raise TrackingUnavailable(f"Missing valid position for {key}")
                if require_tracked and not joint.position_tracked:
                    raise TrackingUnavailable(f"Position is not currently tracked for {key}")
                positions.append(joint.position)
            with np.errstate(over="ignore", invalid="ignore"):
                lengths = [np.linalg.norm(positions[i + 1] - positions[i]) for i in (0, 1)]
            if any(not np.isfinite(length) or length <= EPS for length in lengths):
                raise TrackingUnavailable(f"Degenerate or nonfinite SEW segments for {side}")
            wrist = frame.joints[f"{side}_wrist"]
            if wrist.orientation is None:
                raise TrackingUnavailable(f"Missing valid orientation for {side}_wrist")
            if require_tracked and not wrist.orientation_tracked:
                raise TrackingUnavailable(f"Orientation is not currently tracked for {side}_wrist")

    def _transform_frame(self, frame: TrackingFrame) -> BimanualPose:
        """Transform a snapshot after check_frame, without repeating input checks."""
        try:
            with np.errstate(over="raise", invalid="raise"):
                arms = []
                for side in ("left", "right"):
                    positions = [
                        self.rotation @ frame.joints[f"{side}_{name}"].position + self.translation
                        for name in ("shoulder", "elbow", "wrist")
                    ]
                    orientation = (
                        self.rotation
                        @ frame.joints[f"{side}_wrist"].orientation
                        @ getattr(self, f"{side}_hand_to_tool")
                    )
                    tool = positions[2] + self.tool_length * orientation[:, 0]
                    arms.append(ArmPose(*positions, tool, orientation))
                pose = BimanualPose(*arms)
                pose.points()
                return pose
        except (ValueError, FloatingPointError) as exc:
            raise TrackingUnavailable(
                "Calibrated pose is not finite or has invalid orientation"
            ) from exc

    def to_pose(
        self,
        frame: TrackingFrame,
        *,
        now: float,
        policy: FramePolicy = FramePolicy(),
        require_tracked: bool = False,
    ) -> BimanualPose:
        """Validate a complete input snapshot and transform it to the robot frame."""
        self.check_frame(frame, now=now, policy=policy, require_tracked=require_tracked)
        return self._transform_frame(frame)

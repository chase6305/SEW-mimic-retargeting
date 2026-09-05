"""Shared latest-input lifecycle for live adapters and deterministic replay."""

from __future__ import annotations

from threading import Lock

import numpy as np

from ..filtering import BimanualPoseFilter, OneEuroConfig
from ..safety import BimanualPose
from .config import TrackingStreamConfig
from .model import TrackingFrame, TrackingIdentity
from .motion import TrackingMotionLimits
from .pipeline import (
    CalibrationRequired,
    FramePolicy,
    LatestFrameBuffer,
    TrackingCalibration,
    TrackingUnavailable,
)


def _same_session(first: TrackingIdentity, second: TrackingIdentity) -> bool:
    return (first.source_id, first.session_id) == (second.source_id, second.session_id)


class TrackingPoseStream:
    """Own input ordering, loss notification, calibration and pose-filter history.

    publish() may run on a producer thread; poll() belongs to one control loop.
    Methods serialize internal state, without holding locks during robot solving
    or I/O. A successful poll is an input target, not an engagement or actuator
    authorization. The application still owns its controller and watchdog.
    """

    def __init__(
        self,
        calibration: TrackingCalibration,
        *,
        policy: FramePolicy = FramePolicy(),
        require_tracked: bool = False,
        min_cutoff: float = 1.0,
        beta: float = 0.02,
        derivative_cutoff: float = 1.0,
        rotation_config: OneEuroConfig | None = None,
        motion_limits: TrackingMotionLimits | None = None,
    ) -> None:
        if not isinstance(calibration, TrackingCalibration) or not isinstance(policy, FramePolicy):
            raise ValueError("calibration and policy must be TrackingCalibration and FramePolicy")
        if not isinstance(require_tracked, (bool, np.bool_)):
            raise ValueError("require_tracked must be a boolean")
        if motion_limits is not None and not isinstance(motion_limits, TrackingMotionLimits):
            raise ValueError("motion_limits must be TrackingMotionLimits or None")
        self._config = TrackingStreamConfig(
            policy=policy,
            position_filter=OneEuroConfig(min_cutoff, beta, derivative_cutoff),
            rotation_filter=rotation_config,
            motion_limits=motion_limits,
            require_tracked=require_tracked,
        )
        self._calibration = calibration
        self._policy = policy
        self._require_tracked = bool(require_tracked)
        self._motion_limits = motion_limits
        self._motion_reference: TrackingFrame | None = None
        self._buffer = LatestFrameBuffer(
            calibration.identity.source_id, calibration.identity.session_id, policy=policy
        )
        self._observed_identity = calibration.identity
        self._filter = BimanualPoseFilter(
            min_cutoff,
            beta,
            derivative_cutoff,
            rotation_config=rotation_config,
            tool_length=calibration.tool_length,
        )
        self._sample_floor = -np.inf
        self._last_poll_time = -np.inf
        self._waiting_for_sample = True
        self._pending_fault: str | None = None
        self._calibration_fault: str | None = None
        self._lock = Lock()

    @property
    def config(self) -> TrackingStreamConfig:
        """Return immutable settings suitable for recording and later replay."""
        return self._config

    @classmethod
    def from_config(
        cls, calibration: TrackingCalibration, config: TrackingStreamConfig
    ) -> TrackingPoseStream:
        """Create fresh input/filter state from a complete configuration snapshot."""
        if not isinstance(config, TrackingStreamConfig):
            raise ValueError("config must be TrackingStreamConfig")
        return cls(
            calibration,
            policy=config.policy,
            require_tracked=config.require_tracked,
            min_cutoff=config.position_filter.min_cutoff,
            beta=config.position_filter.beta,
            derivative_cutoff=config.position_filter.derivative_cutoff,
            rotation_config=config.rotation_filter,
            motion_limits=config.motion_limits,
        )

    def _reset_filter(self, frame: TrackingFrame | None) -> None:
        # Keep the motion reference across faults. Only explicit calibration
        # selection may re-anchor it, so a reset cannot disable plausibility.
        self._filter.reset()
        self._waiting_for_sample = True
        if frame is not None:
            self._sample_floor = max(self._sample_floor, frame.sample_time)

    def publish(self, frame: TrackingFrame) -> bool:
        """Accept an ordered frame and preserve faults that occur between polls.

        Rejected foreign/old packets cannot invalidate this source. Accepted
        invalid input creates a pending fault even if good input replaces it
        before the consumer polls. Freshness is also checked again on every poll.
        Optional motion limits check each valid raw snapshot before smoothing.
        """
        if not isinstance(frame, TrackingFrame):
            raise ValueError("frame must be a TrackingFrame")
        with self._lock:
            if _same_session(frame.identity, self._calibration.identity) and (
                frame.identity.space_revision < self._calibration.identity.space_revision
            ):
                return False
            previous_frame = self._buffer.latest()
            if not self._buffer.publish(frame):
                return False
            self._observed_identity = frame.identity
            reset_boundary = frame
            try:
                self._policy.check_timestamps(frame, now=frame.received_time)
            except TrackingUnavailable:
                # A bad mapped time is never a recovery boundary. Discard the
                # payload so waiting for the clock to catch up cannot revive it.
                reset_boundary = previous_frame
                self._buffer.clear()
            try:
                self._calibration.check_frame(
                    frame,
                    now=frame.received_time,
                    policy=self._policy,
                    require_tracked=self._require_tracked,
                )
                if self._motion_limits is not None:
                    if self._motion_reference is not None:
                        self._motion_limits._check(self._motion_reference, frame)
                    # Repeated samples after a reset cannot seed a new reference.
                    if frame.sample_time > self._sample_floor and (
                        self._motion_reference is None
                        or frame.sample_time > self._motion_reference.sample_time
                    ):
                        self._motion_reference = frame
            except CalibrationRequired as exc:
                self._calibration_fault = str(exc)
                self._reset_filter(reset_boundary)
            except TrackingUnavailable as exc:
                if self._pending_fault is None:
                    self._pending_fault = str(exc)
                self._reset_filter(reset_boundary)
            return True

    def poll(self, *, now: float) -> BimanualPose | None:
        """Return a new filtered target, or None for a still-valid repeated sample.

        TrackingUnavailable requests a hold. A transient accepted fault is
        reported at least once; a reference change requires set_calibration().
        Recovery requires a sample newer than the reset boundary. Never refresh
        frame timestamps when calling this method.
        """
        now = float(now)
        if not np.isfinite(now):
            raise ValueError("now must be finite")
        with self._lock:
            if now < self._last_poll_time:
                raise ValueError("control poll timestamps must not decrease")
            self._last_poll_time = now
            if self._calibration_fault is not None:
                raise CalibrationRequired(self._calibration_fault)
            if self._pending_fault is not None:
                reason = self._pending_fault
                self._pending_fault = None
                raise TrackingUnavailable(reason)
            frame = self._buffer.latest()
            if frame is None:
                raise TrackingUnavailable("No frame has arrived since calibration")
            try:
                self._calibration.check_frame(
                    frame, now=now, policy=self._policy, require_tracked=self._require_tracked
                )
                if frame.sample_time <= self._sample_floor:
                    if self._waiting_for_sample:
                        raise TrackingUnavailable("Waiting for a newer tracking sample after reset")
                    return None
                pose = self._calibration._transform_frame(frame)
                result = self._filter.update(frame.sample_time, pose)
            except TrackingUnavailable:
                self._reset_filter(frame)
                raise
            except ValueError as exc:
                self._reset_filter(frame)
                raise TrackingUnavailable(f"Unable to filter tracking pose: {exc}") from exc
            self._sample_floor = frame.sample_time
            self._waiting_for_sample = False
            return result

    def invalidate(self, reason: str) -> None:
        """Latch a decoder/application fault and discard history until fresh input.

        Call on decoder errors or rejected robot solves, so an unchanged input
        is not retried after a filter reset. Fault strings, not exception objects,
        are retained; repeated faults do not accumulate a queue or tracebacks.
        """
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("reason must be a nonempty string")
        with self._lock:
            if self._pending_fault is None:
                self._pending_fault = reason
            self._reset_filter(self._buffer.latest())

    def set_calibration(self, calibration: TrackingCalibration) -> None:
        """Apply an explicitly selected calibration and require fresh observations.

        Within a connection, retain packet ordering and reject obsolete reference
        revisions. Selecting a new source/session explicitly starts new input
        ordering. Also clear the optional motion reference; the next newer valid
        sample establishes it. The control clock must continue monotonically.
        """
        if not isinstance(calibration, TrackingCalibration):
            raise ValueError("calibration must be a TrackingCalibration")
        with self._lock:
            identity = calibration.identity
            observed = self._observed_identity
            if _same_session(identity, observed):
                if identity.space_revision < observed.space_revision or (
                    identity.space_revision == observed.space_revision
                    and identity.space_id != observed.space_id
                ):
                    raise CalibrationRequired(
                        "Cannot apply an obsolete or conflicting reference-space calibration"
                    )
                self._reset_filter(self._buffer.latest())
                self._buffer.clear()
            else:
                self._reset_filter(None)
                self._buffer = LatestFrameBuffer(
                    identity.source_id, identity.session_id, policy=self._policy
                )
                self._sample_floor = -np.inf
            self._calibration = calibration
            self._motion_reference = None
            self._filter.tool_length = calibration.tool_length
            self._observed_identity = identity
            self._pending_fault = None
            self._calibration_fault = None

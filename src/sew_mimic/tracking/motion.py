"""Optional motion plausibility limits applied before adaptive pose smoothing."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .model import TrackingFrame
from .pipeline import TrackingUnavailable


@dataclass(frozen=True)
class TrackingMotionLimits:
    """Application-selected input bounds, independent of robot velocity limits.

    Speeds use canonical positions in m/s and wrist rotations in rad/s. At
    least one speed limit is required. The last valid sample remains the
    reference after rejected input; max_sample_gap bounds how long recovery
    may compare against it. After expiry, explicitly apply calibration again
    to establish a new reference. No device-specific speed limits are assumed.
    """

    max_joint_speed: float | None = None
    max_wrist_angular_speed: float | None = None
    max_sample_gap: float = 0.1

    def __post_init__(self) -> None:
        for name in ("max_joint_speed", "max_wrist_angular_speed", "max_sample_gap"):
            value = getattr(self, name)
            if value is None and name != "max_sample_gap":
                continue
            scalar = np.asarray(value)
            if scalar.ndim != 0 or scalar.dtype.kind not in "fiu":
                raise ValueError(f"{name} must be a finite positive scalar")
            value = float(scalar)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a finite positive scalar")
            object.__setattr__(self, name, value)
        if self.max_joint_speed is None and self.max_wrist_angular_speed is None:
            raise ValueError("At least one tracking speed limit must be configured")

    def _check(self, previous: TrackingFrame, frame: TrackingFrame) -> None:
        """Check two complete, calibrated, same-identity snapshots without mutation."""
        dt = frame.sample_time - previous.sample_time
        if not math.isfinite(dt) or dt > self.max_sample_gap:
            raise TrackingUnavailable(
                "Tracking motion reference expired; apply set_calibration() to re-anchor"
            )
        if dt < 0:
            raise TrackingUnavailable("Tracking motion sample timestamps must not decrease")
        for side in ("left", "right"):
            if self.max_joint_speed is not None:
                for name in ("shoulder", "elbow", "wrist"):
                    key = f"{side}_{name}"
                    # Completeness has already been checked by calibration.
                    with np.errstate(over="ignore", invalid="ignore"):
                        delta = frame.joints[key].position - previous.joints[key].position
                    distance = math.hypot(*delta)
                    if not math.isfinite(distance) or distance > self.max_joint_speed * dt:
                        raise TrackingUnavailable(
                            f"Tracking position jump at {key} exceeds max_joint_speed"
                        )
            if self.max_wrist_angular_speed is not None:
                key = f"{side}_wrist"
                relative = previous.joints[key].orientation.T @ frame.joints[key].orientation
                # atan2 retains small-angle resolution and the shortest angle
                # near pi; identical rotations give an exactly zero skew part.
                sine = 0.5 * math.hypot(
                    relative[2, 1] - relative[1, 2],
                    relative[0, 2] - relative[2, 0],
                    relative[1, 0] - relative[0, 1],
                )
                cosine = 0.5 * (float(np.trace(relative)) - 1.0)
                angle = math.atan2(sine, cosine)
                if angle > self.max_wrist_angular_speed * dt:
                    raise TrackingUnavailable(
                        f"Tracking rotation jump at {key} exceeds max_wrist_angular_speed"
                    )

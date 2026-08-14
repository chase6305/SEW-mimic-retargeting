"""Robot-specific, reusable trajectory profiles for collision demo validation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CollisionDemoProfile:
    """Four joint-space endpoints for one continuous open/collide/open motion."""

    neutral_left: np.ndarray
    neutral_right: np.ndarray
    colliding_left: np.ndarray
    colliding_right: np.ndarray

    def __post_init__(self) -> None:
        for name in ("neutral_left", "neutral_right", "colliding_left", "colliding_right"):
            values = np.asarray(getattr(self, name), dtype=np.float64).reshape(-1).copy()
            if values.shape != (7,) or not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must be a finite seven-joint vector")
            values.flags.writeable = False
            object.__setattr__(self, name, values)

    def trajectory(self, duration: float, fps: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate a smooth target that crosses collision without pausing."""
        if duration <= 0.0 or fps <= 0.0:
            raise ValueError("duration and fps must be positive")
        times = np.linspace(0.0, duration, int(np.ceil(duration * fps)) + 1)
        phase = times / duration
        rising = 10.0 * (2.0 * phase) ** 3 - 15.0 * (2.0 * phase) ** 4 + 6.0 * (2.0 * phase) ** 5
        falling_phase = 2.0 * (1.0 - phase)
        falling = 10.0 * falling_phase**3 - 15.0 * falling_phase**4 + 6.0 * falling_phase**5
        blend = np.where(phase <= 0.5, rising, falling)
        left = self.neutral_left + blend[:, None] * (self.colliding_left - self.neutral_left)
        right = self.neutral_right + blend[:, None] * (self.colliding_right - self.neutral_right)
        return times, left, right


MARVIN_COLLISION_PROFILE = CollisionDemoProfile(
    neutral_left=[-2.58915, 0.51915, 2.60269, -1.76463, 0.97923, 0.70662, -0.64099],
    neutral_right=[-0.55240, -0.51914, 0.53884, -1.76463, 2.48376, -0.84185, -0.41466],
    colliding_left=[-2.87488, 1.38313, 2.89654, -1.29803, 1.05635, 0.78405, -0.64147],
    colliding_right=[-0.26671, -1.38313, 0.24503, -1.29803, 2.55556, -0.93924, -0.28068],
)

OPENARM_COLLISION_PROFILE = CollisionDemoProfile(
    neutral_left=[-1.26172, -0.92937, 1.19838, 1.33681, 1.57079, -0.46509, -0.52772],
    neutral_right=[0.90530, 0.80607, -1.00774, 1.32608, 1.57079, -0.43390, 0.72121],
    colliding_left=[-1.49687, -0.20808, 1.44850, 1.26534, 1.57079, -0.73129, -0.32459],
    colliding_right=[1.20158, 0.15643, -1.37080, 1.25447, 1.57079, -0.68932, 0.57426],
)

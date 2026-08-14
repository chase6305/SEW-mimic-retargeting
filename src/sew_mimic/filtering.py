"""Stateful real-time filters for noisy tracking and robot commands."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .safety import ArmPose, BimanualPose
from .utility import is_rotation_matrix


def _positive_finite(name: str, value: float | np.ndarray) -> np.ndarray:
    values = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError(f"{name} must contain only finite positive values")
    return values


def _smoothing_factor(cutoff: float | np.ndarray, dt: float) -> np.ndarray:
    """Return exact first-order low-pass gain for cutoff in hertz."""
    return 1.0 - np.exp(-2.0 * np.pi * np.asarray(cutoff) * dt)


@dataclass
class OneEuroFilter:
    """Adaptive low-pass filter for scalar or Euclidean vector observations.

    ``min_cutoff`` controls stationary smoothing, while ``beta`` increases the
    cutoff in proportion to filtered component-wise speed. Timestamps are in
    seconds and must increase strictly. Use a dedicated SO(3) filter for
    rotation matrices; Euclidean element-wise filtering does not preserve SO(3).
    """

    min_cutoff: float = 1.0
    beta: float = 0.02
    derivative_cutoff: float = 1.0
    _timestamp: float | None = field(default=None, init=False, repr=False)
    _raw_value: np.ndarray | None = field(default=None, init=False, repr=False)
    _value: np.ndarray | None = field(default=None, init=False, repr=False)
    _derivative: np.ndarray | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        _positive_finite("min_cutoff", self.min_cutoff)
        _positive_finite("derivative_cutoff", self.derivative_cutoff)
        if not np.isfinite(self.beta) or self.beta < 0.0:
            raise ValueError("beta must be finite and nonnegative")

    def reset(self) -> None:
        """Discard timestamp, value, and derivative history."""
        self._timestamp = None
        self._raw_value = None
        self._value = None
        self._derivative = None

    def update(self, timestamp: float, observation: np.ndarray | float) -> np.ndarray:
        """Filter one finite observation and return an owned float64 array."""
        value = np.asarray(observation, dtype=np.float64)
        if not np.isfinite(timestamp) or not np.all(np.isfinite(value)):
            raise ValueError("timestamp and observation must be finite")
        if self._timestamp is None:
            self._timestamp = float(timestamp)
            self._raw_value = value.copy()
            self._value = value.copy()
            self._derivative = np.zeros_like(value)
            return self._value.copy()
        assert self._raw_value is not None and self._value is not None
        assert self._derivative is not None
        if value.shape != self._value.shape:
            raise ValueError(f"observation shape changed from {self._value.shape} to {value.shape}")
        dt = float(timestamp - self._timestamp)
        if dt <= 0.0:
            raise ValueError("timestamps must increase strictly")
        raw_derivative = (value - self._raw_value) / dt
        derivative_alpha = _smoothing_factor(self.derivative_cutoff, dt)
        derivative = self._derivative + derivative_alpha * (raw_derivative - self._derivative)
        cutoff = self.min_cutoff + self.beta * np.abs(derivative)
        value_alpha = _smoothing_factor(cutoff, dt)
        filtered = self._value + value_alpha * (value - self._value)
        self._timestamp = float(timestamp)
        self._raw_value = value.copy()
        self._value = filtered
        self._derivative = derivative
        return filtered.copy()


@dataclass
class JointRateLimiter:
    """Apply exact per-joint velocity bounds to an online command stream."""

    max_velocity: np.ndarray | float
    _timestamp: float | None = field(default=None, init=False, repr=False)
    _command: np.ndarray | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.max_velocity = _positive_finite("max_velocity", self.max_velocity)

    def reset(self, command: np.ndarray, timestamp: float) -> None:
        """Set the known current command without applying a transition."""
        value = np.asarray(command, dtype=np.float64)
        if not np.isfinite(timestamp) or not np.all(np.isfinite(value)):
            raise ValueError("timestamp and command must be finite")
        try:
            np.broadcast_to(self.max_velocity, value.shape)
        except ValueError as exc:
            raise ValueError("max_velocity is not broadcastable to command shape") from exc
        self._timestamp = float(timestamp)
        self._command = value.copy()

    def update(self, timestamp: float, target: np.ndarray) -> np.ndarray:
        """Move from the previous command toward target within ``rad/s`` bounds."""
        value = np.asarray(target, dtype=np.float64)
        if self._timestamp is None or self._command is None:
            raise RuntimeError("reset(command, timestamp) must be called before update")
        if value.shape != self._command.shape or not np.all(np.isfinite(value)):
            raise ValueError("target must be finite and preserve command shape")
        dt = float(timestamp - self._timestamp)
        if not np.isfinite(timestamp) or dt <= 0.0:
            raise ValueError("timestamps must be finite and increase strictly")
        maximum_step = np.broadcast_to(self.max_velocity, value.shape) * dt
        self._command = self._command + np.clip(value - self._command, -maximum_step, maximum_step)
        self._timestamp = float(timestamp)
        return self._command.copy()


def _rotation_quaternion(rotation: np.ndarray) -> np.ndarray:
    """Convert a validated rotation matrix to scalar-first unit quaternion."""
    matrix = np.asarray(rotation, dtype=np.float64)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = 2.0 * np.sqrt(trace + 1.0)
        quaternion = np.array(
            [
                0.25 * scale,
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
            ]
        )
    else:
        index = int(np.argmax(np.diag(matrix)))
        first, second = (index + 1) % 3, (index + 2) % 3
        scale = 2.0 * np.sqrt(
            max(0.0, 1.0 + matrix[index, index] - matrix[first, first] - matrix[second, second])
        )
        quaternion = np.empty(4)
        quaternion[0] = (matrix[second, first] - matrix[first, second]) / scale
        quaternion[index + 1] = 0.25 * scale
        quaternion[first + 1] = (matrix[first, index] + matrix[index, first]) / scale
        quaternion[second + 1] = (matrix[second, index] + matrix[index, second]) / scale
    return quaternion / np.linalg.norm(quaternion)


def _quaternion_rotation(quaternion: np.ndarray) -> np.ndarray:
    """Convert a scalar-first unit quaternion to SO(3)."""
    w, x, y, z = quaternion / np.linalg.norm(quaternion)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ]
    )


def _slerp(first: np.ndarray, second: np.ndarray, fraction: float) -> np.ndarray:
    """Shortest-arc unit-quaternion interpolation."""
    dot = float(first @ second)
    if dot < 0.0:
        second = -second
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        interpolated = first + fraction * (second - first)
        return interpolated / np.linalg.norm(interpolated)
    angle = float(np.arccos(dot))
    sine = np.sin(angle)
    return (np.sin((1.0 - fraction) * angle) * first + np.sin(fraction * angle) * second) / sine


@dataclass
class OneEuroRotationFilter:
    """One Euro filter on SO(3) using adaptive shortest-arc quaternion SLERP.

    Set ``validate_input=False`` only when an upstream component, such as a
    validated FK implementation, already guarantees finite SO(3) matrices.
    """

    min_cutoff: float = 1.0
    beta: float = 0.02
    derivative_cutoff: float = 1.0
    validate_input: bool = True
    _timestamp: float | None = field(default=None, init=False, repr=False)
    _raw_quaternion: np.ndarray | None = field(default=None, init=False, repr=False)
    _quaternion: np.ndarray | None = field(default=None, init=False, repr=False)
    _angular_velocity: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self) -> None:
        _positive_finite("min_cutoff", self.min_cutoff)
        _positive_finite("derivative_cutoff", self.derivative_cutoff)
        if not np.isfinite(self.beta) or self.beta < 0.0:
            raise ValueError("beta must be finite and nonnegative")

    def reset(self) -> None:
        """Discard orientation and angular-velocity history."""
        self._timestamp = None
        self._raw_quaternion = None
        self._quaternion = None
        self._angular_velocity = 0.0

    def update(self, timestamp: float, rotation: np.ndarray) -> np.ndarray:
        """Filter one SO(3) observation and return a valid rotation matrix."""
        if not np.isfinite(timestamp) or (self.validate_input and not is_rotation_matrix(rotation)):
            raise ValueError("timestamp must be finite and rotation must be SO(3)")
        quaternion = _rotation_quaternion(rotation)
        if self._timestamp is None:
            self._timestamp = float(timestamp)
            self._raw_quaternion = quaternion.copy()
            self._quaternion = quaternion.copy()
            return np.asarray(rotation, dtype=np.float64).copy()
        assert self._raw_quaternion is not None and self._quaternion is not None
        dt = float(timestamp - self._timestamp)
        if dt <= 0.0:
            raise ValueError("timestamps must increase strictly")
        relative_dot = float(np.clip(abs(self._raw_quaternion @ quaternion), 0.0, 1.0))
        raw_speed = 2.0 * np.arccos(relative_dot) / dt
        derivative_alpha = float(_smoothing_factor(self.derivative_cutoff, dt))
        self._angular_velocity += derivative_alpha * (raw_speed - self._angular_velocity)
        cutoff = self.min_cutoff + self.beta * self._angular_velocity
        fraction = float(_smoothing_factor(cutoff, dt))
        self._quaternion = _slerp(self._quaternion, quaternion, fraction)
        self._raw_quaternion = quaternion.copy()
        self._timestamp = float(timestamp)
        return _quaternion_rotation(self._quaternion)


@dataclass
class BimanualPoseFilter:
    """Filter eight bimanual keypoints and both tool orientations coherently.

    ``validate_rotations=False`` removes duplicate SO(3) checks for trusted FK
    output. Keep the default enabled for measurements from external trackers.
    """

    min_cutoff: float = 1.0
    beta: float = 0.02
    derivative_cutoff: float = 1.0
    validate_rotations: bool = True
    _points: OneEuroFilter = field(init=False, repr=False)
    _left_orientation: OneEuroRotationFilter = field(init=False, repr=False)
    _right_orientation: OneEuroRotationFilter = field(init=False, repr=False)

    def __post_init__(self) -> None:
        parameters = (self.min_cutoff, self.beta, self.derivative_cutoff)
        self._points = OneEuroFilter(*parameters)
        self._left_orientation = OneEuroRotationFilter(
            *parameters, validate_input=self.validate_rotations
        )
        self._right_orientation = OneEuroRotationFilter(
            *parameters, validate_input=self.validate_rotations
        )

    def reset(self) -> None:
        """Clear position and orientation histories for both arms."""
        self._points.reset()
        self._left_orientation.reset()
        self._right_orientation.reset()

    def update(self, timestamp: float, pose: BimanualPose) -> BimanualPose:
        """Filter a complete bimanual pose while preserving its structure."""
        points = self._points.update(timestamp, pose.keypoints())
        left_orientation = self._left_orientation.update(timestamp, pose.left.tool_orientation)
        right_orientation = self._right_orientation.update(timestamp, pose.right.tool_orientation)
        return BimanualPose(
            ArmPose(*points[:4], left_orientation),
            ArmPose(*points[4:], right_orientation),
        )

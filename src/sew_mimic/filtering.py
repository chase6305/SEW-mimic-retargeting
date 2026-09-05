"""Stateful real-time filters for noisy tracking and robot commands."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import NamedTuple

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
    return -np.expm1(-2.0 * np.pi * np.asarray(cutoff) * dt)


def _scalar_smoothing_factor(cutoff: float, dt: float) -> float:
    """Scalar specialization avoiding NumPy dispatch in SO(3) hot loops."""
    return -math.expm1(-2.0 * math.pi * cutoff * dt)


@dataclass(frozen=True)
class OneEuroConfig:
    """Immutable tuning parameters; cutoffs are Hz, beta depends on signal units."""

    min_cutoff: float = 1.0
    beta: float = 0.02
    derivative_cutoff: float = 1.0

    def __post_init__(self) -> None:
        for name in ("min_cutoff", "beta", "derivative_cutoff"):
            value = np.asarray(getattr(self, name))
            if value.ndim != 0 or value.dtype.kind not in "fiu":
                raise ValueError(f"{name} must be a finite scalar")
            scalar = float(value)
            if not math.isfinite(scalar) or scalar < 0 or (name != "beta" and scalar == 0):
                raise ValueError(
                    f"{name} must be finite and {'nonnegative' if name == 'beta' else 'positive'}"
                )
            object.__setattr__(self, name, scalar)


class _ValueState(NamedTuple):
    timestamp: float
    raw_value: np.ndarray
    value: np.ndarray
    derivative: np.ndarray


class _RotationState(NamedTuple):
    timestamp: float
    raw_quaternion: np.ndarray
    quaternion: np.ndarray
    angular_velocity: float


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
        config = OneEuroConfig(self.min_cutoff, self.beta, self.derivative_cutoff)
        self.min_cutoff, self.beta, self.derivative_cutoff = (
            config.min_cutoff,
            config.beta,
            config.derivative_cutoff,
        )

    def reset(self) -> None:
        """Discard timestamp, value, and derivative history."""
        self._timestamp = None
        self._raw_value = None
        self._value = None
        self._derivative = None

    def update(self, timestamp: float, observation: np.ndarray | float) -> np.ndarray:
        """Filter an observation; rejected input leaves history unchanged."""
        state = self._prepare(timestamp, observation)
        self._commit(state)
        return state.value.copy()

    def _commit(self, state: _ValueState) -> None:
        self._timestamp, self._raw_value, self._value, self._derivative = state

    def _prepare(self, timestamp: float, observation: np.ndarray | float) -> _ValueState:
        """Compute and validate the next state without changing history."""
        timestamp = float(timestamp)
        value = np.asarray(observation, dtype=np.float64)
        if not np.isfinite(timestamp) or not np.all(np.isfinite(value)):
            raise ValueError("timestamp and observation must be finite")
        if self._timestamp is None:
            return _ValueState(timestamp, value.copy(), value.copy(), np.zeros_like(value))
        assert self._raw_value is not None and self._value is not None
        assert self._derivative is not None
        if value.shape != self._value.shape:
            raise ValueError(f"observation shape changed from {self._value.shape} to {value.shape}")
        dt = float(timestamp - self._timestamp)
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("timestamps must increase strictly with a finite interval")
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                raw_derivative = (value - self._raw_value) / dt
                derivative_alpha = _smoothing_factor(self.derivative_cutoff, dt)
                derivative = self._derivative + derivative_alpha * (
                    raw_derivative - self._derivative
                )
                cutoff = self.min_cutoff + self.beta * np.abs(derivative)
                value_alpha = _smoothing_factor(cutoff, dt)
                filtered = self._value + value_alpha * (value - self._value)
        except FloatingPointError as exc:
            raise ValueError("Position filter arithmetic exceeded the finite range") from exc
        return _ValueState(timestamp, value.copy(), filtered, derivative)


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
    return quaternion / math.sqrt(float(quaternion @ quaternion))


def _quaternion_rotation(quaternion: np.ndarray) -> np.ndarray:
    """Convert a scalar-first unit quaternion to SO(3)."""
    w, x, y, z = quaternion
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
        return interpolated / math.sqrt(float(interpolated @ interpolated))
    angle = math.acos(dot)
    sine = math.sin(angle)
    return (math.sin((1.0 - fraction) * angle) * first + math.sin(fraction * angle) * second) / sine


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
        config = OneEuroConfig(self.min_cutoff, self.beta, self.derivative_cutoff)
        self.min_cutoff, self.beta, self.derivative_cutoff = (
            config.min_cutoff,
            config.beta,
            config.derivative_cutoff,
        )

    def reset(self) -> None:
        """Discard orientation and angular-velocity history."""
        self._timestamp = None
        self._raw_quaternion = None
        self._quaternion = None
        self._angular_velocity = 0.0

    def update(self, timestamp: float, rotation: np.ndarray) -> np.ndarray:
        """Filter an SO(3) observation; rejected input leaves history unchanged."""
        state, result = self._prepare(timestamp, rotation)
        self._commit(state)
        return result

    def _commit(self, state: _RotationState) -> None:
        self._timestamp, self._raw_quaternion, self._quaternion, self._angular_velocity = state

    def _prepare(self, timestamp: float, rotation: np.ndarray) -> tuple[_RotationState, np.ndarray]:
        timestamp = float(timestamp)
        if not np.isfinite(timestamp) or (self.validate_input and not is_rotation_matrix(rotation)):
            raise ValueError("timestamp must be finite and rotation must be SO(3)")
        quaternion = _rotation_quaternion(rotation)
        if self._timestamp is None:
            return (
                _RotationState(timestamp, quaternion.copy(), quaternion.copy(), 0.0),
                np.asarray(rotation, dtype=np.float64).copy(),
            )
        assert self._raw_quaternion is not None and self._quaternion is not None
        dt = float(timestamp - self._timestamp)
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("timestamps must increase strictly with a finite interval")
        relative_dot = min(abs(float(self._raw_quaternion @ quaternion)), 1.0)
        raw_speed = 2.0 * math.acos(relative_dot) / dt
        derivative_alpha = _scalar_smoothing_factor(self.derivative_cutoff, dt)
        angular_velocity = self._angular_velocity + derivative_alpha * (
            raw_speed - self._angular_velocity
        )
        cutoff = self.min_cutoff + self.beta * angular_velocity
        if not all(math.isfinite(value) for value in (raw_speed, angular_velocity, cutoff)):
            raise ValueError("Rotation filter arithmetic exceeded the finite range")
        fraction = _scalar_smoothing_factor(cutoff, dt)
        filtered = _slerp(self._quaternion, quaternion, fraction)
        return (
            _RotationState(timestamp, quaternion.copy(), filtered, angular_velocity),
            _quaternion_rotation(filtered),
        )


@dataclass
class BimanualPoseFilter:
    """Filter eight bimanual keypoints and both tool orientations coherently.

    ``validate_rotations=False`` removes duplicate SO(3) checks for trusted FK
    output. Keep the default enabled for measurements from external trackers.
    ``rotation_config`` tunes angular smoothing independently of position.
    With ``tool_length``, filter only the six SEW positions and reconstruct
    virtual +X tool markers from the filtered wrists and orientations.
    """

    min_cutoff: float = 1.0
    beta: float = 0.02
    derivative_cutoff: float = 1.0
    validate_rotations: bool = True
    rotation_config: OneEuroConfig | None = field(default=None, kw_only=True)
    tool_length: float | None = field(default=None, kw_only=True)
    _points: OneEuroFilter = field(init=False, repr=False)
    _left_orientation: OneEuroRotationFilter = field(init=False, repr=False)
    _right_orientation: OneEuroRotationFilter = field(init=False, repr=False)

    def __post_init__(self) -> None:
        parameters = (self.min_cutoff, self.beta, self.derivative_cutoff)
        self._points = OneEuroFilter(*parameters)
        if self.rotation_config is not None:
            if not isinstance(self.rotation_config, OneEuroConfig):
                raise ValueError("rotation_config must be OneEuroConfig or None")
            parameters = (
                self.rotation_config.min_cutoff,
                self.rotation_config.beta,
                self.rotation_config.derivative_cutoff,
            )
        if self.tool_length is not None:
            length = np.asarray(self.tool_length)
            if (
                length.ndim != 0
                or length.dtype.kind not in "fiu"
                or not np.isfinite(length)
                or length <= 0
            ):
                raise ValueError("tool_length must be finite and positive, or None")
            self.tool_length = float(length)
        # The complete observation is validated before any component advances.
        self._left_orientation = OneEuroRotationFilter(*parameters, validate_input=False)
        self._right_orientation = OneEuroRotationFilter(*parameters, validate_input=False)

    def reset(self) -> None:
        """Clear position and orientation histories for both arms."""
        self._points.reset()
        self._left_orientation.reset()
        self._right_orientation.reset()

    def update(self, timestamp: float, pose: BimanualPose) -> BimanualPose:
        """Filter a pose; rejected observations leave all histories unchanged."""
        observation = pose.points() if self.validate_rotations else pose.keypoints()
        if self.tool_length is not None:
            observation = observation[[0, 1, 2, 4, 5, 6]]
        position_state = self._points._prepare(timestamp, observation)
        left_state, left_orientation = self._left_orientation._prepare(
            timestamp, pose.left.tool_orientation
        )
        right_state, right_orientation = self._right_orientation._prepare(
            timestamp, pose.right.tool_orientation
        )
        points = position_state.value.copy()
        if self.tool_length is None:
            result = BimanualPose(
                ArmPose(*points[:4], left_orientation),
                ArmPose(*points[4:], right_orientation),
            )
        else:
            try:
                with np.errstate(over="raise", invalid="raise"):
                    result = BimanualPose(
                        ArmPose(
                            *points[:3],
                            points[2] + self.tool_length * left_orientation[:, 0],
                            left_orientation,
                        ),
                        ArmPose(
                            *points[3:],
                            points[5] + self.tool_length * right_orientation[:, 0],
                            right_orientation,
                        ),
                    )
            except FloatingPointError as exc:
                raise ValueError("Filtered tool marker exceeded the finite range") from exc
        # Commit the whole pose only after every component has succeeded.
        self._points._commit(position_state)
        self._left_orientation._commit(left_state)
        self._right_orientation._commit(right_state)
        return result

"""Bimanual capsule/XPBD safety filter from paper Algorithms 4 and 9-12."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Callable

import numpy as np

from .collision import _capsule_contact_unchecked, _capsule_distances_unchecked
from .utility import EPS, SEWMimicError, is_rotation_matrix, rot, unit

logger = logging.getLogger(__name__)


class CapsuleIndex(IntEnum):
    """Stable indices used when configuring capsule collision pairs."""

    TORSO = 0
    LEFT_UPPER_ARM = 1
    LEFT_LOWER_ARM = 2
    LEFT_HAND = 3
    RIGHT_UPPER_ARM = 4
    RIGHT_LOWER_ARM = 5
    RIGHT_HAND = 6


@dataclass(frozen=True)
class ArmPose:
    """SEW/tool keypoints and tool orientation in one shared base frame."""

    shoulder: np.ndarray
    elbow: np.ndarray
    wrist: np.ndarray
    tool: np.ndarray
    tool_orientation: np.ndarray

    def points(self) -> np.ndarray:
        points = np.asarray([self.shoulder, self.elbow, self.wrist, self.tool], dtype=np.float64)
        if points.shape != (4, 3) or not np.all(np.isfinite(points)):
            raise ValueError("Arm keypoints must be a finite (4, 3) array")
        if not is_rotation_matrix(self.tool_orientation):
            raise ValueError("tool_orientation must be a valid SO(3) matrix")
        return points


@dataclass(frozen=True)
class BimanualPose:
    left: ArmPose
    right: ArmPose

    def points(self) -> np.ndarray:
        return np.vstack((self.left.points(), self.right.points()))


@dataclass(frozen=True)
class CapsuleRadii:
    torso: float
    upper_arm: float
    lower_arm: float
    hand: float

    def __post_init__(self) -> None:
        values = np.asarray([self.torso, self.upper_arm, self.lower_arm, self.hand])
        if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("All capsule radii must be finite and positive")


@dataclass(frozen=True)
class SafetyFilterConfig:
    radii: CapsuleRadii
    torso_start: np.ndarray
    torso_end: np.ndarray
    minimum_distance: float = 0.01
    activation_distance: float = 0.03
    release_distance: float = 0.04
    compliance: float = 1e-6
    tolerance: float = 1e-6
    iterations: int = 20
    interpolation_limit: int = 100
    collision_pairs: tuple[tuple[int, int], ...] | None = None
    collision_pair_indices: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        torso_start = np.asarray(self.torso_start, dtype=np.float64)
        torso_end = np.asarray(self.torso_end, dtype=np.float64)
        if (
            torso_start.shape != (3,)
            or torso_end.shape != (3,)
            or not np.all(np.isfinite(torso_start))
            or not np.all(np.isfinite(torso_end))
        ):
            raise ValueError("torso_start and torso_end must be finite 3-vectors")
        torso_start.flags.writeable = False
        torso_end.flags.writeable = False
        object.__setattr__(self, "torso_start", torso_start)
        object.__setattr__(self, "torso_end", torso_end)
        for name in ("minimum_distance", "activation_distance", "release_distance"):
            if not np.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if not self.minimum_distance <= self.activation_distance <= self.release_distance:
            raise ValueError(
                "Distances must obey minimum_distance <= activation_distance <= release_distance"
            )
        if self.compliance < 0.0 or self.tolerance < 0.0:
            raise ValueError("compliance and tolerance must be nonnegative")
        if self.iterations <= 0 or self.interpolation_limit <= 0:
            raise ValueError("iterations and interpolation_limit must be positive")
        pairs = COLLISION_PAIRS if self.collision_pairs is None else self.collision_pairs
        normalized = tuple((int(first), int(second)) for first, second in pairs)
        if not normalized:
            raise ValueError("collision_pairs must not be empty")
        if any(first < 0 or second > 6 or first >= second for first, second in normalized):
            raise ValueError("collision pairs must satisfy 0 <= first < second <= 6")
        object.__setattr__(self, "collision_pairs", normalized)
        pair_indices = np.asarray(normalized, dtype=np.intp)
        pair_indices.flags.writeable = False
        object.__setattr__(self, "collision_pair_indices", pair_indices)


class SafetyFilterStatus(str, Enum):
    """Machine-readable outcome of one bimanual safety-filter call."""

    ACCEPTED = "accepted"
    CORRECTED = "corrected"
    XPBD_FAILED = "xpbd_failed"
    IK_FAILED = "ik_failed"
    VALIDATION_FAILED = "validation_failed"


@dataclass(frozen=True)
class SafetyFilterResult:
    q_left: np.ndarray
    q_right: np.ndarray
    safe: bool
    iterations: int
    minimum_distance: float
    status: SafetyFilterStatus = SafetyFilterStatus.ACCEPTED

    @property
    def modified(self) -> bool:
        """Whether the desired command was changed by the filter."""
        return self.status is not SafetyFilterStatus.ACCEPTED


# Each moving capsule is described by (name, first keypoint, second keypoint,
# radius field). Point order is left s/e/w/t then right s/e/w/t.
_MOVING_CAPSULES = (
    ("left_upper", 0, 1, "upper_arm"),
    ("left_lower", 1, 2, "lower_arm"),
    ("left_hand", 2, 3, "hand"),
    ("right_upper", 4, 5, "upper_arm"),
    ("right_lower", 5, 6, "lower_arm"),
    ("right_hand", 6, 7, "hand"),
)


def _capsule_arrays(
    points: np.ndarray, config: SafetyFilterConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build compact capsule arrays for the real-time collision hot path."""
    starts = np.empty((7, 3), dtype=np.float64)
    ends = np.empty((7, 3), dtype=np.float64)
    radii = np.empty(7, dtype=np.float64)
    starts[0], ends[0], radii[0] = (
        config.torso_start,
        config.torso_end,
        config.radii.torso,
    )
    for index, (_, start, end, radius_name) in enumerate(_MOVING_CAPSULES, 1):
        starts[index] = points[start]
        ends[index] = points[end]
        radii[index] = getattr(config.radii, radius_name)
    return starts, ends, radii


def _default_collision_pairs() -> tuple[tuple[int, int], ...]:
    # Exclude directly connected same-arm links and torso/upper-arm attachment.
    excluded = {
        (CapsuleIndex.LEFT_UPPER_ARM, CapsuleIndex.LEFT_LOWER_ARM),
        (CapsuleIndex.LEFT_LOWER_ARM, CapsuleIndex.LEFT_HAND),
        (CapsuleIndex.RIGHT_UPPER_ARM, CapsuleIndex.RIGHT_LOWER_ARM),
        (CapsuleIndex.RIGHT_LOWER_ARM, CapsuleIndex.RIGHT_HAND),
        (CapsuleIndex.TORSO, CapsuleIndex.LEFT_UPPER_ARM),
        (CapsuleIndex.TORSO, CapsuleIndex.RIGHT_UPPER_ARM),
    }
    return tuple(
        (first, second)
        for first in range(7)
        for second in range(first + 1, 7)
        if (first, second) not in excluded
    )


COLLISION_PAIRS = _default_collision_pairs()


def minimum_capsule_distance(points: np.ndarray, config: SafetyFilterConfig) -> float:
    """Return the smallest signed distance among configured collision pairs."""
    points = np.asarray(points, dtype=np.float64)
    if points.shape != (8, 3) or not np.all(np.isfinite(points)):
        raise ValueError("Bimanual point arrays must be finite and have shape (8, 3)")
    starts, ends, radii = _capsule_arrays(points, config)
    assert config.collision_pairs is not None
    first = config.collision_pair_indices[:, 0]
    second = config.collision_pair_indices[:, 1]
    distances = _capsule_distances_unchecked(
        starts[first], ends[first], starts[second], ends[second], radii[first], radii[second]
    )
    return float(np.min(distances))


def find_first_collision(
    initial_points: np.ndarray,
    desired_points: np.ndarray,
    config: SafetyFilterConfig,
) -> np.ndarray:
    """Paper Algorithm 9 continuous-time approximation."""
    logger.debug("Continuous collision check started")
    initial = np.asarray(initial_points, dtype=np.float64)
    desired = np.asarray(desired_points, dtype=np.float64)
    if initial.shape != (8, 3) or desired.shape != (8, 3):
        raise ValueError("Bimanual point arrays must have shape (8, 3)")
    point_radii = np.array(
        [
            config.radii.upper_arm,
            config.radii.upper_arm,
            config.radii.lower_arm,
            config.radii.hand,
        ]
        * 2
    )
    ratios = np.linalg.norm(desired - initial, axis=1) / point_radii
    samples = int(np.clip(np.ceil(np.max(ratios)), 1, config.interpolation_limit))
    logger.debug("Continuous collision check sampling: samples=%d", samples)
    last = desired
    for sample_index, fraction in enumerate(np.linspace(0.0, 1.0, samples + 1)[1:], 1):
        candidate = initial + fraction * (desired - initial)
        last = candidate
        distance = minimum_capsule_distance(candidate, config)
        if distance < config.activation_distance:
            logger.debug(
                "Continuous collision activation: sample=%d/%d fraction=%.4f distance=%.6f",
                sample_index,
                samples,
                fraction,
                distance,
            )
            break
    else:
        logger.debug("Continuous collision check completed without activation")
    return last.copy()


def _project_lengths(points: np.ndarray, lengths: np.ndarray, compliance: float) -> None:
    """Paper Algorithm 11 link-length XPBD projection."""
    links = ((0, 1), (1, 2), (2, 3), (4, 5), (5, 6), (6, 7))
    maximum_correction = 0.0
    for (start, end), target_length in zip(links, lengths):
        delta = points[end] - points[start]
        length = float(np.linalg.norm(delta))
        if length <= EPS:
            continue
        constraint = length - target_length
        maximum_correction = max(maximum_correction, abs(constraint))
        # Shoulders are fixed; all other points have unit inverse mass.
        weight_start = 0.0 if start in (0, 4) else 1.0
        weight_end = 1.0
        correction = -constraint / (compliance + weight_start + weight_end)
        gradient = delta / length
        points[start] -= weight_start * correction * gradient
        points[end] += weight_end * correction * gradient
    logger.debug("Link-length projection completed: max_error=%.6g", maximum_correction)


def _xpbd_iteration(
    points: np.ndarray,
    config: SafetyFilterConfig,
    multipliers: np.ndarray,
    lengths: np.ndarray,
) -> None:
    """One capsule constraint pass following paper Algorithm 10."""
    starts, ends, radii = _capsule_arrays(points, config)
    active_constraints = 0
    assert config.collision_pairs is not None
    for pair_index, (first, second) in enumerate(config.collision_pairs):
        distance, normal, _, _, parameter_a, parameter_b = _capsule_contact_unchecked(
            starts[first],
            ends[first],
            starts[second],
            ends[second],
            radii[first],
            radii[second],
        )
        if distance >= config.release_distance or (
            distance >= config.activation_distance and multipliers[pair_index] == 0.0
        ):
            multipliers[pair_index] = 0.0
            continue
        constraint = distance - config.minimum_distance
        if constraint >= 0.0:
            continue
        active_constraints += 1

        gradients: dict[int, np.ndarray] = {}
        for capsule_index, sign, parameter in (
            (first, -1.0, parameter_a),
            (second, 1.0, parameter_b),
        ):
            if capsule_index == 0:  # torso is fixed
                continue
            _, start, end, _ = _MOVING_CAPSULES[capsule_index - 1]
            gradients[start] = gradients.get(start, np.zeros(3)) + sign * (1.0 - parameter) * normal
            gradients[end] = gradients.get(end, np.zeros(3)) + sign * parameter * normal

        denominator = config.compliance
        for point_index, gradient in gradients.items():
            if point_index not in (0, 4):
                denominator += float(gradient @ gradient)
        old = multipliers[pair_index]
        increment = -(constraint + config.compliance * old) / max(denominator, EPS)
        new = max(0.0, old + increment)
        multipliers[pair_index] = new
        for point_index, gradient in gradients.items():
            if point_index not in (0, 4):
                points[point_index] += (new - old) * gradient
        starts, ends, radii = _capsule_arrays(points, config)

    _project_lengths(points, lengths, config.compliance)
    logger.debug("XPBD constraint pass completed: active_constraints=%d", active_constraints)


def recover_tool_orientation(current: np.ndarray, target_direction: np.ndarray) -> np.ndarray:
    """Paper Algorithm 12, preserving roll while changing tool pointing direction."""
    if not is_rotation_matrix(current):
        raise ValueError("current must be a valid SO(3) matrix")
    current_direction = current[:, 0]
    target = unit(target_direction)
    cosine = float(np.clip(current_direction @ target, -1.0, 1.0))
    axis = np.cross(current_direction, target)
    if np.linalg.norm(axis) <= EPS:
        if cosine > 0.0:
            return current.copy()
        axis = unit(np.cross(current_direction, [0.0, 1.0, 0.0]))
    return rot(axis, float(np.arccos(cosine))) @ current


def sew_safety_filter(
    q_left_current: np.ndarray,
    q_right_current: np.ndarray,
    q_left_desired: np.ndarray,
    q_right_desired: np.ndarray,
    *,
    forward_kinematics: Callable[[np.ndarray, np.ndarray], BimanualPose],
    solve_left: Callable[[np.ndarray, ArmPose], np.ndarray],
    solve_right: Callable[[np.ndarray, ArmPose], np.ndarray],
    config: SafetyFilterConfig,
) -> SafetyFilterResult:
    """Paper Algorithm 4 with robot-specific FK and SEW callbacks."""
    logger.debug("SEW safety filter started")
    logger.debug("Safety stage 1/6: desired forward kinematics and fast-path check")
    desired_pose = forward_kinematics(q_left_desired, q_right_desired)
    desired_points = desired_pose.points()
    desired_minimum = minimum_capsule_distance(desired_points, config)
    logger.debug("Desired-pose minimum capsule distance: %.6f m", desired_minimum)
    if desired_minimum >= config.minimum_distance:
        logger.debug(
            "SEW safety filter accepted desired pose on fast path: minimum_distance=%.6f m",
            desired_minimum,
        )
        return SafetyFilterResult(
            np.asarray(q_left_desired).copy(),
            np.asarray(q_right_desired).copy(),
            True,
            0,
            desired_minimum,
            SafetyFilterStatus.ACCEPTED,
        )
    logger.debug("Unsafe target detected; computing current-pose forward kinematics")
    current_pose = forward_kinematics(q_left_current, q_right_current)
    current_points = current_pose.points()
    logger.debug("Safety stage 2/6: continuous-time collision approximation")
    points = find_first_collision(current_points, desired_points, config)
    lengths = np.linalg.norm(points[[1, 2, 3, 5, 6, 7]] - points[[0, 1, 2, 4, 5, 6]], axis=1)
    assert config.collision_pairs is not None
    multipliers = np.zeros(len(config.collision_pairs))

    logger.debug("Safety stage 3/6: XPBD collision and link-length projection")
    minimum_distance = minimum_capsule_distance(points, config)
    used_iterations = 0
    for used_iterations in range(1, config.iterations + 1):
        if minimum_distance >= config.minimum_distance - config.tolerance:
            break
        _xpbd_iteration(points, config, multipliers, lengths)
        minimum_distance = minimum_capsule_distance(points, config)
        logger.debug(
            "XPBD iteration %d/%d: minimum_distance=%.6f m",
            used_iterations,
            config.iterations,
            minimum_distance,
        )

    if minimum_distance < config.minimum_distance - config.tolerance:
        logger.warning(
            "SEW safety filter failed to clear collision after %d iterations: distance=%.6f m; returning current pose",
            used_iterations,
            minimum_distance,
            SafetyFilterStatus.XPBD_FAILED,
        )
        return SafetyFilterResult(
            np.asarray(q_left_current).copy(),
            np.asarray(q_right_current).copy(),
            False,
            used_iterations,
            minimum_distance,
            SafetyFilterStatus.IK_FAILED,
        )

    logger.debug("Safety stage 4/6: recovering tool orientations")
    left_orientation = recover_tool_orientation(
        current_pose.left.tool_orientation, points[3] - points[2]
    )
    right_orientation = recover_tool_orientation(
        current_pose.right.tool_orientation, points[7] - points[6]
    )
    left_pose = ArmPose(*points[0:4], left_orientation)
    right_pose = ArmPose(*points[4:8], right_orientation)
    logger.debug("Safety stage 5/6: resolving adjusted SEW keypoints")
    try:
        q_left = solve_left(np.asarray(q_left_current), left_pose)
        q_right = solve_right(np.asarray(q_right_current), right_pose)
    except SEWMimicError as error:
        logger.warning("Adjusted SEW solve failed (%s); returning current pose", error)
        logger.debug("Adjusted SEW solve traceback", exc_info=True)
        return SafetyFilterResult(
            np.asarray(q_left_current).copy(),
            np.asarray(q_right_current).copy(),
            False,
            used_iterations,
            minimum_distance,
        )
    logger.debug("Safety stage 6/6: validating reconstructed robot pose")
    reconstructed = forward_kinematics(q_left, q_right)
    reconstructed_distance = minimum_capsule_distance(reconstructed.points(), config)
    if reconstructed_distance < config.minimum_distance - config.tolerance:
        logger.warning(
            "Reconstructed pose remains unsafe: distance=%.6f m; returning current pose",
            reconstructed_distance,
        )
        return SafetyFilterResult(
            np.asarray(q_left_current).copy(),
            np.asarray(q_right_current).copy(),
            False,
            used_iterations,
            reconstructed_distance,
            SafetyFilterStatus.VALIDATION_FAILED,
        )
    logger.debug(
        "SEW safety filter completed: iterations=%d minimum_distance=%.6f m",
        used_iterations,
        reconstructed_distance,
    )
    return SafetyFilterResult(
        q_left,
        q_right,
        True,
        used_iterations,
        reconstructed_distance,
        SafetyFilterStatus.CORRECTED,
    )

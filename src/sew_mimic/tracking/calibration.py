"""Offline rigid-reference and hand/tool calibration with explicit fit diagnostics.

Point pairs must describe the same physical landmarks in two coordinate frames,
after unit/basis conversion. Human and robot skeletons are not rigidly congruent
reference objects. These helpers neither infer correspondences nor apply a fit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import _array, _rotation


class CalibrationFitError(ValueError):
    """Observations cannot determine a stable fit or violate an accuracy limit."""


def _errors(values: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or np.any(values < 0):
        raise ValueError(f"{name} must be a nonempty vector of nonnegative residuals")
    return _array(values, values.shape, name)


def _rms(values: np.ndarray) -> float:
    scale = float(np.max(values))
    return 0.0 if scale == 0 else scale * float(np.sqrt(np.mean((values / scale) ** 2)))


def _limit(value: float, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return float(value)


def _require_accuracy(errors: np.ndarray, rms_limit: float, peak_limit: float | None, unit: str):
    rms_limit = _limit(rms_limit, "RMS limit")
    if peak_limit is not None:
        peak_limit = _limit(peak_limit, "maximum residual limit")
    rms, peak = _rms(errors), float(np.max(errors))
    if rms > rms_limit or (peak_limit is not None and peak > peak_limit):
        raise CalibrationFitError(
            f"Calibration residuals exceed limits: RMS={rms:.6g}, max={peak:.6g} {unit}"
        )


@dataclass(frozen=True)
class RigidTransformFit:
    """A proper tracking-to-robot rotation/translation and per-point errors in metres."""

    rotation: np.ndarray
    translation: np.ndarray
    residuals: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "rotation", _rotation(self.rotation, "rotation"))
        object.__setattr__(self, "translation", _array(self.translation, (3,), "translation"))
        object.__setattr__(self, "residuals", _errors(self.residuals, "residuals"))

    @property
    def rms_error(self) -> float:
        return _rms(self.residuals)

    @property
    def max_error(self) -> float:
        return float(np.max(self.residuals))

    def require_accuracy(
        self, *, max_rms_error: float, max_point_error: float | None = None
    ) -> None:
        """Reject a fit outside application-chosen metre limits before applying it."""
        _require_accuracy(self.residuals, max_rms_error, max_point_error, "m")


@dataclass(frozen=True)
class HandToolRotationFit:
    """A local hand-to-tool rotation and per-observation angular errors in radians."""

    rotation: np.ndarray
    angular_residuals: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "rotation", _rotation(self.rotation, "rotation"))
        errors = _errors(self.angular_residuals, "angular_residuals")
        if np.any(errors > np.pi):
            raise ValueError("angular_residuals must not exceed pi")
        object.__setattr__(self, "angular_residuals", errors)

    @property
    def rms_angle(self) -> float:
        return _rms(self.angular_residuals)

    @property
    def max_angle(self) -> float:
        return float(np.max(self.angular_residuals))

    def require_accuracy(self, *, max_rms_angle: float, max_angle: float | None = None) -> None:
        """Reject a fit outside application-chosen radian limits before applying it."""
        _require_accuracy(self.angular_residuals, max_rms_angle, max_angle, "rad")


def _points(value: np.ndarray, name: str) -> np.ndarray:
    points = np.asarray(value, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 3:
        raise ValueError(f"{name} must have shape (N, 3), with at least three corresponding points")
    return _array(points, points.shape, name)


def _proper_rotation(matrix: np.ndarray) -> np.ndarray:
    """Closest SO(3) matrix, rejecting an ambiguous least-squares optimum."""
    u, singular, vt = np.linalg.svd(matrix)
    parity = 1.0 if np.linalg.det(u @ vt) > 0 else -1.0
    if singular[0] < 1e-12 or singular[1] + parity * singular[2] <= 1e-8 * singular[0]:
        raise CalibrationFitError(
            "Calibration observations do not determine a unique stable rotation"
        )
    correction = np.array([1.0, 1.0, parity])
    return (u * correction) @ vt


def _center(
    points: np.ndarray, min_spread: float, name: str
) -> tuple[np.ndarray, np.ndarray, float]:
    # Subtract a nearby reference before averaging to retain small spatial differences.
    offsets = points - points[0]
    mean_offset = np.mean(offsets, axis=0)
    centered = offsets - mean_offset
    scale = float(np.max(np.abs(centered)))
    if scale == 0:
        raise CalibrationFitError(f"{name} have no spatial spread")
    singular = np.linalg.svd(centered / scale, compute_uv=False) / np.sqrt(len(points))
    if singular[1] * scale < min_spread or singular[1] <= 1e-8 * singular[0]:
        raise CalibrationFitError(f"{name} are collinear or have insufficient spatial spread")
    return points[0] + mean_offset, centered, scale


def fit_rigid_transform(
    tracking_points: np.ndarray,
    robot_points: np.ndarray,
    *,
    min_spread: float = 1e-3,
) -> RigidTransformFit:
    """Fit ``p_robot = rotation @ p_tracking + translation`` by least squares.

    Each row pair identifies the same landmark, in metres. At least three
    noncollinear points are needed; planar references are valid. min_spread is
    the minimum RMS spread along the second principal axis of each point set.
    The fit uses a proper rotation and fixed unit scale, with no outlier removal.
    Inspect residuals and call require_accuracy before constructing a calibration.
    """
    source = _points(tracking_points, "tracking_points")
    target = _points(robot_points, "robot_points")
    if source.shape != target.shape:
        raise ValueError("tracking_points and robot_points must have matching shapes")
    min_spread = _limit(min_spread, "min_spread")
    if min_spread == 0:
        raise ValueError("min_spread must be positive")
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            source_mean, source_centered, source_scale = _center(
                source, min_spread, "tracking_points"
            )
            target_mean, target_centered, target_scale = _center(target, min_spread, "robot_points")
            covariance = (
                (target_centered / target_scale).T @ (source_centered / source_scale)
            ) / len(source)
            rotation = _proper_rotation(covariance)
            translation = target_mean - rotation @ source_mean
            residuals = np.linalg.norm(source @ rotation.T + translation - target, axis=1)
            return RigidTransformFit(rotation, translation, residuals)
    except (FloatingPointError, np.linalg.LinAlgError) as exc:
        raise CalibrationFitError(
            "Point calibration failed numerically; check units and coordinates"
        ) from exc


def _orientations(value: np.ndarray, name: str) -> np.ndarray:
    matrices = np.asarray(value, dtype=np.float64)
    if matrices.ndim != 3 or matrices.shape[1:] != (3, 3) or not len(matrices):
        raise ValueError(f"{name} must have shape (N, 3, 3), with at least one paired orientation")
    return np.stack([_rotation(matrix, name) for matrix in matrices])


def fit_hand_tool_rotation(
    tracking_hand_orientations: np.ndarray,
    robot_tool_orientations: np.ndarray,
    *,
    tracking_to_robot_rotation: np.ndarray,
) -> HandToolRotationFit:
    """Fit a fixed local offset with the world-frame calibration held constant.

    Minimize the sum of squared matrix residuals for
    ``R_robot_tool = R_robot_tracking @ R_tracking_hand @ R_hand_tool``.
    Rows are synchronized orientation pairs; fit each hand separately. Reported
    errors are geodesic angles, although the fit minimizes chordal matrix error.
    This does not solve for two unknown world/tool transforms simultaneously.
    """
    hands = _orientations(tracking_hand_orientations, "tracking_hand_orientations")
    tools = _orientations(robot_tool_orientations, "robot_tool_orientations")
    world = _rotation(tracking_to_robot_rotation, "tracking_to_robot_rotation")
    if hands.shape != tools.shape:
        raise ValueError("hand and tool orientation batches must have matching shapes")
    aligned_hands = world @ hands
    relative = np.swapaxes(aligned_hands, 1, 2) @ tools
    try:
        rotation = _proper_rotation(np.mean(relative, axis=0))
    except np.linalg.LinAlgError as exc:
        raise CalibrationFitError("Hand/tool calibration failed numerically") from exc
    errors = np.swapaxes(aligned_hands @ rotation, 1, 2) @ tools
    sine = 0.5 * np.linalg.norm(
        np.column_stack(
            [
                errors[:, 2, 1] - errors[:, 1, 2],
                errors[:, 0, 2] - errors[:, 2, 0],
                errors[:, 1, 0] - errors[:, 0, 1],
            ]
        ),
        axis=1,
    )
    cosine = np.clip((np.trace(errors, axis1=1, axis2=2) - 1) / 2, -1.0, 1.0)
    return HandToolRotationFit(rotation, np.arctan2(sine, cosine))

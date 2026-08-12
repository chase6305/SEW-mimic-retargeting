"""Reusable numerical and rotation utilities for SEW-Mimic."""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence

import numpy as np

EPS = 1e-10
TWO_PI = 2.0 * math.pi


def configure_logging(level: int | str = logging.INFO) -> None:
    """Configure concise stage logging for command-line applications."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )


class SEWMimicError(RuntimeError):
    """Base exception for SEW-Mimic."""


class DegenerateGeometryError(SEWMimicError):
    """Raised for an ill-posed geometric operation."""


class JointLimitError(SEWMimicError):
    """Raised when no equivalent analytical solution obeys joint limits."""


def as_vec3(value: Sequence[float]) -> np.ndarray:
    """Convert an array-like value to a finite float64 3-vector."""
    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    if vector.shape != (3,):
        raise ValueError(f"Expected a 3-vector, got shape {vector.shape}")
    if not np.all(np.isfinite(vector)):
        raise ValueError("Expected a finite 3-vector")
    return vector


def unit(value: Sequence[float], eps: float = EPS) -> np.ndarray:
    """Return a unit vector, rejecting near-zero inputs."""
    vector = as_vec3(value)
    norm = float(np.linalg.norm(vector))
    if norm < eps:
        raise DegenerateGeometryError("Cannot normalize a near-zero vector")
    return vector / norm


def skew(value: Sequence[float]) -> np.ndarray:
    """Return the 3-by-3 cross-product matrix of a vector."""
    x, y, z = as_vec3(value)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)


def rot(axis: Sequence[float], theta: float) -> np.ndarray:
    """Return the Rodrigues rotation ``R(axis, theta)``."""
    axis = unit(axis)
    axis_skew = skew(axis)
    return (
        np.eye(3) + math.sin(theta) * axis_skew + (1.0 - math.cos(theta)) * (axis_skew @ axis_skew)
    )


def wrap_to_pi(theta: float) -> float:
    """Wrap an angle to [-pi, pi]."""
    return math.atan2(math.sin(theta), math.cos(theta))


def is_rotation_matrix(rotation: np.ndarray, tol: float = 1e-7) -> bool:
    """Return whether an array is a finite SO(3) rotation matrix."""
    rotation = np.asarray(rotation, dtype=np.float64)
    if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
        return False
    return bool(
        np.linalg.norm(rotation.T @ rotation - np.eye(3), ord="fro") <= tol
        and abs(np.linalg.det(rotation) - 1.0) <= tol
    )


def equivalent_angles_in_limits(
    theta: float, q_min: float, q_max: float, reference: float
) -> list[float]:
    """Return all equivalent angles inside limits, nearest to reference first."""
    theta, q_min, q_max, reference = map(float, (theta, q_min, q_max, reference))
    if q_min > q_max:
        raise ValueError("q_min must be <= q_max")

    k_min = math.floor((q_min - theta) / TWO_PI) - 1
    k_max = math.ceil((q_max - theta) / TWO_PI) + 1
    values = [
        theta + TWO_PI * k
        for k in range(k_min, k_max + 1)
        if q_min - EPS <= theta + TWO_PI * k <= q_max + EPS
    ]
    values.sort(key=lambda angle: abs(angle - reference))
    return values


def make_frame(
    keypoint_left: Sequence[float],
    keypoint_right: Sequence[float],
    keypoint_bottom: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Create the body-centric frame from paper Algorithm 8."""
    left = as_vec3(keypoint_left)
    right = as_vec3(keypoint_right)
    bottom = as_vec3(keypoint_bottom)
    origin = 0.5 * (left + right)
    y_axis = unit(left - right)
    x_axis = unit(np.cross(y_axis, origin - bottom))
    z_axis = unit(np.cross(x_axis, y_axis))
    return np.column_stack((x_axis, y_axis, z_axis)), origin

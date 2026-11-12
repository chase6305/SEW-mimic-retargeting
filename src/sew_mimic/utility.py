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
    x, y, z = map(float, unit(axis))
    sine = math.sin(theta)
    complement = 1.0 - math.cos(theta)
    xy, xz, yz = complement * x * y, complement * x * z, complement * y * z
    sx, sy, sz = sine * x, sine * y, sine * z
    # Expand I + sin(theta) K + (1 - cos(theta)) K² directly, avoiding
    # temporary skew/identity matrices and a tiny matrix multiplication.
    return np.array(
        [
            [1.0 - complement * (y * y + z * z), xy - sz, xz + sy],
            [xy + sz, 1.0 - complement * (x * x + z * z), yz - sx],
            [xz - sy, yz + sx, 1.0 - complement * (x * x + y * y)],
        ],
        dtype=np.float64,
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


def _nearest_equivalent_angle(
    theta: float, q_min: float, q_max: float, reference: float
) -> float | None:
    """Select the first limit-valid equivalent without enumerating every turn.

    Inputs are finite and theta is a principal analytical angle. Four nearby
    turns suffice at ordinary floating-point resolution. Extremely large
    references can give many turns the same rounded distance; a logarithmic
    search then retains the public enumerator's smaller-turn tie preference.
    At enormous angles, distinct integer turns may themselves round to the
    same float; returned values retain that existing float64 limitation.
    """
    theta, q_min, q_max, reference = map(float, (theta, q_min, q_max, reference))
    if q_min > q_max:
        raise ValueError("q_min must be <= q_max")
    lower, upper = q_min - EPS, q_max + EPS
    # A sub-turn interval containing theta has exactly one equivalent angle.
    # Leave a margin so EPS-expanded endpoints near ±pi still compete normally.
    if upper - lower < TWO_PI - EPS and lower <= theta <= upper:
        return theta
    anchor = min(upper, max(lower, reference))
    base = math.floor((anchor - theta) / TWO_PI)
    if abs(base) <= 2**52:
        turns = range(base - 1, base + 3)
    else:
        # Adjacent integers may collapse to one float. Step to the next
        # representable integer turns instead of converting a huge index to int32.
        center = float(base)
        previous = math.floor(math.nextafter(center, -math.inf))
        following = math.ceil(math.nextafter(center, math.inf))
        last = math.ceil(math.nextafter(float(following), math.inf))
        first_allowed = math.floor((q_min - theta) / TWO_PI) - 1
        last_allowed = math.ceil((q_max - theta) / TWO_PI) + 1
        # An integer inside a very narrow interval can round to a representative
        # outside it. Check the enumerator's endpoints as well as representatives.
        turns = sorted(
            {
                turn
                for turn in (previous, base, following, last, first_allowed, last_allowed)
                if first_allowed <= turn <= last_allowed
            }
        )

    best = None
    best_cost = math.inf
    best_turn = base
    for turn in turns:
        value = theta + TWO_PI * turn
        if lower <= value <= upper:
            cost = abs(value - reference)
            if best is None or cost < best_cost:
                best, best_cost, best_turn = value, cost, turn
    if best is None:
        return None

    previous = theta + TWO_PI * (best_turn - 1)
    if lower <= previous <= upper and abs(previous - reference) == best_cost:
        # Rounded-distance ties can span more than the four local candidates.
        # On this side of the nearest turn, the predicate is monotone.
        first = math.floor((q_min - theta) / TWO_PI) - 1
        last = best_turn
        while first < last:
            middle = (first + last) // 2
            value = theta + TWO_PI * middle
            if value >= lower and abs(value - reference) <= best_cost:
                last = middle
            else:
                first = middle + 1
        best = theta + TWO_PI * first
    return best


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

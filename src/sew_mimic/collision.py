"""Capsule collision geometry used by the SEW-Mimic safety filter."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .utility import EPS, as_vec3

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Capsule:
    """A capsule represented by a line segment and a radius, in metres."""

    start: np.ndarray
    end: np.ndarray
    radius: float
    name: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "start", as_vec3(self.start))
        object.__setattr__(self, "end", as_vec3(self.end))
        if not np.isfinite(self.radius) or self.radius <= 0.0:
            raise ValueError("Capsule radius must be finite and positive")


@dataclass(frozen=True)
class CapsuleContact:
    """Signed capsule distance and closest-point information."""

    distance: float
    normal: np.ndarray
    point_a: np.ndarray
    point_b: np.ndarray
    parameter_a: float
    parameter_b: float

    @property
    def colliding(self) -> bool:
        return self.distance < 0.0


def closest_points_on_segments(
    a0: np.ndarray, a1: np.ndarray, b0: np.ndarray, b1: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Return closest points and parameters on two finite 3-D segments."""
    a0, a1, b0, b1 = map(as_vec3, (a0, a1, b0, b1))
    return _closest_points_on_segments_unchecked(a0, a1, b0, b1)


def _closest_points_on_segments_unchecked(
    a0: np.ndarray, a1: np.ndarray, b0: np.ndarray, b1: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Internal closest-point kernel for already validated float arrays."""
    u, v, w = a1 - a0, b1 - b0, a0 - b0
    aa, bb, cc = float(u @ u), float(u @ v), float(v @ v)
    dd, ee = float(u @ w), float(v @ w)

    if aa <= EPS and cc <= EPS:
        return a0, b0, 0.0, 0.0
    if aa <= EPS:
        s, t = 0.0, float(np.clip(ee / cc, 0.0, 1.0))
    elif cc <= EPS:
        s, t = float(np.clip(-dd / aa, 0.0, 1.0)), 0.0
    else:
        denominator = aa * cc - bb * bb
        s = 0.0 if denominator <= EPS else float(np.clip((bb * ee - cc * dd) / denominator, 0, 1))
        t = (bb * s + ee) / cc
        if t < 0.0:
            t, s = 0.0, float(np.clip(-dd / aa, 0.0, 1.0))
        elif t > 1.0:
            t, s = 1.0, float(np.clip((bb - dd) / aa, 0.0, 1.0))

    return a0 + s * u, b0 + t * v, s, t


def capsule_contact(capsule_a: Capsule, capsule_b: Capsule) -> CapsuleContact:
    """Compute signed distance; negative values indicate intersection."""
    distance, normal, point_a, point_b, parameter_a, parameter_b = _capsule_contact_unchecked(
        capsule_a.start,
        capsule_a.end,
        capsule_b.start,
        capsule_b.end,
        capsule_a.radius,
        capsule_b.radius,
    )

    return CapsuleContact(
        distance=distance,
        normal=normal,
        point_a=point_a,
        point_b=point_b,
        parameter_a=parameter_a,
        parameter_b=parameter_b,
    )


def _capsule_contact_unchecked(
    start_a: np.ndarray,
    end_a: np.ndarray,
    start_b: np.ndarray,
    end_b: np.ndarray,
    radius_a: float,
    radius_b: float,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Internal capsule kernel without allocations for input validation."""
    point_a, point_b, parameter_a, parameter_b = _closest_points_on_segments_unchecked(
        start_a, end_a, start_b, end_b
    )
    delta = point_b - point_a
    center_distance = float(np.sqrt(delta @ delta))
    if center_distance > EPS:
        normal = delta / center_distance
    else:
        # Deterministic fallback for coincident segment points.
        direction = np.cross(end_a - start_a, end_b - start_b)
        if direction @ direction <= EPS * EPS:
            direction = np.cross(end_a - start_a, [1.0, 0.0, 0.0])
        if direction @ direction <= EPS * EPS:
            direction = np.array([0.0, 1.0, 0.0])
        normal = direction / np.sqrt(direction @ direction)
    return (
        center_distance - radius_a - radius_b,
        normal,
        point_a + radius_a * normal,
        point_b - radius_b * normal,
        parameter_a,
        parameter_b,
    )


def _capsule_distances_unchecked(
    starts_a: np.ndarray,
    ends_a: np.ndarray,
    starts_b: np.ndarray,
    ends_b: np.ndarray,
    radii_a: np.ndarray,
    radii_b: np.ndarray,
) -> np.ndarray:
    """Vectorized signed distances for already validated capsule arrays."""
    u = ends_a - starts_a
    v = ends_b - starts_b
    w = starts_a - starts_b
    aa = np.einsum("ij,ij->i", u, u)
    bb = np.einsum("ij,ij->i", u, v)
    cc = np.einsum("ij,ij->i", v, v)
    dd = np.einsum("ij,ij->i", u, w)
    ee = np.einsum("ij,ij->i", v, w)
    s = np.zeros_like(aa)
    t = np.zeros_like(aa)

    a_valid = aa > EPS
    b_valid = cc > EPS
    only_b = ~a_valid & b_valid
    only_a = a_valid & ~b_valid
    both = a_valid & b_valid
    t[only_b] = np.clip(ee[only_b] / cc[only_b], 0.0, 1.0)
    s[only_a] = np.clip(-dd[only_a] / aa[only_a], 0.0, 1.0)

    denominator = aa * cc - bb * bb
    nonparallel = both & (denominator > EPS)
    s[nonparallel] = np.clip(
        (bb[nonparallel] * ee[nonparallel] - cc[nonparallel] * dd[nonparallel])
        / denominator[nonparallel],
        0.0,
        1.0,
    )
    t[both] = (bb[both] * s[both] + ee[both]) / cc[both]
    below = both & (t < 0.0)
    above = both & (t > 1.0)
    t[below] = 0.0
    s[below] = np.clip(-dd[below] / aa[below], 0.0, 1.0)
    t[above] = 1.0
    s[above] = np.clip((bb[above] - dd[above]) / aa[above], 0.0, 1.0)

    delta = (starts_b + t[:, None] * v) - (starts_a + s[:, None] * u)
    return np.sqrt(np.einsum("ij,ij->i", delta, delta)) - radii_a - radii_b


def capsule_from_oobb(mesh_path: str | Path, *, padding: float = 1.05, name: str = "") -> Capsule:
    """Approximate a mesh OOBB with a conservative capsule.

    ``trimesh`` is imported lazily because OOBB fitting is an offline robot
    configuration step, not part of the real-time safety-filter loop.
    """
    if padding < 1.0:
        raise ValueError("padding must be >= 1")
    try:
        import trimesh
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError("OOBB fitting requires the visualization dependencies") from exc

    mesh_path = Path(mesh_path)
    logger.debug("OOBB mesh loading started: %s", mesh_path)
    mesh = trimesh.load_mesh(mesh_path, process=False)
    if hasattr(mesh, "geometry"):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    box = mesh.bounding_box_oriented
    extents = np.asarray(box.primitive.extents, dtype=np.float64)
    transform = np.asarray(box.primitive.transform, dtype=np.float64)
    major = int(np.argmax(extents))
    radius = 0.5 * float(np.max(np.delete(extents, major))) * padding
    half_segment = max(0.0, 0.5 * float(extents[major]) * padding - radius)
    axis = transform[:3, major]
    center = transform[:3, 3]
    capsule = Capsule(center - half_segment * axis, center + half_segment * axis, radius, name)
    logger.debug(
        "OOBB capsule fitted: mesh=%s extents=%s radius=%.6f segment_length=%.6f",
        mesh_path.name,
        extents,
        radius,
        2.0 * half_segment,
    )
    return capsule

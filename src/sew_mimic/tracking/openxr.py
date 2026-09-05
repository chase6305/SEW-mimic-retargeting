"""Decode exported OpenXR joint snapshots without importing a headset SDK."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import IntFlag
from types import MappingProxyType

import numpy as np

from .model import JointSample, TrackingFrame, TrackingIdentity, _array, _index


class LocationFlags(IntFlag):
    ORIENTATION_VALID = 0x1
    POSITION_VALID = 0x2
    ORIENTATION_TRACKED = 0x4
    POSITION_TRACKED = 0x8


# OpenXR: +X right, +Y up, -Z forward. Canonical: +X forward, +Y left, +Z up.
OPENXR_TO_FLU = _array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]], (3, 3), "basis")
# Unity Transform exports have already changed convention; do not convert twice.
UNITY_TO_FLU = _array([[0, 0, 1], [-1, 0, 0], [0, 1, 0]], (3, 3), "basis")

# The upper/lower arm bone origins supply shoulder/elbow, not the clavicle joint.
FB_BODY_JOINT_MAP = MappingProxyType(
    {
        f"{side}_{semantic}": f"XR_BODY_JOINT_{side.upper()}_{bone}_FB"
        for side in ("left", "right")
        for semantic, bone in (
            ("shoulder", "ARM_UPPER"),
            ("elbow", "ARM_LOWER"),
            ("wrist", "HAND_WRIST"),
        )
    }
)


@dataclass(frozen=True)
class OpenXRJoint:
    """Raw SDK values; components without VALID flags may contain garbage.

    Quaternion order is explicitly (x, y, z, w). The decoder, rather than this
    transport container, validates and takes ownership of usable components.
    """

    position: np.ndarray | None
    orientation_xyzw: np.ndarray | None
    flags: int


def _quaternion_rotation(value: np.ndarray) -> np.ndarray:
    q = _array(value, (4,), "orientation_xyzw")
    scale = np.max(np.abs(q))
    if scale < 1e-12:
        raise ValueError("orientation_xyzw must have nonzero norm")
    q = q / scale
    x, y, z, w = q / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


class OpenXRAdapter:
    """Map named joints and coordinate conventions into canonical snapshots.

    This is a decoder, not an OpenXR runtime or network client. Applications
    enumerate extensions and acquire/synchronize SDK data outside this package.
    Both world and local joint coordinate bases are converted by B R B^T.
    """

    def __init__(
        self,
        joint_map: Mapping[str, str] = FB_BODY_JOINT_MAP,
        *,
        basis: np.ndarray = OPENXR_TO_FLU,
        metres_per_unit: float = 1.0,
    ) -> None:
        mapping = dict(joint_map)
        if not mapping or any(
            not isinstance(name, str) or not name or not isinstance(raw, str) or not raw
            for name, raw in mapping.items()
        ):
            raise ValueError("joint_map must map nonempty semantic names to SDK joint names")
        basis = _array(basis, (3, 3), "basis")
        if not np.allclose(basis @ basis.T, np.eye(3), atol=1e-8, rtol=0):
            raise ValueError("basis must be orthogonal (a reflection is allowed)")
        if not np.isfinite(metres_per_unit) or metres_per_unit <= 0:
            raise ValueError("metres_per_unit must be finite and positive")
        self.joint_map = MappingProxyType(mapping)
        self.basis = basis
        self.metres_per_unit = float(metres_per_unit)

    def decode(
        self,
        samples: Mapping[str, OpenXRJoint],
        *,
        identity: TrackingIdentity,
        sequence: int,
        sample_time: float,
        received_time: float,
        active: bool = True,
        confidence: float | None = None,
    ) -> TrackingFrame:
        """Decode only valid components; missing joints remain explicitly absent."""
        if not isinstance(active, (bool, np.bool_)):
            raise ValueError("active must be a boolean")
        if not active:
            # Inactive SDK output cannot supply joint data, even if its storage
            # still contains old VALID bits. Deliver loss without reading it.
            return TrackingFrame(identity, sequence, sample_time, received_time, {}, False)
        joints = {}
        for name, raw_name in self.joint_map.items():
            if raw_name not in samples:
                continue
            raw = samples[raw_name]
            flags = LocationFlags(_index(raw.flags, "location flags"))
            position = orientation = None
            if flags & LocationFlags.POSITION_VALID:
                position = self.metres_per_unit * (self.basis @ _array(raw.position, (3,), name))
            if flags & LocationFlags.ORIENTATION_VALID:
                orientation = self.basis @ _quaternion_rotation(raw.orientation_xyzw) @ self.basis.T
            joints[name] = JointSample(
                position,
                orientation,
                position is not None and bool(flags & LocationFlags.POSITION_TRACKED),
                orientation is not None and bool(flags & LocationFlags.ORIENTATION_TRACKED),
            )
        return TrackingFrame(
            identity, sequence, sample_time, received_time, joints, active, confidence
        )

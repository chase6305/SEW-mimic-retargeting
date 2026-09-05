"""Owned tracking snapshots, independent of device SDKs and network transports."""

from __future__ import annotations

import operator
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from ..utility import is_rotation_matrix


def _array(value: np.ndarray, shape: tuple[int, ...], name: str) -> np.ndarray:
    result = np.array(value, dtype=np.float64, copy=True)
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite array of shape {shape}")
    result.flags.writeable = False
    return result


def _rotation(value: np.ndarray, name: str) -> np.ndarray:
    result = _array(value, (3, 3), name)
    if not is_rotation_matrix(result):
        raise ValueError(f"{name} must be a valid SO(3) matrix")
    return result


def _index(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a nonnegative integer")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a nonnegative integer") from exc
    if result < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return result


@dataclass(frozen=True)
class TrackingIdentity:
    """Calibration scope; increment revision whenever the reference space changes.

    A reconnect must use a new session ID, even for the same physical device.
    """

    source_id: str
    session_id: str
    space_id: str
    space_revision: int = 0

    def __post_init__(self) -> None:
        for name in ("source_id", "session_id", "space_id"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be a nonempty string")
        object.__setattr__(self, "space_revision", _index(self.space_revision, "space_revision"))


@dataclass(frozen=True)
class JointSample:
    """A joint in metres and a common right-handed tracking frame.

    Missing/invalid components are None. Runtime TRACKED flags are preserved
    separately from validity; they do not establish that a joint was directly
    measured rather than inferred by a body-tracking model.
    """

    position: np.ndarray | None = None
    orientation: np.ndarray | None = None
    position_tracked: bool = False
    orientation_tracked: bool = False

    def __post_init__(self) -> None:
        if self.position is not None:
            object.__setattr__(self, "position", _array(self.position, (3,), "position"))
        if self.orientation is not None:
            object.__setattr__(self, "orientation", _rotation(self.orientation, "orientation"))
        for component in ("position", "orientation"):
            tracked = getattr(self, f"{component}_tracked")
            if not isinstance(tracked, (bool, np.bool_)):
                raise ValueError(f"{component}_tracked must be a boolean")
            if tracked and getattr(self, component) is None:
                raise ValueError(f"A tracked {component} must be valid")


@dataclass(frozen=True)
class TrackingFrame:
    """One snapshot with timestamps in the receiver's monotonic clock domain.

    sample_time is the located pose's time, after clock conversion/synchronization;
    received_time is assigned locally on receipt, never trusted from a packet.
    confidence is optional body-level confidence, not invented per-joint quality.
    """

    identity: TrackingIdentity
    sequence: int
    sample_time: float
    received_time: float
    joints: Mapping[str, JointSample]
    active: bool = True
    confidence: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.identity, TrackingIdentity):
            raise ValueError("identity must be a TrackingIdentity")
        object.__setattr__(self, "sequence", _index(self.sequence, "sequence"))
        for name in ("sample_time", "received_time"):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        if not isinstance(self.active, (bool, np.bool_)):
            raise ValueError("active must be a boolean")
        if self.confidence is not None:
            value = float(self.confidence)
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError("confidence must be between zero and one, or None")
            object.__setattr__(self, "confidence", value)
        joints = dict(self.joints)
        if any(
            not isinstance(name, str) or not name or not isinstance(joint, JointSample)
            for name, joint in joints.items()
        ):
            raise ValueError("joints must map nonempty semantic names to JointSample objects")
        object.__setattr__(self, "joints", MappingProxyType(joints))

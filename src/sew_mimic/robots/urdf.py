"""Small dependency-free URDF forward-kinematics helper."""

from __future__ import annotations

import logging
import math
import xml.etree.ElementTree as ET
from collections.abc import Collection
from functools import lru_cache
from pathlib import Path

import numpy as np

from ..utility import skew, unit

logger = logging.getLogger(__name__)
_IDENTITY_3 = np.eye(3)
_IDENTITY_3.flags.writeable = False
JointRecord = tuple[str, str, str, str, np.ndarray, np.ndarray]


def rpy_rotation(rpy: np.ndarray) -> np.ndarray:
    """URDF fixed-axis RPY rotation, ``Rz(yaw) @ Ry(pitch) @ Rx(roll)``."""
    roll, pitch, yaw = np.asarray(rpy, dtype=np.float64)
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def _numbers(value: str | None, default: str = "0 0 0") -> np.ndarray:
    return np.fromstring(default if value is None else value, sep=" ", dtype=np.float64)


def origin_transform(element: ET.Element) -> np.ndarray:
    origin = element.find("origin")
    transform = np.eye(4)
    if origin is not None:
        transform[:3, :3] = rpy_rotation(_numbers(origin.get("rpy")))
        transform[:3, 3] = _numbers(origin.get("xyz"))
    return transform


class URDFKinematics:
    """Forward kinematics for fixed, revolute, continuous and prismatic joints."""

    def __init__(self, urdf_path: str | Path) -> None:
        self.path = Path(urdf_path).expanduser().resolve()
        root = ET.parse(self.path).getroot()
        links = {str(link.get("name")) for link in root.findall("link")}
        self.joints: list[JointRecord] = []
        child_links = set()
        for joint in root.findall("joint"):
            parent = joint.find("parent")
            child = joint.find("child")
            if parent is None or child is None:
                continue
            name = str(joint.get("name"))
            parent_name, child_name = str(parent.get("link")), str(child.get("link"))
            axis = joint.find("axis")
            self.joints.append(
                (
                    name,
                    str(joint.get("type")),
                    parent_name,
                    child_name,
                    origin_transform(joint),
                    _numbers(None if axis is None else axis.get("xyz"), "1 0 0"),
                )
            )
            child_links.add(child_name)
        roots = links - child_links
        if len(roots) != 1:
            raise ValueError(f"Expected one URDF root link, found {sorted(roots)}")
        self.root_link = roots.pop()
        # Resolve topology and Rodrigues constants once instead of rebuilding
        # both for every real-time FK call.
        ordered = []
        resolved_links = {self.root_link}
        pending = self.joints
        while pending:
            remaining = []
            for joint in pending:
                if joint[2] in resolved_links:
                    ordered.append(joint)
                    resolved_links.add(joint[3])
                else:
                    remaining.append(joint)
            if len(remaining) == len(pending):
                missing = sorted({joint[2] for joint in remaining})
                raise ValueError(f"Could not traverse URDF joint tree; missing parents: {missing}")
            pending = remaining
        self.joints = ordered
        self.joint_child_links = {name: child for name, _, _, child, _, _ in self.joints}
        self._joint_by_child = {child: joint for joint in self.joints for child in (joint[3],)}
        self._rotation_terms: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._prismatic_axes: dict[str, np.ndarray] = {}
        for name, joint_type, _, _, _, axis in self.joints:
            if joint_type in ("revolute", "continuous"):
                axis_skew = skew(unit(axis))
                self._rotation_terms[name] = (axis_skew, axis_skew @ axis_skew)
            elif joint_type == "prismatic":
                normalized_axis = unit(axis)
                normalized_axis.flags.writeable = False
                self._prismatic_axes[name] = normalized_axis
        logger.info(
            "URDF kinematics loaded: path=%s root=%s joints=%d",
            self.path,
            self.root_link,
            len(self.joints),
        )

    @lru_cache(maxsize=32)
    def _required_joint_names(self, required_links: frozenset[str]) -> frozenset[str]:
        """Return the ancestor-joint closure needed to resolve target links."""
        unknown = required_links - {self.root_link} - set(self._joint_by_child)
        if unknown:
            raise KeyError(f"Unknown URDF links: {sorted(unknown)}")
        names: set[str] = set()
        for target in required_links:
            link = target
            while link != self.root_link:
                joint = self._joint_by_child[link]
                names.add(joint[0])
                link = joint[2]
        return frozenset(names)

    @lru_cache(maxsize=32)
    def _joint_plan(self, required_links: frozenset[str]) -> tuple[JointRecord, ...]:
        """Compile the ordered FK operations needed for a set of target links."""
        required_names = self._required_joint_names(required_links)
        return tuple(joint for joint in self.joints if joint[0] in required_names)

    def link_transforms(
        self,
        joint_positions: dict[str, float],
        required_links: Collection[str] | None = None,
    ) -> dict[str, np.ndarray]:
        """Compute FK, optionally pruning branches unrelated to target links.

        When ``required_links`` is provided, the result contains the URDF root
        and links on the targets' ancestor chains. This avoids evaluating hand,
        sensor, and other fixed branches in high-rate control loops.
        """
        logger.debug("URDF FK started: commanded_joints=%d", len(joint_positions))
        joint_plan = (
            self.joints if required_links is None else self._joint_plan(frozenset(required_links))
        )
        transforms = {self.root_link: np.eye(4)}
        for name, joint_type, parent, child, origin, axis in joint_plan:
            if joint_type in ("revolute", "continuous"):
                theta = float(joint_positions.get(name, 0.0))
                axis_skew, axis_skew_squared = self._rotation_terms[name]
                rotation = (
                    _IDENTITY_3
                    + math.sin(theta) * axis_skew
                    + (1.0 - math.cos(theta)) * axis_skew_squared
                )
                transform = origin.copy()
                transform[:3, :3] = origin[:3, :3] @ rotation
                transforms[child] = transforms[parent] @ transform
            elif joint_type == "prismatic":
                transform = origin.copy()
                transform[:3, 3] += origin[:3, :3] @ (
                    float(joint_positions.get(name, 0.0)) * self._prismatic_axes[name]
                )
                transforms[child] = transforms[parent] @ transform
            else:
                transforms[child] = transforms[parent] @ origin
        logger.debug("URDF FK completed: resolved_links=%d", len(transforms))
        return transforms

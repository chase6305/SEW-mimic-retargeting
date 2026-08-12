"""Small dependency-free URDF forward-kinematics helper."""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from ..utility import skew, unit

logger = logging.getLogger(__name__)


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
        self.joints: list[tuple[str, str, str, str, np.ndarray, np.ndarray]] = []
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
        self._rotation_terms: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for name, joint_type, _, _, _, axis in self.joints:
            if joint_type in ("revolute", "continuous"):
                axis_skew = skew(unit(axis))
                self._rotation_terms[name] = (axis_skew, axis_skew @ axis_skew)
        logger.info(
            "URDF kinematics loaded: path=%s root=%s joints=%d",
            self.path,
            self.root_link,
            len(self.joints),
        )

    def link_transforms(self, joint_positions: dict[str, float]) -> dict[str, np.ndarray]:
        logger.debug("URDF FK started: commanded_joints=%d", len(joint_positions))
        transforms = {self.root_link: np.eye(4)}
        for name, joint_type, parent, child, origin, axis in self.joints:
            if joint_type in ("revolute", "continuous"):
                theta = float(joint_positions.get(name, 0.0))
                axis_skew, axis_skew_squared = self._rotation_terms[name]
                rotation = (
                    np.eye(3)
                    + np.sin(theta) * axis_skew
                    + (1.0 - np.cos(theta)) * axis_skew_squared
                )
                transform = origin.copy()
                transform[:3, :3] = origin[:3, :3] @ rotation
                transforms[child] = transforms[parent] @ transform
            elif joint_type == "prismatic":
                transform = origin.copy()
                transform[:3, 3] += origin[:3, :3] @ (
                    float(joint_positions.get(name, 0.0)) * unit(axis)
                )
                transforms[child] = transforms[parent] @ transform
            else:
                transforms[child] = transforms[parent] @ origin
        logger.debug("URDF FK completed: resolved_links=%d", len(transforms))
        return transforms

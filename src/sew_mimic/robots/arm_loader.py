"""Shared URDF extraction for serial seven-DoF SEW robot arms."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from ..solver import Serial7DoF
from .urdf import origin_transform


@dataclass(frozen=True)
class LoadedSerialArm:
    """Robot-independent result of extracting one seven-joint URDF chain."""

    robot: Serial7DoF
    base_link: str
    ee_link: str


@dataclass(frozen=True)
class SerialRobotArm:
    """Default concrete :class:`RobotArm` returned by declarative specs."""

    side: Literal["left", "right"]
    robot: Serial7DoF
    joint_names: tuple[str, ...]
    base_link: str
    ee_link: str


@dataclass(frozen=True)
class SerialArmSpec:
    """Declarative left/right joint and tool-link configuration for one robot."""

    left_joint_names: tuple[str, ...]
    right_joint_names: tuple[str, ...]
    left_ee_link: str
    right_ee_link: str

    def __post_init__(self) -> None:
        for side, names in (
            ("left", self.left_joint_names),
            ("right", self.right_joint_names),
        ):
            normalized = tuple(names)
            if len(normalized) != 7 or len(set(normalized)) != 7:
                raise ValueError(f"{side}_joint_names must contain seven unique joints")
            object.__setattr__(self, f"{side}_joint_names", normalized)
        if not self.left_ee_link or not self.right_ee_link:
            raise ValueError("left_ee_link and right_ee_link must not be empty")

    def load(
        self,
        urdf_path: str | Path,
        side: Literal["left", "right"] = "left",
        *,
        R_align: np.ndarray | None = None,
    ) -> SerialRobotArm:
        """Load one configured side through :func:`load_serial_7dof_arm`."""
        if side not in ("left", "right"):
            raise ValueError("side must be 'left' or 'right'")
        joint_names = self.left_joint_names if side == "left" else self.right_joint_names
        ee_link = self.left_ee_link if side == "left" else self.right_ee_link
        loaded = load_serial_7dof_arm(
            urdf_path,
            joint_names,
            ee_link,
            R_align=R_align,
        )
        return SerialRobotArm(
            side=side,
            robot=loaded.robot,
            joint_names=joint_names,
            base_link=loaded.base_link,
            ee_link=loaded.ee_link,
        )


def _joint_axis(joint: ET.Element) -> np.ndarray:
    axis = joint.find("axis")
    value = "1 0 0" if axis is None or axis.get("xyz") is None else axis.get("xyz")
    return np.fromstring(value, sep=" ", dtype=np.float64)


def _fixed_tool_rotation(joints: list[ET.Element], start_link: str, target_link: str) -> np.ndarray:
    """Accumulate the fixed-joint rotation from joint seven to a tool link."""
    outgoing: dict[str, list[ET.Element]] = {}
    for joint in joints:
        parent = joint.find("parent")
        if joint.get("type") == "fixed" and parent is not None:
            outgoing.setdefault(str(parent.get("link")), []).append(joint)
    pending = [(start_link, np.eye(3))]
    visited = set()
    while pending:
        link, rotation = pending.pop()
        if link == target_link:
            return rotation
        if link in visited:
            continue
        visited.add(link)
        for joint in outgoing.get(link, []):
            child = joint.find("child")
            if child is not None:
                pending.append(
                    (
                        str(child.get("link")),
                        rotation @ origin_transform(joint)[:3, :3],
                    )
                )
    raise ValueError(f"No fixed tool chain from {start_link} to {target_link}")


def load_serial_7dof_arm(
    urdf_path: str | Path,
    joint_names: tuple[str, ...],
    ee_link: str,
    *,
    R_align: np.ndarray | None = None,
) -> LoadedSerialArm:
    """Extract and validate a serial seven-revolute-joint model from a URDF.

    ``joint_names`` defines solver order from shoulder to wrist. ``ee_link``
    must be the seventh joint's child or be reachable from it using only fixed
    joints. Continuous joints receive the conventional ``[-pi, pi]`` limits.
    """
    if len(joint_names) != 7 or len(set(joint_names)) != 7:
        raise ValueError("joint_names must contain seven unique joints")
    path = Path(urdf_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Robot URDF not found: {path}")
    root = ET.parse(path).getroot()
    all_joints = root.findall("joint")
    by_name = {str(joint.get("name")): joint for joint in all_joints}

    chain = []
    for name in joint_names:
        joint = by_name.get(name)
        if joint is None:
            raise ValueError(f"Required arm joint is missing: {name}")
        if joint.get("type") not in ("revolute", "continuous"):
            raise ValueError(f"{name} must be revolute/continuous, got {joint.get('type')}")
        chain.append(joint)
    for first, second in zip(chain, chain[1:]):
        child, parent = first.find("child"), second.find("parent")
        if child is None or parent is None or child.get("link") != parent.get("link"):
            raise ValueError(
                f"Arm chain is discontinuous between {first.get('name')} and {second.get('name')}"
            )

    axes, rotations, lower, upper = [], [], [], []
    for joint in chain:
        axes.append(_joint_axis(joint))
        rotations.append(origin_transform(joint)[:3, :3])
        limit = joint.find("limit")
        if joint.get("type") == "continuous":
            lower.append(-np.pi)
            upper.append(np.pi)
        elif limit is None or limit.get("lower") is None or limit.get("upper") is None:
            raise ValueError(f"Joint limits are missing for {joint.get('name')}")
        else:
            lower.append(float(limit.get("lower")))
            upper.append(float(limit.get("upper")))

    first_parent = chain[0].find("parent")
    last_child = chain[-1].find("child")
    assert first_parent is not None and last_child is not None
    tool_rotation = _fixed_tool_rotation(all_joints, str(last_child.get("link")), ee_link)
    return LoadedSerialArm(
        robot=Serial7DoF(
            axes_local=np.asarray(axes),
            R_local=np.asarray(rotations),
            q_min=np.asarray(lower),
            q_max=np.asarray(upper),
            R_7T_local=tool_rotation,
            R_align=np.eye(3) if R_align is None else R_align,
        ),
        base_link=str(first_parent.get("link")),
        ee_link=ee_link,
    )

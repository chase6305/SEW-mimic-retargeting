"""URDF adapter for the Marvin M6 seven-DoF arms.

The SEW solver works in an arm-base frame and only needs joint orientations.
This module extracts those orientations, axes, limits, and the fixed EE
orientation directly from the robot URDF.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from ..solver import Serial7DoF

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MARVIN_URDF = REPOSITORY_ROOT / "assets" / "Marvin_M6_S_CCS_696_V4.0" / "robot_with_ee.urdf"

_JOINT_STEMS = (
    "SHOULDER_PITCH",
    "SHOULDER_ROLL",
    "ELBOW_PITCH",
    "ELBOW_YAW",
    "WRIST_PITCH",
    "WRIST_YAW",
    "WRIST_ROLL",
)


@dataclass(frozen=True)
class MarvinArm:
    """A parsed SEW model plus the URDF names needed by a visualizer."""

    side: Literal["left", "right"]
    robot: Serial7DoF
    joint_names: tuple[str, ...]
    base_link: str
    ee_link: str


def _numbers(value: str | None, default: str) -> np.ndarray:
    return np.fromstring(default if value is None else value, sep=" ", dtype=np.float64)


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


def _origin_rotation(joint: ET.Element) -> np.ndarray:
    origin = joint.find("origin")
    return rpy_rotation(_numbers(None if origin is None else origin.get("rpy"), "0 0 0"))


def load_marvin_arm(
    urdf_path: str | Path = DEFAULT_MARVIN_URDF,
    side: Literal["left", "right"] = "left",
    *,
    R_align: np.ndarray | None = None,
) -> MarvinArm:
    """Build a :class:`Serial7DoF` from either arm in ``robot_with_ee.urdf``.

    Human keypoints and hand orientation passed to :func:`sew_mimic` must be
    represented in the selected arm's ``*_arm_base`` frame. ``R_align`` can be
    supplied when the human hand-frame convention differs from ``*_ee``.
    """
    if side not in ("left", "right"):
        raise ValueError("side must be 'left' or 'right'")

    path = Path(urdf_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Marvin URDF not found: {path}")
    root = ET.parse(path).getroot()
    by_name = {joint.get("name"): joint for joint in root.findall("joint")}
    suffix = "L" if side == "left" else "R"
    joint_names = tuple(f"{stem}_{suffix}_J{i}" for i, stem in enumerate(_JOINT_STEMS, 1))

    joints: list[ET.Element] = []
    for name in joint_names:
        joint = by_name.get(name)
        if joint is None:
            raise ValueError(f"Required {side} arm joint is missing: {name}")
        if joint.get("type") not in ("revolute", "continuous"):
            raise ValueError(f"{name} must be revolute/continuous, got {joint.get('type')}")
        joints.append(joint)

    # Guard against silently accepting the right names in the wrong topology.
    for parent_joint, child_joint in zip(joints, joints[1:]):
        parent_child = parent_joint.find("child")
        child_parent = child_joint.find("parent")
        if (
            parent_child is None
            or child_parent is None
            or parent_child.get("link") != child_parent.get("link")
        ):
            raise ValueError(
                f"Arm chain is discontinuous between {parent_joint.get('name')} and {child_joint.get('name')}"
            )

    axes = []
    rotations = []
    q_min = []
    q_max = []
    for joint in joints:
        axis = joint.find("axis")
        limit = joint.find("limit")
        axes.append(_numbers(None if axis is None else axis.get("xyz"), "1 0 0"))
        rotations.append(_origin_rotation(joint))
        if joint.get("type") == "continuous":
            q_min.append(-np.pi)
            q_max.append(np.pi)
        elif limit is None or limit.get("lower") is None or limit.get("upper") is None:
            raise ValueError(f"Joint limits are missing for {joint.get('name')}")
        else:
            q_min.append(float(limit.get("lower")))
            q_max.append(float(limit.get("upper")))

    ee_name = "LEFT_EE" if side == "left" else "RIGHT_EE"
    ee_joint = by_name.get(ee_name)
    if ee_joint is None or ee_joint.get("type") != "fixed":
        raise ValueError(f"Required fixed end-effector joint is missing: {ee_name}")
    last_child = joints[-1].find("child")
    ee_parent = ee_joint.find("parent")
    if last_child is None or ee_parent is None or last_child.get("link") != ee_parent.get("link"):
        raise ValueError(f"{ee_name} is not attached to {joint_names[-1]}")

    robot = Serial7DoF(
        axes_local=np.asarray(axes),
        R_local=np.asarray(rotations),
        q_min=np.asarray(q_min),
        q_max=np.asarray(q_max),
        R_7T_local=_origin_rotation(ee_joint),
        R_align=np.eye(3) if R_align is None else R_align,
    )
    base_parent = joints[0].find("parent")
    ee_child = ee_joint.find("child")
    assert base_parent is not None and ee_child is not None
    return MarvinArm(
        side=side,
        robot=robot,
        joint_names=joint_names,
        base_link=str(base_parent.get("link")),
        ee_link=str(ee_child.get("link")),
    )

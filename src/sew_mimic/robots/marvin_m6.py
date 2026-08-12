"""URDF adapter for the Marvin M6 seven-DoF arms.

The SEW solver works in an arm-base frame and only needs joint orientations.
This module extracts those orientations, axes, limits, and the fixed EE
orientation directly from the robot URDF.
"""

from __future__ import annotations

import logging
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np

from ..collision import capsule_from_oobb
from ..safety import (
    ArmPose,
    BimanualPose,
    CapsuleRadii,
    SafetyFilterConfig,
    SafetyFilterResult,
    sew_safety_filter,
)
from ..solver import Serial7DoF, sew_mimic
from .urdf import URDFKinematics, rpy_rotation

logger = logging.getLogger(__name__)

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def _default_marvin_urdf() -> Path:
    relative = Path("assets") / "Marvin_M6_S_CCS_696_V4.0" / "robot_with_ee.urdf"
    candidates = (
        REPOSITORY_ROOT / relative,
        Path(sys.prefix) / "share" / "sew-mimic-retargeting" / relative,
    )
    return next((path for path in candidates if path.is_file()), candidates[0])


DEFAULT_MARVIN_URDF = _default_marvin_urdf()

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
    logger.debug("Marvin arm loading started: side=%s urdf=%s", side, urdf_path)
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
    arm = MarvinArm(
        side=side,
        robot=robot,
        joint_names=joint_names,
        base_link=str(base_parent.get("link")),
        ee_link=str(ee_child.get("link")),
    )
    logger.info(
        "Marvin arm loaded: side=%s joints=%d base=%s ee=%s",
        side,
        len(joint_names),
        arm.base_link,
        arm.ee_link,
    )
    return arm


def marvin_bimanual_pose(
    kinematics: URDFKinematics,
    left_arm: MarvinArm,
    right_arm: MarvinArm,
    q_left: np.ndarray,
    q_right: np.ndarray,
) -> BimanualPose:
    """Return paper SEW/tool keypoints in the common Marvin URDF root frame."""
    logger.debug("Marvin bimanual position FK started")
    positions = {
        **dict(zip(left_arm.joint_names, np.asarray(q_left, dtype=np.float64))),
        **dict(zip(right_arm.joint_names, np.asarray(q_right, dtype=np.float64))),
    }
    transforms = kinematics.link_transforms(positions)

    def arm_pose(arm: MarvinArm) -> ArmPose:
        # Joint actuator locations are the origins of child-link frames.
        shoulder_link = _joint_child(kinematics, arm.joint_names[0])
        elbow_link = _joint_child(kinematics, arm.joint_names[3])
        wrist_link = _joint_child(kinematics, arm.joint_names[5])
        shoulder = transforms[shoulder_link][:3, 3]
        elbow = transforms[elbow_link][:3, 3]
        wrist = transforms[wrist_link][:3, 3]
        tool_transform = transforms[arm.ee_link]
        return ArmPose(
            shoulder,
            elbow,
            wrist,
            tool_transform[:3, 3],
            tool_transform[:3, :3] @ arm.robot.R_align,
        )

    pose = BimanualPose(arm_pose(left_arm), arm_pose(right_arm))
    logger.debug("Marvin bimanual position FK completed")
    return pose


def _joint_child(kinematics: URDFKinematics, joint_name: str) -> str:
    for name, _, _, child, _, _ in kinematics.joints:
        if name == joint_name:
            return child
    raise KeyError(joint_name)


@lru_cache(maxsize=8)
def _cached_marvin_capsule_config(path_string: str, padding: float) -> SafetyFilterConfig:
    """Cached implementation; OOBB fitting is invariant for one asset/padding pair."""
    path = Path(path_string)
    logger.info("Marvin OOBB capsule estimation started: urdf=%s padding=%.3f", path, padding)
    collision_dir = path.parent / "collision"

    def radius(*stems: str) -> float:
        return max(
            capsule_from_oobb(collision_dir / f"{stem}.obj", padding=padding).radius
            for stem in stems
        )

    radii = CapsuleRadii(
        torso=radius("torso_base"),
        upper_arm=radius("shoulder_roll_l_j2_link", "elbow_pitch_l_j3_link"),
        lower_arm=radius("elbow_yaw_l_j4_link", "wrist_pitch_l_j5_link"),
        hand=radius("wrist_yaw_l_j6_link", "wrist_roll_l_j7_link", "left_hand_base_link"),
    )
    torso_local = capsule_from_oobb(collision_dir / "torso_base.obj", padding=padding, name="torso")
    transforms = URDFKinematics(path).link_transforms({})
    torso_transform = transforms["torso_base"]

    def transform_point(point: np.ndarray) -> np.ndarray:
        return torso_transform[:3, :3] @ point + torso_transform[:3, 3]

    config = SafetyFilterConfig(
        radii=radii,
        torso_start=transform_point(torso_local.start),
        torso_end=transform_point(torso_local.end),
    )
    logger.info(
        "Marvin OOBB capsule estimation completed: torso=%.4f upper=%.4f lower=%.4f hand=%.4f m",
        radii.torso,
        radii.upper_arm,
        radii.lower_arm,
        radii.hand,
    )
    return config


def estimate_marvin_capsule_config(
    urdf_path: str | Path = DEFAULT_MARVIN_URDF,
    *,
    padding: float = 1.05,
) -> SafetyFilterConfig:
    """Estimate and cache Marvin capsule parameters from collision-mesh OOBBs."""
    path = Path(urdf_path).expanduser().resolve()
    return _cached_marvin_capsule_config(str(path), float(padding))


class MarvinSafetyFilter:
    """Ready-to-use bimanual Marvin safety filter with OOBB capsule parameters."""

    def __init__(
        self,
        urdf_path: str | Path = DEFAULT_MARVIN_URDF,
        *,
        padding: float = 1.05,
        config: SafetyFilterConfig | None = None,
    ) -> None:
        logger.info("Marvin safety-filter initialization started")
        self.urdf_path = Path(urdf_path).expanduser().resolve()
        self.left = load_marvin_arm(self.urdf_path, "left")
        self.right = load_marvin_arm(self.urdf_path, "right")
        self.kinematics = URDFKinematics(self.urdf_path)
        self.config = (
            estimate_marvin_capsule_config(self.urdf_path, padding=padding)
            if config is None
            else config
        )
        zero_transforms = self.kinematics.link_transforms({})
        self._base_transforms = {
            "left": zero_transforms[self.left.base_link],
            "right": zero_transforms[self.right.base_link],
        }
        logger.info("Marvin safety-filter initialization completed")

    def forward_kinematics(self, q_left: np.ndarray, q_right: np.ndarray) -> BimanualPose:
        return marvin_bimanual_pose(self.kinematics, self.left, self.right, q_left, q_right)

    def _solve(self, arm: MarvinArm, q_current: np.ndarray, pose: ArmPose) -> np.ndarray:
        base = self._base_transforms[arm.side]
        rotation_base_world = base[:3, :3].T
        position_world_base = base[:3, 3]

        def point_in_base(point: np.ndarray) -> np.ndarray:
            return rotation_base_world @ (point - position_world_base)

        return sew_mimic(
            arm.robot,
            q_current,
            point_in_base(pose.shoulder),
            point_in_base(pose.elbow),
            point_in_base(pose.wrist),
            rotation_base_world @ pose.tool_orientation,
        )

    def filter(
        self,
        q_left_current: np.ndarray,
        q_right_current: np.ndarray,
        q_left_desired: np.ndarray,
        q_right_desired: np.ndarray,
    ) -> SafetyFilterResult:
        """Filter one desired bimanual pose; return current pose on failure."""
        logger.debug("Marvin safety-filter frame started")
        result = sew_safety_filter(
            q_left_current,
            q_right_current,
            q_left_desired,
            q_right_desired,
            forward_kinematics=self.forward_kinematics,
            solve_left=lambda q, pose: self._solve(self.left, q, pose),
            solve_right=lambda q, pose: self._solve(self.right, q, pose),
            config=self.config,
        )
        logger.debug(
            "Marvin safety-filter frame completed: safe=%s iterations=%d distance=%.6f m",
            result.safe,
            result.iterations,
            result.minimum_distance,
        )
        return result

    def __call__(
        self,
        q_left_current: np.ndarray,
        q_right_current: np.ndarray,
        q_left_desired: np.ndarray,
        q_right_desired: np.ndarray,
    ) -> SafetyFilterResult:
        """Alias for :meth:`filter` for use in real-time callback pipelines."""
        return self.filter(
            q_left_current,
            q_right_current,
            q_left_desired,
            q_right_desired,
        )

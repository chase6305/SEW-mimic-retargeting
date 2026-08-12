"""OpenArm seven-DoF SEW-Mimic adapter."""

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


def _default_openarm_urdf() -> Path:
    relative = Path("assets") / "OpenArm" / "robot.urdf"
    candidates = (
        REPOSITORY_ROOT / relative,
        Path(sys.prefix) / "share" / "sew-mimic-retargeting" / relative,
    )
    return next((path for path in candidates if path.is_file()), candidates[0])


DEFAULT_OPENARM_URDF = _default_openarm_urdf()
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
class OpenArmArm:
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


def _fixed_tool_rotation(joints: list[ET.Element], start_link: str, target_link: str) -> np.ndarray:
    """Accumulate a unique fixed-joint path from joint seven to the tracked hand link."""
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
                pending.append((str(child.get("link")), rotation @ _origin_rotation(joint)))
    raise ValueError(f"No fixed tool chain from {start_link} to {target_link}")


def load_openarm_arm(
    urdf_path: str | Path = DEFAULT_OPENARM_URDF,
    side: Literal["left", "right"] = "left",
    *,
    R_align: np.ndarray | None = None,
) -> OpenArmArm:
    """Load one OpenArm seven-DoF chain, tracking its hand-base link."""
    if side not in ("left", "right"):
        raise ValueError("side must be 'left' or 'right'")
    path = Path(urdf_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"OpenArm URDF not found: {path}")
    root = ET.parse(path).getroot()
    all_joints = root.findall("joint")
    by_name = {joint.get("name"): joint for joint in all_joints}
    suffix = "L" if side == "left" else "R"
    names = tuple(f"{stem}_{suffix}_J{i}" for i, stem in enumerate(_JOINT_STEMS, 1))
    joints = []
    for name in names:
        joint = by_name.get(name)
        if joint is None or joint.get("type") not in ("revolute", "continuous"):
            raise ValueError(f"Required revolute OpenArm joint is missing: {name}")
        joints.append(joint)
    for first, second in zip(joints, joints[1:]):
        child, parent = first.find("child"), second.find("parent")
        if child is None or parent is None or child.get("link") != parent.get("link"):
            raise ValueError(f"OpenArm chain is discontinuous near {second.get('name')}")

    axes, rotations, lower, upper = [], [], [], []
    for joint in joints:
        axis, limit = joint.find("axis"), joint.find("limit")
        axes.append(_numbers(None if axis is None else axis.get("xyz"), "1 0 0"))
        rotations.append(_origin_rotation(joint))
        if limit is None or limit.get("lower") is None or limit.get("upper") is None:
            raise ValueError(f"Joint limits are missing for {joint.get('name')}")
        lower.append(float(limit.get("lower")))
        upper.append(float(limit.get("upper")))

    first_parent = joints[0].find("parent")
    last_child = joints[-1].find("child")
    assert first_parent is not None and last_child is not None
    # Track the hand base requested by the OpenArm convention. It is 100.1 mm
    # beyond the seventh wrist-roll frame; the TCP remains another 80 mm ahead
    # and is intentionally excluded.
    ee_link = f"{side}_hand_base"
    tool_rotation = _fixed_tool_rotation(all_joints, str(last_child.get("link")), ee_link)
    arm = OpenArmArm(
        side=side,
        robot=Serial7DoF(
            axes_local=np.asarray(axes),
            R_local=np.asarray(rotations),
            q_min=np.asarray(lower),
            q_max=np.asarray(upper),
            R_7T_local=tool_rotation,
            R_align=np.eye(3) if R_align is None else R_align,
        ),
        joint_names=names,
        base_link=str(first_parent.get("link")),
        ee_link=ee_link,
    )
    logger.info("OpenArm arm loaded: side=%s base=%s ee=%s", side, arm.base_link, arm.ee_link)
    return arm


def openarm_bimanual_pose(
    kinematics: URDFKinematics,
    left_arm: OpenArmArm,
    right_arm: OpenArmArm,
    q_left: np.ndarray,
    q_right: np.ndarray,
) -> BimanualPose:
    """Return OpenArm SEW/tool keypoints in the common URDF root frame."""
    positions = {
        **dict(zip(left_arm.joint_names, np.asarray(q_left, dtype=np.float64))),
        **dict(zip(right_arm.joint_names, np.asarray(q_right, dtype=np.float64))),
    }
    transforms = kinematics.link_transforms(positions)

    def arm_pose(arm: OpenArmArm) -> ArmPose:
        links = kinematics.joint_child_links
        tool = transforms[arm.ee_link]
        return ArmPose(
            transforms[links[arm.joint_names[0]]][:3, 3],
            transforms[links[arm.joint_names[3]]][:3, 3],
            transforms[links[arm.joint_names[5]]][:3, 3],
            tool[:3, 3],
            tool[:3, :3] @ arm.robot.R_align,
        )

    return BimanualPose(arm_pose(left_arm), arm_pose(right_arm))


@lru_cache(maxsize=8)
def _cached_openarm_capsule_config(path_string: str, padding: float) -> SafetyFilterConfig:
    path = Path(path_string)
    collision = path.parent / "collision"

    def fitted(relative: str):
        return capsule_from_oobb(collision / relative, padding=padding)

    def radius(*relative_paths: str) -> float:
        return max(fitted(relative).radius for relative in relative_paths)

    radii = CapsuleRadii(
        torso=radius("torso/torso_base.stl"),
        upper_arm=radius(
            "left_arm/shoulder_roll_l_j2_link.stl",
            "left_arm/elbow_pitch_l_j3_link.stl",
        ),
        lower_arm=radius(
            "left_arm/elbow_yaw_l_j4_link.stl",
            "left_arm/wrist_pitch_l_j5_link.stl",
        ),
        hand=radius(
            "left_arm/wrist_yaw_l_j6_link.stl",
            "left_arm/wrist_roll_l_j7_link.stl",
            "hand/hand_base.stl",
        ),
    )
    torso = fitted("torso/torso_base.stl")
    transforms = URDFKinematics(path).link_transforms({})
    transform = transforms["torso_base"]

    def world(point: np.ndarray) -> np.ndarray:
        return transform[:3, :3] @ point + transform[:3, 3]

    return SafetyFilterConfig(radii, world(torso.start), world(torso.end))


def estimate_openarm_capsule_config(
    urdf_path: str | Path = DEFAULT_OPENARM_URDF, *, padding: float = 1.05
) -> SafetyFilterConfig:
    """Estimate and cache OpenArm capsules from its collision-mesh OOBBs."""
    path = Path(urdf_path).expanduser().resolve()
    return _cached_openarm_capsule_config(str(path), float(padding))


class OpenArmSafetyFilter:
    """Ready-to-use OpenArm bimanual capsule/XPBD safety filter."""

    def __init__(
        self,
        urdf_path: str | Path = DEFAULT_OPENARM_URDF,
        *,
        padding: float = 1.05,
        config: SafetyFilterConfig | None = None,
    ) -> None:
        self.urdf_path = Path(urdf_path).expanduser().resolve()
        self.left = load_openarm_arm(self.urdf_path, "left")
        self.right = load_openarm_arm(self.urdf_path, "right")
        self.kinematics = URDFKinematics(self.urdf_path)
        self.config = config or estimate_openarm_capsule_config(self.urdf_path, padding=padding)
        zero = self.kinematics.link_transforms({})
        self._base_transforms = {
            "left": zero[self.left.base_link],
            "right": zero[self.right.base_link],
        }

    def forward_kinematics(self, q_left: np.ndarray, q_right: np.ndarray) -> BimanualPose:
        return openarm_bimanual_pose(self.kinematics, self.left, self.right, q_left, q_right)

    def _solve(self, arm: OpenArmArm, current: np.ndarray, pose: ArmPose) -> np.ndarray:
        base = self._base_transforms[arm.side]
        rotation = base[:3, :3].T
        origin = base[:3, 3]

        def local(point: np.ndarray) -> np.ndarray:
            return rotation @ (point - origin)

        return sew_mimic(
            arm.robot,
            current,
            local(pose.shoulder),
            local(pose.elbow),
            local(pose.wrist),
            rotation @ pose.tool_orientation,
        )

    def filter(
        self,
        q_left_current: np.ndarray,
        q_right_current: np.ndarray,
        q_left_desired: np.ndarray,
        q_right_desired: np.ndarray,
    ) -> SafetyFilterResult:
        return sew_safety_filter(
            q_left_current,
            q_right_current,
            q_left_desired,
            q_right_desired,
            forward_kinematics=self.forward_kinematics,
            solve_left=lambda q, pose: self._solve(self.left, q, pose),
            solve_right=lambda q, pose: self._solve(self.right, q, pose),
            config=self.config,
        )

    __call__ = filter

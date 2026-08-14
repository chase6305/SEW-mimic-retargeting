"""OpenArm seven-DoF SEW-Mimic adapter."""

from __future__ import annotations

import logging
import sys
from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np

from ..safety import BimanualPose, SafetyFilterConfig
from .arm_loader import SerialArmSpec, SerialRobotArm
from .capsule_config import CapsuleMeshSpec, estimate_capsule_config_from_meshes
from .pose import URDFBimanualPoseEvaluator, urdf_bimanual_pose
from .safety_filter import RobotSafetyFilter
from .urdf import URDFKinematics

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
_ARM_SPEC = SerialArmSpec(
    left_joint_names=tuple(f"{stem}_L_J{i}" for i, stem in enumerate(_JOINT_STEMS, 1)),
    right_joint_names=tuple(f"{stem}_R_J{i}" for i, stem in enumerate(_JOINT_STEMS, 1)),
    left_ee_link="left_hand_base",
    right_ee_link="right_hand_base",
)
_CAPSULE_MESH_SPEC = CapsuleMeshSpec(
    torso="torso/torso_base.stl",
    upper_arm=(
        "left_arm/shoulder_roll_l_j2_link.stl",
        "left_arm/elbow_pitch_l_j3_link.stl",
    ),
    lower_arm=(
        "left_arm/elbow_yaw_l_j4_link.stl",
        "left_arm/wrist_pitch_l_j5_link.stl",
    ),
    hand=(
        "left_arm/wrist_yaw_l_j6_link.stl",
        "left_arm/wrist_roll_l_j7_link.stl",
        "hand/hand_base.stl",
    ),
)


OpenArmArm = SerialRobotArm


def load_openarm_arm(
    urdf_path: str | Path = DEFAULT_OPENARM_URDF,
    side: Literal["left", "right"] = "left",
    *,
    R_align: np.ndarray | None = None,
) -> OpenArmArm:
    """Load one OpenArm seven-DoF chain, tracking its hand-base link."""
    if side not in ("left", "right"):
        raise ValueError("side must be 'left' or 'right'")
    # Track the hand base requested by the OpenArm convention. It is 100.1 mm
    # beyond the seventh wrist-roll frame; the TCP remains another 80 mm ahead
    # and is intentionally excluded.
    arm = _ARM_SPEC.load(urdf_path, side, R_align=R_align)
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
    return urdf_bimanual_pose(kinematics, left_arm, right_arm, q_left, q_right)


@lru_cache(maxsize=8)
def _cached_openarm_capsule_config(path_string: str, padding: float) -> SafetyFilterConfig:
    path = Path(path_string)
    return estimate_capsule_config_from_meshes(
        path,
        path.parent / "collision",
        _CAPSULE_MESH_SPEC,
        padding=padding,
    )


def estimate_openarm_capsule_config(
    urdf_path: str | Path = DEFAULT_OPENARM_URDF, *, padding: float = 1.05
) -> SafetyFilterConfig:
    """Estimate and cache OpenArm capsules from its collision-mesh OOBBs."""
    path = Path(urdf_path).expanduser().resolve()
    return _cached_openarm_capsule_config(str(path), float(padding))


class OpenArmSafetyFilter(RobotSafetyFilter):
    """Ready-to-use OpenArm bimanual capsule/XPBD safety filter.

    ``backend`` explicitly selects both SEW recovery and collision/XPBD
    implementation. The tracked tool link is each arm's ``*_hand_base``;
    ``*_hand_tcp`` is intentionally excluded from the safety keypoint chain.
    """

    def __init__(
        self,
        urdf_path: str | Path = DEFAULT_OPENARM_URDF,
        *,
        padding: float = 1.05,
        config: SafetyFilterConfig | None = None,
        backend: str = "python",
    ) -> None:
        path = Path(urdf_path).expanduser().resolve()
        left = load_openarm_arm(path, "left")
        right = load_openarm_arm(path, "right")
        kinematics = URDFKinematics(path)
        resolved_config = config or estimate_openarm_capsule_config(path, padding=padding)
        pose_evaluator = URDFBimanualPoseEvaluator(kinematics, left, right)
        super().__init__(
            path,
            left=left,
            right=right,
            kinematics=kinematics,
            config=resolved_config,
            pose_function=pose_evaluator.pose_function,
            backend=backend,
        )

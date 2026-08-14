"""URDF adapter for the Marvin M6 seven-DoF arms.

The SEW solver works in an arm-base frame and only needs joint orientations.
This module extracts those orientations, axes, limits, and the fixed EE
orientation directly from the robot URDF.
"""

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
_ARM_SPEC = SerialArmSpec(
    left_joint_names=tuple(f"{stem}_L_J{i}" for i, stem in enumerate(_JOINT_STEMS, 1)),
    right_joint_names=tuple(f"{stem}_R_J{i}" for i, stem in enumerate(_JOINT_STEMS, 1)),
    left_ee_link="left_ee",
    right_ee_link="right_ee",
)
_CAPSULE_MESH_SPEC = CapsuleMeshSpec(
    torso="torso_base.obj",
    upper_arm=("shoulder_roll_l_j2_link.obj", "elbow_pitch_l_j3_link.obj"),
    lower_arm=("elbow_yaw_l_j4_link.obj", "wrist_pitch_l_j5_link.obj"),
    hand=("wrist_yaw_l_j6_link.obj", "wrist_roll_l_j7_link.obj", "left_hand_base_link.obj"),
)


MarvinArm = SerialRobotArm


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

    arm = _ARM_SPEC.load(urdf_path, side, R_align=R_align)
    logger.info(
        "Marvin arm loaded: side=%s joints=%d base=%s ee=%s",
        side,
        len(arm.joint_names),
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
    return urdf_bimanual_pose(kinematics, left_arm, right_arm, q_left, q_right)


@lru_cache(maxsize=8)
def _cached_marvin_capsule_config(path_string: str, padding: float) -> SafetyFilterConfig:
    """Cached implementation; OOBB fitting is invariant for one asset/padding pair."""
    path = Path(path_string)
    logger.info("Marvin OOBB capsule estimation started: urdf=%s padding=%.3f", path, padding)
    config = estimate_capsule_config_from_meshes(
        path,
        path.parent / "collision",
        _CAPSULE_MESH_SPEC,
        padding=padding,
    )
    radii = config.radii
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


class MarvinSafetyFilter(RobotSafetyFilter):
    """Ready-to-use bimanual Marvin safety filter with OOBB capsule parameters.

    ``backend`` explicitly selects both SEW recovery and collision/XPBD
    implementation. Use ``"python"`` for the reference path or ``"cpp"`` for
    the installed native extension; no automatic fallback is performed.
    """

    def __init__(
        self,
        urdf_path: str | Path = DEFAULT_MARVIN_URDF,
        *,
        padding: float = 1.05,
        config: SafetyFilterConfig | None = None,
        backend: str = "python",
    ) -> None:
        logger.info("Marvin safety-filter initialization started")
        path = Path(urdf_path).expanduser().resolve()
        left = load_marvin_arm(path, "left")
        right = load_marvin_arm(path, "right")
        kinematics = URDFKinematics(path)
        resolved_config = (
            estimate_marvin_capsule_config(path, padding=padding) if config is None else config
        )
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
        logger.info("Marvin safety-filter initialization completed")

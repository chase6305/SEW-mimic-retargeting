"""Generic bimanual keypoint extraction from a parsed URDF tree."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from ..safety import ArmPose, BimanualPose
from .kinematics import CompiledKinematics
from .registry import RobotArm
from .urdf import URDFKinematics


def _landmark_links(
    kinematics: URDFKinematics,
    left: RobotArm,
    right: RobotArm,
    indices: tuple[int, int, int],
) -> tuple[str, ...]:
    if len(indices) != 3 or any(index < 0 or index >= 7 for index in indices):
        raise ValueError("three landmark joint indices must be in [0, 6]")
    if left.side != "left" or right.side != "right":
        raise ValueError("left and right adapters must have matching side labels")
    links = kinematics.joint_child_links
    return tuple(
        link
        for arm in (left, right)
        for link in (*[links[arm.joint_names[index]] for index in indices], arm.ee_link)
    )


def _joint_values(q_left: np.ndarray, q_right: np.ndarray) -> np.ndarray:
    values = np.asarray([q_left, q_right], dtype=np.float64)
    if values.shape != (2, 7):
        raise ValueError("q_left and q_right must each have shape (7,)")
    return values.reshape(14)


def _bimanual_pose(
    transforms: Sequence[np.ndarray], left: RobotArm, right: RobotArm
) -> BimanualPose:
    def arm_pose(arm: RobotArm, offset: int) -> ArmPose:
        shoulder, elbow, wrist, tool = transforms[offset : offset + 4]
        return ArmPose(
            shoulder[:3, 3],
            elbow[:3, 3],
            wrist[:3, 3],
            tool[:3, 3],
            tool[:3, :3] @ arm.robot.R_align,
        )

    return BimanualPose(arm_pose(left, 0), arm_pose(right, 4))


@dataclass(frozen=True)
class URDFBimanualPoseEvaluator:
    """Precomputed, allocation-light bimanual landmark FK evaluator.

    Link lookup, landmark validation, and FK branch selection happen once at
    construction instead of on every control frame.
    """

    kinematics: URDFKinematics
    left_arm: RobotArm
    right_arm: RobotArm
    landmark_indices: tuple[int, int, int] = (0, 3, 5)
    backend: str = field(default="python", kw_only=True)
    _required_links: tuple[str, ...] = field(init=False)
    _compiled: CompiledKinematics = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "_required_links",
            _landmark_links(self.kinematics, self.left_arm, self.right_arm, self.landmark_indices),
        )
        object.__setattr__(
            self,
            "_compiled",
            self.kinematics.compile(
                (*self.left_arm.joint_names, *self.right_arm.joint_names),
                self._required_links,
                backend=self.backend,
            ),
        )

    def evaluate(self, q_left: np.ndarray, q_right: np.ndarray) -> BimanualPose:
        """Evaluate both arms in their shared URDF root frame."""
        transforms = self._compiled.evaluate(_joint_values(q_left, q_right))
        return _bimanual_pose(transforms, self.left_arm, self.right_arm)

    def pose_function(
        self,
        kinematics: URDFKinematics,
        left_arm: RobotArm,
        right_arm: RobotArm,
        q_left: np.ndarray,
        q_right: np.ndarray,
    ) -> BimanualPose:
        """Adapt :meth:`evaluate` to the stable robot pose-function contract."""
        if (
            kinematics is not self.kinematics
            or left_arm is not self.left_arm
            or right_arm is not self.right_arm
        ):
            raise ValueError("pose evaluator was called with different robot models")
        return self.evaluate(q_left, q_right)


def urdf_bimanual_pose(
    kinematics: URDFKinematics,
    left_arm: RobotArm,
    right_arm: RobotArm,
    q_left: np.ndarray,
    q_right: np.ndarray,
    *,
    shoulder_joint_index: int = 0,
    elbow_joint_index: int = 3,
    wrist_joint_index: int = 5,
) -> BimanualPose:
    """Return SEW/tool landmarks for two registered arms in the URDF root frame.

    Landmark indices refer to each arm's ordered seven-joint chain. Defaults
    match the paper convention used by Marvin and OpenArm: J1 shoulder, J4
    elbow, J6 wrist, plus the configured tracked tool link.

    This one-shot helper uses the parsed tree directly. Construct a persistent
    ``URDFBimanualPoseEvaluator`` for a control loop or native FK execution.
    """
    links = _landmark_links(
        kinematics,
        left_arm,
        right_arm,
        (shoulder_joint_index, elbow_joint_index, wrist_joint_index),
    )
    values = _joint_values(q_left, q_right)
    if not np.all(np.isfinite(values)):
        raise ValueError("q must contain only finite values")
    names = (*left_arm.joint_names, *right_arm.joint_names)
    transforms = kinematics.link_transforms(dict(zip(names, values)), links)
    return _bimanual_pose([transforms[link] for link in links], left_arm, right_arm)

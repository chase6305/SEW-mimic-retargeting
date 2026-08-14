"""Generic bimanual keypoint extraction from a parsed URDF tree."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..safety import ArmPose, BimanualPose
from .registry import RobotArm
from .urdf import URDFKinematics


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
    _landmark_links: tuple[tuple[str, str, str], tuple[str, str, str]] = field(init=False)
    _required_links: tuple[str, ...] = field(init=False)

    def __post_init__(self) -> None:
        if any(index < 0 or index >= 7 for index in self.landmark_indices):
            raise ValueError("landmark joint indices must be in [0, 6]")
        if self.left_arm.side != "left" or self.right_arm.side != "right":
            raise ValueError("left and right adapters must have matching side labels")
        links = self.kinematics.joint_child_links
        landmark_links = tuple(
            tuple(links[arm.joint_names[index]] for index in self.landmark_indices)
            for arm in (self.left_arm, self.right_arm)
        )
        object.__setattr__(self, "_landmark_links", landmark_links)
        object.__setattr__(
            self,
            "_required_links",
            (*landmark_links[0], self.left_arm.ee_link, *landmark_links[1], self.right_arm.ee_link),
        )

    def evaluate(self, q_left: np.ndarray, q_right: np.ndarray) -> BimanualPose:
        """Evaluate both arms in their shared URDF root frame."""
        positions = dict(zip(self.left_arm.joint_names, np.asarray(q_left, dtype=np.float64)))
        positions.update(zip(self.right_arm.joint_names, np.asarray(q_right, dtype=np.float64)))
        transforms = self.kinematics.link_transforms(positions, self._required_links)

        def arm_pose(arm: RobotArm, links: tuple[str, str, str]) -> ArmPose:
            shoulder_link, elbow_link, wrist_link = links
            tool = transforms[arm.ee_link]
            return ArmPose(
                transforms[shoulder_link][:3, 3],
                transforms[elbow_link][:3, 3],
                transforms[wrist_link][:3, 3],
                tool[:3, 3],
                tool[:3, :3] @ arm.robot.R_align,
            )

        return BimanualPose(
            arm_pose(self.left_arm, self._landmark_links[0]),
            arm_pose(self.right_arm, self._landmark_links[1]),
        )

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
    """
    evaluator = URDFBimanualPoseEvaluator(
        kinematics,
        left_arm,
        right_arm,
        (shoulder_joint_index, elbow_joint_index, wrist_joint_index),
    )
    return evaluator.evaluate(q_left, q_right)

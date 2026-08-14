"""Reusable high-level safety-filter orchestration for bimanual robot adapters."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np

from ..backends import solve
from ..safety import (
    ArmPose,
    BimanualPose,
    SafetyFilterConfig,
    SafetyFilterResult,
    sew_safety_filter,
)
from .registry import RobotArm
from .urdf import URDFKinematics

logger = logging.getLogger(__name__)

BimanualPoseFunction = Callable[
    [URDFKinematics, RobotArm, RobotArm, np.ndarray, np.ndarray], BimanualPose
]


class RobotSafetyFilter:
    """Robot-independent bimanual FK, frame conversion, and SEW orchestration.

    Robot adapters supply two arm models, a bimanual FK function, and one
    immutable capsule configuration. The class owns all shared per-frame logic
    and explicitly dispatches both collision and SEW work to ``backend``.
    """

    def __init__(
        self,
        urdf_path: str | Path,
        *,
        left: RobotArm,
        right: RobotArm,
        kinematics: URDFKinematics,
        config: SafetyFilterConfig,
        pose_function: BimanualPoseFunction,
        backend: str = "python",
    ) -> None:
        if left.side != "left" or right.side != "right":
            raise ValueError("left and right adapters must have matching side labels")
        self.urdf_path = Path(urdf_path).expanduser().resolve()
        self.backend = backend
        self.left = left
        self.right = right
        self.kinematics = kinematics
        self.config = config
        self._pose_function = pose_function

        zero = self.kinematics.link_transforms({}, (self.left.base_link, self.right.base_link))
        self._base_rotations = {
            "left": zero[self.left.base_link][:3, :3].T.copy(),
            "right": zero[self.right.base_link][:3, :3].T.copy(),
        }
        self._base_origins = {
            "left": zero[self.left.base_link][:3, 3].copy(),
            "right": zero[self.right.base_link][:3, 3].copy(),
        }

    def forward_kinematics(self, q_left: np.ndarray, q_right: np.ndarray) -> BimanualPose:
        """Evaluate robot-specific bimanual FK in the shared URDF root frame."""
        return self._pose_function(self.kinematics, self.left, self.right, q_left, q_right)

    def _solve_arm(self, arm: RobotArm, current: np.ndarray, pose: ArmPose) -> np.ndarray:
        """Transform a projected world-frame pose and solve it in one arm base."""
        rotation = self._base_rotations[arm.side]
        origin = self._base_origins[arm.side]
        return solve(
            arm.robot,
            current,
            rotation @ (pose.shoulder - origin),
            rotation @ (pose.elbow - origin),
            rotation @ (pose.wrist - origin),
            rotation @ pose.tool_orientation,
            backend=self.backend,
        )

    def _solve_left(self, current: np.ndarray, pose: ArmPose) -> np.ndarray:
        return self._solve_arm(self.left, current, pose)

    def _solve_right(self, current: np.ndarray, pose: ArmPose) -> np.ndarray:
        return self._solve_arm(self.right, current, pose)

    def filter(
        self,
        q_left_current: np.ndarray,
        q_right_current: np.ndarray,
        q_left_desired: np.ndarray,
        q_right_desired: np.ndarray,
    ) -> SafetyFilterResult:
        """Filter one desired bimanual command and return a structured result."""
        result = sew_safety_filter(
            q_left_current,
            q_right_current,
            q_left_desired,
            q_right_desired,
            forward_kinematics=self.forward_kinematics,
            solve_left=self._solve_left,
            solve_right=self._solve_right,
            config=self.config,
            backend=self.backend,
        )
        logger.debug(
            "Robot safety-filter frame completed: safe=%s status=%s iterations=%d distance=%.6f m",
            result.safe,
            result.status.value,
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
        """Alias for :meth:`filter` for real-time callback pipelines."""
        return self.filter(
            q_left_current,
            q_right_current,
            q_left_desired,
            q_right_desired,
        )

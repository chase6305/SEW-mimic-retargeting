"""Prepared FK execution shared by URDF adapters and numerical backends."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..backends import CppKinematicsBackend
from ..utility import is_rotation_matrix, skew


@dataclass(frozen=True)
class _KinematicsPlan:
    """Indexed tree after fixed and uncommanded joints have been folded.

    Operation rows contain (parent transform index, motion type, q index).
    Transform zero is the root; operation i writes transform i+1. Motion types
    are 1 for rotation and 2 for translation. Targets retain caller order.
    """

    joint_names: tuple[str, ...]
    link_names: tuple[str, ...]
    operations: np.ndarray
    origins: np.ndarray
    axes: np.ndarray
    targets: np.ndarray
    offsets: np.ndarray

    def __post_init__(self) -> None:
        for name in ("operations", "origins", "axes", "targets", "offsets"):
            dtype = np.int32 if name in ("operations", "targets") else np.float64
            value = np.array(getattr(self, name), dtype=dtype, copy=True)
            if not np.all(np.isfinite(value)):
                raise ValueError(f"FK {name} must contain only finite values")
            if name in ("origins", "offsets") and any(
                not is_rotation_matrix(transform[:3, :3])
                or not np.array_equal(transform[3], [0, 0, 0, 1])
                for transform in value
            ):
                raise ValueError(f"FK {name} must contain rigid transforms")
            value.flags.writeable = False
            object.__setattr__(self, name, value)


class _PythonKinematics:
    def __init__(self, plan: _KinematicsPlan) -> None:
        self._plan = plan
        self._parents = plan.operations[:, 0].tolist()
        self._indices = plan.operations[:, 2]
        self._sines = np.zeros((len(plan.operations), 3, 3))
        self._cosines = np.zeros_like(self._sines)
        self._shifts = np.zeros((len(plan.operations), 3))
        self._has_prismatic = bool(np.any(plan.operations[:, 1] == 2))
        for i, ((_, motion, _), origin, axis) in enumerate(
            zip(plan.operations, plan.origins, plan.axes)
        ):
            rotation = origin[:3, :3]
            if motion == 1:
                cross = skew(axis)
                self._sines[i] = rotation @ cross
                self._cosines[i] = self._sines[i] @ cross
            else:
                self._shifts[i] = rotation @ axis

    def evaluate(self, q: np.ndarray) -> np.ndarray:
        # Scratch belongs to this call: one plan can serve concurrent streams.
        if not self._parents:
            return self._plan.offsets.copy()
        angles = q[self._indices]
        local = self._plan.origins.copy()
        # Construct all local rotations together; only the dependency-ordered
        # tree accumulation remains a Python loop.
        local[:, :3, :3] += (
            np.sin(angles)[:, None, None] * self._sines
            + (1.0 - np.cos(angles))[:, None, None] * self._cosines
        )
        if self._has_prismatic:
            local[:, :3, 3] += angles[:, None] * self._shifts
        world = np.empty((len(self._parents) + 1, 4, 4))
        world[0] = np.eye(4)
        for child, parent in enumerate(self._parents, start=1):
            world[child] = world[parent] @ local[child - 1]
        return world[self._plan.targets] @ self._plan.offsets


class CompiledKinematics:
    """Reusable FK evaluator constructed by :meth:`URDFKinematics.compile`.

    Model preparation and backend selection happen once. Each call accepts
    joint values in ``joint_names`` order and returns owned ``(N, 4, 4)``
    transforms in ``link_names`` order. The captured model is a snapshot;
    compile again after changing calibration. Evaluation has no shared scratch.
    """

    def __init__(self, plan: _KinematicsPlan, *, backend: str = "python") -> None:
        self._plan = plan
        if backend == "python":
            self._executor = _PythonKinematics(plan)
        elif backend == "cpp":
            self._executor = CppKinematicsBackend(
                plan.operations,
                plan.origins,
                plan.axes,
                plan.targets,
                plan.offsets,
                len(plan.joint_names),
            )
        else:
            raise ValueError("backend must be one of: python, cpp")

    @property
    def joint_names(self) -> tuple[str, ...]:
        return self._plan.joint_names

    @property
    def link_names(self) -> tuple[str, ...]:
        return self._plan.link_names

    def evaluate(self, q: np.ndarray) -> np.ndarray:
        """Evaluate a finite joint vector without redoing name lookup or FK planning."""
        values = np.asarray(q, dtype=np.float64)
        if values.shape != (len(self.joint_names),):
            raise ValueError(f"q must have shape ({len(self.joint_names)},)")
        if not np.all(np.isfinite(values)):
            raise ValueError("q must contain only finite values")
        return self._executor.evaluate(values)

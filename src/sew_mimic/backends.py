"""Runtime selection for the interchangeable Python and C++ solver backends."""

from __future__ import annotations

import os
from collections import OrderedDict
from dataclasses import dataclass
from typing import Literal, Protocol, Sequence

import numpy as np

from .solver import Serial7DoF
from .solver import sew_mimic as _python_sew_mimic
from .utility import SEWMimicError

BackendName = Literal["python", "cpp"]


class SolverBackend(Protocol):
    """Stable contract implemented by every SEW-Mimic backend."""

    name: str

    def solve(
        self,
        robot: Serial7DoF,
        q0: Sequence[float],
        shoulder: Sequence[float],
        elbow: Sequence[float],
        wrist: Sequence[float],
        hand_orientation: np.ndarray,
    ) -> np.ndarray: ...

    def solve_batch(
        self,
        robot: Serial7DoF,
        q0: Sequence[float],
        shoulders: np.ndarray,
        elbows: np.ndarray,
        wrists: np.ndarray,
        hand_orientations: np.ndarray,
    ) -> np.ndarray: ...


@dataclass(frozen=True)
class PythonBackend:
    """Reference NumPy implementation."""

    name: str = "python"

    def solve(
        self,
        robot: Serial7DoF,
        q0: Sequence[float],
        shoulder: Sequence[float],
        elbow: Sequence[float],
        wrist: Sequence[float],
        hand_orientation: np.ndarray,
    ) -> np.ndarray:
        return _python_sew_mimic(robot, q0, shoulder, elbow, wrist, hand_orientation)

    def solve_batch(
        self,
        robot: Serial7DoF,
        q0: Sequence[float],
        shoulders: np.ndarray,
        elbows: np.ndarray,
        wrists: np.ndarray,
        hand_orientations: np.ndarray,
    ) -> np.ndarray:
        current = np.asarray(q0, dtype=np.float64)
        output = np.empty((len(shoulders), 7), dtype=np.float64)
        for index, (shoulder, elbow, wrist, hand) in enumerate(
            zip(shoulders, elbows, wrists, hand_orientations, strict=True)
        ):
            current = self.solve(robot, current, shoulder, elbow, wrist, hand)
            output[index] = current
        return output


class CppBackend:
    """Optional pybind11 implementation with the same public contract."""

    name = "cpp"

    def __init__(self) -> None:
        try:
            from . import _sew_mimic_cpp
        except ImportError as exc:
            raise RuntimeError(
                "The C++ backend is not installed. Build with "
                "`SEW_MIMIC_BUILD_CPP=1 pip install -e '.[cpp]'`, or select "
                "backend='python'."
            ) from exc
        self._native = _sew_mimic_cpp
        self._solvers: OrderedDict[int, tuple[Serial7DoF, object]] = OrderedDict()
        self._maximum_cached_solvers = 16

    def _solver_for(self, robot: Serial7DoF):
        """Return one persistent native solver per calibrated robot object."""
        key = id(robot)
        cached = self._solvers.get(key)
        if cached is not None and cached[0] is robot:
            self._solvers.move_to_end(key)
            return cached[1]
        solver = self._native.SewMimicSolver(
            robot.axes_local,
            robot.R_local,
            robot.q_min,
            robot.q_max,
            robot.R_7T_local,
            robot.R_align,
        )
        self._solvers[key] = (robot, solver)
        if len(self._solvers) > self._maximum_cached_solvers:
            self._solvers.popitem(last=False)
        return solver

    def clear_cache(self) -> None:
        """Release all persistent native robot models owned by this backend."""
        self._solvers.clear()

    def solve(
        self,
        robot: Serial7DoF,
        q0: Sequence[float],
        shoulder: Sequence[float],
        elbow: Sequence[float],
        wrist: Sequence[float],
        hand_orientation: np.ndarray,
    ) -> np.ndarray:
        try:
            result = self._solver_for(robot).solve(q0, shoulder, elbow, wrist, hand_orientation)
        except RuntimeError as exc:
            raise SEWMimicError(str(exc)) from exc
        return np.asarray(result, dtype=np.float64)

    def solve_batch(
        self,
        robot: Serial7DoF,
        q0: Sequence[float],
        shoulders: np.ndarray,
        elbows: np.ndarray,
        wrists: np.ndarray,
        hand_orientations: np.ndarray,
    ) -> np.ndarray:
        try:
            result = self._solver_for(robot).solve_batch(
                q0, shoulders, elbows, wrists, hand_orientations
            )
        except RuntimeError as exc:
            raise SEWMimicError(str(exc)) from exc
        return np.asarray(result, dtype=np.float64)


class CppCollisionBackend:
    """Native capsule-distance and XPBD projection operations."""

    name = "cpp"

    def __init__(self) -> None:
        try:
            from . import _sew_mimic_cpp
        except ImportError as exc:
            raise RuntimeError(
                "The C++ collision backend is not installed. Build with "
                "`SEW_MIMIC_BUILD_CPP=1 pip install -e '.[cpp]'`."
            ) from exc
        self._native = _sew_mimic_cpp

    def minimum_distance(
        self,
        starts: np.ndarray,
        ends: np.ndarray,
        radii: np.ndarray,
        pairs: np.ndarray,
    ) -> float:
        return float(
            self._native.minimum_capsule_distance(
                starts, ends, radii, np.asarray(pairs, dtype=np.int32)
            )
        )

    def project_xpbd(
        self,
        points: np.ndarray,
        torso_start: np.ndarray,
        torso_end: np.ndarray,
        radii: np.ndarray,
        pairs: np.ndarray,
        *,
        minimum_distance: float,
        activation_distance: float,
        release_distance: float,
        compliance: float,
        tolerance: float,
        iterations: int,
    ) -> tuple[np.ndarray, int, float]:
        projected, used_iterations, distance = self._native.project_xpbd(
            points,
            torso_start,
            torso_end,
            radii,
            np.asarray(pairs, dtype=np.int32),
            minimum_distance,
            activation_distance,
            release_distance,
            compliance,
            tolerance,
            iterations,
        )
        return np.asarray(projected), int(used_iterations), float(distance)


def cpp_backend_available() -> bool:
    """Return whether the optional native extension can be imported."""
    try:
        from . import _sew_mimic_cpp  # noqa: F401
    except ImportError:
        return False
    return True


def backend_status() -> dict[str, str | bool]:
    """Return deployment diagnostics without constructing the C++ backend."""
    available = cpp_backend_available()
    implementation = "unavailable"
    if available:
        from . import _sew_mimic_cpp

        implementation = str(_sew_mimic_cpp.implementation)
    return {
        "default": os.getenv("SEW_MIMIC_BACKEND", "python").strip().lower(),
        "cpp_available": available,
        "cpp_implementation": implementation,
    }


def get_backend(name: BackendName | str | None = None) -> SolverBackend:
    """Resolve an explicit backend or ``SEW_MIMIC_BACKEND`` (default: python)."""
    selected = (name or os.getenv("SEW_MIMIC_BACKEND", "python")).strip().lower()
    if selected == "python":
        return _PYTHON_BACKEND
    if selected == "cpp":
        global _CPP_BACKEND
        if _CPP_BACKEND is None:
            _CPP_BACKEND = CppBackend()
        return _CPP_BACKEND
    raise ValueError("backend must be one of: python, cpp")


_PYTHON_BACKEND = PythonBackend()
_CPP_BACKEND: CppBackend | None = None
_CPP_COLLISION_BACKEND: CppCollisionBackend | None = None


def get_cpp_collision_backend() -> CppCollisionBackend:
    """Return the process-wide native collision backend, failing if unavailable."""
    global _CPP_COLLISION_BACKEND
    if _CPP_COLLISION_BACKEND is None:
        _CPP_COLLISION_BACKEND = CppCollisionBackend()
    return _CPP_COLLISION_BACKEND


def solve(
    robot: Serial7DoF,
    q0: Sequence[float],
    shoulder: Sequence[float],
    elbow: Sequence[float],
    wrist: Sequence[float],
    hand_orientation: np.ndarray,
    *,
    backend: BackendName | str | None = None,
) -> np.ndarray:
    """Solve with a selected backend while preserving the existing data model."""
    return get_backend(backend).solve(robot, q0, shoulder, elbow, wrist, hand_orientation)


def solve_batch(
    robot: Serial7DoF,
    q0: Sequence[float],
    shoulders: np.ndarray,
    elbows: np.ndarray,
    wrists: np.ndarray,
    hand_orientations: np.ndarray,
    *,
    backend: BackendName | str | None = None,
) -> np.ndarray:
    """Solve a trajectory sequentially, carrying each result into the next frame.

    Keypoint arrays must have shape ``(N, 3)`` and hand orientations must have
    shape ``(N, 3, 3)``. The result has shape ``(N, 7)``. Frame ``i`` uses the
    solution from frame ``i-1`` as its branch-selection reference; only frame
    zero uses ``q0``.
    """
    current = np.asarray(q0, dtype=np.float64).reshape(-1)
    if current.shape != (7,):
        raise ValueError(f"q0 must have shape (7,), got {current.shape}")
    if not np.all(np.isfinite(current)):
        raise ValueError("q0 must contain only finite values")

    arrays = [
        np.asarray(value, dtype=np.float64)
        for value in (shoulders, elbows, wrists, hand_orientations)
    ]
    sample_count = len(arrays[0])
    if (
        arrays[0].shape != (sample_count, 3)
        or any(value.shape != (sample_count, 3) for value in arrays[1:3])
        or arrays[3].shape != (sample_count, 3, 3)
    ):
        raise ValueError(
            "shoulders, elbows, and wrists must have shape (N, 3), and "
            "hand_orientations must have shape (N, 3, 3)"
        )
    if any(not np.all(np.isfinite(value)) for value in arrays):
        raise ValueError("Batch inputs must contain only finite values")
    return get_backend(backend).solve_batch(robot, current, *arrays)

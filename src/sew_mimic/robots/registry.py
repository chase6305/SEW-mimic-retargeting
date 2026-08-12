"""Unified robot-adapter registry used by applications and demos."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Literal, Protocol, runtime_checkable

import numpy as np

from ..solver import Serial7DoF

RobotSide = Literal["left", "right"]


@runtime_checkable
class RobotArm(Protocol):
    """Structural interface shared by SEW-compatible robot adapters."""

    side: RobotSide
    robot: Serial7DoF
    joint_names: tuple[str, ...]
    base_link: str
    ee_link: str


ArmLoader = Callable[[str | Path, RobotSide], RobotArm]
TrajectoryGenerator = Callable[[str, float, float, float | None], tuple[np.ndarray, np.ndarray]]


@dataclass(frozen=True)
class RobotAdapter:
    """Metadata and loader for one registered robot family."""

    name: str
    display_name: str
    default_urdf: Path
    load_arm: ArmLoader
    trajectory_profile: str | TrajectoryGenerator
    keypoint_profile: Literal["solver", "urdf"] = "urdf"

    def __post_init__(self) -> None:
        canonical = _normalize_name(self.name)
        if not canonical or canonical != self.name:
            raise ValueError("RobotAdapter.name must be a normalized non-empty name")
        if not self.display_name.strip():
            raise ValueError("RobotAdapter.display_name must not be empty")
        if not callable(self.load_arm):
            raise TypeError("RobotAdapter.load_arm must be callable")
        if not callable(self.trajectory_profile) and not self.trajectory_profile.strip():
            raise ValueError("RobotAdapter.trajectory_profile must not be empty")
        if self.keypoint_profile not in ("solver", "urdf"):
            raise ValueError("RobotAdapter.keypoint_profile must be 'solver' or 'urdf'")
        object.__setattr__(self, "default_urdf", Path(self.default_urdf).expanduser().resolve())


_ADAPTERS: dict[str, RobotAdapter] = {}
_ALIASES: dict[str, str] = {}
_REGISTRY_LOCK = RLock()
_BUILTINS_REGISTERED = False


def _normalize_name(name: str) -> str:
    """Normalize user-facing robot names and aliases."""
    return name.strip().lower().replace("_", "-")


def register_robot_adapter(
    adapter: RobotAdapter,
    *,
    aliases: tuple[str, ...] = (),
    replace: bool = False,
) -> None:
    """Register a robot family without modifying the package registry source.

    Registration is process-local and thread-safe. Set ``replace=True`` only
    when an application intentionally overrides an existing integration.
    """
    _ensure_builtin_adapters()
    _register_robot_adapter(adapter, aliases=aliases, replace=replace)


def _register_robot_adapter(
    adapter: RobotAdapter,
    *,
    aliases: tuple[str, ...],
    replace: bool,
) -> None:
    """Internal registration primitive that does not bootstrap built-ins."""
    canonical = _normalize_name(adapter.name)
    normalized_aliases = tuple(_normalize_name(alias) for alias in aliases)
    if any(not alias for alias in normalized_aliases):
        raise ValueError("Robot aliases must not be empty")
    if canonical in normalized_aliases:
        raise ValueError("Robot aliases must differ from the canonical name")
    with _REGISTRY_LOCK:
        conflicts = {
            name
            for name in (canonical, *normalized_aliases)
            if name in _ADAPTERS or name in _ALIASES
        }
        if conflicts and not replace:
            raise ValueError(f"Robot names already registered: {sorted(conflicts)}")
        if replace:
            _ALIASES.update({name: canonical for name in normalized_aliases if name != canonical})
        _ADAPTERS[canonical] = adapter
        for alias in normalized_aliases:
            _ALIASES[alias] = canonical


def _ensure_builtin_adapters() -> None:
    """Lazily install bundled adapters exactly once."""
    global _BUILTINS_REGISTERED
    with _REGISTRY_LOCK:
        if _BUILTINS_REGISTERED:
            return
        # Imports stay local to keep adapter modules independently importable.
        from .marvin_m6 import DEFAULT_MARVIN_URDF, load_marvin_arm
        from .openarm import DEFAULT_OPENARM_URDF, load_openarm_arm

        _register_robot_adapter(
            RobotAdapter(
                "marvin",
                "Marvin M6",
                DEFAULT_MARVIN_URDF,
                load_marvin_arm,
                "marvin_humanlike",
                "solver",
            ),
            aliases=("marvin-m6", "m6"),
            replace=False,
        )
        _register_robot_adapter(
            RobotAdapter(
                "openarm",
                "OpenArm",
                DEFAULT_OPENARM_URDF,
                load_openarm_arm,
                "openarm_safe",
                "urdf",
            ),
            aliases=("open-arm",),
            replace=False,
        )
        _BUILTINS_REGISTERED = True


def available_robots() -> tuple[str, ...]:
    """Return stable robot names accepted by :func:`get_robot_adapter`."""
    _ensure_builtin_adapters()
    with _REGISTRY_LOCK:
        return tuple(_ADAPTERS)


def get_robot_adapter(name: str) -> RobotAdapter:
    """Return a registered adapter by case-insensitive name."""
    if not isinstance(name, str):
        raise TypeError("Robot name must be a string")
    _ensure_builtin_adapters()
    normalized = _normalize_name(name)
    with _REGISTRY_LOCK:
        normalized = _ALIASES.get(normalized, normalized)
        adapter = _ADAPTERS.get(normalized)
    if adapter is not None:
        return adapter
    choices = ", ".join(available_robots())
    raise ValueError(f"Unknown robot {name!r}; available robots: {choices}")


def resolve_robot_urdf(name: str, urdf_path: str | Path | None = None) -> Path:
    """Resolve an override or the registered default URDF and validate it."""
    path = (
        get_robot_adapter(name).default_urdf
        if urdf_path is None
        else Path(urdf_path).expanduser().resolve()
    )
    if not path.is_file():
        raise FileNotFoundError(f"Robot URDF not found: {path}")
    return path


def load_robot_arm(
    name: str,
    side: RobotSide = "left",
    urdf_path: str | Path | None = None,
) -> RobotArm:
    """Load any registered SEW-compatible arm through one stable API."""
    adapter = get_robot_adapter(name)
    return adapter.load_arm(resolve_robot_urdf(name, urdf_path), side)

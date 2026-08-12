"""Unified robot-adapter registry used by applications and demos."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

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


@dataclass(frozen=True)
class RobotAdapter:
    """Metadata and loader for one registered robot family."""

    name: str
    display_name: str
    default_urdf: Path
    load_arm: ArmLoader
    trajectory_profile: str


def _registry() -> dict[str, RobotAdapter]:
    # Imports stay local to keep adapter modules independently importable.
    from .marvin_m6 import DEFAULT_MARVIN_URDF, load_marvin_arm
    from .openarm import DEFAULT_OPENARM_URDF, load_openarm_arm

    return {
        "marvin": RobotAdapter(
            "marvin", "Marvin M6", DEFAULT_MARVIN_URDF, load_marvin_arm, "marvin_humanlike"
        ),
        "openarm": RobotAdapter(
            "openarm", "OpenArm", DEFAULT_OPENARM_URDF, load_openarm_arm, "openarm_safe"
        ),
    }


def available_robots() -> tuple[str, ...]:
    """Return stable robot names accepted by :func:`get_robot_adapter`."""
    return tuple(_registry())


def get_robot_adapter(name: str) -> RobotAdapter:
    """Return a registered adapter by case-insensitive name."""
    normalized = name.strip().lower().replace("_", "-")
    aliases = {"marvin-m6": "marvin", "m6": "marvin", "open-arm": "openarm"}
    normalized = aliases.get(normalized, normalized)
    try:
        return _registry()[normalized]
    except KeyError as exc:
        choices = ", ".join(available_robots())
        raise ValueError(f"Unknown robot {name!r}; available robots: {choices}") from exc


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

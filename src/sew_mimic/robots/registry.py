"""Unified robot-adapter registry used by applications and demos."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

import numpy as np

from ..solver import Serial7DoF
from .demo_profiles import CollisionDemoProfile

if TYPE_CHECKING:
    from .safety_filter import RobotSafetyFilter

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
SafetyFilterFactory = Callable[..., object]


@dataclass(frozen=True)
class RobotAdapter:
    """Metadata and loader for one registered robot family."""

    name: str
    display_name: str
    default_urdf: Path
    load_arm: ArmLoader
    trajectory_profile: str | TrajectoryGenerator
    keypoint_profile: Literal["solver", "urdf"] = "urdf"
    safety_filter_factory: SafetyFilterFactory | None = None
    collision_profile: CollisionDemoProfile | None = None

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
        if self.safety_filter_factory is not None and not callable(self.safety_filter_factory):
            raise TypeError("RobotAdapter.safety_filter_factory must be callable")
        object.__setattr__(self, "default_urdf", Path(self.default_urdf).expanduser().resolve())


@dataclass(frozen=True)
class RobotValidationReport:
    """Serializable structural and numerical checks for one robot adapter."""

    name: str
    urdf: str
    left_base_link: str
    right_base_link: str
    left_ee_link: str
    right_ee_link: str
    minimum_joint_range: float
    maximum_consecutive_axis_dot: float
    keypoints_finite: bool


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
    if len(set(normalized_aliases)) != len(normalized_aliases):
        raise ValueError("Robot aliases must be unique after normalization")
    if canonical in normalized_aliases:
        raise ValueError("Robot aliases must differ from the canonical name")
    with _REGISTRY_LOCK:
        canonical_alias_conflicts = set(normalized_aliases) & set(_ADAPTERS)
        if canonical_alias_conflicts:
            raise ValueError(
                f"Robot aliases conflict with canonical names: {sorted(canonical_alias_conflicts)}"
            )
        conflicts = {
            name
            for name in (canonical, *normalized_aliases)
            if name in _ADAPTERS or name in _ALIASES
        }
        if conflicts and not replace:
            raise ValueError(f"Robot names already registered: {sorted(conflicts)}")
        if replace:
            # Discard aliases formerly owned by this family and ensure its
            # canonical name cannot remain an alias of another family.
            _ALIASES.pop(canonical, None)
            for alias, owner in tuple(_ALIASES.items()):
                if owner == canonical:
                    del _ALIASES[alias]
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
        from .demo_profiles import MARVIN_COLLISION_PROFILE, OPENARM_COLLISION_PROFILE
        from .marvin_m6 import DEFAULT_MARVIN_URDF, MarvinSafetyFilter, load_marvin_arm
        from .openarm import DEFAULT_OPENARM_URDF, OpenArmSafetyFilter, load_openarm_arm

        _register_robot_adapter(
            RobotAdapter(
                "marvin",
                "Marvin M6",
                DEFAULT_MARVIN_URDF,
                load_marvin_arm,
                "marvin_humanlike",
                keypoint_profile="solver",
                safety_filter_factory=MarvinSafetyFilter,
                collision_profile=MARVIN_COLLISION_PROFILE,
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
                keypoint_profile="urdf",
                safety_filter_factory=OpenArmSafetyFilter,
                collision_profile=OPENARM_COLLISION_PROFILE,
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


def create_robot_safety_filter(
    name: str,
    urdf_path: str | Path | None = None,
    **kwargs: object,
) -> RobotSafetyFilter:
    """Construct a registered robot's safety filter through one stable API.

    Extra keyword arguments are forwarded to the adapter factory, for example
    ``backend="cpp"``, ``padding=1.05``, or a custom collision configuration.
    """
    adapter = get_robot_adapter(name)
    factory = adapter.safety_filter_factory
    if factory is None:
        raise ValueError(f"Robot {adapter.name!r} does not provide a safety filter")
    return factory(resolve_robot_urdf(adapter.name, urdf_path), **kwargs)


def validate_robot_adapter(
    name: str,
    urdf_path: str | Path | None = None,
) -> RobotValidationReport:
    """Load both arms and validate their common URDF FK contract.

    This is intentionally independent of visualization and collision-mesh
    dependencies, making it suitable as the first acceptance check for a new
    robot integration.
    """
    from .pose import URDFBimanualPoseEvaluator
    from .urdf import URDFKinematics

    adapter = get_robot_adapter(name)
    path = resolve_robot_urdf(adapter.name, urdf_path)
    left = adapter.load_arm(path, "left")
    right = adapter.load_arm(path, "right")
    if left.side != "left" or right.side != "right":
        raise ValueError("robot loader returned incorrect left/right side labels")
    kinematics = URDFKinematics(path)
    q_left_neutral = np.clip(np.zeros(7), left.robot.q_min, left.robot.q_max)
    q_right_neutral = np.clip(np.zeros(7), right.robot.q_min, right.robot.q_max)
    pose = URDFBimanualPoseEvaluator(kinematics, left, right).evaluate(
        q_left_neutral, q_right_neutral
    )
    # The complete check validates both finite positions and tool SO(3).
    keypoints = pose.points()
    ranges = np.concatenate(
        (left.robot.q_max - left.robot.q_min, right.robot.q_max - right.robot.q_min)
    )
    axis_dot = max(
        float(np.max(np.abs(left.robot.consecutive_axis_dot_products()))),
        float(np.max(np.abs(right.robot.consecutive_axis_dot_products()))),
    )
    return RobotValidationReport(
        name=adapter.name,
        urdf=str(path),
        left_base_link=left.base_link,
        right_base_link=right.base_link,
        left_ee_link=left.ee_link,
        right_ee_link=right.ee_link,
        minimum_joint_range=float(np.min(ranges)),
        maximum_consecutive_axis_dot=axis_dot,
        keypoints_finite=bool(np.all(np.isfinite(keypoints))),
    )

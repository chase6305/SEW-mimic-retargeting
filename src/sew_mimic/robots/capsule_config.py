"""Reusable OOBB-to-capsule configuration for robot collision meshes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..collision import Capsule, capsule_from_oobb
from ..safety import CapsuleRadii, SafetyFilterConfig
from .urdf import URDFKinematics


@dataclass(frozen=True)
class CapsuleMeshSpec:
    """Collision-mesh groups representing torso, arm, forearm, and hand."""

    torso: str
    upper_arm: tuple[str, ...]
    lower_arm: tuple[str, ...]
    hand: tuple[str, ...]
    torso_link: str = "torso_base"

    def __post_init__(self) -> None:
        if not self.torso or not self.torso_link:
            raise ValueError("torso mesh and torso_link must not be empty")
        if not self.upper_arm or not self.lower_arm or not self.hand:
            raise ValueError("upper_arm, lower_arm, and hand mesh groups must not be empty")


def estimate_capsule_config_from_meshes(
    urdf_path: str | Path,
    collision_directory: str | Path,
    spec: CapsuleMeshSpec,
    *,
    padding: float = 1.05,
) -> SafetyFilterConfig:
    """Fit one OOBB per unique mesh and return a root-frame capsule config."""
    path = Path(urdf_path).expanduser().resolve()
    collision_root = Path(collision_directory).expanduser().resolve()
    fitted_cache: dict[str, Capsule] = {}

    def fitted(relative: str) -> Capsule:
        if relative not in fitted_cache:
            fitted_cache[relative] = capsule_from_oobb(
                collision_root / relative,
                padding=padding,
                name=Path(relative).stem,
            )
        return fitted_cache[relative]

    def maximum_radius(relative_paths: tuple[str, ...]) -> float:
        return max(fitted(relative).radius for relative in relative_paths)

    torso = fitted(spec.torso)
    radii = CapsuleRadii(
        torso=torso.radius,
        upper_arm=maximum_radius(spec.upper_arm),
        lower_arm=maximum_radius(spec.lower_arm),
        hand=maximum_radius(spec.hand),
    )
    transform = URDFKinematics(path).link_transforms({}, (spec.torso_link,))[spec.torso_link]

    def root_point(point: np.ndarray) -> np.ndarray:
        return transform[:3, :3] @ point + transform[:3, 3]

    return SafetyFilterConfig(
        radii=radii,
        torso_start=root_point(torso.start),
        torso_end=root_point(torso.end),
    )

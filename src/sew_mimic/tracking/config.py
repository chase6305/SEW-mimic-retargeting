"""Immutable, versioned settings shared by live tracking and recorded replay."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields
from typing import ClassVar

import numpy as np

from ..filtering import OneEuroConfig
from .motion import TrackingMotionLimits
from .pipeline import FramePolicy


def _object(pairs: list[tuple[str, object]]) -> dict:
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError("Duplicate tracking configuration field")
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"Nonfinite tracking configuration value: {value}")


def _load(value: str) -> object:
    if not isinstance(value, str) or len(value) > TrackingStreamConfig.MAX_JSON_CHARS:
        raise ValueError("Tracking configuration must be a JSON string of at most 4096 characters")
    try:
        return json.loads(value, object_pairs_hook=_object, parse_constant=_reject_constant)
    except (ValueError, RecursionError) as exc:
        raise ValueError(f"Invalid tracking configuration JSON: {exc}") from exc


def _parameters(cls, value: object, *, exact: bool = True) -> dict:
    names = {item.name for item in fields(cls)}
    if not isinstance(value, dict) or set(value) - names or (exact and set(value) != names):
        raise ValueError(f"Invalid {cls.__name__} fields")
    return value


@dataclass(frozen=True)
class TrackingStreamConfig:
    """Complete input policy and filter tuning; calibration remains an event.

    JSON version 1 stores every setting, including explicit nulls. Strict field
    checks prevent missing settings from silently adopting different defaults.
    Rotation settings of None intentionally inherit the position parameters.
    """

    MAX_JSON_CHARS: ClassVar[int] = 4096

    policy: FramePolicy = field(default_factory=FramePolicy)
    position_filter: OneEuroConfig = field(default_factory=OneEuroConfig)
    rotation_filter: OneEuroConfig | None = None
    motion_limits: TrackingMotionLimits | None = None
    require_tracked: bool = False

    def __post_init__(self) -> None:
        for name, cls, optional in (
            ("policy", FramePolicy, False),
            ("position_filter", OneEuroConfig, False),
            ("rotation_filter", OneEuroConfig, True),
            ("motion_limits", TrackingMotionLimits, True),
        ):
            value = getattr(self, name)
            if not (optional and value is None) and not isinstance(value, cls):
                raise ValueError(
                    f"{name} must be {cls.__name__}" + (" or None" if optional else "")
                )
        if not isinstance(self.require_tracked, (bool, np.bool_)):
            raise ValueError("require_tracked must be a boolean")
        object.__setattr__(self, "require_tracked", bool(self.require_tracked))

    def to_json(self) -> str:
        """Return a deterministic, complete configuration snapshot."""
        return json.dumps(
            {"version": 1, **asdict(self)}, allow_nan=False, sort_keys=True, separators=(",", ":")
        )

    @classmethod
    def from_json(cls, value: str) -> TrackingStreamConfig:
        """Load a bounded snapshot, rejecting unknown versions and partial data."""
        data = _load(value)
        if (
            not isinstance(data, dict)
            or type(data.get("version")) is not int
            or data["version"] != 1
        ):
            raise ValueError("Unsupported tracking configuration version")
        values = _parameters(cls, {name: item for name, item in data.items() if name != "version"})
        return cls(
            policy=FramePolicy(**_parameters(FramePolicy, values["policy"])),
            position_filter=OneEuroConfig(**_parameters(OneEuroConfig, values["position_filter"])),
            rotation_filter=None
            if values["rotation_filter"] is None
            else OneEuroConfig(**_parameters(OneEuroConfig, values["rotation_filter"])),
            motion_limits=None
            if values["motion_limits"] is None
            else TrackingMotionLimits(**_parameters(TrackingMotionLimits, values["motion_limits"])),
            require_tracked=values["require_tracked"],
        )

    def to_metadata(self) -> dict[str, str]:
        """Encode settings within the recording's string-valued metadata format.

        Keep the legacy motion entry for readers that only understand it. Both
        entries come from the same snapshot and are checked for agreement.
        """
        return {
            "tracking_config": self.to_json(),
            "motion_limits": json.dumps(
                None if self.motion_limits is None else asdict(self.motion_limits),
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
        }

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, str]) -> TrackingStreamConfig:
        """Restore complete settings, or the earlier motion-only recording format."""
        if not isinstance(metadata, Mapping):
            raise ValueError("metadata must be a mapping of strings to strings")
        config = (
            cls.from_json(metadata["tracking_config"]) if "tracking_config" in metadata else None
        )
        motion = None
        if "motion_limits" in metadata:
            value = _load(metadata["motion_limits"])
            if value is not None:
                motion = TrackingMotionLimits(
                    **_parameters(TrackingMotionLimits, value, exact=False)
                )
            if config is not None and config.motion_limits != motion:
                raise ValueError("Conflicting tracking_config and motion_limits metadata")
        return config if config is not None else cls(motion_limits=motion)

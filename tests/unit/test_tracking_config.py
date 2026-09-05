import json
from dataclasses import replace

import numpy as np
import pytest

from sew_mimic import OneEuroConfig
from sew_mimic.tracking import (
    FramePolicy,
    TrackingCalibration,
    TrackingIdentity,
    TrackingMotionLimits,
    TrackingPoseStream,
    TrackingStreamConfig,
)


@pytest.fixture
def config():
    return TrackingStreamConfig(
        policy=FramePolicy(0.08, 0.06, 0.02, 0.7),
        position_filter=OneEuroConfig(3.0, 0.4, 2.0),
        rotation_filter=OneEuroConfig(5.0, 0.8, 3.0),
        motion_limits=TrackingMotionLimits(2.0, 6.0, 0.25),
        require_tracked=True,
    )


def test_config_round_trip_preserves_every_setting_and_legacy_constructor(config):
    encoded = config.to_json()
    assert TrackingStreamConfig.from_json(encoded) == config
    assert TrackingStreamConfig.from_json(encoded).to_json() == encoded
    assert TrackingStreamConfig.from_metadata(config.to_metadata()) == config
    identity = TrackingIdentity("tracker", "session", "local")
    calibration = TrackingCalibration(identity)
    stream = TrackingPoseStream.from_config(calibration, config)
    assert stream.config == config
    stream.set_calibration(replace(calibration, tool_length=0.2))
    assert stream.config == config  # Calibration remains a separate recorded event.
    legacy = TrackingPoseStream(
        calibration,
        policy=config.policy,
        min_cutoff=3.0,
        beta=0.4,
        derivative_cutoff=2.0,
        rotation_config=config.rotation_filter,
        motion_limits=config.motion_limits,
        require_tracked=True,
    )
    assert legacy.config.to_metadata() == config.to_metadata()


def test_optional_rotation_inheritance_and_disabled_motion_round_trip():
    config = TrackingStreamConfig(position_filter=OneEuroConfig(5.0, 0.4))
    restored = TrackingStreamConfig.from_metadata(config.to_metadata())
    assert restored == config
    assert restored.rotation_filter is None
    assert restored.motion_limits is None


@pytest.mark.parametrize(
    "section", [None, "policy", "position_filter", "rotation_filter", "motion_limits"]
)
@pytest.mark.parametrize("mutation", ["missing", "unknown"])
def test_partial_or_unknown_configuration_fields_do_not_adopt_defaults(config, section, mutation):
    data = json.loads(config.to_json())
    target = data if section is None else data[section]
    if mutation == "missing":
        del target[next(key for key in target if key != "version")]
    else:
        target["typo"] = 1.0
    with pytest.raises(ValueError, match="fields"):
        TrackingStreamConfig.from_json(json.dumps(data))


@pytest.mark.parametrize("version", [None, True, 1.0, "1", 2])
def test_config_requires_supported_integer_version(config, version):
    data = json.loads(config.to_json())
    data["version"] = version
    with pytest.raises(ValueError, match="version"):
        TrackingStreamConfig.from_json(json.dumps(data))


@pytest.mark.parametrize(
    "value",
    [
        "{}",
        "null",
        "[]",
        "{",
        '{"version":1,"version":1}',
        '{"value":NaN}',
        '{"value":Infinity}',
        "[" * 1100 + "]" * 1100,
        " " * (TrackingStreamConfig.MAX_JSON_CHARS + 1),
    ],
)
def test_invalid_or_unbounded_config_json_is_rejected(value):
    with pytest.raises(ValueError):
        TrackingStreamConfig.from_json(value)


def test_legacy_metadata_restores_only_the_settings_it_contains():
    assert TrackingStreamConfig.from_metadata({"robot": "marvin"}) == TrackingStreamConfig()
    config = TrackingStreamConfig.from_metadata({"motion_limits": '{"max_joint_speed":2}'})
    assert config == TrackingStreamConfig(motion_limits=TrackingMotionLimits(max_joint_speed=2))


def test_full_metadata_cannot_conflict_with_legacy_motion_settings(config):
    metadata = config.to_metadata()
    metadata["motion_limits"] = "null"
    with pytest.raises(ValueError, match="Conflicting"):
        TrackingStreamConfig.from_metadata(metadata)
    assert TrackingStreamConfig.from_metadata({"tracking_config": config.to_json()}) == config


@pytest.mark.parametrize(
    "name,value",
    [
        ("policy", {}),
        ("position_filter", None),
        ("rotation_filter", {}),
        ("motion_limits", {}),
        ("require_tracked", "false"),
        ("require_tracked", 1),
    ],
)
def test_config_rejects_untyped_nested_settings(name, value):
    with pytest.raises(ValueError, match=name):
        TrackingStreamConfig(**{name: value})


def test_frame_policy_owns_scalar_settings_in_configuration_snapshot():
    cutoff = np.array(0.08)
    confidence = np.array(0.7)
    config = TrackingStreamConfig(
        policy=FramePolicy(max_sample_age=cutoff, min_confidence=confidence),
        require_tracked=np.bool_(True),
    )
    before = config.to_json()
    cutoff[...] = 100.0
    confidence[...] = 0.0
    assert config.policy.max_sample_age == 0.08
    assert config.policy.min_confidence == 0.7
    assert config.to_json() == before


@pytest.mark.parametrize(
    "name", ["max_sample_age", "max_receive_age", "max_prediction", "min_confidence"]
)
@pytest.mark.parametrize("value", [True, [0.1], "0.1", np.inf])
def test_frame_policy_requires_numeric_scalar_limits(name, value):
    with pytest.raises(ValueError, match=name):
        FramePolicy(**{name: value})

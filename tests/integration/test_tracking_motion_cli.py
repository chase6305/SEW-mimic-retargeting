import json
import sys
from dataclasses import replace

import pytest

from examples.demo_tracking_input import main
from sew_mimic import OneEuroConfig
from sew_mimic.tracking import (
    FramePolicy,
    TrackingMotionLimits,
    TrackingRecorder,
    TrackingRecordingReader,
    TrackingStreamConfig,
)


def _run(monkeypatch, capsys, *arguments):
    monkeypatch.setattr(sys, "argv", ["demo_tracking_input", *map(str, arguments)])
    main()
    return capsys.readouterr().out


def _digest(output):
    return next(line for line in output.splitlines() if line.startswith("command_digest="))


def test_cli_rejects_unsupported_backend_before_creating_recording(tmp_path, monkeypatch, capsys):
    path = tmp_path / "invalid.jsonl"
    with pytest.raises(SystemExit) as error:
        _run(monkeypatch, capsys, "--backend", "auto", "--record", path)
    assert error.value.code == 2
    assert "invalid choice: 'auto'" in capsys.readouterr().err
    assert not path.exists()


def test_cli_restores_recorded_limits_and_allows_explicit_comparison(tmp_path, monkeypatch, capsys):
    path = tmp_path / "motion.jsonl"
    original = _run(
        monkeypatch,
        capsys,
        "--robot",
        "openarm",
        "--max-joint-speed",
        "2",
        "--max-wrist-angular-speed",
        "6",
        "--max-motion-gap",
        "0.25",
        "--record",
        path,
    )
    with path.open() as stream:
        assert json.loads(TrackingRecordingReader(stream).metadata["motion_limits"]) == {
            "max_joint_speed": 2.0,
            "max_wrist_angular_speed": 6.0,
            "max_sample_gap": 0.25,
        }
    assert _run(monkeypatch, capsys, "--replay", path) == original
    tighter = _run(monkeypatch, capsys, "--replay", path, "--max-motion-gap", "0.01")
    assert "solved=1 held=59" in tighter
    assert "'max_sample_gap': 0.01" in tighter
    assert "'max_joint_speed': 2.0" in tighter  # Unchanged fields are inherited.
    assert _digest(tighter) != _digest(original)
    disabled = _run(monkeypatch, capsys, "--replay", path, "--no-motion-limits")
    assert "motion_limits=None" in disabled
    assert _digest(disabled) == _digest(original)


@pytest.mark.parametrize(
    "arguments",
    [
        ("--max-joint-speed", "0"),
        ("--max-wrist-angular-speed", "nan"),
        ("--max-motion-gap", "0.1"),
        ("--no-motion-limits", "--max-joint-speed", "2"),
    ],
)
def test_bad_limit_options_do_not_create_partial_recording(
    tmp_path, monkeypatch, capsys, arguments
):
    path = tmp_path / "invalid.jsonl"
    with pytest.raises(SystemExit) as error:
        _run(monkeypatch, capsys, "--record", path, *arguments)
    assert error.value.code == 2
    assert not path.exists()


@pytest.mark.parametrize("metadata", [{}, {"motion_limits": "null"}])
def test_old_recording_metadata_keeps_motion_checks_disabled(
    tmp_path, monkeypatch, capsys, metadata
):
    path = tmp_path / "legacy.jsonl"
    with (
        path.open("w") as stream,
        TrackingRecorder(stream, metadata={"robot": "marvin", **metadata}),
    ):
        pass
    result = _run(monkeypatch, capsys, "--replay", path)
    assert "motion_limits=None" in result


@pytest.mark.parametrize(
    "limits",
    [
        "[]",
        "{}",
        '{"unknown": 1}',
        '{"max_joint_speed": "bad"}',
        '{"max_joint_speed": 1, "max_joint_speed": 2}',
        '{"max_joint_speed": NaN}',
        '{"max_joint_speed":',
    ],
)
def test_invalid_recorded_limits_are_reported_by_cli(tmp_path, monkeypatch, capsys, limits):
    path = tmp_path / "bad-limits.jsonl"
    with (
        path.open("w") as stream,
        TrackingRecorder(stream, metadata={"robot": "marvin", "motion_limits": limits}),
    ):
        pass
    with pytest.raises(SystemExit) as error:
        _run(monkeypatch, capsys, "--replay", path)
    assert error.value.code == 2


def test_cli_records_complete_config_and_file_override_is_explicit(tmp_path, monkeypatch, capsys):
    config = TrackingStreamConfig(
        policy=FramePolicy(max_receive_age=0.05),
        position_filter=OneEuroConfig(3.0, 0.4, 2.0),
        rotation_filter=OneEuroConfig(5.0, 0.8, 3.0),
        motion_limits=TrackingMotionLimits(2.0, 6.0, 0.25),
        require_tracked=True,
    )
    config_path = tmp_path / "settings.json"
    config_path.write_text(config.to_json())
    recording_path = tmp_path / "configured.jsonl"
    original = _run(
        monkeypatch,
        capsys,
        "--config",
        config_path,
        "--max-joint-speed",
        "3",
        "--record",
        recording_path,
    )
    expected = replace(config, motion_limits=replace(config.motion_limits, max_joint_speed=3))
    with recording_path.open() as stream:
        reader = TrackingRecordingReader(stream)
        assert TrackingStreamConfig.from_metadata(reader.metadata) == expected
    assert _run(monkeypatch, capsys, "--replay", recording_path) == original
    no_motion = _run(monkeypatch, capsys, "--replay", recording_path, "--no-motion-limits")
    effective = next(
        line.removeprefix("tracking_config=")
        for line in no_motion.splitlines()
        if line.startswith("tracking_config=")
    )
    assert TrackingStreamConfig.from_json(effective) == replace(expected, motion_limits=None)

    defaults_path = tmp_path / "defaults.json"
    defaults_path.write_text(TrackingStreamConfig().to_json())
    changed = _run(monkeypatch, capsys, "--replay", recording_path, "--config", defaults_path)
    assert _digest(changed) != _digest(original)
    assert "solved=50 held=5 unchanged=5" in changed


@pytest.mark.parametrize("value", ["{}", '{"version":2}', " " * 4097])
def test_invalid_config_file_does_not_create_a_recording(tmp_path, monkeypatch, capsys, value):
    config_path = tmp_path / "invalid.json"
    config_path.write_text(value)
    recording_path = tmp_path / "output.jsonl"
    with pytest.raises(SystemExit) as error:
        _run(monkeypatch, capsys, "--config", config_path, "--record", recording_path)
    assert error.value.code == 2
    assert not recording_path.exists()


def test_cli_rejects_conflicting_full_and_legacy_settings(tmp_path, monkeypatch, capsys):
    path = tmp_path / "conflict.jsonl"
    metadata = TrackingStreamConfig().to_metadata()
    metadata["motion_limits"] = '{"max_joint_speed":2}'
    with (
        path.open("w") as stream,
        TrackingRecorder(stream, metadata={"robot": "marvin", **metadata}),
    ):
        pass
    with pytest.raises(SystemExit) as error:
        _run(monkeypatch, capsys, "--replay", path)
    assert error.value.code == 2

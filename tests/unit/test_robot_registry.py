from pathlib import Path

import pytest

from sew_mimic.robots import (
    RobotAdapter,
    available_robots,
    get_robot_adapter,
    register_robot_adapter,
    resolve_robot_urdf,
)


def _unused_loader(path, side):
    raise AssertionError("loader should not be called by registry metadata tests")


def test_application_can_register_robot_adapter_and_alias(tmp_path: Path):
    urdf = tmp_path / "robot.urdf"
    urdf.write_text('<robot name="test-platform"><link name="base"/></robot>', encoding="utf-8")
    adapter = RobotAdapter(
        name="test-platform",
        display_name="Test Platform",
        default_urdf=urdf,
        load_arm=_unused_loader,
        trajectory_profile="test_profile",
    )

    register_robot_adapter(adapter, aliases=("test_platform_alias",))

    assert "test-platform" in available_robots()
    assert get_robot_adapter("TEST_PLATFORM_ALIAS") is adapter
    assert resolve_robot_urdf("test-platform") == urdf.resolve()


def test_registry_rejects_duplicate_names(tmp_path: Path):
    adapter = RobotAdapter(
        name="marvin",
        display_name="Duplicate",
        default_urdf=tmp_path / "unused.urdf",
        load_arm=_unused_loader,
        trajectory_profile="test_profile",
    )
    with pytest.raises(ValueError, match="already registered"):
        register_robot_adapter(adapter)


@pytest.mark.parametrize("alias", ["marvin-m6", "M6", "open_arm"])
def test_bundled_robot_aliases_remain_compatible(alias):
    expected = "openarm" if "open" in alias.lower() else "marvin"
    assert get_robot_adapter(alias).name == expected


def test_bundled_adapters_preserve_robot_specific_keypoint_conventions():
    assert get_robot_adapter("marvin").keypoint_profile == "solver"
    assert get_robot_adapter("openarm").keypoint_profile == "urdf"

from pathlib import Path

import numpy as np
import pytest

from sew_mimic.robots import (
    CollisionDemoProfile,
    RobotAdapter,
    available_robots,
    create_robot_safety_filter,
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


def test_registry_rejects_aliases_equal_after_normalization(tmp_path: Path):
    adapter = RobotAdapter(
        name="duplicate-alias-test",
        display_name="Duplicate Alias Test",
        default_urdf=tmp_path / "unused.urdf",
        load_arm=_unused_loader,
        trajectory_profile="test_profile",
    )
    with pytest.raises(ValueError, match="unique after normalization"):
        register_robot_adapter(adapter, aliases=("test_alias", "test-alias"))


def test_replace_removes_stale_aliases(tmp_path: Path):
    first = RobotAdapter(
        name="replace-test",
        display_name="First",
        default_urdf=tmp_path / "first.urdf",
        load_arm=_unused_loader,
        trajectory_profile="test_profile",
    )
    second = RobotAdapter(
        name="replace-test",
        display_name="Second",
        default_urdf=tmp_path / "second.urdf",
        load_arm=_unused_loader,
        trajectory_profile="test_profile",
    )
    register_robot_adapter(first, aliases=("old-replace-alias",))
    register_robot_adapter(second, aliases=("new-replace-alias",), replace=True)

    assert get_robot_adapter("replace-test") is second
    assert get_robot_adapter("new-replace-alias") is second
    with pytest.raises(ValueError, match="Unknown robot"):
        get_robot_adapter("old-replace-alias")


def test_replace_cannot_shadow_another_canonical_name(tmp_path: Path):
    adapter = RobotAdapter(
        name="canonical-shadow-test",
        display_name="Canonical Shadow Test",
        default_urdf=tmp_path / "unused.urdf",
        load_arm=_unused_loader,
        trajectory_profile="test_profile",
    )
    with pytest.raises(ValueError, match="conflict with canonical names"):
        register_robot_adapter(adapter, aliases=("marvin",), replace=True)
    assert get_robot_adapter("marvin").name == "marvin"


@pytest.mark.parametrize("alias", ["marvin-m6", "M6", "open_arm"])
def test_bundled_robot_aliases_remain_compatible(alias):
    expected = "openarm" if "open" in alias.lower() else "marvin"
    assert get_robot_adapter(alias).name == expected


def test_bundled_adapters_preserve_robot_specific_keypoint_conventions():
    assert get_robot_adapter("marvin").keypoint_profile == "solver"
    assert get_robot_adapter("openarm").keypoint_profile == "urdf"


def test_bundled_adapters_expose_collision_demo_capabilities():
    for name in ("marvin", "openarm"):
        adapter = get_robot_adapter(name)
        assert adapter.safety_filter_factory is not None
        assert adapter.collision_profile is not None


def test_generic_safety_filter_factory_selects_robot_and_backend():
    safety_filter = create_robot_safety_filter("m6", backend="python")
    assert safety_filter.backend == "python"
    assert safety_filter.left.side == "left"
    assert safety_filter.right.side == "right"


def test_collision_profile_generates_continuous_round_trip():
    profile = CollisionDemoProfile(
        neutral_left=np.zeros(7),
        neutral_right=np.ones(7),
        colliding_left=np.full(7, 0.5),
        colliding_right=np.full(7, 1.5),
    )
    times, left, right = profile.trajectory(duration=2.0, fps=10.0)

    assert len(times) == 21
    assert left[0] == pytest.approx(profile.neutral_left)
    assert left[10] == pytest.approx(profile.colliding_left)
    assert left[-1] == pytest.approx(profile.neutral_left)
    assert right[10] == pytest.approx(profile.colliding_right)
    assert not profile.neutral_left.flags.writeable

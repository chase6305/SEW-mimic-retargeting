import zipfile
from pathlib import Path

import pytest

from scripts.verify_wheel import verify_wheel

REQUIRED_ASSETS = (
    "data/assets/Marvin_M6_S_CCS_696_V4.0/robot_with_ee.urdf",
    "data/assets/OpenArm/robot.urdf",
    "data/assets/Marvin_M6_S_CCS_696_V4.0/collision/torso_base.obj",
    "data/assets/OpenArm/collision/torso/torso_base.stl",
)
PACKAGE_FILES = ("sew_mimic/__init__.py", "sew_mimic/py.typed")


def _wheel(path: Path, names: tuple[str, ...]) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        for name in names:
            archive.writestr(name, "test")
    return path


def test_wheel_verifier_accepts_runtime_package_and_assets(tmp_path: Path):
    wheel = _wheel(tmp_path / "valid.whl", (*PACKAGE_FILES, *REQUIRED_ASSETS))
    verify_wheel(wheel)


def test_wheel_verifier_rejects_build_only_cpp_package(tmp_path: Path):
    wheel = _wheel(tmp_path / "cpp.whl", (*REQUIRED_ASSETS, "cpp/sew_mimic_cpp.cpp"))
    with pytest.raises(ValueError, match="build-only"):
        verify_wheel(wheel)


def test_wheel_verifier_requires_typing_marker(tmp_path: Path):
    wheel = _wheel(tmp_path / "untyped.whl", ("sew_mimic/__init__.py", *REQUIRED_ASSETS))
    with pytest.raises(ValueError, match="runtime package files"):
        verify_wheel(wheel)


def test_wheel_verifier_reports_missing_assets(tmp_path: Path):
    wheel = _wheel(tmp_path / "missing.whl", PACKAGE_FILES)
    with pytest.raises(ValueError, match="missing required robot assets"):
        verify_wheel(wheel)


def test_wheel_verifier_distinguishes_native_and_pure_wheels(tmp_path: Path):
    pure = _wheel(tmp_path / "pure.whl", (*PACKAGE_FILES, *REQUIRED_ASSETS))
    native = _wheel(
        tmp_path / "native.whl",
        (
            *PACKAGE_FILES,
            *REQUIRED_ASSETS,
            "sew_mimic/_sew_mimic_cpp.cpython-310-x86_64-linux-gnu.so",
        ),
    )
    verify_wheel(pure, expect_native=False)
    verify_wheel(native, expect_native=True)
    with pytest.raises(ValueError, match=r"missing.*C\+\+"):
        verify_wheel(pure, expect_native=True)
    with pytest.raises(ValueError, match="unexpectedly contains"):
        verify_wheel(native, expect_native=False)


def test_wheel_verifier_compares_complete_asset_inventory(tmp_path: Path):
    assets = tmp_path / "assets"
    for packaged_name in REQUIRED_ASSETS:
        source = assets / packaged_name.removeprefix("data/assets/")
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text("test", encoding="utf-8")
    extra = assets / "OpenArm" / "visual" / "extra.glb"
    extra.parent.mkdir(parents=True, exist_ok=True)
    extra.write_text("test", encoding="utf-8")
    wheel = _wheel(tmp_path / "incomplete.whl", (*PACKAGE_FILES, *REQUIRED_ASSETS))

    with pytest.raises(ValueError, match="inventory differs"):
        verify_wheel(wheel, assets_dir=assets)

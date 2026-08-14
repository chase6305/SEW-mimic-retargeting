"""Validate runtime packages and bundled robot assets in a built wheel."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


def verify_wheel(
    path: Path,
    *,
    assets_dir: Path | None = None,
    expect_native: bool | None = None,
) -> None:
    """Raise when a wheel has the wrong code or robot-asset contents."""
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
    if any(name.startswith("cpp/") for name in names):
        raise ValueError("wheel contains the build-only src/cpp directory as a package")
    required_package_files = ("sew_mimic/__init__.py", "sew_mimic/py.typed")
    missing_package_files = [name for name in required_package_files if name not in names]
    if missing_package_files:
        raise ValueError(f"wheel is missing runtime package files: {missing_package_files}")
    native_extensions = {
        name
        for name in names
        if name.startswith("sew_mimic/_sew_mimic_cpp") and name.lower().endswith((".so", ".pyd"))
    }
    if expect_native is True and not native_extensions:
        raise ValueError("native wheel is missing the sew_mimic C++ extension")
    if expect_native is False and native_extensions:
        raise ValueError("pure Python wheel unexpectedly contains the C++ extension")
    required_suffixes = (
        "assets/Marvin_M6_S_CCS_696_V4.0/robot_with_ee.urdf",
        "assets/OpenArm/robot.urdf",
        "assets/Marvin_M6_S_CCS_696_V4.0/collision/torso_base.obj",
        "assets/OpenArm/collision/torso/torso_base.stl",
    )
    missing = [suffix for suffix in required_suffixes if not any(n.endswith(suffix) for n in names)]
    if missing:
        raise ValueError(f"wheel is missing required robot assets: {missing}")
    if assets_dir is not None:
        assets_dir = assets_dir.resolve()
        if not assets_dir.is_dir():
            raise FileNotFoundError(f"asset source directory not found: {assets_dir}")
        expected_assets = {
            f"assets/{source.relative_to(assets_dir).as_posix()}"
            for source in assets_dir.rglob("*")
            if source.is_file()
        }
        packaged_assets = {
            name[name.index("assets/") :]
            for name in names
            if "assets/" in name and not name.endswith("/")
        }
        missing_assets = sorted(expected_assets - packaged_assets)
        unexpected_assets = sorted(packaged_assets - expected_assets)
        if missing_assets or unexpected_assets:
            raise ValueError(
                "wheel asset inventory differs from source: "
                f"missing={missing_assets}, unexpected={unexpected_assets}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument("--assets", type=Path, default=None, help="compare the complete asset tree")
    kind = parser.add_mutually_exclusive_group()
    kind.add_argument("--expect-native", action="store_true")
    kind.add_argument("--expect-pure", action="store_true")
    args = parser.parse_args()
    expected_native = True if args.expect_native else False if args.expect_pure else None
    verify_wheel(args.wheel, assets_dir=args.assets, expect_native=expected_native)
    print(f"Wheel contents verified: {args.wheel}")


if __name__ == "__main__":
    main()

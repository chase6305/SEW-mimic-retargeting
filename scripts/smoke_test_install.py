"""Smoke-test an installed wheel from outside the source tree."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from sew_mimic import backend_status
from sew_mimic.cli import deployment_info
from sew_mimic.robots import (
    DEFAULT_MARVIN_URDF,
    DEFAULT_OPENARM_URDF,
    URDFKinematics,
    load_marvin_arm,
    load_openarm_arm,
    urdf_bimanual_pose,
)


def _check_robot(urdf: Path, loader) -> None:
    if not urdf.is_relative_to(Path(sys.prefix)):
        raise RuntimeError(f"robot asset is outside the installation prefix: {urdf}")
    left, right = loader(urdf, "left"), loader(urdf, "right")
    pose = urdf_bimanual_pose(URDFKinematics(urdf), left, right, np.zeros(7), np.zeros(7))
    if pose.keypoints().shape != (8, 3):
        raise RuntimeError("installed robot FK did not return eight bimanual keypoints")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expect-native", action="store_true")
    args = parser.parse_args()
    status = backend_status()
    if args.expect_native and not status["cpp_available"]:
        raise RuntimeError("installed native wheel cannot load its C++ extension")
    _check_robot(DEFAULT_MARVIN_URDF, load_marvin_arm)
    _check_robot(DEFAULT_OPENARM_URDF, load_openarm_arm)
    info = deployment_info()
    if not all(robot["urdf_exists"] for robot in info["robots"].values()):
        raise RuntimeError("installed deployment diagnostics report missing robot assets")
    print(f"Installed wheel smoke test passed: prefix={sys.prefix} backend={status}")


if __name__ == "__main__":
    main()

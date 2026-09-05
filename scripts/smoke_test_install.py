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
    URDFBimanualPoseEvaluator,
    URDFKinematics,
    load_marvin_arm,
    load_openarm_arm,
    urdf_bimanual_pose,
)
from sew_mimic.tracking import (
    TrackingCalibration,
    TrackingIdentity,
    TrackingPoseStream,
    TrackingStreamConfig,
)


def _check_robot(urdf: Path, loader, *, backend: str) -> None:
    if not urdf.is_relative_to(Path(sys.prefix)):
        raise RuntimeError(f"robot asset is outside the installation prefix: {urdf}")
    left, right = loader(urdf, "left"), loader(urdf, "right")
    kinematics = URDFKinematics(urdf)
    q_left, q_right = np.linspace(-0.2, 0.3, 7), np.linspace(0.3, -0.2, 7)
    pose = urdf_bimanual_pose(kinematics, left, right, q_left, q_right)
    if pose.keypoints().shape != (8, 3):
        raise RuntimeError("installed robot FK did not return eight bimanual keypoints")
    compiled = URDFBimanualPoseEvaluator(kinematics, left, right, backend=backend)
    evaluated = compiled.evaluate(q_left, q_right)
    np.testing.assert_allclose(evaluated.keypoints(), pose.keypoints(), rtol=0, atol=1e-12)
    for actual, expected in ((evaluated.left, pose.left), (evaluated.right, pose.right)):
        np.testing.assert_allclose(
            actual.tool_orientation, expected.tool_orientation, rtol=0, atol=1e-12
        )


def _check_tracking() -> None:
    config = TrackingStreamConfig(require_tracked=True)
    restored = TrackingStreamConfig.from_metadata(config.to_metadata())
    calibration = TrackingCalibration(TrackingIdentity("smoke", "installed", "stage"))
    stream = TrackingPoseStream.from_config(calibration, restored)
    if stream.config != config:
        raise RuntimeError("installed tracking configuration did not round-trip")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expect-native", action="store_true")
    args = parser.parse_args()
    status = backend_status()
    if args.expect_native and not status["cpp_available"]:
        raise RuntimeError("installed native wheel cannot load its C++ extension")
    backend = "cpp" if args.expect_native else "python"
    _check_robot(DEFAULT_MARVIN_URDF, load_marvin_arm, backend=backend)
    _check_robot(DEFAULT_OPENARM_URDF, load_openarm_arm, backend=backend)
    _check_tracking()
    info = deployment_info()
    if not all(robot["urdf_exists"] for robot in info["robots"].values()):
        raise RuntimeError("installed deployment diagnostics report missing robot assets")
    print(f"Installed wheel smoke test passed: prefix={sys.prefix} backend={status}")


if __name__ == "__main__":
    main()

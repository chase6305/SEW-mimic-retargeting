"""Fit a synthetic tracking reference and both hand/tool offsets without hardware."""

from __future__ import annotations

import numpy as np

from sew_mimic import rot
from sew_mimic.tracking import (
    TrackingCalibration,
    TrackingIdentity,
    fit_hand_tool_rotation,
    fit_rigid_transform,
)


def main() -> None:
    rng = np.random.default_rng(503)
    expected_rotation = rot([0.4, -0.2, 1.0], 0.65)
    expected_translation = np.array([0.3, -0.4, 0.2])
    tracking_points = rng.uniform(-0.5, 0.5, (20, 3))
    robot_points = tracking_points @ expected_rotation.T + expected_translation
    robot_points += rng.normal(0, 0.0003, robot_points.shape)

    world = fit_rigid_transform(tracking_points, robot_points)
    # Limits here are chosen for the synthetic noise level, not a hardware specification.
    world.require_accuracy(max_rms_error=0.002, max_point_error=0.005)
    print(
        f"Reference fit: RMS={world.rms_error * 1000:.3f} mm, max={world.max_error * 1000:.3f} mm"
    )

    hands = np.stack([rot(rng.normal(size=3), rng.uniform(-1.5, 1.5)) for _ in range(12)])
    hand_fits = {}
    for side, offset in (
        ("left", rot([1, 2, 0], 0.4)),
        ("right", rot([0, 1, 2], -0.7)),
    ):
        tools = expected_rotation @ hands @ offset
        tools = np.stack([tool @ rot(rng.normal(size=3), rng.normal(0, 0.001)) for tool in tools])
        fit = fit_hand_tool_rotation(hands, tools, tracking_to_robot_rotation=world.rotation)
        fit.require_accuracy(max_rms_angle=np.deg2rad(1), max_angle=np.deg2rad(2))
        hand_fits[side] = fit
        print(
            f"{side.capitalize()} tool fit: RMS={np.rad2deg(fit.rms_angle):.4f} deg, max={np.rad2deg(fit.max_angle):.4f} deg"
        )

    calibration = TrackingCalibration(
        TrackingIdentity("synthetic-tracker", "calibration-demo", "local"),
        rotation=world.rotation,
        translation=world.translation,
        left_hand_to_tool=hand_fits["left"].rotation,
        right_hand_to_tool=hand_fits["right"].rotation,
    )
    print(f"Calibration ready for identity={calibration.identity}")
    print("Synthetic observations only; no device or actuator connection was opened.")


if __name__ == "__main__":
    main()

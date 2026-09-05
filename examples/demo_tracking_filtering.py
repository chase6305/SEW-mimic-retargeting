"""Compare VR pose smoothing on repeatable synthetic noise and step inputs.

RMS is measured after one second of warmup. Step response is measured separately
without noise; it is filter response time, not headset-to-robot latency. The
profiles illustrate tuning and are not device-specific calibrated settings.
"""

from __future__ import annotations

import argparse

import numpy as np

from sew_mimic import ArmPose, BimanualPose, BimanualPoseFilter, OneEuroConfig, rot


def _pose(position: float, angle: float) -> BimanualPose:
    orientation = rot(np.array([0.0, 0.0, 1.0]), angle)
    arms = []
    for y in (0.3, -0.3):
        shoulder = np.array([position, y, 1.4])
        elbow = shoulder + [0.2, 0.0, -0.3]
        wrist = elbow + [0.3, 0.0, 0.1]
        arms.append(ArmPose(shoulder, elbow, wrist, wrist + 0.1 * orientation[:, 0], orientation))
    return BimanualPose(*arms)


def _signal(pose: BimanualPose) -> np.ndarray:
    rotation = pose.left.tool_orientation
    return np.array([pose.left.shoulder[0], np.arctan2(rotation[1, 0], rotation[0, 0])])


def measure(filter_: BimanualPoseFilter | None, rate: int) -> tuple[np.ndarray, np.ndarray]:
    """Return stationary RMS (metres/radians) and 90% step times (seconds)."""
    rng = np.random.default_rng(42)
    noise, timestamp = [], 0.0
    for index in range(4 * rate):
        timestamp += rng.uniform(0.8, 1.2) / rate
        observation = _pose(rng.normal(0.0, 0.003), rng.normal(0.0, np.deg2rad(0.8)))
        result = observation if filter_ is None else filter_.update(timestamp, observation)
        if index >= rate:
            noise.append(_signal(result))
    rms = np.sqrt(np.mean(np.square(noise), axis=0))

    if filter_ is None:
        return rms, np.zeros(2)
    filter_.reset()
    filter_.update(0.0, _pose(0.0, 0.0))
    amplitude = np.array([0.2, np.deg2rad(30)])
    target = _pose(*amplitude)
    response_times = np.full(2, np.nan)
    for index in range(1, 2 * rate + 1):
        elapsed = index / rate
        progress = _signal(filter_.update(elapsed, target)) / amplitude
        reached = np.isnan(response_times) & (progress >= 0.9)
        response_times[reached] = elapsed
        if np.all(np.isfinite(response_times)):
            break
    return rms, response_times


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rate", type=int, choices=(72, 90, 120), default=90)
    args = parser.parse_args()
    profiles = {
        "raw": None,
        "fixed": BimanualPoseFilter(
            min_cutoff=1.0, beta=0.0, rotation_config=OneEuroConfig(1.5, 0.0), tool_length=0.1
        ),
        "adaptive": BimanualPoseFilter(
            min_cutoff=1.0, beta=5.0, rotation_config=OneEuroConfig(1.5, 1.0), tool_length=0.1
        ),
    }
    print(f"Synthetic input: {args.rate} Hz, +/-20% sample intervals, seed=42")
    print("profile   position RMS mm   rotation RMS deg   position t90 ms   rotation t90 ms")
    for name, filter_ in profiles.items():
        rms, response_times = measure(filter_, args.rate)
        print(
            f"{name:<9} {rms[0] * 1000:>15.3f} {np.rad2deg(rms[1]):>18.3f}"
            f" {response_times[0] * 1000:>17.1f} {response_times[1] * 1000:>17.1f}"
        )
    print("Step input: 0.2 m / 30 deg; response times exclude device, network and robot latency.")


if __name__ == "__main__":
    main()

"""Reproducible Python/C++ SEW, capsule, and safety-filter microbenchmarks."""

from __future__ import annotations

import argparse
import logging
import time

import numpy as np

from examples.demo_robot_collision_avoidance import (
    collision_test_trajectory,
    plan_filtered_trajectory,
)
from sew_mimic import backend_status, minimum_capsule_distance, solve
from sew_mimic.robots import MarvinSafetyFilter, load_marvin_arm


def timed(callback, iterations: int) -> tuple[float, float]:
    start = time.perf_counter()
    for _ in range(iterations):
        callback()
    elapsed = time.perf_counter() - start
    return iterations / elapsed, elapsed / iterations * 1e3


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=20_000)
    parser.add_argument("--trajectory-fps", type=float, default=30.0)
    args = parser.parse_args()
    logging.disable(logging.WARNING)
    print("Backend status:", backend_status())

    arm = load_marvin_arm(side="left")
    q_reference = np.array([-1.2, -0.8, 1.0, -1.2, 0.0, 0.0, 0.0])
    shoulder = np.zeros(3)
    elbow = 0.287 * arm.robot.axis_world(q_reference, 3)
    wrist = elbow + 0.314 * arm.robot.axis_world(q_reference, 5)
    hand = arm.robot.tool_orientation(q_reference)
    for backend in ("python", "cpp"):
        q_current = np.zeros(7)

        def solve_frame() -> None:
            nonlocal q_current
            q_current = solve(
                arm.robot,
                q_current,
                shoulder,
                elbow,
                wrist,
                hand,
                backend=backend,
            )

        rate, latency = timed(solve_frame, args.iterations)
        print(f"{backend:6s} SEW:       {rate:10.1f} Hz  {latency:9.4f} ms/frame")

    _, left, right = collision_test_trajectory(12.0, args.trajectory_fps)
    for backend in ("python", "cpp"):
        safety_filter = MarvinSafetyFilter(backend=backend)
        pose = safety_filter.forward_kinematics(left[len(left) // 2], right[len(right) // 2])
        rate, latency = timed(
            lambda: minimum_capsule_distance(pose.points(), safety_filter.config, backend=backend),
            args.iterations,
        )
        print(f"{backend:6s} capsules:  {rate:10.1f} Hz  {latency:9.4f} ms/frame")
        start = time.perf_counter()
        plan_filtered_trajectory(safety_filter, left, right)
        elapsed = time.perf_counter() - start
        print(
            f"{backend:6s} safety:    {len(left) / elapsed:10.1f} Hz  {elapsed / len(left) * 1e3:9.4f} ms/frame"
        )


if __name__ == "__main__":
    main()

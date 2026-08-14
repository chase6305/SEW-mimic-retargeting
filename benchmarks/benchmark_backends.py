"""Reproducible Python/C++ SEW, capsule, and safety-filter microbenchmarks."""

from __future__ import annotations

import argparse
import logging
import statistics
import time
from dataclasses import dataclass

import numpy as np

from examples.demo_robot_collision_avoidance import (
    collision_test_trajectory,
    plan_filtered_trajectory,
)
from sew_mimic import backend_status, minimum_capsule_distance, solve
from sew_mimic.robots import MarvinSafetyFilter, load_marvin_arm


@dataclass(frozen=True)
class Measurement:
    """Median latency plus observed repeat range for one benchmark workload."""

    latency_ms: float
    minimum_ms: float
    maximum_ms: float

    @property
    def rate(self) -> float:
        return 1e3 / self.latency_ms


def _measurement(latencies_ms: list[float]) -> Measurement:
    return Measurement(
        statistics.median(latencies_ms),
        min(latencies_ms),
        max(latencies_ms),
    )


def timed(callback, iterations: int, repeats: int) -> Measurement:
    """Measure one operation repeatedly and report the median repeat."""
    if iterations <= 0 or repeats <= 0:
        raise ValueError("iterations and repeats must be positive")
    callback()  # Exclude lazy backend/model construction from steady-state timing.
    latencies = []
    for _ in range(repeats):
        start = time.perf_counter()
        for _ in range(iterations):
            callback()
        latencies.append((time.perf_counter() - start) / iterations * 1e3)
    return _measurement(latencies)


def timed_batch(callback, operations: int, repeats: int) -> Measurement:
    """Measure a callback that processes a fixed-size batch of operations."""
    if operations <= 0 or repeats <= 0:
        raise ValueError("operations and repeats must be positive")
    callback()  # Match scalar timing: exclude lazy initialization and cold caches.
    latencies = []
    for _ in range(repeats):
        start = time.perf_counter()
        callback()
        latencies.append((time.perf_counter() - start) / operations * 1e3)
    return _measurement(latencies)


def print_measurement(
    backend: str,
    workload: str,
    result: Measurement,
    *,
    rate_unit: str,
    latency_unit: str,
) -> None:
    """Print throughput and latency with workload-specific operation units."""
    print(
        f"{backend:6s} {workload:9s} {result.rate:10.1f} {rate_unit:9s}  "
        f"{result.latency_ms:9.4f} {latency_unit:8s}  "
        f"range=[{result.minimum_ms:.4f}, {result.maximum_ms:.4f}] {latency_unit}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=20_000)
    parser.add_argument("--trajectory-fps", type=float, default=30.0)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.iterations <= 0 or args.trajectory_fps <= 0.0 or args.repeats <= 0:
        parser.error("iterations, trajectory-fps, and repeats must be positive")
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

        print_measurement(
            backend,
            "SEW:",
            timed(solve_frame, args.iterations, args.repeats),
            rate_unit="solves/s",
            latency_unit="ms/solve",
        )

    _, left, right = collision_test_trajectory(12.0, args.trajectory_fps)
    for backend in ("python", "cpp"):
        safety_filter = MarvinSafetyFilter(backend=backend)
        pose = safety_filter.forward_kinematics(left[len(left) // 2], right[len(right) // 2])
        capsule_result = timed(
            lambda: minimum_capsule_distance(pose.points(), safety_filter.config, backend=backend),
            args.iterations,
            args.repeats,
        )
        print_measurement(
            backend,
            "capsules:",
            capsule_result,
            rate_unit="queries/s",
            latency_unit="ms/query",
        )
        safety_result = timed_batch(
            lambda: plan_filtered_trajectory(safety_filter, left, right),
            len(left),
            args.repeats,
        )
        print_measurement(
            backend,
            "safety:",
            safety_result,
            rate_unit="frames/s",
            latency_unit="ms/frame",
        )


if __name__ == "__main__":
    main()

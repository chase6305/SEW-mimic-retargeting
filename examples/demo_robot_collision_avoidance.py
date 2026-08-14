"""Robot-selectable viser demo of bimanual self-collision avoidance.

The translucent red robot follows an intentionally unsafe target. The solid
robot shows the command allowed by the capsule/XPBD safety filter.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np

from sew_mimic import configure_logging
from sew_mimic.robots import (
    RobotSafetyFilter,
    available_robots,
    create_robot_safety_filter,
    get_robot_adapter,
    resolve_robot_urdf,
)


def collision_test_trajectory(
    duration: float, fps: float, robot: str = "marvin"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One continuous open -> collide -> open motion with no target pause."""
    profile = get_robot_adapter(robot).collision_profile
    if profile is None:
        raise ValueError(f"Robot {robot!r} has no collision demo profile")
    return profile.trajectory(duration, fps)


def plan_filtered_trajectory(
    safety_filter: RobotSafetyFilter,
    desired_left: np.ndarray,
    desired_right: np.ndarray,
    solve_timings: list[float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply the stateful safety filter sequentially to a desired trajectory."""
    safe_left = np.empty_like(desired_left)
    safe_right = np.empty_like(desired_right)
    target_distances = np.empty(len(desired_left))
    command_distances = np.empty(len(desired_left))
    accepted = np.empty(len(desired_left), dtype=bool)
    current_left, current_right = desired_left[0].copy(), desired_right[0].copy()

    for index, (target_left, target_right) in enumerate(zip(desired_left, desired_right)):
        solve_start = time.perf_counter()
        result = safety_filter(current_left, current_right, target_left, target_right)
        solve_elapsed = time.perf_counter() - solve_start
        if solve_timings is not None:
            solve_timings.append(solve_elapsed)
        current_left, current_right = result.q_left, result.q_right
        safe_left[index], safe_right[index] = current_left, current_right
        accepted[index] = result.safe
        if result.desired_minimum_distance is None or result.command_minimum_distance is None:
            raise RuntimeError("safety filter did not report clearance diagnostics")
        target_distances[index] = result.desired_minimum_distance
        command_distances[index] = result.command_minimum_distance
    return safe_left, safe_right, target_distances, command_distances, accepted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", choices=available_robots(), default="marvin")
    parser.add_argument("--backend", choices=("python", "cpp"), default="python")
    parser.add_argument("--urdf", type=Path, default=None, help="override the selected robot URDF")
    parser.add_argument("--duration", type=float, default=12.0)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--padding", type=float, default=1.05)
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    args = parser.parse_args()
    configure_logging(getattr(logging, args.log_level))
    args.urdf = resolve_robot_urdf(args.robot, args.urdf)

    try:
        import viser
        from viser.extras import ViserUrdf
    except ImportError as exc:
        raise SystemExit(
            "Install demo dependencies with: pip install -e '.[visualization,safety]'"
        ) from exc

    logger = logging.getLogger(__name__)
    logger.info("Collision-avoidance demo planning started")
    adapter = get_robot_adapter(args.robot)
    if adapter.safety_filter_factory is None or adapter.collision_profile is None:
        raise SystemExit(
            f"Robot {args.robot!r} does not provide a safety filter and collision profile"
        )
    safety_filter = create_robot_safety_filter(
        args.robot, args.urdf, padding=args.padding, backend=args.backend
    )
    if not isinstance(safety_filter, RobotSafetyFilter):
        raise TypeError("collision demo safety filters must inherit RobotSafetyFilter")
    times, desired_left, desired_right = collision_test_trajectory(
        args.duration, args.fps, args.robot
    )
    solve_timings: list[float] = []
    safe_left, safe_right, target_distance, command_distance, accepted = plan_filtered_trajectory(
        safety_filter,
        desired_left,
        desired_right,
        solve_timings,
    )
    solve_timings_array = np.asarray(solve_timings)
    mean_solve_seconds = float(np.mean(solve_timings_array))
    solve_fps = 1.0 / mean_solve_seconds
    mean_solve_ms = mean_solve_seconds * 1e3
    p95_solve_ms = float(np.percentile(solve_timings_array, 95.0) * 1e3)
    logger.info(
        "Collision-avoidance demo planning completed: samples=%d target_min=%.4f m command_min=%.4f m blocked=%d solve_rate=%.1f Hz mean=%.3f ms p95=%.3f ms",
        len(times),
        float(np.min(target_distance)),
        float(np.min(command_distance)),
        int(np.count_nonzero(~accepted)),
        solve_fps,
        mean_solve_ms,
        p95_solve_ms,
    )

    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("+z")
    server.scene.add_grid("/ground", width=2.5, height=2.5)
    safe_robot = ViserUrdf(server, args.urdf, root_node_name="/safe_robot")
    target_robot = ViserUrdf(
        server,
        args.urdf,
        root_node_name="/unsafe_target",
        mesh_color_override=(1.0, 0.15, 0.15, 0.28),
    )
    joint_names = safe_robot.get_actuated_joint_names()
    left_indices = [joint_names.index(name) for name in safety_filter.left.joint_names]
    right_indices = [joint_names.index(name) for name in safety_filter.right.joint_names]

    target_points = server.scene.add_point_cloud(
        "/collision_debug/unsafe_target_keypoints",
        points=np.zeros((8, 3), dtype=np.float32),
        colors=(255, 50, 50),
        point_size=0.045,
    )
    safe_points = server.scene.add_point_cloud(
        "/collision_debug/safe_command_keypoints",
        points=np.zeros((8, 3), dtype=np.float32),
        colors=(50, 255, 100),
        point_size=0.035,
    )

    with server.gui.add_folder("Collision avoidance"):
        play = server.gui.add_checkbox("Play", initial_value=True)
        loop = server.gui.add_checkbox("Loop", initial_value=True)
        timeline = server.gui.add_slider(
            "Time [s]", min=0.0, max=args.duration, step=1.0 / args.fps, initial_value=0.0
        )
        target_metric = server.gui.add_number(
            "Unsafe target distance [m]", initial_value=0.0, disabled=True
        )
        command_metric = server.gui.add_number(
            "Safe command distance [m]", initial_value=0.0, disabled=True
        )
        state = server.gui.add_text("Filter state", initial_value="SAFE", disabled=True)
        server.gui.add_markdown(
            "**Red translucent:** unfiltered target  \n"
            "**Solid robot / green points:** filtered command  \n"
            "Negative target distance means capsule penetration."
        )

    with server.gui.add_folder("Safety solver performance"):
        server.gui.add_number(
            "End-to-end safety solve rate [Hz]",
            initial_value=round(solve_fps, 1),
            step=0.1,
            disabled=True,
        )
        server.gui.add_number(
            "Mean safety solve [ms/frame]",
            initial_value=round(mean_solve_ms, 3),
            step=0.001,
            disabled=True,
        )
        server.gui.add_number(
            "P95 safety solve [ms/frame]",
            initial_value=round(p95_solve_ms, 3),
            step=0.001,
            disabled=True,
        )
        current_solve_metric = server.gui.add_number(
            "Current sample solve [ms]",
            initial_value=0.0,
            step=0.001,
            disabled=True,
        )
        server.gui.add_markdown(
            "Measures the **complete safety-filter call**: FK, capsule checks, "
            "continuous sampling, XPBD, optional SEW reconstruction, and final "
            "validation. It excludes trajectory generation and viser rendering."
        )

    def show(index: int) -> None:
        index = int(np.clip(index, 0, len(times) - 1))
        safe_config = np.zeros(len(joint_names))
        target_config = np.zeros(len(joint_names))
        safe_config[left_indices], safe_config[right_indices] = safe_left[index], safe_right[index]
        target_config[left_indices], target_config[right_indices] = (
            desired_left[index],
            desired_right[index],
        )
        with server.atomic():
            safe_robot.update_cfg(safe_config)
            target_robot.update_cfg(target_config)
            target_points.points = (
                safety_filter.forward_kinematics(desired_left[index], desired_right[index])
                .points()
                .astype(np.float32)
            )
            safe_points.points = (
                safety_filter.forward_kinematics(safe_left[index], safe_right[index])
                .points()
                .astype(np.float32)
            )
            target_metric.value = float(target_distance[index])
            command_metric.value = float(command_distance[index])
            state.value = "SAFE" if accepted[index] else "BLOCKED / HOLDING LAST SAFE POSE"
            current_solve_metric.value = float(solve_timings_array[index] * 1e3)

    @timeline.on_update
    def _(_) -> None:
        show(int(round(timeline.value * args.fps)))

    show(0)
    logger.info("Open the viser URL above; Ctrl+C exits")
    last_tick = time.monotonic()
    try:
        while True:
            now = time.monotonic()
            if play.value:
                next_time = timeline.value + now - last_tick
                if next_time >= args.duration:
                    if loop.value:
                        next_time %= args.duration
                    else:
                        next_time = args.duration
                        play.value = False
                timeline.value = float(next_time)
            last_tick = now
            time.sleep(1.0 / args.fps)
    except KeyboardInterrupt:
        logger.info("Collision-avoidance demo stopped")


if __name__ == "__main__":
    main()

"""Viser demo of Marvin M6 bimanual self-collision avoidance.

The translucent red robot follows an intentionally unsafe target. The solid
robot shows the command allowed by the capsule/XPBD safety filter.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np

from sew_mimic import configure_logging, minimum_capsule_distance
from sew_mimic.robots import DEFAULT_MARVIN_URDF, MarvinSafetyFilter

NEUTRAL_LEFT = np.array([-0.1974, -0.3738, 0.3438, -0.4240, 0.0, 0.0, 0.0])
NEUTRAL_RIGHT = np.array([0.1974, -0.3738, -0.3438, -0.4240, 0.0, 0.0, 0.0])

# Both arms converge at mid-chest height. Their target capsules penetrate with
# the default OOBB fit without raising the wrists above the shoulders.
COLLIDING_LEFT = np.array([-0.00391, -1.68366, 0.42503, -0.65085, 1.63001, 0.08318, -1.48791])
COLLIDING_RIGHT = np.array([-0.76247, -1.38874, 0.70495, -1.58591, 0.63242, -0.54190, -0.70514])


def minimum_jerk(value: np.ndarray) -> np.ndarray:
    return 10.0 * value**3 - 15.0 * value**4 + 6.0 * value**5


def collision_test_trajectory(
    duration: float, fps: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One continuous open -> collide -> open motion with no target pause."""
    if duration <= 0.0 or fps <= 0.0:
        raise ValueError("duration and fps must be positive")
    times = np.linspace(0.0, duration, int(np.ceil(duration * fps)) + 1)
    phase = times / duration
    # The target never stops. Only the filtered robot pauses at the safety
    # boundary while the unsafe target passes through the collision region.
    blend = np.where(
        phase <= 0.5,
        minimum_jerk(2.0 * phase),
        minimum_jerk(2.0 * (1.0 - phase)),
    )
    left = NEUTRAL_LEFT + blend[:, None] * (COLLIDING_LEFT - NEUTRAL_LEFT)
    right = NEUTRAL_RIGHT + blend[:, None] * (COLLIDING_RIGHT - NEUTRAL_RIGHT)
    return times, left, right


def plan_filtered_trajectory(
    safety_filter: MarvinSafetyFilter,
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
        target_pose = safety_filter.forward_kinematics(target_left, target_right)
        target_distances[index] = minimum_capsule_distance(
            target_pose.points(), safety_filter.config
        )
        solve_start = time.perf_counter()
        result = safety_filter(current_left, current_right, target_left, target_right)
        solve_elapsed = time.perf_counter() - solve_start
        if solve_timings is not None:
            solve_timings.append(solve_elapsed)
        current_left, current_right = result.q_left, result.q_right
        safe_left[index], safe_right[index] = current_left, current_right
        accepted[index] = result.safe
        command_pose = safety_filter.forward_kinematics(current_left, current_right)
        command_distances[index] = minimum_capsule_distance(
            command_pose.points(), safety_filter.config
        )
    return safe_left, safe_right, target_distances, command_distances, accepted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_MARVIN_URDF)
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

    try:
        import viser
        from viser.extras import ViserUrdf
    except ImportError as exc:
        raise SystemExit(
            "Install demo dependencies with: pip install -e '.[visualization,safety]'"
        ) from exc

    logger = logging.getLogger(__name__)
    logger.info("Collision-avoidance demo planning started")
    safety_filter = MarvinSafetyFilter(args.urdf, padding=args.padding)
    times, desired_left, desired_right = collision_test_trajectory(args.duration, args.fps)
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

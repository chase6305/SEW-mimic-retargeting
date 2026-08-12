"""Interactive robot-selectable SEW-Mimic visualization with viser."""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np

from sew_mimic import (
    Serial7DoF,
    alignment_diagnostics,
    configure_logging,
    solve,
    solve_batch,
)
from sew_mimic.robots import (
    RobotArm,
    URDFKinematics,
    available_robots,
    get_robot_adapter,
    load_robot_arm,
    resolve_robot_urdf,
)


def rotation_to_wxyz(R: np.ndarray) -> tuple[float, float, float, float]:
    """Convert a rotation matrix to the normalized quaternion format viser uses."""
    R = np.asarray(R, dtype=np.float64)
    w = np.sqrt(max(0.0, 1.0 + np.trace(R))) / 2.0
    x = np.copysign(np.sqrt(max(0.0, 1.0 + R[0, 0] - R[1, 1] - R[2, 2])) / 2.0, R[2, 1] - R[1, 2])
    y = np.copysign(np.sqrt(max(0.0, 1.0 - R[0, 0] + R[1, 1] - R[2, 2])) / 2.0, R[0, 2] - R[2, 0])
    z = np.copysign(np.sqrt(max(0.0, 1.0 - R[0, 0] - R[1, 1] + R[2, 2])) / 2.0, R[1, 0] - R[0, 1])
    q = np.array([w, x, y, z])
    q /= np.linalg.norm(q)
    return tuple(float(value) for value in q)


def minimum_jerk_trajectory(
    q_start: np.ndarray,
    q_goal: np.ndarray,
    duration: float,
    fps: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Plan a synchronized joint trajectory with zero endpoint velocity/acceleration."""
    if duration <= 0.0:
        raise ValueError("duration must be positive")
    if fps <= 0.0:
        raise ValueError("fps must be positive")
    q_start = np.asarray(q_start, dtype=np.float64)
    q_goal = np.asarray(q_goal, dtype=np.float64)
    if q_start.shape != q_goal.shape:
        raise ValueError("q_start and q_goal must have the same shape")

    times = np.linspace(0.0, duration, max(2, int(np.ceil(duration * fps)) + 1))
    tau = times / duration
    blend = 10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5
    trajectory = q_start[None, :] + blend[:, None] * (q_goal - q_start)[None, :]
    return times, trajectory


def periodic_waypoint_trajectory(
    waypoints: np.ndarray,
    duration: float,
    fps: float,
    cycle_duration: float,
    *,
    tension: float = 0.45,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate a closed waypoint program with a velocity-continuous cardinal spline."""
    waypoints = np.asarray(waypoints, dtype=np.float64)
    if waypoints.ndim != 2 or len(waypoints) < 2:
        raise ValueError("waypoints must be a two-dimensional array with at least two rows")
    if duration <= 0.0 or fps <= 0.0 or cycle_duration <= 0.0:
        raise ValueError("duration, fps, and cycle_duration must be positive")
    if not np.allclose(waypoints[0], waypoints[-1]):
        raise ValueError("the final waypoint must duplicate the first")
    if not 0.0 <= tension <= 1.0:
        raise ValueError("tension must be in [0, 1]")

    times = np.linspace(0.0, duration, max(2, int(np.ceil(duration * fps)) + 1))
    controls = waypoints[:-1]
    segment_count = len(controls)
    curve_position = np.mod(times, cycle_duration) / cycle_duration * segment_count
    segment = np.floor(curve_position).astype(int) % segment_count
    u = curve_position - segment
    p0 = controls[(segment - 1) % segment_count]
    p1 = controls[segment]
    p2 = controls[(segment + 1) % segment_count]
    p3 = controls[(segment + 2) % segment_count]
    tangent_scale = 0.5 * (1.0 - tension)
    m1 = tangent_scale * (p2 - p0)
    m2 = tangent_scale * (p3 - p1)
    u = u[:, None]
    trajectory = (
        (2.0 * u**3 - 3.0 * u**2 + 1.0) * p1
        + (u**3 - 2.0 * u**2 + u) * m1
        + (-2.0 * u**3 + 3.0 * u**2) * p2
        + (u**3 - u**2) * m2
    )
    return times, trajectory


def humanlike_reference_trajectory(
    side: str,
    duration: float,
    fps: float,
    cycle_duration: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a continuous chest-expansion and vertical arm-swing cycle.

    These poses are only used to generate reachable human shoulder-elbow-wrist
    directions and hand orientations. Every output sample is subsequently
    recovered by SEW-Mimic; it is not sent directly to the visualizer.

    A periodic Catmull-Rom curve is used instead of independent minimum-jerk
    segments. It carries velocity through poses rather than stopping at every
    waypoint. Small joint-dependent delays mimic motion propagating from the
    shoulder through the elbow to the wrist.
    """
    if cycle_duration is None:
        cycle_duration = min(60.0, duration)
    if cycle_duration <= 0.0:
        raise ValueError("cycle_duration must be positive")
    sign = 1.0 if side == "left" else -1.0
    neutral = np.array([-0.1974 * sign, -0.3738, 0.3438 * sign, -0.4240, 0, 0, 0])
    chest_cross = np.array([0.1325 * sign, -0.6901, -0.0554 * sign, -1.6296, 0, 0, 0])
    chest_open = np.array([0.2606 * sign, -0.1620, -0.2405 * sign, 0.0549, 0, 0, 0])
    # This pose is intentionally asymmetric in world coordinates: left rises,
    # right lowers, while both wrists remain above the base-link top.
    opposing_swing = (
        np.array([-1.3776, -1.2135, -0.0640, -0.1790, 0, 0, 0])
        if side == "left"
        else np.array([-0.8851, -0.6821, -0.2593, -0.4023, 0, 0, 0])
    )
    swing_mid = 0.5 * (neutral + opposing_swing)
    # Deep guard and near-full extension create a clearly readable punch.
    guard = np.array([-0.7403 * sign, -0.0871, 0.6278 * sign, -2.0206, 0, 0, 0])
    punch = np.array([-0.0425 * sign, -1.3652, 0.0053 * sign, -0.1534, 0, 0, 0])
    run_forward = np.array([-0.0461 * sign, -1.0773, -0.0796 * sign, -0.8984, 0, 0, 0])
    run_back = np.array([0.1194 * sign, 0.3696, -0.1652 * sign, -1.3691, 0, 0, 0])
    # Zombie-style synchronized reach: wrists keep the same height and lateral
    # lanes. Forward/backward travel comes from elbow extension/flexion.
    pull_back = np.array([-0.2309 * sign, -0.1978, 0.1508 * sign, -1.9378, 0, 0, 0])
    push_front = np.array([-0.0548 * sign, -1.1405, 0.0110 * sign, -0.4486, 0, 0, 0])

    # One 60-second program. Every exercise group contains exactly two cycles.
    waypoints = np.stack(
        [
            neutral,
            chest_open,
            chest_cross,
            chest_open,
            chest_cross,
            chest_open,
            neutral,
            swing_mid,
            opposing_swing,
            swing_mid,
            opposing_swing,
            swing_mid,
            neutral,
            guard,
            punch if side == "right" else guard,
            guard,
            punch if side == "left" else guard,
            guard,
            punch if side == "right" else guard,
            guard,
            punch if side == "left" else guard,
            guard,
            neutral,
            pull_back,
            push_front,
            pull_back,
            push_front,
            pull_back,
            neutral,
            run_forward if side == "left" else run_back,
            run_back if side == "left" else run_forward,
            run_forward if side == "left" else run_back,
            run_back if side == "left" else run_forward,
            neutral,
        ]
    )
    # The last waypoint duplicates the first to describe a closed gesture.
    controls = waypoints[:-1]
    times = np.linspace(0.0, duration, max(2, int(np.ceil(duration * fps)) + 1))
    trajectory = np.empty((len(times), waypoints.shape[1]), dtype=np.float64)
    propagation_delays = np.array([0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06])
    arm_delay = 0.0
    for joint_index in range(waypoints.shape[1]):
        phase_time = np.mod(times - arm_delay - propagation_delays[joint_index], cycle_duration)
        curve_position = phase_time / cycle_duration * len(controls)
        i1 = np.floor(curve_position).astype(int) % len(controls)
        u = curve_position - np.floor(curve_position)
        i0 = (i1 - 1) % len(controls)
        i2 = (i1 + 1) % len(controls)
        i3 = (i1 + 2) % len(controls)
        p0 = controls[i0, joint_index]
        p1 = controls[i1, joint_index]
        p2 = controls[i2, joint_index]
        p3 = controls[i3, joint_index]
        trajectory[:, joint_index] = 0.5 * (
            2.0 * p1
            + (-p0 + p2) * u
            + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * u**2
            + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * u**3
        )
    return times, trajectory


def openarm_reference_trajectory(
    side: str,
    duration: float,
    fps: float,
    cycle_duration: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Readable dual-arm exercise program fitted to OpenArm Cartesian poses."""
    if cycle_duration is None:
        cycle_duration = min(60.0, duration)
    if side not in ("left", "right"):
        raise ValueError("side must be 'left' or 'right'")

    if side == "left":
        neutral = np.array([-1.26172, -0.92937, 1.19838, 1.33681, 1.57079, -0.46509, -0.52772])
        chest_open = np.array([-1.24066, -1.22790, 1.24012, 0.75371, 1.57079, -0.46509, -0.52772])
        chest_cross = np.array([-1.49687, -0.20808, 1.44850, 1.26534, 1.57079, -0.73129, -0.32459])
        guard = np.array([-1.27456, -1.08608, 1.50561, 1.51821, 1.57079, -0.46509, -0.52772])
        reach = np.array([-1.47036, -0.30722, 1.57079, 0.21797, 1.57079, -0.46509, -0.52772])
        arm_up = np.array([-2.27412, -1.06364, 1.38618, 1.23989, 1.57079, -0.46509, -0.52772])
        arm_down = np.array([-0.81102, -0.79007, 1.57079, 0.81933, 1.57079, -0.46509, -0.52772])
        pull = np.array([-1.35464, -1.03884, 1.49253, 1.43208, 1.57079, -0.46509, -0.52772])
        push = np.array([-1.46465, -0.43522, 1.57079, 0.29252, 1.57079, -0.46509, -0.52772])
        distal_solutions = np.array(
            [
                [0.0, 0.0, 0.0],
                [-0.10254, -0.04179, -0.86499],
                [-0.01287, 0.10761, 0.67405],
                [-0.07669, 0.23206, 0.02105],
                [-0.10942, 0.04685, -0.47737],
                [-0.41451, -0.78539, 0.11273],
                [-0.17273, 0.65388, -0.51222],
                [-0.10832, 0.15639, -0.00712],
                [-0.11602, 0.03164, -0.52919],
            ]
        )
    else:
        neutral = np.array([0.90530, 0.80607, -1.00774, 1.32608, 1.57079, -0.43390, 0.72121])
        chest_open = np.array([0.82241, 1.05278, -0.36818, 0.74904, 1.57079, -0.43390, 0.72121])
        chest_cross = np.array([1.20158, 0.15643, -1.37080, 1.25447, 1.57079, -0.68932, 0.57426])
        guard = np.array([0.79265, 0.98154, -0.85500, 1.51730, 1.57079, -0.43390, 0.72121])
        reach = np.array([1.31775, 0.20603, -0.57904, 0.17137, 1.57079, -0.43390, 0.72121])
        arm_up = np.array([1.86739, 0.94219, -0.75311, 1.24007, 1.57079, -0.43390, 0.72121])
        arm_down = np.array([0.66111, 0.77253, -1.31202, 0.81127, 1.57079, -0.43390, 0.72121])
        pull = np.array([0.90711, 0.93346, -0.85986, 1.43132, 1.57079, -0.43390, 0.72121])
        push = np.array([1.28141, 0.33548, -0.64148, 0.26140, 1.57079, -0.43390, 0.72121])
        distal_solutions = np.array(
            [
                [0.0, 0.0, 0.0],
                [-0.38473, 0.46028, 0.87182],
                [0.00082, -0.20455, -0.62950],
                [0.03810, 0.07049, -0.00967],
                [-0.60210, 0.16038, 0.49278],
                [0.64411, 0.78539, -0.61779],
                [0.22859, -0.45664, 0.60742],
                [0.04756, 0.16078, -0.00703],
                [-0.52217, 0.21142, 0.54109],
            ]
        )

    named_poses = [neutral, chest_open, chest_cross, guard, reach, arm_up, arm_down, pull, push]
    for pose, distal in zip(named_poses, distal_solutions):
        pose[2] = np.clip(pose[2], -1.45, 1.45)
        pose[4:] = distal

    # Each paired arm receives the complementary waypoint at the same index.
    # This produces chest opening/crossing, opposing vertical swings,
    # alternating reaches, level-wrist push/pull, and running-style swings.
    waypoints = np.stack(
        [
            neutral,
            chest_open,
            chest_cross,
            chest_open,
            neutral,
            arm_up if side == "left" else arm_down,
            neutral,
            arm_down if side == "left" else arm_up,
            neutral,
            guard,
            reach if side == "right" else guard,
            guard,
            reach if side == "left" else guard,
            guard,
            pull,
            push,
            pull,
            neutral,
            reach if side == "left" else pull,
            pull if side == "left" else reach,
            reach if side == "left" else pull,
            neutral,
        ]
    )
    # J5-J7 above compensate the shoulder/elbow rotation so the hand-base
    # frames stay mutually aligned and nearly level throughout each gesture.
    return periodic_waypoint_trajectory(waypoints, duration, fps, cycle_duration)


def retarget_reference_trajectory(
    robot: Serial7DoF,
    reference: np.ndarray,
    solve_timings: list[float] | None = None,
    backend: str = "python",
) -> tuple[np.ndarray, float]:
    """Convert a reachable reference motion to SEW inputs and solve each frame."""
    shoulders = np.zeros((len(reference), 3), dtype=np.float64)
    elbows = np.empty_like(shoulders)
    wrists = np.empty_like(shoulders)
    hands = np.empty((len(reference), 3, 3), dtype=np.float64)
    for index, q_reference in enumerate(reference):
        elbows[index] = 0.287 * robot.axis_world(q_reference, 3)
        wrists[index] = elbows[index] + 0.314 * robot.axis_world(q_reference, 5)
        hands[index] = robot.tool_orientation(q_reference)

    if backend == "cpp":
        solve_start = time.perf_counter()
        solved_array = solve_batch(
            robot,
            np.zeros(7),
            shoulders,
            elbows,
            wrists,
            hands,
            backend="cpp",
        )
        elapsed_per_frame = (time.perf_counter() - solve_start) / len(reference)
        if solve_timings is not None:
            solve_timings.extend([elapsed_per_frame] * len(reference))
    else:
        solved = []
        q_previous = np.zeros(7)
        for shoulder, elbow, wrist, hand in zip(shoulders, elbows, wrists, hands):
            solve_start = time.perf_counter()
            q_previous = solve(
                robot,
                q_previous,
                shoulder,
                elbow,
                wrist,
                hand,
                backend=backend,
            )
            if solve_timings is not None:
                solve_timings.append(time.perf_counter() - solve_start)
            solved.append(q_previous.copy())
        solved_array = np.asarray(solved)

    max_error = 0.0
    for q_solution, shoulder, elbow, wrist, hand in zip(
        solved_array, shoulders, elbows, wrists, hands
    ):
        errors = alignment_diagnostics(robot, q_solution, shoulder, elbow, wrist, hand)
        max_error = max(
            max_error,
            *(
                errors["upper_vector_l2"],
                errors["lower_vector_l2"],
                errors["tool_rotation_fro"],
            ),
        )
    return solved_array, max_error


def reference_keypoint_trajectory(robot: Serial7DoF, reference: np.ndarray) -> np.ndarray:
    """Return shoulder/elbow/wrist targets in the selected arm-base frame."""
    keypoints = np.empty((len(reference), 3, 3), dtype=np.float64)
    for index, q_reference in enumerate(reference):
        shoulder = np.zeros(3)
        elbow = shoulder + 0.287 * robot.axis_world(q_reference, 3)
        wrist = elbow + 0.314 * robot.axis_world(q_reference, 5)
        keypoints[index] = np.stack((shoulder, elbow, wrist))
    return keypoints


def urdf_reference_keypoint_trajectory(
    kinematics: URDFKinematics,
    arm: RobotArm,
    reference: np.ndarray,
) -> np.ndarray:
    """Return physical shoulder, J4 elbow, and configured hand landmark."""
    links = kinematics.joint_child_links
    shoulder_link = links[arm.joint_names[0]]
    elbow_link = links[arm.joint_names[3]]
    wrist_link = arm.ee_link
    zero_transforms = kinematics.link_transforms({})
    T_arm_world = np.linalg.inv(zero_transforms[arm.base_link])
    keypoints = np.empty((len(reference), 3, 3), dtype=np.float64)
    for index, q_reference in enumerate(reference):
        transforms = kinematics.link_transforms(
            dict(zip(arm.joint_names, q_reference)),
            (shoulder_link, elbow_link, wrist_link),
        )
        points_world = np.stack(
            [
                transforms[shoulder_link][:3, 3],
                transforms[elbow_link][:3, 3],
                transforms[wrist_link][:3, 3],
            ]
        )
        keypoints[index] = points_world @ T_arm_world[:3, :3].T + T_arm_world[:3, 3]
    return keypoints


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", choices=available_robots(), default="marvin")
    parser.add_argument("--backend", choices=("python", "cpp"), default="python")
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
        help="library and demo logging verbosity",
    )
    parser.add_argument("--urdf", type=Path, default=None, help="override the selected robot URDF")
    parser.add_argument("--side", choices=("left", "right", "both"), default="both")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--duration", type=float, default=60.0, help="total trajectory duration in seconds"
    )
    parser.add_argument("--fps", type=float, default=60.0, help="trajectory sampling/playback rate")
    parser.add_argument(
        "--playback-speed", type=float, default=1.75, help="viser playback speed multiplier"
    )
    parser.add_argument(
        "--motion-cycle",
        type=float,
        default=60.0,
        help="seconds per gesture cycle; independent of total duration",
    )
    args = parser.parse_args()
    configure_logging(getattr(logging, args.log_level))
    logger = logging.getLogger(__name__)
    logger.info(
        "Viser demo initialization: robot=%s backend=%s sides=%s duration=%.1f s fps=%.1f",
        args.robot,
        args.backend,
        args.side,
        args.duration,
        args.fps,
    )
    adapter = get_robot_adapter(args.robot)
    args.urdf = resolve_robot_urdf(args.robot, args.urdf)

    try:
        import viser
        from viser.extras import ViserUrdf
    except ImportError as exc:
        raise SystemExit(
            "Install visualization dependencies with: pip install -e '.[visualization]'"
        ) from exc

    sides = ("left", "right") if args.side == "both" else (args.side,)
    arms = {side: load_robot_arm(args.robot, side, args.urdf) for side in sides}
    trajectories = {}
    target_keypoints = {}
    solve_timings: list[float] = []
    max_errors = {}
    urdf_fk = URDFKinematics(args.urdf)
    trajectory_times = None
    for side, arm in arms.items():
        logger.info("Trajectory planning started: side=%s backend=%s", side, args.backend)
        trajectory_generators = {
            "marvin_humanlike": humanlike_reference_trajectory,
            "openarm_safe": openarm_reference_trajectory,
        }
        if callable(adapter.trajectory_profile):
            trajectory_generator = adapter.trajectory_profile
        else:
            try:
                trajectory_generator = trajectory_generators[adapter.trajectory_profile]
            except KeyError as exc:
                raise ValueError(
                    f"Robot {adapter.name!r} uses unknown trajectory profile "
                    f"{adapter.trajectory_profile!r}; register a callable trajectory_profile"
                ) from exc
        times, reference = trajectory_generator(side, args.duration, args.fps, args.motion_cycle)
        trajectory, max_error = retarget_reference_trajectory(
            arm.robot, reference, solve_timings, args.backend
        )
        trajectory_times = times
        trajectories[side] = trajectory
        target_keypoints[side] = (
            urdf_reference_keypoint_trajectory(urdf_fk, arm, reference)
            if adapter.keypoint_profile == "urdf"
            else reference_keypoint_trajectory(arm.robot, reference)
        )
        max_errors[side] = max_error
        logger.info(
            "Trajectory planning completed: side=%s frames=%d max_error=%.3g",
            side,
            len(trajectory),
            max_error,
        )
    assert trajectory_times is not None
    total_solve_seconds = float(np.sum(solve_timings))
    arm_solve_fps = len(solve_timings) / total_solve_seconds
    pose_pair_fps = len(trajectory_times) / total_solve_seconds
    mean_solve_ms = total_solve_seconds / len(solve_timings) * 1e3
    logger.info(
        "SEW planning performance: backend=%s arm_rate=%.1f Hz pair_rate=%.1f Hz mean=%.4f ms/arm",
        args.backend,
        arm_solve_fps,
        pose_pair_fps,
        mean_solve_ms,
    )

    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("+z")
    server.scene.add_grid("/ground", width=2.0, height=2.0)
    viser_urdf = ViserUrdf(server, urdf_or_path=args.urdf)
    all_names = viser_urdf.get_actuated_joint_names()
    cfg = np.zeros(len(all_names), dtype=np.float64)
    arm_indices = {
        side: [all_names.index(name) for name in arm.joint_names] for side, arm in arms.items()
    }
    sliders: dict[str, list] = {}
    coordinate_frames: dict[str, list[tuple[str, object]]] = {}
    keypoint_handles = {}
    bone_handles = {}

    def update_coordinate_frames() -> dict[str, np.ndarray]:
        joint_positions = {name: float(cfg[index]) for index, name in enumerate(all_names)}
        transforms = urdf_fk.link_transforms(joint_positions)
        for side in sides:
            for link_name, handle in coordinate_frames[side]:
                T = transforms[link_name]
                handle.position = tuple(float(value) for value in T[:3, 3])
                handle.wxyz = rotation_to_wxyz(T[:3, :3])
        return transforms

    def update_target_skeletons(index: int, transforms: dict[str, np.ndarray]) -> None:
        for side, arm in arms.items():
            T_world_arm = transforms[arm.base_link]
            points_arm = target_keypoints[side][index]
            points_world = points_arm @ T_world_arm[:3, :3].T + T_world_arm[:3, 3]
            keypoint_handles[side].points = points_world.astype(np.float32)
            bone_handles[side].points = np.stack(
                ((points_world[0], points_world[1]), (points_world[1], points_world[2]))
            ).astype(np.float32)

    def update_robot_from_sliders() -> dict[str, np.ndarray]:
        for side in sides:
            cfg[arm_indices[side]] = [slider.value for slider in sliders[side]]
        viser_urdf.update_cfg(cfg)
        return update_coordinate_frames()

    def show_trajectory_sample(index: int) -> None:
        index = int(np.clip(index, 0, len(trajectory_times) - 1))
        with server.atomic():
            for side in sides:
                for slider, value in zip(sliders[side], trajectories[side][index]):
                    slider.value = float(value)
        transforms = update_robot_from_sliders()
        update_target_skeletons(index, transforms)

    for side, arm in arms.items():
        sliders[side] = []
        with server.gui.add_folder(f"{side.title()} arm joints"):
            for name, value, lower, upper_limit in zip(
                arm.joint_names,
                trajectories[side][0],
                arm.robot.q_min,
                arm.robot.q_max,
            ):
                slider = server.gui.add_slider(
                    name,
                    min=float(lower),
                    max=float(upper_limit),
                    step=1e-3,
                    initial_value=float(value),
                )
                slider.on_update(lambda _: update_robot_from_sliders())
                sliders[side].append(slider)

    # Draw the arm base and each moving joint frame with compact axes. The EE
    # frame is intentionally larger so its target orientation is easy to read.
    for side, arm in arms.items():
        coordinate_frames[side] = []
        small_links = [arm.base_link] + [
            urdf_fk.joint_child_links[name] for name in arm.joint_names
        ]
        for link_name in small_links:
            handle = server.scene.add_frame(
                f"/coordinate_frames/{side}/{link_name}",
                axes_length=0.035,
                axes_radius=0.0015,
            )
            coordinate_frames[side].append((link_name, handle))
        ee_handle = server.scene.add_frame(
            f"/coordinate_frames/{side}/{arm.ee_link}",
            axes_length=0.09,
            axes_radius=0.004,
        )
        coordinate_frames[side].append((arm.ee_link, ee_handle))

        # Large semantic keypoints make the human target readable alongside
        # the robot: shoulder=red, elbow=yellow, wrist=cyan.
        initial_points = np.zeros((3, 3), dtype=np.float32)
        keypoint_handles[side] = server.scene.add_point_cloud(
            f"/human_target/{side}/keypoints",
            points=initial_points,
            colors=np.array(
                [[255, 70, 70], [255, 210, 40], [40, 210, 255]],
                dtype=np.uint8,
            ),
            point_size=0.055,
            point_shape="circle",
            point_shading="gradient",
        )
        bone_color = (255, 90, 190) if side == "left" else (70, 160, 255)
        bone_handles[side] = server.scene.add_line_segments(
            f"/human_target/{side}/bones",
            points=np.zeros((2, 2, 3), dtype=np.float32),
            colors=bone_color,
            line_width=6.0,
        )

    with server.gui.add_folder("Coordinate frames"):
        show_frames = server.gui.add_checkbox("Show frames", initial_value=True)

        @show_frames.on_update
        def _(_) -> None:
            for entries in coordinate_frames.values():
                for _, handle in entries:
                    handle.visible = show_frames.value

    with server.gui.add_folder("Human target"):
        show_target = server.gui.add_checkbox("Show shoulder / elbow / wrist", initial_value=True)

        @show_target.on_update
        def _(_) -> None:
            for side in sides:
                keypoint_handles[side].visible = show_target.value
                bone_handles[side].visible = show_target.value

    with server.gui.add_folder("Trajectory"):
        play = server.gui.add_checkbox("Play", initial_value=True)
        loop = server.gui.add_checkbox("Loop", initial_value=True)
        playback_speed = server.gui.add_slider(
            "Playback speed [x]",
            min=0.25,
            max=3.0,
            step=0.05,
            initial_value=float(args.playback_speed),
        )
        timeline = server.gui.add_slider(
            "Time [s]",
            min=0.0,
            max=float(args.duration),
            step=1.0 / args.fps,
            initial_value=0.0,
        )
        restart = server.gui.add_button("Restart")

        @timeline.on_update
        def _(_) -> None:
            index = int(round(timeline.value / args.duration * (len(trajectory_times) - 1)))
            show_trajectory_sample(index)

        @restart.on_click
        def _(_) -> None:
            timeline.value = 0.0
            play.value = True

    with server.gui.add_folder("SEW-Mimic performance"):
        server.gui.add_number(
            "Single-arm solve rate [Hz]",
            initial_value=round(arm_solve_fps, 1),
            step=0.1,
            disabled=True,
        )
        server.gui.add_number(
            "Dual-arm pose-pair rate [pairs/s]",
            initial_value=round(pose_pair_fps, 1),
            step=0.1,
            disabled=True,
        )
        server.gui.add_number(
            "Mean solve latency [ms/arm]",
            initial_value=round(mean_solve_ms, 3),
            step=0.001,
            disabled=True,
        )
        server.gui.add_markdown(
            "Rates measure **SEW-Mimic solving only**. The dual-arm rate assumes "
            "left and right arms are solved sequentially; it is not the viser rendering FPS."
        )

    show_trajectory_sample(0)
    print(f"{adapter.display_name} {', '.join(sides)} arm(s) loaded from {args.urdf}")
    print("Maximum SEW errors:", max_errors)
    print(
        f"Planned {len(trajectory_times)} samples over {args.duration:.2f} s at {args.fps:.1f} Hz"
    )
    print(
        f"SEW-Mimic solve-only throughput: {arm_solve_fps:,.1f} Hz (single-arm solves), "
        f"{pose_pair_fps:,.1f} pairs/s ({len(sides)}-arm sequential pose pairs), "
        f"{mean_solve_ms:.3f} ms/arm mean latency; excludes URDF FK and rendering"
    )
    print(f"Open the viser URL shown above (port {args.port}); Ctrl+C exits.")
    last_tick = time.monotonic()
    try:
        while True:
            now = time.monotonic()
            if play.value:
                next_time = timeline.value + (now - last_tick) * playback_speed.value
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
        pass


if __name__ == "__main__":
    main()

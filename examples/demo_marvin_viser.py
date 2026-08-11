"""Interactive Marvin M6 SEW-Mimic/URDF visualization with viser."""

from __future__ import annotations

import argparse
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from sew_mimic import Serial7DoF, alignment_diagnostics, rot, sew_mimic
from sew_mimic.robots import DEFAULT_MARVIN_URDF, load_marvin_arm, rpy_rotation


class UrdfForwardKinematics:
    """Small URDF FK helper used to place viser coordinate-frame overlays."""

    def __init__(self, urdf_path: Path) -> None:
        root = ET.parse(urdf_path).getroot()
        child_links = set()
        self.joints = []
        self.joint_child_links: dict[str, str] = {}
        for joint in root.findall("joint"):
            parent_element = joint.find("parent")
            child_element = joint.find("child")
            if parent_element is None or child_element is None:
                continue
            origin = joint.find("origin")
            axis = joint.find("axis")
            xyz = np.fromstring("0 0 0" if origin is None else origin.get("xyz", "0 0 0"), sep=" ")
            rpy = np.fromstring("0 0 0" if origin is None else origin.get("rpy", "0 0 0"), sep=" ")
            axis_xyz = np.fromstring("1 0 0" if axis is None else axis.get("xyz", "1 0 0"), sep=" ")
            parent = str(parent_element.get("link"))
            child = str(child_element.get("link"))
            child_links.add(child)
            self.joint_child_links[str(joint.get("name"))] = child
            self.joints.append(
                (
                    str(joint.get("name")),
                    str(joint.get("type")),
                    parent,
                    child,
                    xyz,
                    rpy_rotation(rpy),
                    axis_xyz,
                )
            )
        all_links = {link.get("name") for link in root.findall("link")}
        roots = all_links - child_links
        if len(roots) != 1:
            raise ValueError(f"Expected one URDF root link, found {sorted(roots)}")
        self.root_link = str(roots.pop())

    def link_transforms(self, joint_positions: dict[str, float]) -> dict[str, np.ndarray]:
        transforms = {self.root_link: np.eye(4)}
        pending = list(self.joints)
        while pending:
            progressed = False
            for joint in pending[:]:
                name, joint_type, parent, child, xyz, origin_R, axis = joint
                if parent not in transforms:
                    continue
                T_parent_child = np.eye(4)
                T_parent_child[:3, :3] = origin_R
                T_parent_child[:3, 3] = xyz
                value = float(joint_positions.get(name, 0.0))
                if joint_type in ("revolute", "continuous"):
                    T_motion = np.eye(4)
                    T_motion[:3, :3] = rot(axis, value)
                    T_parent_child = T_parent_child @ T_motion
                elif joint_type == "prismatic":
                    T_parent_child[:3, 3] += origin_R @ (axis * value)
                transforms[child] = transforms[parent] @ T_parent_child
                pending.remove(joint)
                progressed = True
            if not progressed:
                missing = [joint[0] for joint in pending]
                raise ValueError(f"Could not traverse URDF joint tree: {missing}")
        return transforms


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


def retarget_reference_trajectory(
    robot: Serial7DoF,
    reference: np.ndarray,
    solve_timings: list[float] | None = None,
) -> tuple[np.ndarray, float]:
    """Convert a reachable reference motion to SEW inputs and solve each frame."""
    solved = []
    q_previous = np.zeros(7)
    max_error = 0.0
    for q_reference in reference:
        upper = robot.axis_world(q_reference, 3)
        lower = robot.axis_world(q_reference, 5)
        hand = robot.tool_orientation(q_reference)
        shoulder = np.zeros(3)
        elbow = shoulder + 0.287 * upper
        wrist = elbow + 0.314 * lower
        solve_start = time.perf_counter()
        q_previous = sew_mimic(robot, q_previous, shoulder, elbow, wrist, hand)
        if solve_timings is not None:
            solve_timings.append(time.perf_counter() - solve_start)
        errors = alignment_diagnostics(robot, q_previous, shoulder, elbow, wrist, hand)
        max_error = max(
            max_error,
            *(
                errors["upper_vector_l2"],
                errors["lower_vector_l2"],
                errors["tool_rotation_fro"],
            ),
        )
        solved.append(q_previous.copy())
    return np.asarray(solved), max_error


def reference_keypoint_trajectory(robot: Serial7DoF, reference: np.ndarray) -> np.ndarray:
    """Return shoulder/elbow/wrist targets in the selected arm-base frame."""
    keypoints = np.empty((len(reference), 3, 3), dtype=np.float64)
    for index, q_reference in enumerate(reference):
        shoulder = np.zeros(3)
        elbow = shoulder + 0.287 * robot.axis_world(q_reference, 3)
        wrist = elbow + 0.314 * robot.axis_world(q_reference, 5)
        keypoints[index] = np.stack((shoulder, elbow, wrist))
    return keypoints


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_MARVIN_URDF)
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

    try:
        import viser
        from viser.extras import ViserUrdf
    except ImportError as exc:
        raise SystemExit(
            "Install visualization dependencies with: pip install -e '.[visualization]'"
        ) from exc

    sides = ("left", "right") if args.side == "both" else (args.side,)
    arms = {side: load_marvin_arm(args.urdf, side) for side in sides}
    trajectories = {}
    target_keypoints = {}
    solve_timings: list[float] = []
    max_errors = {}
    urdf_fk = UrdfForwardKinematics(args.urdf)
    trajectory_times = None
    for side, arm in arms.items():
        times, reference = humanlike_reference_trajectory(
            side, args.duration, args.fps, args.motion_cycle
        )
        trajectory, max_error = retarget_reference_trajectory(arm.robot, reference, solve_timings)
        trajectory_times = times
        trajectories[side] = trajectory
        target_keypoints[side] = reference_keypoint_trajectory(arm.robot, reference)
        max_errors[side] = max_error
    assert trajectory_times is not None
    total_solve_seconds = float(np.sum(solve_timings))
    arm_solve_fps = len(solve_timings) / total_solve_seconds
    pose_pair_fps = len(trajectory_times) / total_solve_seconds
    mean_solve_ms = total_solve_seconds / len(solve_timings) * 1e3

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
    print(f"Marvin {', '.join(sides)} arm(s) loaded from {args.urdf}")
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

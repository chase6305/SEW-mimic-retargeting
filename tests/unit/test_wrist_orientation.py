from dataclasses import replace

import numpy as np
import pytest

from sew_mimic import (
    Serial7DoF,
    align_wrist_orientation,
    align_wrist_parallel,
    cpp_backend_available,
    rot,
    solve,
)
from sew_mimic.robots import load_robot_arm, resolve_robot_urdf


def _rotations(robot, q):
    """Independent ordered matrix composition, without solver FK helpers."""
    frames = [np.eye(3)]
    for origin, axis, angle in zip(robot.R_local, robot.axes_local, q):
        frames.append(frames[-1] @ origin @ rot(axis, angle))
    return frames


def test_legacy_wrist_entry_point_is_the_same_function():
    assert align_wrist_parallel is align_wrist_orientation


@pytest.mark.parametrize("last_axis", [[1, 0, 0], [0, 0, 1]], ids=["XYX", "XYZ"])
@pytest.mark.parametrize("mount_angle", [0.0, np.pi / 2])
@pytest.mark.parametrize("backend", ["python", "cpp"])
def test_wrist_orders_and_fixed_mounts_recover_full_orientation(last_axis, mount_angle, backend):
    if backend == "cpp" and not cpp_backend_available():
        pytest.skip("native extension is not built")
    robot = Serial7DoF(
        axes_local=np.array(
            [[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 1, 0], last_axis]
        ),
        R_local=np.repeat(np.eye(3)[None], 7, axis=0),
        q_min=np.full(7, -np.pi),
        q_max=np.full(7, np.pi),
        R_7T_local=rot([0, 1, 0], mount_angle),
        R_align=rot([1, 1, 0], 0.37),
    )
    rng = np.random.default_rng(107)
    singular = 0.0 if last_axis[0] else np.pi / 2
    middles = [singular, singular - 1e-3, singular + 1e-3, *rng.uniform(-1, 1, 20)]
    for middle in middles:
        target = np.array([0.2, -0.3, 0.1, -0.4, 0.3, middle, -0.2])
        frames = _rotations(robot, target)
        hand = frames[7] @ robot.R_7T_local @ robot.R_align
        seed = target.copy()
        seed[4:] += 1e-4
        if backend == "python":
            result = seed.copy()
            result[4:] = align_wrist_orientation(robot, seed, hand)
            np.testing.assert_array_equal(result[:4], target[:4])
            actual = _rotations(robot, result)[7] @ robot.R_7T_local @ robot.R_align
            np.testing.assert_allclose(actual, hand, rtol=0, atol=1e-8)
        elbow = 0.3 * (frames[3] @ robot.axes_local[2])
        wrist = elbow + 0.25 * (frames[5] @ robot.axes_local[4])
        result = solve(robot, seed, np.zeros(3), elbow, wrist, hand, backend=backend)
        actual = _rotations(robot, result)[7] @ robot.R_7T_local @ robot.R_align
        np.testing.assert_allclose(actual, hand, rtol=0, atol=1e-8)
        assert np.all(result >= robot.q_min) and np.all(result <= robot.q_max)


@pytest.mark.parametrize("name", ["marvin", "openarm"])
@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("backend", ["python", "cpp"])
def test_bundled_wrist_targets_from_independent_urdf_fk(name, side, backend):
    yourdfpy = pytest.importorskip("yourdfpy")
    if backend == "cpp" and not cpp_backend_available():
        pytest.skip("native extension is not built")
    arm = load_robot_arm(name, side)
    robot = replace(arm.robot, R_align=rot([1, 2, 3], 0.29))
    tree = yourdfpy.URDF.load(resolve_robot_urdf(name), load_meshes=False)
    tree.update_cfg(dict.fromkeys(arm.joint_names, 0.0))
    final_joint = tree.joint_map[arm.joint_names[6]]
    axis = tree.get_transform(final_joint.child, arm.base_link)[:3, :3] @ final_joint.axis
    tool_x = tree.get_transform(arm.ee_link, arm.base_link)[:3, 0]
    assert abs(axis @ tool_x) == pytest.approx(1.0 if name == "marvin" else 0.0, abs=1e-9)
    rng = np.random.default_rng(108)
    boundary = 0.5 * (robot.q_min + robot.q_max)
    boundary[4:] = [robot.q_min[4] + 1e-3, 0.35, robot.q_max[6] - 1e-3]
    for target in [boundary, *rng.uniform(robot.q_min + 1e-3, robot.q_max - 1e-3, size=(30, 7))]:
        tree.update_cfg(dict(zip(arm.joint_names, target)))
        hand = tree.get_transform(arm.ee_link, arm.base_link)[:3, :3] @ robot.R_align
        seed = target.copy()
        seed[4:] += 1e-4
        if backend == "python":
            result = seed.copy()
            result[4:] = align_wrist_orientation(robot, seed, hand)
            tree.update_cfg(dict(zip(arm.joint_names, result)))
            actual = tree.get_transform(arm.ee_link, arm.base_link)[:3, :3] @ robot.R_align
            np.testing.assert_allclose(actual, hand, rtol=0, atol=1e-8)
        frames = _rotations(robot, target)
        elbow = 0.3 * (frames[3] @ robot.axes_local[2])
        wrist = elbow + 0.25 * (frames[5] @ robot.axes_local[4])
        result = solve(robot, seed, np.zeros(3), elbow, wrist, hand, backend=backend)
        tree.update_cfg(dict(zip(arm.joint_names, result)))
        actual = tree.get_transform(arm.ee_link, arm.base_link)[:3, :3] @ robot.R_align
        np.testing.assert_allclose(actual, hand, rtol=0, atol=1e-8)
        assert np.all(result >= robot.q_min) and np.all(result <= robot.q_max)

import math

import numpy as np
import pytest

from sew_mimic import (
    Serial7DoF,
    align_axis,
    align_wrist_orientation,
    cpp_backend_available,
    equivalent_angles_in_limits,
    rot,
    sew_mimic,
    solve,
    sp1,
    sp2,
    sp4,
)


def build_robot():
    axes = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )
    return Serial7DoF(
        axes_local=axes,
        R_local=np.repeat(np.eye(3)[None, :, :], 7, axis=0),
        q_min=-np.pi * np.ones(7),
        q_max=np.pi * np.ones(7),
    )


def test_sp1_quarter_turn():
    theta, _ = sp1([1, 0, 0], [0, 1, 0], [0, 0, 1])
    assert abs(theta - math.pi / 2) < 1e-10


def test_sp4_exact_solution():
    # Rotate x around z until y^T R x = 1 -> theta = +pi/2.
    sols, is_ls = sp4([0, 1, 0], [1, 0, 0], [0, 0, 1], 1.0)
    assert not is_ls
    residuals = [
        abs(np.array([0, 1, 0]) @ (rot([0, 0, 1], t) @ np.array([1, 0, 0])) - 1.0) for t in sols
    ]
    assert min(residuals) < 1e-10


@pytest.mark.parametrize(
    "p1,p2,count,is_ls,residual",
    [
        ([1, 0, 0], [1, 0, 0], 2, False, 0.0),
        ([0, 0, 1], [1, 0, 0], 1, False, 0.0),
        ([0, 0, 1], [0, 1, 0], 1, True, np.sqrt(2.0)),
        ([2, 0, 0], [1, 0, 0], 2, True, 1.0),
    ],
)
def test_sp2_preserves_single_paired_and_least_squares_solutions(p1, p2, count, is_ls, residual):
    solutions, actual_ls = sp2(p1, p2, [0, 0, 1], [0, 1, 0])
    assert len(solutions) == count
    assert actual_ls == is_ls
    for first, second in solutions:
        actual = np.linalg.norm(rot([0, 0, 1], first) @ p1 - rot([0, 1, 0], second) @ p2)
        assert actual == pytest.approx(residual, abs=1e-12)


def test_sp2_random_exact_cases():
    rng = np.random.default_rng(0)
    for _ in range(200):
        k1 = rng.normal(size=3)
        k1 /= np.linalg.norm(k1)
        k2 = rng.normal(size=3)
        k2 /= np.linalg.norm(k2)
        if abs(k1 @ k2) > 0.95:
            continue
        p1 = rng.normal(size=3)
        p1 /= np.linalg.norm(p1)
        a = rng.uniform(-math.pi, math.pi)
        b = rng.uniform(-math.pi, math.pi)
        common = rot(k1, a) @ p1
        p2 = rot(k2, -b) @ common

        sols, _ = sp2(p1, p2, k1, k2)
        best = min(np.linalg.norm(rot(k1, aa) @ p1 - rot(k2, bb) @ p2) for aa, bb in sols)
        assert best < 1e-8


def test_end_to_end_fk_to_sew_to_solver():
    robot = build_robot()
    rng = np.random.default_rng(42)

    for _ in range(100):
        q_gt = rng.uniform(-1.0, 1.0, size=7)
        q0 = rng.uniform(-1.0, 1.0, size=7)

        upper_direction = robot.axis_world(q_gt, 3)
        lower_direction = robot.axis_world(q_gt, 5)
        H = robot.tool_orientation(q_gt)

        s = np.zeros(3)
        e = s + 0.33 * upper_direction
        w = e + 0.27 * lower_direction

        q = sew_mimic(robot, q0, s, e, w, H)

        assert np.linalg.norm(robot.axis_world(q, 3) - upper_direction) < 1e-8
        assert np.linalg.norm(robot.axis_world(q, 5) - lower_direction) < 1e-8
        assert np.linalg.norm(robot.tool_orientation(q) - H, ord="fro") < 1e-8


def test_axis_alignment_selects_nearest_multi_turn_equivalents():
    robot = build_robot()
    robot.q_min[:] = -20.0 * np.pi
    robot.q_max[:] = 20.0 * np.pi
    expected = np.array([0.3 + 12.0 * np.pi, 0.4 - 10.0 * np.pi, 0, 0, 0, 0, 0])
    target = robot.axis_world(expected, 3)
    current = expected.copy()
    current[:2] += 0.01
    assert align_axis(robot, 3, current, target) == pytest.approx(expected[:2], abs=1e-12)


def _solve_with_independent_rotations(robot, current, shoulder, elbow, wrist, hand):
    """Recover each stage separately, including the full six-joint wrist FK."""
    result = current.copy()
    result[:2] = align_axis(robot, 3, result, elbow - shoulder)
    result[2:4] = align_axis(robot, 5, result, wrist - elbow)
    desired = hand @ robot.R_align.T @ robot.R_7T_local.T
    result[4:6] = align_axis(robot, 7, result, desired @ robot.axes_local[6])
    target = desired.T @ (robot.R0(result, 6) @ robot.axes_local[5])
    theta, _ = sp1(robot.R_local[6].T @ robot.axes_local[5], target, -robot.axes_local[6])
    result[6] = equivalent_angles_in_limits(theta, robot.q_min[6], robot.q_max[6], current[6])[0]
    return result


@pytest.mark.parametrize("negative_axes", [False, True])
@pytest.mark.parametrize("backend", ["python", "cpp"])
def test_solver_prefix_matches_independent_fk_with_calibrated_frames(negative_axes, backend):
    if backend == "cpp" and not cpp_backend_available():
        pytest.skip("native extension is not built")
    rng = np.random.default_rng(410)
    frames = [np.eye(3)] + [rot(rng.normal(size=3), rng.uniform(-3, 3)) for _ in range(7)]
    axes = build_robot().axes_local
    if negative_axes:
        axes[::2] *= -1.0
    robot = Serial7DoF(
        axes_local=np.array([frames[i + 1].T @ axis for i, axis in enumerate(axes)]),
        R_local=np.array([frames[i].T @ frames[i + 1] for i in range(7)]),
        q_min=np.full(7, -4.0 * np.pi),
        q_max=np.full(7, 4.0 * np.pi),
        R_7T_local=frames[7].T @ rot([1, 2, -3], 0.41),
        R_align=rot([-2, 1, 3], -0.29),
    )
    for index, reference in enumerate(rng.uniform(-1.3, 1.3, size=(16, 7))):
        if index < 2:
            reference[5] = 0.0  # Wrist singularity: q5 and q7 share the remaining spin.
        current = rng.uniform(-9, 9, size=7)
        original = current.copy()
        shoulder = rng.normal(size=3)
        upper = robot.axis_world(reference, 3)
        lower = robot.axis_world(reference, 5)
        elbow = shoulder + 0.33 * upper
        wrist = elbow + 0.27 * lower
        hand = robot.tool_orientation(reference)

        expected = _solve_with_independent_rotations(robot, current, shoulder, elbow, wrist, hand)
        result = solve(robot, current, shoulder, elbow, wrist, hand, backend=backend)
        np.testing.assert_allclose(result, expected, rtol=0.0, atol=1e-10)
        np.testing.assert_array_equal(current, original)
        np.testing.assert_allclose(robot.axis_world(result, 3), upper, atol=1e-9)
        np.testing.assert_allclose(robot.axis_world(result, 5), lower, atol=1e-9)
        np.testing.assert_allclose(robot.tool_orientation(result), hand, atol=1e-9)
        np.testing.assert_allclose(
            align_wrist_orientation(robot, expected, hand), result[4:], atol=1e-10
        )
        assert np.all(result >= robot.q_min - 1e-10)
        assert np.all(result <= robot.q_max + 1e-10)


def test_public_wrist_accepts_readonly_joints_without_mutating_input():
    robot = build_robot()
    reference = np.array([0.2, -0.3, 0.4, -0.5, 0.6, -0.7, 0.8])
    current = reference.copy()
    current[4:] += 0.1
    original = current.copy()
    current.flags.writeable = False
    hand = robot.tool_orientation(reference)
    wrist = align_wrist_orientation(robot, current, hand)
    np.testing.assert_array_equal(current, original)
    np.testing.assert_allclose(
        robot.tool_orientation(np.concatenate((current[:4], wrist))), hand, atol=1e-12
    )


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_public_wrist_rejects_nonfinite_initial_joints(value):
    current = np.zeros(7)
    current[6] = value
    with pytest.raises(ValueError, match="finite"):
        align_wrist_orientation(build_robot(), current, np.eye(3))


def test_solver_prefix_uses_current_model_after_calibration_changes():
    robot = build_robot()
    reference = np.array([0.2, -0.3, 0.4, -0.5, 0.6, -0.7, 0.8])
    current = np.zeros(7)
    for changed in (False, True):
        if changed:
            robot.R_local[0] = rot([1, 2, 3], 0.37) @ robot.R_local[0]
            robot.axes_local[6] *= -1.0
            robot.R_7T_local[:] = rot([-1, 1, 2], -0.48)
            robot.R_align[:] = rot([2, -1, 3], 0.21)
        shoulder = np.zeros(3)
        elbow = 0.33 * robot.axis_world(reference, 3)
        wrist = elbow + 0.27 * robot.axis_world(reference, 5)
        hand = robot.tool_orientation(reference)
        expected = _solve_with_independent_rotations(robot, current, shoulder, elbow, wrist, hand)
        result = sew_mimic(robot, current, shoulder, elbow, wrist, hand)
        np.testing.assert_allclose(result, expected, atol=1e-12)
        np.testing.assert_allclose(robot.tool_orientation(result), hand, atol=1e-12)
        current = result


def test_relative_rotations_and_world_axes_match_full_fk_with_fixed_transforms():
    rng = np.random.default_rng(103)
    robot = Serial7DoF(
        axes_local=rng.normal(size=(7, 3)),
        R_local=np.array([rot(rng.normal(size=3), rng.uniform(-3, 3)) for _ in range(7)]),
        q_min=np.full(7, -np.pi),
        q_max=np.full(7, np.pi),
    )
    for _ in range(10):
        q = rng.uniform(-3, 3, size=7)
        full = [np.eye(3)]
        for index in range(7):
            full.append(full[-1] @ robot.R_local[index] @ rot(robot.axes_local[index], q[index]))
        for first in range(8):
            for second in range(8):
                np.testing.assert_allclose(
                    robot.R_between(q, first, second), full[first].T @ full[second], atol=1e-14
                )
        for index in range(7):
            np.testing.assert_allclose(
                robot.axis_world(q, index + 1),
                full[index + 1] @ robot.axes_local[index],
                atol=1e-14,
            )


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_solver_rejects_nonfinite_initial_joints(value):
    q = np.zeros(7)
    q[0] = value
    with pytest.raises(ValueError, match="finite"):
        sew_mimic(build_robot(), q, [0, 0, 0], [1, 0, 0], [2, 0, 0], np.eye(3))


@pytest.mark.parametrize("field", ["q_min", "q_max"])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_robot_rejects_nonfinite_limits(field, value):
    from dataclasses import replace

    limits = np.zeros(7)
    limits[0] = value
    with pytest.raises(ValueError, match="finite"):
        replace(build_robot(), **{field: limits})

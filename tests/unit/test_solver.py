import math

import numpy as np

from sew_mimic import Serial7DoF, rot, sew_mimic, sp1, sp2, sp4


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

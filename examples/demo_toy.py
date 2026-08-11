"""Runnable synthetic example for the :mod:`sew_mimic` package.

This toy robot is not a commercial robot. It is intentionally chosen to obey
SEW-Mimic's alternating/perpendicular-axis assumptions and to make the
FK -> SEW target -> closed-form retargeting consistency test easy to inspect.
"""

import numpy as np

from sew_mimic import Serial7DoF, alignment_diagnostics, sew_mimic


def build_toy_robot() -> Serial7DoF:
    # Consecutive axes are perpendicular. Each axis is expressed in its own frame.
    axes = np.array(
        [
            [0.0, 0.0, 1.0],  # h1
            [0.0, 1.0, 0.0],  # h2
            [1.0, 0.0, 0.0],  # h3 -> upper-arm proxy
            [0.0, 1.0, 0.0],  # h4
            [1.0, 0.0, 0.0],  # h5 -> lower-arm proxy
            [0.0, 1.0, 0.0],  # h6
            [1.0, 0.0, 0.0],  # h7 -> wrist/tool pointing axis
        ],
        dtype=np.float64,
    )

    R_local = np.repeat(np.eye(3)[None, :, :], 7, axis=0)
    q_min = -np.pi * np.ones(7)
    q_max = np.pi * np.ones(7)

    return Serial7DoF(
        axes_local=axes,
        R_local=R_local,
        q_min=q_min,
        q_max=q_max,
        R_7T_local=np.eye(3),
        R_align=np.eye(3),
    )


def main() -> None:
    np.set_printoptions(precision=6, suppress=True)
    robot = build_toy_robot()

    # A ground-truth robot pose is converted into the SEW representation.
    q_gt = np.array([0.35, -0.55, 0.42, 0.70, -0.33, 0.27, 0.61])
    upper = robot.axis_world(q_gt, 3)
    lower = robot.axis_world(q_gt, 5)
    H = robot.tool_orientation(q_gt)

    # Artificial human keypoints with the same limb directions. Link lengths do
    # not matter for the SEW orientation objective.
    shoulder = np.zeros(3)
    elbow = shoulder + 0.31 * upper
    wrist = elbow + 0.26 * lower

    q0 = np.array([-0.4, 0.1, -0.2, 0.2, 0.3, -0.1, -0.2])
    q = sew_mimic(robot, q0, shoulder, elbow, wrist, H)

    print("q0   =", q0)
    print("q_gt =", q_gt)
    print("q    =", q)
    print("diagnostics =", alignment_diagnostics(robot, q, shoulder, elbow, wrist, H))
    print("consecutive-axis dot products =", robot.consecutive_axis_dot_products())


if __name__ == "__main__":
    main()

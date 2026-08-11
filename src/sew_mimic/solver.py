"""SEW-Mimic core retargeting solver (paper reproduction).

Implements the core single-arm, parallel-wrist path from:
"A Closed-Form Geometric Retargeting Solver for Upper Body Humanoid Robot Teleoperation".

Included:
- Rodrigues rotations
- IK-Geo canonical Subproblems 1, 2, and 4
- Algorithm 2: AlignAxis
- Algorithm 3: parallel-wrist AlignWrist
- Algorithm 1: SEW-Mimic
- Algorithm 8 helper: MakeFrame

Not included:
- XPBD/capsule safety filter (Algorithms 4, 9-12)
- perpendicular-wrist Euler decomposition (Appendix C), because it requires
  robot-specific Euler-axis ordering and frame conventions.

The solver assumes a 7-DoF serial revolute chain with consecutive joint axes
perpendicular in the sense required by the paper.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

EPS = 1e-10
TWO_PI = 2.0 * math.pi


class SEWMimicError(RuntimeError):
    """Base exception for the solver."""


class DegenerateGeometryError(SEWMimicError):
    """Raised for an ill-posed geometric subproblem."""


class JointLimitError(SEWMimicError):
    """Raised when no equivalent analytical solution obeys joint limits."""


def _as_vec3(v: Sequence[float]) -> np.ndarray:
    a = np.asarray(v, dtype=np.float64).reshape(-1)
    if a.shape != (3,):
        raise ValueError(f"Expected a 3-vector, got shape {a.shape}")
    return a


def unit(v: Sequence[float], eps: float = EPS) -> np.ndarray:
    v = _as_vec3(v)
    n = float(np.linalg.norm(v))
    if n < eps:
        raise DegenerateGeometryError("Cannot normalize a near-zero vector")
    return v / n


def skew(v: Sequence[float]) -> np.ndarray:
    x, y, z = _as_vec3(v)
    return np.array(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]],
        dtype=np.float64,
    )


def rot(k: Sequence[float], theta: float) -> np.ndarray:
    """Rodrigues rotation R(k, theta)."""
    k = unit(k)
    K = skew(k)
    return np.eye(3) + math.sin(theta) * K + (1.0 - math.cos(theta)) * (K @ K)


def wrap_to_pi(theta: float) -> float:
    return math.atan2(math.sin(theta), math.cos(theta))


def _is_rotation_matrix(R: np.ndarray, tol: float = 1e-7) -> bool:
    R = np.asarray(R, dtype=np.float64)
    if R.shape != (3, 3):
        return False
    return (
        np.linalg.norm(R.T @ R - np.eye(3), ord="fro") <= tol and abs(np.linalg.det(R) - 1.0) <= tol
    )


def make_frame(
    keypoint_left: Sequence[float],
    keypoint_right: Sequence[float],
    keypoint_bottom: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray]:
    """Paper Algorithm 8: create a body-centric frame from 3 keypoints.

    Returns (R, p), where columns of R are front/left/up axes expressed in the
    input coordinate system, and p is the midpoint of the left/right points.

    Note: the paper's printed Algorithm 8 line 7 shows [ux, uy, uy]; this code
    uses [ux, uy, uz], which is consistent with lines 4-6 and orthonormality.
    """
    kl = _as_vec3(keypoint_left)
    kr = _as_vec3(keypoint_right)
    kb = _as_vec3(keypoint_bottom)

    p = 0.5 * (kl + kr)
    uy = unit(kl - kr)
    ux = unit(np.cross(uy, p - kb))
    uz = unit(np.cross(ux, uy))
    R = np.column_stack((ux, uy, uz))
    return R, p


@dataclass
class Serial7DoF:
    """Minimal orientation-only robot model needed by SEW-Mimic.

    Parameters follow the paper convention:
        R^(i-1,i)(q_i) = R_local[i] @ R(h_i, q_i)

    axes_local[i]
        Joint axis h_i expressed in the i-th joint/link local frame.
    R_local[i]
        Fixed 3x3 orientation from child frame i+1 to parent frame i at zero
        joint angle, using Python's 0-based array indexing.
    q_min, q_max
        Joint limits in radians.
    R_7T_local
        Fixed end-effector/tool-frame orientation in the 7th joint frame.
    R_align
        Fixed right-hand-rule alignment transform used by the paper.

    The tool orientation is implemented as:
        T(q) = R^(0,7)(q) @ R_7T_local @ R_align
    """

    axes_local: np.ndarray
    R_local: np.ndarray
    q_min: np.ndarray
    q_max: np.ndarray
    R_7T_local: np.ndarray = None
    R_align: np.ndarray = None

    def __post_init__(self) -> None:
        self.axes_local = np.asarray(self.axes_local, dtype=np.float64)
        self.R_local = np.asarray(self.R_local, dtype=np.float64)
        self.q_min = np.asarray(self.q_min, dtype=np.float64).reshape(-1)
        self.q_max = np.asarray(self.q_max, dtype=np.float64).reshape(-1)
        self.R_7T_local = (
            np.eye(3) if self.R_7T_local is None else np.asarray(self.R_7T_local, dtype=np.float64)
        )
        self.R_align = (
            np.eye(3) if self.R_align is None else np.asarray(self.R_align, dtype=np.float64)
        )

        if self.axes_local.shape != (7, 3):
            raise ValueError(f"axes_local must have shape (7,3), got {self.axes_local.shape}")
        if self.R_local.shape != (7, 3, 3):
            raise ValueError(f"R_local must have shape (7,3,3), got {self.R_local.shape}")
        if self.q_min.shape != (7,) or self.q_max.shape != (7,):
            raise ValueError("q_min and q_max must each have shape (7,)")
        if np.any(self.q_min > self.q_max):
            raise ValueError("Every q_min must be <= q_max")

        self.axes_local = np.stack([unit(a) for a in self.axes_local], axis=0)
        for i, R in enumerate(self.R_local):
            if not _is_rotation_matrix(R):
                raise ValueError(f"R_local[{i}] is not a valid SO(3) rotation matrix")
        if not _is_rotation_matrix(self.R_7T_local):
            raise ValueError("R_7T_local is not a valid SO(3) rotation matrix")
        if not _is_rotation_matrix(self.R_align):
            raise ValueError("R_align is not a valid SO(3) rotation matrix")

    @property
    def dof(self) -> int:
        return 7

    def joint_rotation(self, joint_index: int, q_i: float) -> np.ndarray:
        """Parent<-child rotation for one joint, joint_index is 0-based."""
        return self.R_local[joint_index] @ rot(self.axes_local[joint_index], q_i)

    def R0(self, q: Sequence[float], frame: int) -> np.ndarray:
        """Rotation R^(0,frame); frame is 0..7."""
        q = np.asarray(q, dtype=np.float64).reshape(-1)
        if q.shape != (7,):
            raise ValueError("q must have shape (7,)")
        if not 0 <= frame <= 7:
            raise ValueError("frame must be in [0, 7]")

        R = np.eye(3)
        for j in range(frame):
            R = R @ self.joint_rotation(j, q[j])
        return R

    def R_between(self, q: Sequence[float], frame_a: int, frame_b: int) -> np.ndarray:
        """Rotation R^(a,b), mapping a vector in frame b into frame a."""
        return self.R0(q, frame_a).T @ self.R0(q, frame_b)

    def axis_world(self, q: Sequence[float], joint_number: int) -> np.ndarray:
        """World/base-frame direction of h_i; joint_number is paper-style 1..7."""
        if not 1 <= joint_number <= 7:
            raise ValueError("joint_number must be in [1, 7]")
        return self.R0(q, joint_number) @ self.axes_local[joint_number - 1]

    def tool_orientation(self, q: Sequence[float]) -> np.ndarray:
        return self.R0(q, 7) @ self.R_7T_local @ self.R_align

    def consecutive_axis_dot_products(self) -> np.ndarray:
        """Check the paper's consecutive-perpendicular-axis assumption at q=0.

        Each next axis is first expressed in the preceding joint frame.
        Values should be close to zero for the ideal SEW-Mimic morphology.
        """
        out = []
        for i in range(6):
            h_prev = self.axes_local[i]
            h_next_in_prev = self.R_local[i + 1] @ self.axes_local[i + 1]
            out.append(float(np.dot(unit(h_prev), unit(h_next_in_prev))))
        return np.asarray(out)


# ---------------------------------------------------------------------------
# Canonical geometric subproblems
# ---------------------------------------------------------------------------


def sp1(p1: Sequence[float], p2: Sequence[float], k: Sequence[float]) -> Tuple[float, bool]:
    """IK-Geo Subproblem 1.

    Solve min_theta ||R(k,theta)p1 - p2||.
    Returns (theta, is_least_squares).
    """
    p1 = _as_vec3(p1)
    p2 = _as_vec3(p2)
    k = unit(k)

    kxp = np.cross(k, p1)
    A = np.column_stack((kxp, -np.cross(k, kxp)))
    x = A.T @ p2

    if np.linalg.norm(x) < EPS:
        # Ill-posed: p1 or p2 is effectively parallel to k. Any angle has the
        # same projected objective; choose 0 for continuity.
        theta = 0.0
        return theta, True

    theta = math.atan2(float(x[0]), float(x[1]))
    is_ls = (
        abs(float(np.linalg.norm(p1)) - float(np.linalg.norm(p2))) > 1e-8
        or abs(float(np.dot(k, p1) - np.dot(k, p2))) > 1e-8
    )
    return theta, is_ls


def sp4(
    h: Sequence[float],
    p: Sequence[float],
    k: Sequence[float],
    d: float,
) -> Tuple[np.ndarray, bool]:
    """IK-Geo Subproblem 4 (circle-plane), using reference argument order.

    Solve min_theta |h^T R(k,theta)p - d|.

    IMPORTANT:
    The SEW-Mimic PDF's Algorithm 7 heading prints SP4(p,h,k,d), while its
    Algorithm 6 call order matches the IK-Geo reference implementation
    SP4(h,p,k,d). This function follows the IK-Geo reference order.
    """
    h = unit(h)
    p = _as_vec3(p)
    k = unit(k)
    d = float(d)

    A11 = np.cross(k, p)
    A1 = np.column_stack((A11, -np.cross(k, A11)))
    A = h @ A1
    b = d - float(np.dot(h, k) * np.dot(k, p))
    norm_A_2 = float(np.dot(A, A))

    if norm_A_2 < EPS:
        # Ill-posed: objective is independent of theta.
        current_residual = abs(float(np.dot(h, p)) - d)
        return np.array([0.0]), current_residual > 1e-8

    # This is the scaled least-squares vector used by the IK-Geo reference.
    x_ls_tilde = A1.T @ (h * b)

    if norm_A_2 > b * b + 1e-12:
        xi = math.sqrt(max(0.0, norm_A_2 - b * b))
        x_N_prime_tilde = np.array([A[1], -A[0]], dtype=np.float64)
        sc1 = x_ls_tilde + xi * x_N_prime_tilde
        sc2 = x_ls_tilde - xi * x_N_prime_tilde
        theta = np.array(
            [
                math.atan2(float(sc1[0]), float(sc1[1])),
                math.atan2(float(sc2[0]), float(sc2[1])),
            ],
            dtype=np.float64,
        )
        return theta, False

    theta = np.array(
        [math.atan2(float(x_ls_tilde[0]), float(x_ls_tilde[1]))],
        dtype=np.float64,
    )
    # The tangent case norm_A_2 == b^2 has one exact solution. The MATLAB
    # reference groups equality into its one-solution branch, so determine the
    # least-squares flag from the actual residual instead of branch identity.
    residual = abs(float(h @ (rot(k, float(theta[0])) @ p)) - d)
    return theta, residual > 1e-8


def sp2(
    p1: Sequence[float],
    p2: Sequence[float],
    k1: Sequence[float],
    k2: Sequence[float],
) -> Tuple[List[Tuple[float, float]], bool]:
    """IK-Geo Subproblem 2.

    Solve min ||R(k1,theta1)p1 - R(k2,theta2)p2||.
    Returns up to two paired solutions and a least-squares flag.
    """
    p1 = _as_vec3(p1)
    p2 = _as_vec3(p2)
    k1 = unit(k1)
    k2 = unit(k2)

    # Reference implementation is ill-posed when k1 || k2.
    if np.linalg.norm(np.cross(k1, k2)) < EPS:
        raise DegenerateGeometryError("SP2 is ill-posed because k1 and k2 are parallel")

    p1_n = unit(p1)
    p2_n = unit(p2)

    theta1, ls1 = sp4(k2, p1_n, k1, float(np.dot(k2, p2_n)))
    theta2, ls2 = sp4(k1, p2_n, k2, float(np.dot(k1, p1_n)))

    # IK-Geo pairs the two circle intersections in opposite order.
    if len(theta1) > 1 or len(theta2) > 1:
        t1 = np.array([theta1[0], theta1[-1]], dtype=np.float64)
        t2 = np.array([theta2[-1], theta2[0]], dtype=np.float64)
    else:
        t1 = theta1
        t2 = theta2

    solutions = [(float(a), float(b)) for a, b in zip(t1, t2)]
    is_ls = abs(float(np.linalg.norm(p1)) - float(np.linalg.norm(p2))) > 1e-8 or ls1 or ls2
    return solutions, is_ls


# ---------------------------------------------------------------------------
# Joint limits and SEW solver
# ---------------------------------------------------------------------------


def equivalent_angles_in_limits(
    theta: float,
    q_min: float,
    q_max: float,
    reference: float,
) -> List[float]:
    """All theta + 2*pi*k values inside limits, sorted by closeness to reference."""
    theta = float(theta)
    q_min = float(q_min)
    q_max = float(q_max)
    reference = float(reference)

    k_min = math.floor((q_min - theta) / TWO_PI) - 1
    k_max = math.ceil((q_max - theta) / TWO_PI) + 1
    vals = []
    for k in range(k_min, k_max + 1):
        a = theta + TWO_PI * k
        if q_min - 1e-10 <= a <= q_max + 1e-10:
            vals.append(a)
    vals.sort(key=lambda a: abs(a - reference))
    return vals


def _bound_angle(theta: float, robot: Serial7DoF, idx: int, reference: float) -> float:
    vals = equivalent_angles_in_limits(theta, robot.q_min[idx], robot.q_max[idx], reference)
    if not vals:
        raise JointLimitError(
            f"No equivalent solution for q{idx + 1}={theta:.6f} rad lies inside "
            f"[{robot.q_min[idx]:.6f}, {robot.q_max[idx]:.6f}]"
        )
    return vals[0]


def align_axis(
    robot: Serial7DoF,
    joint_number_i: int,
    q0: Sequence[float],
    target_vector: Sequence[float],
) -> Tuple[float, float]:
    """Paper Algorithm 2: align axis h_i with target_vector.

    joint_number_i is paper-style 1-based and should be one of 3, 5, 7 for
    standard SEW-Mimic usage. Solves q_{i-2}, q_{i-1}.

    Implementation detail:
    The two joints being solved are zeroed when constructing the local SP2
    geometry. This makes SP2 return absolute values for q_{i-2}, q_{i-1},
    matching Algorithm 2's BoundJoints/closest-to-q0 interpretation. The PDF
    does not explicitly state this zeroing step, but it is needed to turn the
    printed local geometry into absolute joint-angle solutions.
    """
    if joint_number_i not in (3, 5, 7):
        raise ValueError("SEW-Mimic AlignAxis expects i in {3, 5, 7}")

    q0 = np.asarray(q0, dtype=np.float64).reshape(-1)
    if q0.shape != (7,):
        raise ValueError("q0 must have shape (7,)")
    v = unit(target_vector)

    idx_a = joint_number_i - 3  # q_{i-2}, 0-based
    idx_b = joint_number_i - 2  # q_{i-1}, 0-based
    frame = joint_number_i - 2  # frame i-2, paper numbering

    q_base = q0.copy()
    q_base[idx_a] = 0.0
    q_base[idx_b] = 0.0

    # v^(i-2)
    v_f = robot.R0(q_base, frame).T @ v

    # h_i expressed in frame i-2
    h_i_f = robot.R_between(q_base, frame, joint_number_i) @ robot.axes_local[joint_number_i - 1]

    # h_{i-1} expressed in frame i-2.
    # This is R^(i-2,i-1) h_{i-1}; the PDF's Algorithm 2 line 3 appears to
    # contain a frame-index typo in the printed rotation superscripts.
    h_prev_f = (
        robot.R_between(q_base, frame, joint_number_i - 1) @ robot.axes_local[joint_number_i - 2]
    )

    # In frame i-2, changing q_{i-2} rotates the target by -h_{i-2}; changing
    # q_{i-1} rotates h_i around h_{i-1}.
    h_first = robot.axes_local[joint_number_i - 3]
    raw_solutions, _ = sp2(v_f, h_i_f, -h_first, h_prev_f)

    candidates: List[Tuple[float, float, float]] = []
    for qa_raw, qb_raw in raw_solutions:
        qa_vals = equivalent_angles_in_limits(
            qa_raw, robot.q_min[idx_a], robot.q_max[idx_a], q0[idx_a]
        )
        qb_vals = equivalent_angles_in_limits(
            qb_raw, robot.q_min[idx_b], robot.q_max[idx_b], q0[idx_b]
        )
        for qa in qa_vals:
            for qb in qb_vals:
                cost = abs(q0[idx_a] - qa) + abs(q0[idx_b] - qb)
                candidates.append((cost, qa, qb))

    if not candidates:
        raise JointLimitError(
            f"AlignAxis(i={joint_number_i}) found analytical solutions, but none obey joint limits"
        )

    candidates.sort(key=lambda x: x[0])
    return candidates[0][1], candidates[0][2]


def align_wrist_parallel(
    robot: Serial7DoF,
    q0: Sequence[float],
    H: np.ndarray,
) -> np.ndarray:
    """Paper Algorithm 3 for the parallel-wrist case.

    H is the desired human hand orientation in the robot/body-centric base frame.
    Returns [q5, q6, q7].

    The PDF uses R_des[:,1] in 1-based notation to align h7. Here we use the
    coordinate-free equivalent R_des @ h7, which reduces to the first column
    when h7 is the local +X axis and also supports other local h7 conventions.
    """
    q = np.asarray(q0, dtype=np.float64).reshape(-1).copy()
    H = np.asarray(H, dtype=np.float64)
    if q.shape != (7,):
        raise ValueError("q0 must have shape (7,)")
    if not _is_rotation_matrix(H):
        raise ValueError("H must be a valid 3x3 SO(3) rotation matrix")

    # Desired orientation of the 7th joint frame.
    R07_des = H @ robot.R_align.T @ robot.R_7T_local.T

    # Align h7 first using q5,q6.
    h7_world_des = R07_des @ robot.axes_local[6]
    q5, q6 = align_axis(robot, 7, q, h7_world_des)
    q[4] = q5
    q[5] = q6

    # Desired expression of h6 in frame 7.
    u6_in_7 = R07_des.T @ (robot.R0(q, 6) @ robot.axes_local[5])

    # h6 expressed in frame 7 when q7=0; R_local[6] is R^(6,7)_local.
    h6_in_7_at_q7_zero = robot.R_local[6].T @ robot.axes_local[5]

    q7_raw, _ = sp1(h6_in_7_at_q7_zero, u6_in_7, -robot.axes_local[6])
    q7 = _bound_angle(q7_raw, robot, 6, q0[6])

    return np.array([q5, q6, q7], dtype=np.float64)


def sew_mimic(
    robot: Serial7DoF,
    q0: Sequence[float],
    shoulder: Sequence[float],
    elbow: Sequence[float],
    wrist: Sequence[float],
    hand_orientation: np.ndarray,
) -> np.ndarray:
    """Paper Algorithm 1: single-arm SEW-Mimic retargeting.

    Inputs must already be in a shared body-centric base frame:
      shoulder, elbow, wrist: 3D human keypoints
      hand_orientation: 3x3 human hand orientation H
      q0: current/initial robot joint configuration

    Returns a 7-element joint vector q.
    """
    q = np.asarray(q0, dtype=np.float64).reshape(-1).copy()
    if q.shape != (7,):
        raise ValueError("q0 must have shape (7,)")

    s = _as_vec3(shoulder)
    e = _as_vec3(elbow)
    w = _as_vec3(wrist)
    H = np.asarray(hand_orientation, dtype=np.float64)

    upper_direction = unit(e - s)
    lower_direction = unit(w - e)

    q[0:2] = align_axis(robot, 3, q, upper_direction)
    q[2:4] = align_axis(robot, 5, q, lower_direction)
    q[4:7] = align_wrist_parallel(robot, q, H)
    return q


def alignment_diagnostics(
    robot: Serial7DoF,
    q: Sequence[float],
    shoulder: Sequence[float],
    elbow: Sequence[float],
    wrist: Sequence[float],
    hand_orientation: np.ndarray,
) -> dict:
    """Useful numerical checks for a solved pose."""
    s = _as_vec3(shoulder)
    e = _as_vec3(elbow)
    w = _as_vec3(wrist)
    upper_direction = unit(e - s)
    lower_direction = unit(w - e)
    H = np.asarray(hand_orientation, dtype=np.float64)

    upper = unit(robot.axis_world(q, 3))
    lower = unit(robot.axis_world(q, 5))
    T = robot.tool_orientation(q)

    return {
        "upper_vector_l2": float(np.linalg.norm(upper - upper_direction)),
        "lower_vector_l2": float(np.linalg.norm(lower - lower_direction)),
        "tool_rotation_fro": float(np.linalg.norm(T - H, ord="fro")),
        "upper_cosine": float(np.clip(np.dot(upper, upper_direction), -1.0, 1.0)),
        "lower_cosine": float(np.clip(np.dot(lower, lower_direction), -1.0, 1.0)),
    }

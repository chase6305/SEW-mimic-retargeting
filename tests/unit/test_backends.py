from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from sew_mimic import (
    Serial7DoF,
    SEWMimicError,
    backend_status,
    cpp_backend_available,
    equivalent_angles_in_limits,
    get_backend,
    rot,
    solve,
    solve_batch,
)
from sew_mimic.backends import PythonBackend


def make_robot():
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


def test_default_backend_is_python(monkeypatch):
    monkeypatch.delenv("SEW_MIMIC_BACKEND", raising=False)
    assert get_backend().name == "python"


def test_environment_selects_backend(monkeypatch):
    monkeypatch.setenv("SEW_MIMIC_BACKEND", "python")
    assert get_backend().name == "python"
    assert get_backend("python") is get_backend("python")
    assert backend_status()["default"] == "python"


def test_explicit_python_backend_matches_reference():
    robot = make_robot()
    q_reference = np.array([0.2, -0.3, 0.4, -0.5, 0.15, 0.25, -0.1])
    shoulder = np.zeros(3)
    elbow = 0.287 * robot.axis_world(q_reference, 3)
    wrist = elbow + 0.314 * robot.axis_world(q_reference, 5)
    hand = robot.tool_orientation(q_reference)
    result = solve(robot, np.zeros(7), shoulder, elbow, wrist, hand, backend="python")
    direct = PythonBackend().solve(robot, np.zeros(7), shoulder, elbow, wrist, hand)
    assert np.allclose(result, direct)


@pytest.mark.parametrize("backend", ["python", "cpp"])
def test_solver_selects_nearby_equivalents_for_multi_turn_joint_limits(backend):
    if backend == "cpp" and not cpp_backend_available():
        pytest.skip("native extension is not built")
    robot = make_robot()
    robot.q_min[:] = -20.0 * np.pi
    robot.q_max[:] = 20.0 * np.pi
    target = np.array([0.3, 0.4, -0.2, 0.7, 0.1, -0.5, 0.2])
    target += 2.0 * np.pi * np.array([6, -5, 4, -3, 2, -1, 1])
    shoulder = np.zeros(3)
    elbow = 0.287 * robot.axis_world(target, 3)
    wrist = elbow + 0.314 * robot.axis_world(target, 5)
    result = solve(
        robot,
        target + 0.01,
        shoulder,
        elbow,
        wrist,
        robot.tool_orientation(target),
        backend=backend,
    )
    np.testing.assert_allclose(result, target, rtol=0, atol=1e-12)


@pytest.mark.parametrize("backend", ["python", "cpp"])
@pytest.mark.parametrize("reference", [-1e18, 1e18])
def test_both_solvers_preserve_enumerated_distance_plateaus(backend, reference, monkeypatch):
    if backend == "cpp" and not cpp_backend_available():
        pytest.skip("native extension is not built")
    robot = make_robot()
    robot.q_min[:] = -1000.0
    robot.q_max[:] = 1000.0
    target = np.array([0.3, 0.4, -0.2, 0.7, 0.1, -0.5, 0.2])
    shoulder = np.zeros(3)
    elbow = 0.287 * robot.axis_world(target, 3)
    wrist = elbow + 0.314 * robot.axis_world(target, 5)
    hand = robot.tool_orientation(target)
    current = np.full(7, reference)

    def enumerated_nearest(theta, lower, upper, previous):
        values = equivalent_angles_in_limits(theta, lower, upper, previous)
        return values[0] if values else None

    # Use the unchanged public enumerator throughout all coupled solve stages
    # as the oracle; large references make several turns' float costs equal.
    with monkeypatch.context() as context:
        context.setattr("sew_mimic.solver._nearest_equivalent_angle", enumerated_nearest)
        expected = solve(robot, current, shoulder, elbow, wrist, hand, backend="python")
    actual = solve(robot, current, shoulder, elbow, wrist, hand, backend=backend)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-10)
    assert np.all(np.isfinite(actual))
    assert np.all(actual >= robot.q_min - 1e-10)
    assert np.all(actual <= robot.q_max + 1e-10)


@pytest.mark.parametrize("backend", ["python", "cpp"])
@pytest.mark.parametrize("limit", [1e12, 1e300])
def test_both_solvers_accept_wide_finite_limits_without_enumerating_turns(backend, limit):
    if backend == "cpp" and not cpp_backend_available():
        pytest.skip("native extension is not built")
    robot = make_robot()
    robot.q_min[:] = -limit
    robot.q_max[:] = limit
    target = np.array([0.3, 0.4, -0.2, 0.7, 0.1, -0.5, 0.2])
    shoulder = np.zeros(3)
    elbow = 0.287 * robot.axis_world(target, 3)
    wrist = elbow + 0.314 * robot.axis_world(target, 5)
    hand = robot.tool_orientation(target)
    with np.errstate(all="raise"):
        actual = solve(robot, target + 0.01, shoulder, elbow, wrist, hand, backend=backend)
    np.testing.assert_allclose(actual, target, rtol=0.0, atol=1e-10)
    assert np.all(np.isfinite(actual))


def test_empty_python_batch_preserves_documented_output_shape():
    robot = make_robot()
    result = solve_batch(
        robot,
        np.zeros(7),
        np.empty((0, 3)),
        np.empty((0, 3)),
        np.empty((0, 3)),
        np.empty((0, 3, 3)),
        backend="python",
    )
    assert result.shape == (0, 7)
    assert result.dtype == np.float64


@pytest.mark.parametrize(
    ("q0", "message"),
    [
        (np.zeros(6), r"q0 must have shape \(7,\)"),
        (np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, np.nan]), "finite"),
    ],
)
def test_batch_rejects_invalid_initial_configuration(q0, message):
    robot = make_robot()
    with pytest.raises(ValueError, match=message):
        solve_batch(
            robot,
            q0,
            np.empty((0, 3)),
            np.empty((0, 3)),
            np.empty((0, 3)),
            np.empty((0, 3, 3)),
            backend="python",
        )


def test_batch_rejects_non_finite_keypoints_before_backend_dispatch():
    robot = make_robot()
    shoulders = np.zeros((1, 3))
    shoulders[0, 0] = np.inf
    with pytest.raises(ValueError, match="finite"):
        solve_batch(
            robot,
            np.zeros(7),
            shoulders,
            np.ones((1, 3)),
            np.full((1, 3), 2.0),
            np.eye(3)[None, :, :],
            backend="python",
        )


def test_explicit_missing_cpp_backend_fails_clearly():
    if cpp_backend_available():
        pytest.skip("native extension is installed")
    with pytest.raises(RuntimeError, match=r"C\+\+ backend is not installed"):
        get_backend("cpp")


@pytest.mark.skipif(not cpp_backend_available(), reason="native extension is not built")
def test_native_cpp_backend_matches_python_reference():
    robot = make_robot()
    cpp_backend = get_backend("cpp")
    assert cpp_backend is get_backend("cpp")
    q_current = np.zeros(7)
    rng = np.random.default_rng(42)
    for _ in range(100):
        q_reference = rng.uniform(-1.0, 1.0, size=7)
        shoulder = np.zeros(3)
        elbow = 0.287 * robot.axis_world(q_reference, 3)
        wrist = elbow + 0.314 * robot.axis_world(q_reference, 5)
        hand = robot.tool_orientation(q_reference)
        python_result = solve(robot, q_current, shoulder, elbow, wrist, hand, backend="python")
        cpp_result = solve(robot, q_current, shoulder, elbow, wrist, hand, backend="cpp")
        assert np.allclose(cpp_result, python_result, atol=1e-10)
        q_current = python_result
    assert len(cpp_backend._solvers) >= 1
    assert backend_status()["cpp_implementation"] == "native"


@pytest.mark.skipif(not cpp_backend_available(), reason="native extension is not built")
def test_native_batch_matches_python_sequential_solver():
    robot = make_robot()
    rng = np.random.default_rng(7)
    references = rng.uniform(-0.8, 0.8, size=(64, 7))
    shoulders = np.zeros((len(references), 3))
    elbows = np.asarray([0.287 * robot.axis_world(q, 3) for q in references])
    wrists = elbows + np.asarray([0.314 * robot.axis_world(q, 5) for q in references])
    hands = np.asarray([robot.tool_orientation(q) for q in references])
    python_result = solve_batch(
        robot, np.zeros(7), shoulders, elbows, wrists, hands, backend="python"
    )
    cpp_result = solve_batch(robot, np.zeros(7), shoulders, elbows, wrists, hands, backend="cpp")
    assert np.allclose(cpp_result, python_result, atol=1e-10)


@pytest.mark.skipif(not cpp_backend_available(), reason="native extension is not built")
def test_native_solver_rejects_invalid_runtime_inputs():
    robot = make_robot()
    with pytest.raises(SEWMimicError, match="valid rotation matrix"):
        solve(
            robot,
            np.zeros(7),
            np.zeros(3),
            np.array([1.0, 0.0, 0.0]),
            np.array([2.0, 0.0, 0.0]),
            np.zeros((3, 3)),
            backend="cpp",
        )
    with pytest.raises(SEWMimicError, match="finite"):
        solve(
            robot,
            np.zeros(7),
            np.zeros(3),
            np.array([np.nan, 0.0, 0.0]),
            np.array([2.0, 0.0, 0.0]),
            np.eye(3),
            backend="cpp",
        )


@pytest.mark.skipif(not cpp_backend_available(), reason="native extension is not built")
def test_native_solver_is_safe_for_parallel_arm_workers():
    robot = make_robot()
    rng = np.random.default_rng(17)
    references = rng.uniform(-0.6, 0.6, size=(128, 7))
    shoulders = np.zeros((len(references), 3))
    elbows = np.asarray([0.287 * robot.axis_world(q, 3) for q in references])
    wrists = elbows + np.asarray([0.314 * robot.axis_world(q, 5) for q in references])
    hands = np.asarray([robot.tool_orientation(q) for q in references])

    def worker():
        return solve_batch(
            robot,
            np.zeros(7),
            shoulders,
            elbows,
            wrists,
            hands,
            backend="cpp",
        )

    expected = worker()
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(lambda _: worker(), range(8)))
    assert all(np.allclose(result, expected, atol=1e-12) for result in results)


def test_invalid_backend_name_is_rejected():
    with pytest.raises(ValueError, match="python, cpp"):
        get_backend("cuda")


@pytest.mark.parametrize("backend", ["python", "cpp"])
@pytest.mark.parametrize("direct", [False, True])
@pytest.mark.parametrize("invalid", ["scalar", "shape", "nonfinite", "empty_q0"])
def test_batch_validation_is_consistent_for_all_entry_points(backend, direct, invalid):
    if backend == "cpp" and not cpp_backend_available():
        pytest.skip("native extension is not built")
    q0 = np.zeros(7)
    shoulders = np.empty((0, 3))
    if invalid == "scalar":
        shoulders = 0.0
    elif invalid == "shape":
        shoulders = np.empty((0, 4))
    elif invalid == "nonfinite":
        q0[0] = np.nan
    else:
        q0 = np.zeros(6)
    args = (make_robot(), q0, shoulders, np.empty((0, 3)), np.empty((0, 3)), np.empty((0, 3, 3)))
    with pytest.raises(ValueError, match="shape|finite"):
        if direct:
            get_backend(backend).solve_batch(*args)
        else:
            solve_batch(*args, backend=backend)


@pytest.mark.parametrize("backend", ["python", "cpp"])
def test_solver_is_invariant_to_local_joint_frame_conventions(backend):
    if backend == "cpp" and not cpp_backend_available():
        pytest.skip("native extension is not built")
    rng = np.random.default_rng(302)
    reference = make_robot()
    frames = [np.eye(3)] + [rot(rng.normal(size=3), rng.uniform(-3, 3)) for _ in range(7)]
    calibrated = Serial7DoF(
        axes_local=np.array(
            [frames[i + 1].T @ axis for i, axis in enumerate(reference.axes_local)]
        ),
        R_local=np.array([frames[i].T @ frames[i + 1] for i in range(7)]),
        q_min=reference.q_min,
        q_max=reference.q_max,
        R_7T_local=frames[7].T,
    )
    for _ in range(30):
        target = rng.uniform(-1.0, 1.0, size=7)
        current = rng.uniform(-1.0, 1.0, size=7)
        shoulder = np.zeros(3)
        elbow = 0.3 * reference.axis_world(target, 3)
        wrist = elbow + 0.25 * reference.axis_world(target, 5)
        hand = reference.tool_orientation(target)
        result = solve(calibrated, current, shoulder, elbow, wrist, hand, backend=backend)
        expected = solve(reference, current, shoulder, elbow, wrist, hand, backend="python")
        np.testing.assert_allclose(result, expected, atol=1e-9)
        np.testing.assert_allclose(calibrated.tool_orientation(result), hand, atol=1e-9)

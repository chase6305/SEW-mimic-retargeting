from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from sew_mimic import (
    Serial7DoF,
    SEWMimicError,
    backend_status,
    cpp_backend_available,
    get_backend,
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

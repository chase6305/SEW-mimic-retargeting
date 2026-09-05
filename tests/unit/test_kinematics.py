from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from sew_mimic import cpp_backend_available
from sew_mimic.robots import (
    URDFBimanualPoseEvaluator,
    URDFKinematics,
    load_robot_arm,
    resolve_robot_urdf,
)


@pytest.fixture(params=["python", "cpp"])
def backend(request):
    if request.param == "cpp" and not cpp_backend_available():
        pytest.skip("native extension is not built")
    return request.param


@pytest.fixture
def tree(tmp_path):
    path = tmp_path / "tree.urdf"
    path.write_text("""<robot name="mixed">
      <link name="base"/><link name="mount"/><link name="arm"/>
      <link name="tip"/><link name="slide"/><link name="sensor"/>
      <joint name="tool" type="fixed"><parent link="arm"/><child link="tip"/>
        <origin xyz="0.3 0.1 0.2" rpy="0.2 -0.3 0.4"/></joint>
      <joint name="mount" type="fixed"><parent link="base"/><child link="mount"/>
        <origin xyz="1 2 3" rpy="0.3 0.2 0.1"/></joint>
      <joint name="turn" type="continuous"><parent link="mount"/><child link="arm"/>
        <origin xyz="0 0.2 0.1" rpy="0.1 0.3 -0.2"/><axis xyz="1 2 3"/></joint>
      <joint name="slide" type="prismatic"><parent link="mount"/><child link="slide"/>
        <origin xyz="0.5 0 0" rpy="0.3 0.7 -0.4"/><axis xyz="0 2 0"/></joint>
      <joint name="sensor" type="revolute"><parent link="slide"/><child link="sensor"/>
        <origin xyz="0 0 0.3" rpy="0.4 -0.3 0.2"/><axis xyz="0 0 1"/></joint>
    </robot>""")
    return URDFKinematics(path)


def test_prepared_fk_matches_tree_with_folded_fixed_and_uncommanded_joints(tree, backend):
    names = ("slide", "turn")  # Deliberately differs from tree traversal order.
    links = ("tip", "base", "sensor", "tip", "mount")
    model = tree.compile(names, links, backend=backend)
    assert model.joint_names == names
    assert model.link_names == links
    for values in np.random.default_rng(40).uniform(-2, 2, size=(50, 2)):
        reference = tree.link_transforms(dict(zip(names, values)))
        actual = model.evaluate(values)
        np.testing.assert_allclose(actual, [reference[link] for link in links], atol=1e-13)


def test_prepared_fk_handles_static_root_empty_targets_and_pruned_commands(tree, backend):
    static = tree.compile((), ("sensor", "base"), backend=backend)
    reference = tree.link_transforms({})
    np.testing.assert_allclose(
        static.evaluate([]), [reference["sensor"], reference["base"]], atol=1e-14
    )
    empty = tree.compile(("turn",), (), backend=backend)
    assert empty.evaluate([1.0]).shape == (0, 4, 4)
    pruned = tree.compile(("turn", "slide"), ("arm",), backend=backend)
    np.testing.assert_allclose(
        pruned.evaluate([0.5, 100]), pruned.evaluate([0.5, -100]), atol=1e-14
    )


def test_prepared_fk_owns_its_model_and_output(tree, backend):
    model = tree.compile(("turn",), ("tip",), backend=backend)
    expected = model.evaluate([0.4])
    # Changing the parsed model or an old result cannot mutate the prepared model.
    tree.joints[0][4][:3, 3] = 100.0
    result = model.evaluate([0.4])
    result[:] = 0.0
    np.testing.assert_array_equal(model.evaluate([0.4]), expected)
    updated = tree.compile(("turn",), ("tip",), backend=backend)
    assert not np.allclose(updated.evaluate([0.4]), expected)


@pytest.mark.parametrize("values", [0.0, [], [0, 1], [[0.0]], [np.nan], [np.inf]])
def test_prepared_fk_rejects_invalid_commands(tree, backend, values):
    model = tree.compile(("turn",), ("tip",), backend=backend)
    with pytest.raises(ValueError, match="shape|finite"):
        model.evaluate(values)


def test_prepared_fk_checks_names_and_backend(tree):
    with pytest.raises(ValueError, match="unique"):
        tree.compile(("turn", "turn"), ("tip",))
    with pytest.raises(KeyError, match="Unknown movable"):
        tree.compile(("missing",), ("tip",))
    with pytest.raises(KeyError, match="Unknown URDF links"):
        tree.compile(("turn",), ("missing",))
    with pytest.raises(ValueError, match="backend"):
        tree.compile(("turn",), ("tip",), backend="invalid")


@pytest.mark.parametrize("invalid", ["nonfinite", "nonrigid"])
def test_prepared_fk_rejects_invalid_calibration(tree, backend, invalid):
    if invalid == "nonfinite":
        tree.joints[0][4][0, 3] = np.nan
    else:
        tree.joints[0][4][:3, :3] *= 2.0
    with pytest.raises(ValueError, match="finite|rigid"):
        tree.compile(("turn",), ("tip",), backend=backend)


def test_prepared_fk_is_safe_for_independent_workers(tree, backend):
    model = tree.compile(("turn", "slide"), ("tip", "sensor"), backend=backend)
    commands = np.random.default_rng(4).normal(size=(128, 2))
    expected = [model.evaluate(command) for command in commands]
    with ThreadPoolExecutor(max_workers=4) as pool:
        actual = list(pool.map(model.evaluate, commands))
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("robot", ["marvin", "openarm"])
def test_bimanual_prepared_fk_matches_full_urdf_for_both_robots(robot, backend):
    kinematics = URDFKinematics(resolve_robot_urdf(robot))
    left, right = (load_robot_arm(robot, side=side) for side in ("left", "right"))
    evaluator = URDFBimanualPoseEvaluator(kinematics, left, right, backend=backend)
    names = (*left.joint_names, *right.joint_names)
    for values in np.random.default_rng(75).uniform(-1, 1, size=(50, 14)):
        reference = kinematics.link_transforms(dict(zip(names, values)))
        pose = evaluator.evaluate(values[:7], values[7:])
        expected = [reference[link][:3, 3] for link in evaluator._required_links]
        np.testing.assert_allclose(pose.keypoints(), expected, atol=1e-13)
        for arm, result in ((left, pose.left), (right, pose.right)):
            np.testing.assert_allclose(
                result.tool_orientation,
                reference[arm.ee_link][:3, :3] @ arm.robot.R_align,
                atol=1e-13,
            )
    with pytest.raises(ValueError, match="shape"):
        evaluator.evaluate(np.zeros(6), np.zeros(6))
    with pytest.raises(ValueError, match="finite"):
        evaluator.evaluate(np.full(7, np.nan), np.zeros(7))

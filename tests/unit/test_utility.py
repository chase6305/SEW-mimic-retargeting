import math

import numpy as np
import pytest

from sew_mimic import (
    DegenerateGeometryError,
    as_vec3,
    equivalent_angles_in_limits,
    is_rotation_matrix,
    make_frame,
    rot,
    unit,
)


def test_as_vec3_rejects_non_finite_values():
    with pytest.raises(ValueError, match="finite"):
        as_vec3([0.0, np.nan, 1.0])


def test_unit_rejects_degenerate_vector():
    with pytest.raises(DegenerateGeometryError):
        unit([0.0, 0.0, 0.0])


def test_rotation_validation():
    assert is_rotation_matrix(rot([0.0, 0.0, 1.0], 0.3))
    assert not is_rotation_matrix(np.diag([1.0, 1.0, -1.0]))
    assert not is_rotation_matrix(np.full((3, 3), np.nan))


@pytest.mark.parametrize("theta", [0.0, 1e-12, -1e-6, 0.7, -2.4, math.pi, 2.0 * math.pi])
def test_rodrigues_rotation_matches_matrix_identity(theta):
    for axis in ([1.0, 0.0, 0.0], [0.0, -7.0, 0.0], [1.0, -2.0, 3.0]):
        normalized = np.asarray(axis) / np.linalg.norm(axis)
        cross_matrix = np.cross(normalized, np.eye(3), axis=0)
        expected = (
            np.eye(3)
            + math.sin(theta) * cross_matrix
            + (1.0 - math.cos(theta)) * (cross_matrix @ cross_matrix)
        )
        actual = rot(axis, theta)
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=5e-16)
        np.testing.assert_allclose(actual @ normalized, normalized, rtol=0.0, atol=5e-16)
        np.testing.assert_allclose(actual.T @ actual, np.eye(3), rtol=0.0, atol=2e-15)
        assert np.linalg.det(actual) == pytest.approx(1.0, abs=2e-15)
        assert actual.dtype == np.float64


def test_rodrigues_rotation_obeys_right_hand_rule_and_axis_scaling():
    expected = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    np.testing.assert_allclose(rot([0.0, 0.0, 12.0], math.pi / 2), expected, atol=1e-15)
    np.testing.assert_allclose(rot([0.0, 0.0, -12.0], -math.pi / 2), expected, atol=1e-15)


@pytest.mark.parametrize(
    "axis", [[1.0, 0.0], [1.0, 0.0, 0.0, 0.0], [np.nan, 0.0, 1.0], [0.0, np.inf, 1.0]]
)
def test_rodrigues_rotation_rejects_invalid_axes(axis):
    with pytest.raises(ValueError, match="3-vector|finite"):
        rot(axis, 0.3)


@pytest.mark.parametrize("axis", [[0.0, 0.0, 0.0], [1e-12, 0.0, 0.0]])
def test_rodrigues_rotation_rejects_near_zero_axes(axis):
    with pytest.raises(DegenerateGeometryError, match="near-zero"):
        rot(axis, 0.3)


def test_equivalent_angles_are_sorted_around_reference():
    values = equivalent_angles_in_limits(0.2, -3 * math.pi, 3 * math.pi, 6.4)
    assert values == pytest.approx([0.2 + 2 * math.pi, 0.2, 0.2 - 2 * math.pi])


def test_make_frame_is_orthonormal():
    rotation, origin = make_frame([0.0, 1.0, 1.0], [0.0, -1.0, 1.0], [0.0, 0.0, 0.0])
    assert is_rotation_matrix(rotation)
    assert origin == pytest.approx([0.0, 0.0, 1.0])

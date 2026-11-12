"""Geometric objective checks for the public scalar subproblem kernels."""

import math

import numpy as np
import pytest

from sew_mimic import rot, sp1, sp4
from sew_mimic.utility import EPS


def _cylindrical_basis():
    axis = np.array([2.0, -1.0, 3.0])
    axis /= np.linalg.norm(axis)
    first = np.cross(axis, [0.0, 1.0, 0.0])
    first /= np.linalg.norm(first)
    return first, np.cross(axis, first), axis


@pytest.mark.parametrize("radius,height,is_ls", [(2.3, 0.4, False), (0.7, -0.2, True)])
def test_sp1_recovers_cylindrical_angle_and_known_minimum(radius, height, is_ls):
    first, second, axis = _cylindrical_basis()
    angle = -0.73
    source = 2.3 * first + 0.4 * axis
    target = radius * (math.cos(angle) * first + math.sin(angle) * second) + height * axis
    actual, actual_ls = sp1(source, target, 7.0 * axis)
    assert actual == pytest.approx(angle, rel=0.0, abs=1e-14)
    assert actual_ls is is_ls
    # Rotation preserves axial position and perpendicular radius; these two
    # differences are orthogonal and determine the global minimum analytically.
    minimum = math.hypot(2.3 - radius, 0.4 - height)
    residual = np.linalg.norm(rot(axis, actual) @ source - target)
    assert residual == pytest.approx(minimum, rel=0.0, abs=2e-15)


@pytest.mark.parametrize("reverse", [False, True])
def test_sp1_zero_projected_objective_selects_zero_for_continuity(reverse):
    source, target = np.array([0.0, 0.0, 2.0]), np.array([3.0, 4.0, 1.0])
    if reverse:
        source, target = target, source
    angle, is_ls = sp1(source, target, [0.0, 0.0, 4.0])
    assert angle == 0.0
    assert is_ls
    assert np.linalg.norm(rot([0, 0, 1], angle) @ source - target) == pytest.approx(
        math.sqrt(26.0), rel=0.0, abs=1e-14
    )


@pytest.mark.parametrize("magnitude,angle", [(0.5 * EPS, 0.0), (2.0 * EPS, math.pi / 2)])
def test_sp1_preserves_small_projection_threshold(magnitude, angle):
    actual, is_ls = sp1([1, 0, 0], [0, magnitude, 0], [0, 0, 3])
    assert actual == pytest.approx(angle, rel=0.0, abs=1e-15)
    assert is_ls  # The source and target have different lengths.


@pytest.mark.parametrize(
    "fraction,count,is_ls",
    [(-1.4, 1, True), (-1, 1, False), (0.3, 2, False), (1, 1, False), (1.4, 1, True)],
)
def test_sp4_attains_known_circle_plane_objective(fraction, count, is_ls):
    first, second, axis = _cylindrical_basis()
    phase, radius, height = 0.41, 2.5, -0.7
    normal = 0.6 * first + 0.8 * axis
    point = radius * (math.cos(phase) * first + math.sin(phase) * second) + height * axis
    amplitude, offset = 0.6 * radius, 0.8 * height
    target = offset + fraction * amplitude
    angles, actual_ls = sp4(9.0 * normal, point, 4.0 * axis, target)
    assert len(angles) == count
    assert actual_ls is is_ls
    assert angles.dtype == np.float64
    # In this basis the plane coordinate is offset + amplitude*cos(phase+theta).
    # Its attainable interval gives both the exact and least-squares minima.
    minimum = max(0.0, abs(target - offset) - amplitude)
    for angle in angles:
        projected = float(normal @ (rot(axis, angle) @ point))
        assert abs(projected - target) == pytest.approx(minimum, rel=0.0, abs=2e-14)
        assert projected == pytest.approx(
            offset + amplitude * math.cos(phase + angle), rel=0.0, abs=2e-14
        )
    if count == 2:
        sine = math.sqrt(1.0 - fraction * fraction)
        np.testing.assert_allclose(
            np.sort(np.sin(phase + angles)), [-sine, sine], rtol=0.0, atol=2e-14
        )


@pytest.mark.parametrize("residual,is_ls", [(0.0, False), (0.5e-8, False), (2.0e-8, True)])
def test_sp4_constant_objective_preserves_residual_flag_threshold(residual, is_ls):
    angles, actual_ls = sp4([0, 0, 5], [2, -1, 3], [0, 0, 7], 3.0 + residual)
    np.testing.assert_array_equal(angles, [0.0])
    assert actual_ls is is_ls


def test_sp4_axis_parallel_point_has_constant_plane_coordinate():
    angles, is_ls = sp4([3, 0, 4], [0, 0, 2], [0, 0, 7], 1.0)
    np.testing.assert_array_equal(angles, [0.0])
    assert is_ls  # Every rotation leaves the plane coordinate at 1.6.


@pytest.mark.parametrize("scale,count,is_ls", [(0.5, 1, True), (2.0, 2, False)])
def test_sp4_preserves_near_constant_projection_threshold(scale, count, is_ls):
    radius = scale * math.sqrt(EPS)
    angles, actual_ls = sp4([1, 0, 0], [radius, 0, 0], [0, 0, 3], 0.0)
    assert len(angles) == count
    assert actual_ls is is_ls
    if count == 1:
        np.testing.assert_array_equal(angles, [0.0])
    else:
        assert np.max(np.abs(np.cos(angles))) < 1e-14


@pytest.mark.parametrize("gap,count", [(0.25e-12, 1), (4.0e-12, 2)])
def test_sp4_preserves_near_tangent_solution_count(gap, count):
    target = math.sqrt(1.0 - gap)
    angles, is_ls = sp4([1, 0, 0], [1, 0, 0], [0, 0, 2], target)
    assert len(angles) == count
    assert not is_ls
    assert np.max(np.abs(np.cos(angles) - target)) <= 1e-12
